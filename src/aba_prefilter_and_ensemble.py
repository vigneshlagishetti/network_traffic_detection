"""
Run ABA on a prefiltered top-K (f_classif) of processed arrays and build an ensemble.

Outputs:
- models/aba_selected_indices.npy
- models/ensemble_lgb.joblib
- models/ensemble_xgb.joblib (if xgboost present)
- models/ensemble_stack_meta.joblib
- results/aba_ensemble_results.json

This script uses conservative defaults to keep runtime reasonable. Adjust K, pop_size, n_iter if you want.
"""
from pathlib import Path
import numpy as np
import joblib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz_tr.exists():
    raise FileNotFoundError('train_processed.npz not found; run preprocessing first')
if not npz_te.exists():
    raise FileNotFoundError('test_processed.npz not found; run preprocessing first')

arr_tr = np.load(npz_tr, allow_pickle=True)
arr_te = np.load(npz_te, allow_pickle=True)
X_tr_full, y_tr_raw = arr_tr['X'], arr_tr['y']
X_te_full, y_te_raw = arr_te['X'], arr_te['y']

def map_to_binary(yarr):
    try:
        svals = [str(v).lower() for v in yarr[:1000]] if len(yarr)>0 else []
        if any('normal' in s for s in svals):
            return np.array([0 if 'normal' in str(v).lower() else 1 for v in yarr], dtype=int)
    except Exception:
        pass
    try:
        y_int = np.array(yarr, dtype=int)
        maj = int(np.argmax(np.bincount(y_int)))
        return np.array([0 if int(v)==maj else 1 for v in y_int], dtype=int)
    except Exception:
        try:
            yf = np.array(yarr, dtype=float)
            return np.array([0 if float(v)==0.0 else 1 for v in yf], dtype=int)
        except Exception:
            first = yarr[0]
            return np.array([0 if v==first else 1 for v in yarr], dtype=int)

y_tr = map_to_binary(y_tr_raw)
y_te = map_to_binary(y_te_raw)
print('train binary counts:', dict(Counter(y_tr)))

# prefilter top-K using f_classif
from sklearn.feature_selection import f_classif
K = 500
F, p = f_classif(X_tr_full, y_tr)
topk_idx = np.argsort(F)[-K:][::-1]
X_tr_topk = X_tr_full[:, topk_idx]
X_te_topk = X_te_full[:, topk_idx]
print('Top-K shapes:', X_tr_topk.shape, X_te_topk.shape)

# run ABA on top-K
import sys
sys.path.insert(0, str(ROOT))
from src.feature_selection.aba import ArtificialButterfly

def fitness_lgb_cv(X_sub, y_sub):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    try:
        import lightgbm as lgb
    except Exception:
        raise RuntimeError('lightgbm required for ABA fitness')
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=1, n_jobs=1)
    scores = cross_val_score(clf, X_sub, y_sub, cv=skf, scoring='f1_macro', n_jobs=1)
    return float(np.mean(scores))

pop_size = 12
n_iter = 20
aba = ArtificialButterfly(pop_size=pop_size, n_iter=n_iter, random_state=1)
print('Running ABA (pop', pop_size, 'iter', n_iter, ') on top-K...')
best_mask_topk, best_score = aba.fit(X_tr_topk, y_tr, fitness_lgb_cv)
print('ABA done: best_score=', best_score, 'n_features=', int(best_mask_topk.sum()))

selected_topk_idx = topk_idx[best_mask_topk.astype(bool)]
np.save(MOD / 'aba_selected_indices.npy', selected_topk_idx)

# prepare selected matrices
X_tr_sel = X_tr_full[:, selected_topk_idx]
X_te_sel = X_te_full[:, selected_topk_idx]

# train LightGBM
from collections import Counter
cnt = Counter(y_tr.tolist()); total = len(y_tr)
class_w = {k: total/(len(cnt)*v) for k,v in cnt.items()}
sw_tr = np.array([class_w[int(yy)] for yy in y_tr])

import lightgbm as lgb
clf_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, random_state=42, n_jobs=1)
clf_lgb.fit(X_tr_sel, y_tr, sample_weight=sw_tr)
joblib.dump(clf_lgb, MOD / 'ensemble_lgb.joblib')

# try xgboost
has_xgb = False
try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    has_xgb = False

if has_xgb:
    clf_xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf_xgb.fit(X_tr_sel, y_tr)
    joblib.dump(clf_xgb, MOD / 'ensemble_xgb.joblib')

# stacking OOF
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

oof_preds = np.zeros((X_tr_sel.shape[0], 0))
test_preds = np.zeros((X_te_sel.shape[0], 0))

# LGB OOF
oof = np.zeros(X_tr_sel.shape[0])
test_fold_preds = np.zeros((X_te_sel.shape[0], skf.n_splits))
for i, (tr, val) in enumerate(skf.split(X_tr_sel, y_tr)):
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, random_state=42, n_jobs=1)
    clf.fit(X_tr_sel[tr], y_tr[tr], sample_weight=np.array([class_w[int(yy)] for yy in y_tr[tr]]))
    oof[val] = clf.predict_proba(X_tr_sel[val])[:,1]
    test_fold_preds[:, i] = clf.predict_proba(X_te_sel)[:,1]
oof_preds = np.column_stack([oof_preds, oof])
test_preds = np.column_stack([test_preds, test_fold_preds.mean(axis=1)])

if has_xgb:
    oof = np.zeros(X_tr_sel.shape[0])
    test_fold_preds = np.zeros((X_te_sel.shape[0], skf.n_splits))
    for i, (tr, val) in enumerate(skf.split(X_tr_sel, y_tr)):
        clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
        clf.fit(X_tr_sel[tr], y_tr[tr])
        oof[val] = clf.predict_proba(X_tr_sel[val])[:,1]
        test_fold_preds[:, i] = clf.predict_proba(X_te_sel)[:,1]
    oof_preds = np.column_stack([oof_preds, oof])
    test_preds = np.column_stack([test_preds, test_fold_preds.mean(axis=1)])

# meta learner
meta = LogisticRegression(max_iter=400)
meta.fit(oof_preds, y_tr)
joblib.dump(meta, MOD / 'ensemble_stack_meta.joblib')

meta_test_proba = meta.predict_proba(test_preds)[:,1]
meta_test_pred = (meta_test_proba >= 0.5).astype(int)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
res = {'aba_best_score': float(best_score), 'n_selected_features': int(len(selected_topk_idx))}
res.update({'lgb_test_acc': float(accuracy_score(y_te, (test_preds[:,0] >= 0.5).astype(int))), 'lgb_test_f1': float(f1_score(y_te, (test_preds[:,0] >= 0.5).astype(int)))})
if has_xgb:
    res.update({'xgb_test_acc': float(accuracy_score(y_te, (test_preds[:,1] >= 0.5).astype(int))), 'xgb_test_f1': float(f1_score(y_te, (test_preds[:,1] >= 0.5).astype(int)))})
res.update({'stack_test_acc': float(accuracy_score(y_te, meta_test_pred)), 'stack_test_f1': float(f1_score(y_te, meta_test_pred)), 'stack_test_auc': float(roc_auc_score(y_te, meta_test_proba))})

with open(RES / 'aba_ensemble_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)

print('ABA+ensemble results:', res)
