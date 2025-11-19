"""Blend existing LightGBM models and CatBoost into a small stacking ensemble.

Strategy:
- Load processed arrays (train/test)
- Load selected feature indices from models/catboost_wrapper.joblib or selected_idx.joblib
- Load all LightGBM joblib models (ensemble_base_lgb_fold*, bag_lgb_*.joblib)
- Load CatBoost model file `models/catboost_model.cbm` if present
- Create a holdout split from the train set (20%) to train a meta LogisticRegression
- Meta features: mean LGB proba across LGB models, CatBoost proba
- Fit meta on holdout, tune threshold on holdout, evaluate on test
- Save results and meta model
"""
from pathlib import Path
import numpy as np
import joblib, json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)

def load_processed():
    tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
    te = ROOT / 'data' / 'processed' / 'test_processed.npz'
    if not tr.exists() or not te.exists():
        raise FileNotFoundError('Processed arrays not found; run preprocessing first')
    arr = np.load(tr, allow_pickle=True)
    arrt = np.load(te, allow_pickle=True)
    return arr['X'], arr['y'], arrt['X'], arrt['y']

X, y_raw, Xte, yte_raw = load_processed()

# Load selected indices (try several artifact locations)
sel_idx = None
for p in [MOD / 'catboost_wrapper.joblib', MOD / 'selected_idx.joblib', MOD / 'best_model_for_99.joblib', MOD / 'best_model_for_99.joblib']:
    if p.exists():
        try:
            w = joblib.load(p)
            if isinstance(w, dict) and 'selected_idx' in w:
                sel_idx = np.array(w['selected_idx'], dtype=int)
                break
        except Exception:
            continue

if sel_idx is None:
    # fallback: use first 800 features if available
    sel_idx = np.arange(min(800, X.shape[1]))

X_sel = X[:, sel_idx]
Xte_sel = Xte[:, sel_idx]

# map labels to binary using existing utility if possible
def to_binary(yarr):
    try:
        from src.run_best_binary import map_to_binary
        return map_to_binary(yarr)
    except Exception:
        # simple: treat string 'normal' as 0
        out = []
        for v in yarr:
            s = str(v).lower()
            out.append(0 if 'normal' in s else 1)
        return np.array(out, dtype=int)

y = to_binary(y_raw)
yte = to_binary(yte_raw)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
tr_idx, hold_idx = next(sss.split(X_sel, y))
X_tr, y_tr = X_sel[tr_idx], y[tr_idx]
X_hold, y_hold = X_sel[hold_idx], y[hold_idx]

# Load LGB models
import glob
lgb_models = []
print('Looking for LGB model files in', MOD)
print('ensemble_base_lgb_fold matches:', [str(x) for x in sorted(MOD.glob('ensemble_base_lgb_fold*.joblib'))])
print('bag_lgb matches:', [str(x) for x in sorted(MOD.glob('bag_lgb_*.joblib'))])
for p in sorted(MOD.glob('ensemble_base_lgb_fold*.joblib')):
    try:
        lgb_models.append(joblib.load(p))
    except Exception:
        continue
for p in sorted(MOD.glob('bag_lgb_*.joblib')):
    try:
        lgb_models.append(joblib.load(p))
    except Exception:
        continue

if len(lgb_models) == 0:
    raise RuntimeError('No LightGBM models found in models/')

# compute mean LGB proba on hold and test
def mean_proba(models, X_):
    preds = []
    for m in models:
        try:
            p = m.predict_proba(X_)[:,1]
        except Exception:
            # sklearn wrapper might require DataFrame; try that
            import pandas as pd
            p = m.predict_proba(pd.DataFrame(X_))[:,1]
        preds.append(p)
    return np.mean(np.column_stack(preds), axis=1)

lgb_hold = mean_proba(lgb_models, X_hold)
lgb_test = mean_proba(lgb_models, Xte_sel)

# Load CatBoost model if available
cat_proba_hold = np.zeros_like(lgb_hold)
cat_proba_test = np.zeros_like(lgb_test)
try:
    from catboost import CatBoostClassifier
    cb_path = MOD / 'catboost_model.cbm'
    if cb_path.exists():
        cb = CatBoostClassifier()
        cb.load_model(str(cb_path))
        # CatBoost expects same selected features; pass numpy
        cat_proba_hold = cb.predict_proba(X_hold)[:,1]
        cat_proba_test = cb.predict_proba(Xte_sel)[:,1]
    else:
        print('No catboost model file found; proceeding without it')
except Exception:
    print('CatBoost not available in this environment; proceeding without it')

X_meta_hold = np.column_stack([lgb_hold, cat_proba_hold])
X_meta_test = np.column_stack([lgb_test, cat_proba_test])

# Train meta learner
from sklearn.linear_model import LogisticRegression
meta = LogisticRegression(max_iter=1000)
meta.fit(X_meta_hold, y_hold)

# Tune threshold on hold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
proba_hold = meta.predict_proba(X_meta_hold)[:,1]
best_acc, best_t = -1, 0.5
for t in np.linspace(0.3, 0.7, 41):
    pr = (proba_hold >= t).astype(int)
    acc = accuracy_score(y_hold, pr)
    if acc > best_acc:
        best_acc, best_t = acc, float(t)

# Evaluate on test
proba_test = meta.predict_proba(X_meta_test)[:,1]
pred_test = (proba_test >= best_t).astype(int)
res = {
    'hold_best_threshold': best_t,
    'hold_best_acc': float(best_acc),
    'test_acc': float(accuracy_score(yte, pred_test)),
    'test_f1': float(f1_score(yte, pred_test)),
    'test_auc': float(roc_auc_score(yte, proba_test)),
    'n_lgb_models': len(lgb_models),
}

with open(RES / 'mixed_ensemble_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)
joblib.dump({'meta': meta, 'selected_idx': sel_idx.tolist(), 'best_threshold': best_t}, MOD / 'mixed_ensemble_meta.joblib')
print('Saved mixed ensemble results:', res)
