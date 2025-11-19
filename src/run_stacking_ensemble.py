"""
Stacking ensemble runner to push binary detection accuracy higher.

Steps:
- load processed train/test arrays
- map labels to binary
- load selected feature indices from existing wrapper (models/best_model_for_99.joblib) if present
- train base models (LightGBM and XGBoost if available) with OOF CV and collect OOF preds
- train meta learner (LogisticRegression) on OOF preds, tune threshold for accuracy
- retrain base models on full train, save final ensemble wrapper and results

Outputs:
- models/ensemble_base_lgb_fold{fold}.joblib
- models/ensemble_base_xgb_fold{fold}.joblib (if xgboost)
- models/ensemble_meta.joblib
- results/ensemble_stack_results.json
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

# load processed
tr_npz = ROOT / 'data' / 'processed' / 'train_processed.npz'
te_npz = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not tr_npz.exists():
    raise FileNotFoundError('train_processed.npz not found')
arr = np.load(tr_npz, allow_pickle=True)
X = arr['X']; y = arr['y']
y_bin = map_to_binary(y)
print('Train', X.shape, 'classes', dict(Counter(y_bin)))

# try to load selected indices
sel_idx = None
wrapper_p = MOD / 'best_model_for_99.joblib'
if wrapper_p.exists():
    try:
        w = joblib.load(wrapper_p)
        if isinstance(w, dict) and 'selected_idx' in w:
            sel_idx = np.array(w['selected_idx'], dtype=int)
            print('Loaded selected_idx length', sel_idx.shape[0])
    except Exception as e:
        print('Could not load selected_idx from wrapper:', e)

if sel_idx is None:
    # fallback: top-K prefilter same as before
    from sklearn.feature_selection import f_classif
    K = 800
    F, pval = f_classif(X, y_bin)
    sel_idx = np.argsort(F)[-K:][::-1]
    print('Computed top-K sel_idx length', sel_idx.shape[0])

X_sel = X[:, sel_idx]

# class weights
from collections import Counter as _C
cnt = _C(y_bin.tolist()); total = len(y_bin)
class_w = {k: total/(len(cnt)*v) for k,v in cnt.items()}

# prepare CV
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# base models: LGB
import lightgbm as lgb
lgb_params = {'n_estimators':800, 'learning_rate':0.02, 'num_leaves':63, 'random_state':42, 'n_jobs':1, 'force_row_wise':True}

oof_preds = np.zeros((X_sel.shape[0], 0))
test_probas = None

# OOF for LGB
oof_lgb = np.zeros(X_sel.shape[0])
test_p_lgb = []
for fold, (tr, val) in enumerate(skf.split(X_sel, y_bin)):
    print('LGB fold', fold)
    sw = np.array([class_w[int(yy)] for yy in y_bin[tr]])
    clf = lgb.LGBMClassifier(**lgb_params)
    # use callbacks when available to silence verbose logs
    try:
        cb = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    except Exception:
        cb = []
    clf.fit(X_sel[tr], y_bin[tr], sample_weight=sw, eval_set=[(X_sel[val], y_bin[val])], eval_metric='auc', callbacks=cb)
    try:
        oof_lgb[val] = clf.predict_proba(X_sel[val])[:,1]
    except Exception:
        oof_lgb[val] = clf.predict(X_sel[val])
    # save fold model
    joblib.dump(clf, MOD / f"ensemble_base_lgb_fold{fold}.joblib")
    # collect test predictions for fold
    try:
        test_p = clf.predict_proba(X_sel)[:,1]
    except Exception:
        test_p = clf.predict(X_sel)
    test_p_lgb.append(test_p)

oof_preds = np.column_stack([oof_preds, oof_lgb])
test_mean_lgb = np.column_stack(test_p_lgb).mean(axis=1)
test_probas = test_mean_lgb.reshape(-1,1)

# try XGBoost
has_xgb = False
try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    has_xgb = False

if has_xgb:
    oof_xgb = np.zeros(X_sel.shape[0])
    test_p_xgb = []
    for fold, (tr,val) in enumerate(skf.split(X_sel, y_bin)):
        print('XGB fold', fold)
        clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
        clf.fit(X_sel[tr], y_bin[tr])
        try:
            oof_xgb[val] = clf.predict_proba(X_sel[val])[:,1]
        except Exception:
            oof_xgb[val] = clf.predict(X_sel[val])
        joblib.dump(clf, MOD / f"ensemble_base_xgb_fold{fold}.joblib")
        try:
            test_p = clf.predict_proba(X_sel)[:,1]
        except Exception:
            test_p = clf.predict(X_sel)
        test_p_xgb.append(test_p)
    oof_preds = np.column_stack([oof_preds, oof_xgb])
    test_probas = np.column_stack([test_probas, np.column_stack(test_p_xgb).mean(axis=1)])

# meta learner
from sklearn.linear_model import LogisticRegression
meta = LogisticRegression(max_iter=400)
meta.fit(oof_preds, y_bin)

# find best threshold on training OOF
from sklearn.metrics import accuracy_score
best_t, best_acc = 0.5, 0.0
proba_oof = meta.predict_proba(oof_preds)[:,1]
for t in np.linspace(0.01, 0.99, 99):
    acc = accuracy_score(y_bin, (proba_oof >= t).astype(int))
    if acc > best_acc:
        best_acc = acc; best_t = t
print('Best OOF threshold', best_t, 'acc', best_acc)

# evaluate on processed test set
res = {}
if te_npz.exists():
    arrt = np.load(te_npz, allow_pickle=True)
    Xte = arrt['X']; yte = arrt['y']
    yte_bin = map_to_binary(yte)
    Xte_sel = Xte[:, sel_idx]
    # build base test columns by loading fold models
    base_test_cols = []
    # LGB folds
    lgb_fold_preds = []
    for fold in range(skf.n_splits):
        mpath = MOD / f"ensemble_base_lgb_fold{fold}.joblib"
        if mpath.exists():
            clf = joblib.load(mpath)
            try:
                lgb_fold_preds.append(clf.predict_proba(Xte_sel)[:,1])
            except Exception:
                lgb_fold_preds.append(clf.predict(Xte_sel))
    if lgb_fold_preds:
        lgb_mean = np.column_stack(lgb_fold_preds).mean(axis=1)
        base_test_cols.append(lgb_mean)
    if has_xgb:
        xgb_fold_preds = []
        for fold in range(skf.n_splits):
            mpath = MOD / f"ensemble_base_xgb_fold{fold}.joblib"
            if mpath.exists():
                clf = joblib.load(mpath)
                try:
                    xgb_fold_preds.append(clf.predict_proba(Xte_sel)[:,1])
                except Exception:
                    xgb_fold_preds.append(clf.predict(Xte_sel))
        if xgb_fold_preds:
            base_test_cols.append(np.column_stack(xgb_fold_preds).mean(axis=1))

    if base_test_cols:
        X_meta_test = np.column_stack(base_test_cols)
        meta_proba_test = meta.predict_proba(X_meta_test)[:,1]
        thr = best_t
        pred = (meta_proba_test >= thr).astype(int)
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        res = {'test_acc': float(accuracy_score(yte_bin, pred)), 'test_f1': float(f1_score(yte_bin, pred)), 'test_auc': float(roc_auc_score(yte_bin, meta_proba_test)), 'best_threshold': float(thr)}
    else:
        res = {'note': 'no base test preds available'}
else:
    res = {'note': 'no test npz found'}

# save meta and results
joblib.dump(meta, MOD / 'ensemble_meta.joblib')
with open(RES / 'ensemble_stack_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)
print('Saved ensemble results to', RES / 'ensemble_stack_results.json')
