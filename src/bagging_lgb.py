"""
Bagging ensemble of LightGBM models trained on the full training set with different seeds and random feature/bagging fractions.

Workflow:
- load processed train/test arrays
- get selected feature indices (from models/best_model_for_99.joblib or selected_idx.joblib) or compute top-K
- split train into train/val (StratifiedShuffleSplit 90/10) to tune threshold
- train N LightGBM models with different seeds and randomized feature_fraction/bagging_fraction
- average validation predictions to find best threshold maximizing accuracy
- average test predictions to evaluate final accuracy/f1/auc
- save models and results

Outputs:
- models/bag_lgb_{i}.joblib
- results/bagging_results.json
"""
from pathlib import Path
import numpy as np
import joblib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

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

train_npz = DATA / 'train_processed.npz'
test_npz = DATA / 'test_processed.npz'
if not train_npz.exists():
    raise FileNotFoundError('train_processed.npz missing')
arr = np.load(train_npz, allow_pickle=True)
X = arr['X']; y = arr['y']
y_bin = map_to_binary(y)
print('Train', X.shape, 'labels', Counter(y_bin))

# load selected indices
sel_idx = None
best_wrapper = MODELS / 'best_model_for_99.joblib'
sel_job = MODELS / 'selected_idx.joblib'
if best_wrapper.exists():
    try:
        w = joblib.load(best_wrapper)
        if isinstance(w, dict) and 'selected_idx' in w:
            sel_idx = np.array(w['selected_idx'], dtype=int)
            print('Loaded selected_idx from best wrapper', sel_idx.shape[0])
    except Exception:
        pass
if sel_idx is None and sel_job.exists():
    try:
        s = joblib.load(sel_job)
        if isinstance(s, dict) and 'selected_idx' in s:
            sel_idx = np.array(s['selected_idx'], dtype=int)
            print('Loaded selected_idx from selected_idx.joblib', sel_idx.shape[0])
    except Exception:
        pass
if sel_idx is None:
    from sklearn.feature_selection import f_classif
    K = 800
    F, p = f_classif(X, y_bin)
    sel_idx = np.argsort(F)[-K:][::-1]
    print('Computed top-K sel_idx', sel_idx.shape[0])

X_sel = X[:, sel_idx]

# create train/val split for threshold tuning
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(sss.split(X_sel, y_bin))
X_tr, y_tr = X_sel[train_idx], y_bin[train_idx]
X_val, y_val = X_sel[val_idx], y_bin[val_idx]

# train bagging ensemble
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import random

n_models = 12
n_estimators = 400
base_seed = 1000
val_preds = np.zeros((X_val.shape[0], n_models))
test_preds = []
if test_npz.exists():
    tr = np.load(test_npz, allow_pickle=True)
    X_test = tr['X']; y_test = tr['y']
    y_test_bin = map_to_binary(y_test)
    X_test_sel = X_test[:, sel_idx]
else:
    X_test = None
    y_test_bin = None
    X_test_sel = None

for i in range(n_models):
    seed = base_seed + i
    # randomize some fractions for diversity
    feat_frac = max(0.4, min(1.0, 0.6 + random.uniform(-0.2, 0.2)))
    bag_frac = max(0.4, min(1.0, 0.7 + random.uniform(-0.2, 0.2)))
    bag_freq = random.choice([0,1,2,5])
    params = {
        'n_estimators': n_estimators,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'feature_fraction': feat_frac,
        'bagging_fraction': bag_frac,
        'bagging_freq': bag_freq,
        'random_state': seed,
        'n_jobs': 1,
        'force_row_wise': True,
    }
    print(f'Training bag model {i+1}/{n_models} seed={seed} feat_frac={feat_frac:.3f} bag_frac={bag_frac:.3f} bag_freq={bag_freq}')
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, y_tr)
    joblib.dump(clf, MODELS / f'bag_lgb_{i}.joblib')
    # val pred
    try:
        val_p = clf.predict_proba(X_val)[:,1]
    except Exception:
        val_p = clf.predict(X_val)
    val_preds[:, i] = val_p
    # test pred
    if X_test_sel is not None:
        try:
            test_p = clf.predict_proba(X_test_sel)[:,1]
        except Exception:
            test_p = clf.predict(X_test_sel)
        test_preds.append(test_p)

# average validation preds and tune threshold
val_mean = val_preds.mean(axis=1)
best_t, best_acc = 0.5, 0.0
for t in np.linspace(0.01,0.99,99):
    acc = accuracy_score(y_val, (val_mean >= t).astype(int))
    if acc > best_acc:
        best_acc = acc; best_t = t
print('Best val threshold', best_t, 'val_acc', best_acc)

res = {'n_models': n_models, 'n_estimators': n_estimators, 'best_val_threshold': float(best_t), 'best_val_acc': float(best_acc)}

if X_test_sel is not None and len(test_preds)>0:
    test_mean = np.column_stack(test_preds).mean(axis=1)
    pred = (test_mean >= best_t).astype(int)
    res.update({'test_acc': float(accuracy_score(y_test_bin, pred)), 'test_f1': float(f1_score(y_test_bin, pred)), 'test_auc': float(roc_auc_score(y_test_bin, test_mean))})

with open(RESULTS / 'bagging_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)

print('Saved bagging results:', res)
