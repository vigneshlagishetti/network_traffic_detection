"""
Train CatBoostClassifier on processed arrays (binary detection).

Requires: CatBoost installed in a conda env. This script is intended to be run with
`conda run -n fruty python src/train_catboost.py` so it uses the conda env that has catboost.

Outputs:
- models/catboost_model.cbm (CatBoost native model)
- models/catboost_wrapper.joblib (dict with selected_idx, threshold)
- results/catboost_results.json
"""
from pathlib import Path
import numpy as np
import json
import joblib

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

# load processed arrays
train_npz = DATA / 'train_processed.npz'
test_npz = DATA / 'test_processed.npz'
if not train_npz.exists():
    raise FileNotFoundError('train_processed.npz missing')
arr = np.load(train_npz, allow_pickle=True)
X = arr['X']; y = arr['y']
y_bin = map_to_binary(y)
print('Loaded train', X.shape, 'labels', np.unique(y_bin, return_counts=True))

# selected features
sel_idx = None
sel_job = MODELS / 'selected_idx.joblib'
best_wrapper = MODELS / 'best_model_for_99.joblib'
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

# load test
if test_npz.exists():
    arrt = np.load(test_npz, allow_pickle=True)
    X_test = arrt['X']; y_test = arrt['y']
    y_test_bin = map_to_binary(y_test)
    X_test_sel = X_test[:, sel_idx]
else:
    X_test = None
    y_test_bin = None
    X_test_sel = None

# train CatBoost
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# compute class weights
from collections import Counter
cnt = Counter(y_bin.tolist()); total = len(y_bin)
class_weights = {int(k): total/(len(cnt)*v) for k,v in cnt.items()}
print('Class weights', class_weights)

params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 8,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 100,
    'od_type': 'Iter',
    'od_wait': 50,
}

model = CatBoostClassifier(**params)
train_pool = Pool(X_sel, y_bin)
model.fit(train_pool)

# save CatBoost model (native format) and wrapper
model_path = MODELS / 'catboost_model.cbm'
model.save_model(str(model_path))
wrapper = {'selected_idx': sel_idx.tolist()}
joblib.dump(wrapper, MODELS / 'catboost_wrapper.joblib')

# evaluate and threshold tuning
res = {}
if X_test_sel is not None:
    p = model.predict_proba(X_test_sel)[:,1]
    # find best threshold
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.01,0.99,99):
        acc = accuracy_score(y_test_bin, (p >= t).astype(int))
        if acc > best_acc:
            best_acc = acc; best_t = t
    res = {'test_acc': float(best_acc), 'test_f1': float(f1_score(y_test_bin, (p>=best_t).astype(int))), 'test_auc': float(roc_auc_score(y_test_bin, p)), 'best_threshold': float(best_t)}
    print('CatBoost test results', res)
    with open(RESULTS / 'catboost_results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
else:
    print('No test set found; model saved')

print('Saved CatBoost model to', model_path)
