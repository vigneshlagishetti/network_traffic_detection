"""
Retrain and finalize the best model (LightGBM) with early stopping.
- loads processed train/test npz
- maps labels to binary (robust heuristic)
- does a stratified train/val split, trains LGBM with early stopping
- evaluates on test set, saves model and results
"""
from pathlib import Path
import numpy as np
import json
import joblib
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

npz = ROOT / 'data' / 'processed' / 'train_processed.npz'
if not npz.exists():
    raise FileNotFoundError(f"Processed train file not found: {npz}\nRun src/preprocessing.py first")
arr = np.load(npz, allow_pickle=True)
X = arr['X']
y = arr['y']

def map_to_binary(yarr):
    # try simple string detection
    try:
        svals = [str(v).lower() for v in yarr[:1000]] if len(yarr)>0 else []
        if any('normal' in s for s in svals):
            return np.array([0 if 'normal' in str(v).lower() else 1 for v in yarr], dtype=int)
    except Exception:
        pass
    # numeric-majority fallback
    try:
        y_int = np.array(yarr, dtype=int)
        maj = int(np.argmax(np.bincount(y_int)))
        return np.array([0 if int(v)==maj else 1 for v in y_int], dtype=int)
    except Exception:
        # last resort: treat zeros as normal
        try:
            yf = np.array(yarr, dtype=float)
            return np.array([0 if float(v)==0.0 else 1 for v in yf], dtype=int)
        except Exception:
            first = yarr[0]
            return np.array([0 if v==first else 1 for v in yarr], dtype=int)

y_bin = map_to_binary(y)
print('Train binary counts:', dict(Counter(y_bin)))

# stratified train/val split
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(sss.split(X, y_bin))
X_tr, X_val = X[tr_idx], X[val_idx]
y_tr, y_val = y_bin[tr_idx], y_bin[val_idx]

from collections import Counter
cnt = Counter(y_tr.tolist())
total = len(y_tr)
class_w = {k: total/(len(cnt)*v) for k,v in cnt.items()}
sw_tr = np.array([class_w[int(yy)] for yy in y_tr])

try:
    import lightgbm as lgb
    has_lgb = True
except Exception:
    has_lgb = False

if not has_lgb:
    raise RuntimeError('LightGBM is required for this script')

clf = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    random_state=42,
    n_jobs=1,
    force_row_wise=True,
)

print('Training LightGBM with early stopping...')
# sklearn API in some lightgbm versions expects callbacks for early stopping
callbacks = []
try:
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
except Exception:
    callbacks = []

clf.fit(
    X_tr, y_tr,
    sample_weight=sw_tr,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=callbacks
)

# evaluate on test set if available
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
if npz_te.exists():
    arrt = np.load(npz_te, allow_pickle=True)
    Xte = arrt['X']; yte = arrt['y']
    yte_bin = map_to_binary(yte)
    ypred = clf.predict(Xte)
    try:
        yproba = clf.predict_proba(Xte)[:,1]
        auc = float(roc_auc_score(yte_bin, yproba))
    except Exception:
        auc = None
    res = {
        'test_accuracy': float(accuracy_score(yte_bin, ypred)),
        'test_f1': float(f1_score(yte_bin, ypred)),
        'test_auc': auc,
        'n_train_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'best_iteration': int(getattr(clf, 'best_iteration_', getattr(clf, 'n_estimators', None)) or 0)
    }
    print('Test eval:', res)
else:
    res = {'note': 'no processed test file found'}

# save model and results
joblib.dump(clf, MOD / 'final_lgb_model.joblib')
with open(RES / 'final_lgb_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)
print('Saved final model to', MOD / 'final_lgb_model.joblib')
print('Saved results to', RES / 'final_lgb_results.json')
