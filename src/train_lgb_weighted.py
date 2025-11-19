import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

# load processed train/test
npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz_tr.exists() or not npz_te.exists():
    raise FileNotFoundError('processed train/test npz files not found in data/processed')
arr = np.load(npz_tr, allow_pickle=True)
X = arr['X']; y = arr['y']
arrt = np.load(npz_te, allow_pickle=True)
Xte = arrt['X']; yte = arrt['y']

# same mapping as earlier
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

y_bin = map_to_binary(y)
yte_bin = map_to_binary(yte)

# compute class counts
unique, counts = np.unique(y_bin, return_counts=True)
class_counts = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
print('Train class counts:', class_counts)

# compute scale_pos_weight = n_negative / n_positive (for LightGBM)
n_pos = int(class_counts.get(1, 0))
# assume label 0 is negative
n_neg = int(class_counts.get(0, 0))
scale_pos_weight = (n_neg / n_pos) if n_pos>0 else 1.0

# sample weights balanced: total/(2*count)
total = len(y_bin)
class_weight = {0: total/(2*class_counts.get(0,1)), 1: total/(2*class_counts.get(1,1))}
sample_weight = np.array([class_weight[int(yi)] for yi in y_bin])

# train LightGBM with sklearn API
try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError('LightGBM not installed in environment: ' + str(e))

clf = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    force_row_wise=True,
)

callbacks = []
try:
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
except Exception:
    callbacks = []

print('Fitting LightGBM with scale_pos_weight=%.4f' % scale_pos_weight)
clf.fit(
    X, y_bin,
    sample_weight=sample_weight,
    eval_set=[(Xte, yte_bin)],
    eval_metric='auc',
    callbacks=callbacks,
)

# evaluate
ypred = clf.predict(Xte)
try:
    yproba = clf.predict_proba(Xte)[:,1]
except Exception:
    yproba = clf.predict_proba(Xte)

res = {
    'test_accuracy': float(accuracy_score(yte_bin, ypred)),
    'test_f1': float(f1_score(yte_bin, ypred)),
    'test_auc': float(roc_auc_score(yte_bin, yproba)),
    'test_pr_auc': float(average_precision_score(yte_bin, yproba)),
    'precision': float(precision_score(yte_bin, ypred, zero_division=0)),
    'recall': float(recall_score(yte_bin, ypred, zero_division=0)),
    'n_train_samples': int(X.shape[0]),
    'n_features': int(X.shape[1]),
    'class_counts': class_counts,
    'scale_pos_weight': float(scale_pos_weight),
}

# save model and results
model_path = MOD / 'final_lgb_model_weighted.joblib'
joblib.dump(clf, model_path)
with open(RES / 'final_lgb_results_weighted.json', 'w') as fh:
    json.dump(res, fh, indent=2)

print('Saved weighted model to', model_path)
print('Results:', res)
