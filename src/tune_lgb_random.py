import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, make_scorer, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz_tr.exists() or not npz_te.exists():
    raise FileNotFoundError('processed train/test npz files not found in data/processed')
arr = np.load(npz_tr, allow_pickle=True)
X = arr['X']; y = arr['y']
arrt = np.load(npz_te, allow_pickle=True)
Xte = arrt['X']; yte = arrt['y']

# map to binary
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

# sample weights
unique, counts = np.unique(y_bin, return_counts=True)
class_counts = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
print('Train class counts:', class_counts)

total = len(y_bin)
class_weight = {k: total/(2*v) for k,v in class_counts.items()}
sample_weight = np.array([class_weight[int(yi)] for yi in y_bin])

# composite scorer: average of roc_auc and accuracy on validation predictions
from sklearn.base import clone

def composite_score(estimator, Xv, yv):
    # estimator.predict_proba may not exist until fitted; estimator is fitted by CV
    ypred = estimator.predict(Xv)
    try:
        yproba = estimator.predict_proba(Xv)[:,1]
    except Exception:
        # fallback to decision_function
        try:
            yproba = estimator.decision_function(Xv)
        except Exception:
            yproba = ypred
    roc = roc_auc_score(yv, yproba)
    acc = accuracy_score(yv, ypred)
    return (roc + acc) / 2.0

from sklearn.metrics import make_scorer
scorer = make_scorer(composite_score, greater_is_better=True)

# model and param distributions
import lightgbm as lgb
clf = lgb.LGBMClassifier(n_estimators=2000, random_state=42)

param_dist = {
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
    'num_leaves': [31, 48, 64, 80, 96, 128],
    'max_depth': [5, 6, 8, 10, 12, 14],
    'min_child_samples': [5, 10, 20, 40, 80],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
    'reg_alpha': [0.0, 0.1, 0.5, 1.0, 5.0],
    'reg_lambda': [0.0, 0.1, 0.5, 1.0, 5.0]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rs = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=40, scoring=scorer, cv=cv, verbose=2, n_jobs=1, random_state=42)

# fit
rs.fit(X, y_bin, sample_weight=sample_weight)

print('Best params (random search):', rs.best_params_)

best_clf = rs.best_estimator_
# evaluate on test
ypred = best_clf.predict(Xte)
try:
    yproba = best_clf.predict_proba(Xte)[:,1]
except Exception:
    try:
        yproba = best_clf.decision_function(Xte)
    except Exception:
        yproba = ypred

res = {
    'best_params': rs.best_params_,
    'test_accuracy': float(accuracy_score(yte_bin, ypred)),
    'test_f1': float(f1_score(yte_bin, ypred)),
    'test_auc': float(roc_auc_score(yte_bin, yproba)),
    'test_pr_auc': float(average_precision_score(yte_bin, yproba)),
    'precision': float(precision_score(yte_bin, ypred, zero_division=0)),
    'recall': float(recall_score(yte_bin, ypred, zero_division=0)),
    'n_train_samples': int(X.shape[0]),
    'n_features': int(X.shape[1]),
}

joblib.dump(best_clf, MOD / 'final_lgb_random_tuned.joblib')
with open(RES / 'final_lgb_random_tuned.json', 'w') as fh:
    json.dump(res, fh, indent=2)

print('Saved tuned model to', MOD / 'final_lgb_random_tuned.joblib')
print('Results:', res)
