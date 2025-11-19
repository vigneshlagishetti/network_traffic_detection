import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz_tr.exists() or not npz_te.exists():
    raise FileNotFoundError('processed train/test npz not found')
arr = np.load(npz_tr, allow_pickle=True)
X = arr['X']; y = arr['y']
arrt = np.load(npz_te, allow_pickle=True)
Xte = arrt['X']; yte = arrt['y']

# label mapping
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

# sample weight
unique, counts = np.unique(y_bin, return_counts=True)
class_counts = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
print('Train class counts:', class_counts)
total = len(y_bin)
class_weight = {k: total/(2*v) for k,v in class_counts.items()}
sample_weight = np.array([class_weight[int(yi)] for yi in y_bin])

# tune LGB for AUC using RandomizedSearchCV
import lightgbm as lgb
from scipy.stats import randint, uniform

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

clf = lgb.LGBMClassifier(n_estimators=1000, random_state=42, n_jobs=1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rs = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=24, scoring='roc_auc', cv=cv, verbose=2, n_jobs=1, random_state=42)
print('Starting RandomizedSearchCV for LGB (AUC)')
rs.fit(X, y_bin, sample_weight=sample_weight)
print('Best params:', rs.best_params_)

best_lgb = lgb.LGBMClassifier(n_estimators=2000, random_state=42, n_jobs=-1, **rs.best_params_)
# weighted LGB (from earlier)
scale_pos_weight = class_counts.get(0,1)/class_counts.get(1,1)
weighted_lgb = lgb.LGBMClassifier(n_estimators=1000, random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight)

estimators = [('lgb_tuned', best_lgb), ('lgb_weighted', weighted_lgb)]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=1, passthrough=False)

print('Fitting stacking classifier on full train')
# fit with sample weights? StackingClassifier doesn't accept sample_weight directly; we can fit without weights for stacking base estimators because they will be fitted internally; but we can pass sample_weight via fit_params for base estimators using sklearn 1.2+? Skipping weights here for stacking speed.
stack.fit(X, y_bin)

# evaluate on test
proba = stack.predict_proba(Xte)[:,1]
ypred = (proba >= 0.5).astype(int)
acc = float(accuracy_score(yte_bin, ypred))
f1 = float(f1_score(yte_bin, ypred))
prec = float(precision_score(yte_bin, ypred, zero_division=0))
rec = float(recall_score(yte_bin, ypred, zero_division=0))
auc = float(roc_auc_score(yte_bin, proba))

res = {
    'accuracy': acc,
    'f1': f1,
    'precision': prec,
    'recall': rec,
    'roc_auc': auc,
    'best_params': rs.best_params_
}

# save
joblib.dump(stack, MOD / 'stack_lgb_tuned_weighted.joblib')
with open(RES / 'stack_lgb_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)

print('Saved stacked model and results')
print(res)
