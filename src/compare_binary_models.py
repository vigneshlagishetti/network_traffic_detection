"""
Compare LightGBM vs CatBoost for binary detection (normal vs attack).
Saves results to results/compare_binary_models.json and prints a summary.
"""
import json
from pathlib import Path
import numpy as np
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

npz = ROOT / 'data' / 'processed' / 'train_processed.npz'
if not npz.exists():
    raise FileNotFoundError(f"Processed train file not found: {npz}\nRun src/preprocessing.py first")
arr = np.load(npz, allow_pickle=True)
X = arr['X']
y = arr['y']

# robust binary mapping
def to_binary_labels(yarr):
    out = []
    for v in yarr:
        try:
            s = str(v).lower().strip()
        except Exception:
            s = ''
        s_clean = s.strip(' \"\'\n\r\t,')
        if 'normal' in s_clean:
            out.append(0)
        else:
            out.append(1)
    return np.array(out, dtype=int)

y_bin = to_binary_labels(y)
print('Binary label counts:', np.bincount(y_bin))

# subsample for CV if large
CV_SUBSAMPLE = 20000
if X.shape[0] > CV_SUBSAMPLE:
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=CV_SUBSAMPLE, random_state=1)
    sub_idx, _ = next(sss.split(X, y_bin))
    X_cv = X[sub_idx]; y_cv = y_bin[sub_idx]
else:
    X_cv = X; y_cv = y_bin

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
results = {}

# LightGBM
try:
    import lightgbm as lgb
    has_lgb = True
except Exception:
    has_lgb = False

if has_lgb:
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=1, n_jobs=1)
    print('Running LightGBM CV...')
    acc = cross_val_score(clf, X_cv, y_cv, cv=skf, scoring=make_scorer(accuracy_score), n_jobs=1)
    f1 = cross_val_score(clf, X_cv, y_cv, cv=skf, scoring=make_scorer(f1_score), n_jobs=1)
    try:
        auc = cross_val_score(clf, X_cv, y_cv, cv=skf, scoring=make_scorer(roc_auc_score), n_jobs=1)
    except Exception:
        auc = [None]*len(acc)
    results['lightgbm'] = {
        'acc_mean': float(acc.mean()), 'acc_std': float(acc.std()),
        'f1_mean': float(f1.mean()), 'f1_std': float(f1.std()),
        'auc_mean': float(np.nanmean([a if a is not None else np.nan for a in auc])) if auc is not None else None
    }
else:
    print('LightGBM not installed; skipping')

# CatBoost
try:
    from catboost import CatBoostClassifier
    has_cat = True
except Exception:
    has_cat = False

if has_cat:
    clf_c = CatBoostClassifier(iterations=200, random_seed=1, verbose=False)
    print('Running CatBoost CV...')
    acc = cross_val_score(clf_c, X_cv, y_cv, cv=skf, scoring=make_scorer(accuracy_score), n_jobs=1)
    f1 = cross_val_score(clf_c, X_cv, y_cv, cv=skf, scoring=make_scorer(f1_score), n_jobs=1)
    try:
        auc = cross_val_score(clf_c, X_cv, y_cv, cv=skf, scoring=make_scorer(roc_auc_score), n_jobs=1)
    except Exception:
        auc = [None]*len(acc)
    results['catboost'] = {
        'acc_mean': float(acc.mean()), 'acc_std': float(acc.std()),
        'f1_mean': float(f1.mean()), 'f1_std': float(f1.std()),
        'auc_mean': float(np.nanmean([a if a is not None else np.nan for a in auc])) if auc is not None else None
    }
else:
    print('CatBoost not installed; skipping')

# Save and print
with open(RES / 'compare_binary_models.json', 'w') as fh:
    json.dump(results, fh, indent=2)

print('Results saved to results/compare_binary_models.json')
print(json.dumps(results, indent=2))

# recommend better model
best = None
if 'lightgbm' in results and 'catboost' in results:
    # pick by f1_mean
    best = 'lightgbm' if results['lightgbm']['f1_mean'] >= results['catboost']['f1_mean'] else 'catboost'
elif 'lightgbm' in results:
    best = 'lightgbm'
elif 'catboost' in results:
    best = 'catboost'

if best:
    print('\nRecommendation: use', best)
else:
    print('\nNo models were run; install lightgbm and/or catboost and retry.')
