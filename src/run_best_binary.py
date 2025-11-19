"""
Create a simple runner that:
- loads processed train/test npz
- maps labels to binary robustly (string 'normal' OR numeric majority-as-normal fallback)
- runs 5-fold Stratified CV for LightGBM and CatBoost (if available)
- fits the best model on full train and evaluates on processed test
- saves results JSON to results/best_binary_run.json and best model to models/best_binary_model.joblib

Designed to be conservative for interactive runs (subsample CV if dataset is large).
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

# robust mapping to binary
def map_to_binary(yarr):
    # try string check first
    try:
        # see if any label, when str-lowered, contains 'normal'
        svals = [str(v).lower() for v in yarr[:1000]] if len(yarr) > 0 else []
        if any('normal' in s for s in svals):
            out = []
            for v in yarr:
                try:
                    s = str(v).lower().strip()
                except Exception:
                    s = ''
                # strip surrounding whitespace and common punctuation/quotes
                s_clean = s.strip(" '\"\\n\\r\\t,")
                out.append(0 if 'normal' in s_clean else 1)
            return np.array(out, dtype=int)
    except Exception:
        pass
    # fallback: numeric factor codes -> majority as normal
    try:
        y_int = np.array(yarr, dtype=int)
        binc = np.bincount(y_int)
        maj = int(np.argmax(binc))
        return np.array([0 if int(v)==maj else 1 for v in y_int], dtype=int)
    except Exception:
        # last fallback: treat any zero as normal
        try:
            y_int = np.array(yarr, dtype=float)
            return np.array([0 if float(v)==0.0 else 1 for v in y_int], dtype=int)
        except Exception:
            # as ultimate fallback, map everything not equal to first label as attack
            first = yarr[0]
            return np.array([0 if v==first else 1 for v in yarr], dtype=int)

y_bin = map_to_binary(y)
print('Binary mapping counts (train):', dict(Counter(y_bin)))

# prepare CV subsample
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
models_tried = {}

# LightGBM
try:
    import lightgbm as lgb
    has_lgb = True
except Exception:
    has_lgb = False

if has_lgb:
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=1, n_jobs=1)
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
    models_tried['lightgbm'] = True
else:
    print('LightGBM not available; skipping')

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
    models_tried['catboost'] = True
else:
    print('CatBoost not available; skipping')

# choose best by f1_mean
best = None
if 'lightgbm' in results and 'catboost' in results:
    best = 'lightgbm' if results['lightgbm']['f1_mean'] >= results['catboost']['f1_mean'] else 'catboost'
elif 'lightgbm' in results:
    best = 'lightgbm'
elif 'catboost' in results:
    best = 'catboost'

# fit best on full train and evaluate on processed test
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if best is not None:
    print('Best model by CV f1:', best)
    if best == 'lightgbm':
        final_clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=1, n_jobs=1)
    else:
        final_clf = CatBoostClassifier(iterations=400, random_seed=1, verbose=False)
    # compute class-based sample weights for training
    cnt = Counter(y_bin.tolist()); total = len(y_bin)
    class_w = {k: total/(len(cnt)*v) for k,v in cnt.items()}
    try:
        sw_full = np.array([class_w[int(yy)] for yy in y_bin])
    except Exception:
        sw_full = None
    if sw_full is not None:
        final_clf.fit(X, y_bin, sample_weight=sw_full)
    else:
        final_clf.fit(X, y_bin)

    # evaluate on test if exists
    if npz_te.exists():
        arrt = np.load(npz_te, allow_pickle=True)
        Xte = arrt['X']; yte = arrt['y']
        yte_bin = map_to_binary(yte)
        ypred = final_clf.predict(Xte)
        try:
            yproba = final_clf.predict_proba(Xte)[:,1]
            auc_full = float(roc_auc_score(yte_bin, yproba))
        except Exception:
            auc_full = None
        acc_full = float(accuracy_score(yte_bin, ypred))
        f1_full = float(f1_score(yte_bin, ypred))
        eval_res = {'test_accuracy': acc_full, 'test_f1': f1_full, 'test_auc': auc_full}
        print('Evaluation on test set ->', eval_res)
    else:
        eval_res = None
        print('No processed test file; only CV metrics available')

    # save model and results
    joblib.dump(final_clf, MOD / 'best_binary_model.joblib')
    out = {'cv_results': results, 'best': best, 'eval_on_test': eval_res}
    with open(RES / 'best_binary_run.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('Saved results to', RES / 'best_binary_run.json')
    print('Saved best model to', MOD / 'best_binary_model.joblib')
else:
    print('No models ran; results:', results)
    with open(RES / 'best_binary_run.json', 'w') as fh:
        json.dump({'cv_results': results}, fh, indent=2)
    print('Saved results to', RES / 'best_binary_run.json')
