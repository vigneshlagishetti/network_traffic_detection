"""
Finalize a model using the best params found earlier.

This script:
- loads processed train/test arrays
- maps labels to binary
- prefilters top-K by f_classif
- runs ABA (if available) to select features
- trains a final LightGBM using params from existing wrapper (if present) or provided defaults
- saves a wrapper with model, params, selected_idx, and threshold
- evaluates on test and writes results to results/best_for_99_results.json

Use when you have a best-params dict (e.g. from a prior run) but the saved wrapper lacks selected_idx.
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

# load processed arrays
npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz_tr.exists():
    raise FileNotFoundError('train_processed.npz not found')
arr = np.load(npz_tr, allow_pickle=True)
X = arr['X']; y = arr['y']
y_bin = map_to_binary(y)
print('Train shape', X.shape, 'binary counts', dict(Counter(y_bin)))

# prefilter
from sklearn.feature_selection import f_classif
K = 800
F, p = f_classif(X, y_bin)
topk_idx = np.argsort(F)[-K:][::-1]
X_topk = X[:, topk_idx]
print('Top-K shape', X_topk.shape)

# ABA selection if available
selected_idx = topk_idx
aba_best = None
try:
    from src.feature_selection.aba import ArtificialButterfly
    aba = ArtificialButterfly(pop_size=24, n_iter=40, random_state=42)
    def fitness_wrapper(Xsub, ysub):
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        import lightgbm as lgb
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=1)
        scores = cross_val_score(clf, Xsub, ysub, cv=skf, scoring='accuracy', n_jobs=1)
        return float(np.mean(scores))
    print('Running ABA...')
    mask, score = aba.fit(X_topk, y_bin, fitness_wrapper)
    aba_best = float(score)
    selected_idx = topk_idx[mask.astype(bool)]
    print('ABA selected', selected_idx.shape[0], 'features, score', aba_best)
except Exception as e:
    print('ABA unavailable or failed:', e)
    selected_idx = topk_idx

# train final LGB with best params if available
wrapper_path = MOD / 'best_model_for_99.joblib'
best_params = {'n_estimators':500, 'learning_rate':0.03, 'num_leaves':31}
best_threshold = 0.5
if wrapper_path.exists():
    try:
        w = joblib.load(wrapper_path)
        if isinstance(w, dict):
            if 'params' in w and w['params'] is not None:
                best_params = w['params']
            if 'threshold' in w:
                best_threshold = w['threshold']
            if 'thr' in w:
                best_threshold = w['thr']
    except Exception as e:
        print('Could not read existing wrapper params:', e)

X_sel = X[:, selected_idx]
print('Training final model on selected features:', X_sel.shape)

from collections import Counter
cnt = Counter(y_bin.tolist()); total = len(y_bin)
class_w = {k: total/(len(cnt)*v) for k,v in cnt.items()}
sw = np.array([class_w[int(yy)] for yy in y_bin])

import lightgbm as lgb
clf = lgb.LGBMClassifier(random_state=42, n_jobs=1, force_row_wise=True, **best_params)
print('Fitting final model...')
clf.fit(X_sel, y_bin, sample_weight=sw)

# save wrapper with selected_idx
wrapper = {'model': clf, 'params': best_params, 'selected_idx': selected_idx, 'threshold': float(best_threshold)}
joblib.dump(wrapper, wrapper_path)
print('Saved wrapper to', wrapper_path)

# evaluate on test
if npz_te.exists():
    arrt = np.load(npz_te, allow_pickle=True)
    Xte = arrt['X']; yte = arrt['y']
    yte_bin = map_to_binary(yte)
    Xte_sel = Xte[:, selected_idx]
    try:
        proba = clf.predict_proba(Xte_sel)[:,1]
        pred = (proba >= best_threshold).astype(int)
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        res = {
            'test_acc': float(accuracy_score(yte_bin, pred)),
            'test_f1': float(f1_score(yte_bin, pred)),
            'test_auc': float(roc_auc_score(yte_bin, proba)),
            'aba_best_score': aba_best,
            'n_selected_features': int(selected_idx.shape[0]),
            'params': best_params,
            'threshold': float(best_threshold)
        }
    except Exception as e:
        res = {'eval_error': str(e)}
else:
    res = {'note': 'no test npz found'}

with open(RES / 'best_for_99_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)
print('Wrote results to', RES / 'best_for_99_results.json')
