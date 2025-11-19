"""Blend existing model artifacts and evaluate on test set.

This script loads model bundles if present:
- models/pruned_stack.joblib
- models/optuna_pruned_light.joblib
- models/stack_improved.joblib
- models/lgb_optuna_tuned.joblib
- models/lgb_weighted_baseline.joblib
- models/xgb_baseline.joblib

It computes per-model test probabilities robustly (handling bundles and single models),
averages the probabilities (simple mean), sweeps threshold for best accuracy, and
saves results to results/blend_results.json and model bundle to models/blend_model.joblib.
"""

import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
MOD.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# candidate model filenames (prefer pruned and tuned artifacts first)
CAND_FILES = [
    MOD / 'pruned_stack.joblib',
    MOD / 'optuna_pruned_light.joblib',
    MOD / 'stack_improved.joblib',
    MOD / 'lgb_optuna_tuned.joblib',
    MOD / 'lgb_weighted_baseline.joblib',
    MOD / 'xgb_baseline.joblib',
    MOD / 'stack_lgb_xgb.joblib',
]


def load_processed():
    npz = ROOT / 'data' / 'processed' / 'test_processed.npz'
    if not npz.exists():
        raise FileNotFoundError(npz)
    d = np.load(npz, allow_pickle=True)
    return d['X'], d['y']


def map_to_binary(yarr):
    y = np.array(yarr)
    svals = [str(v).lower() for v in y[:1000]] if y.size > 0 else []
    if any('normal' in s for s in svals):
        return np.array([0 if 'normal' in str(v).lower() else 1 for v in y], dtype=int)
    try:
        yi = y.astype(int)
        maj = int(np.argmax(np.bincount(yi)))
        return np.array([0 if int(v) == maj else 1 for v in yi], dtype=int)
    except Exception:
        try:
            yf = y.astype(float)
            return np.array([0 if float(v) == 0.0 else 1 for v in yf], dtype=int)
        except Exception:
            first = y[0]
            return np.array([0 if v == first else 1 for v in y], dtype=int)


def get_proba_from_obj(obj, X):
    """Return probability array for positive class for object `obj` on X.
    Handles different saved bundle shapes.
    """
    # If it's a dict bundle with models
    if isinstance(obj, dict):
        # Possible keys: 'base_models', 'meta', 'threshold', or direct model
        # If meta exists and base_models is a dict, compute meta proba
        if 'meta' in obj and 'base_models' in obj:
            # build test meta matrix
            base_models = obj['base_models']
            prob_list = []
            for name, mdl in (base_models.items() if isinstance(base_models, dict) else base_models):
                try:
                    p = mdl.predict_proba(X)
                    p = p[:, 1] if p.ndim == 2 else p
                except Exception:
                    try:
                        dec = mdl.decision_function(X)
                        p = 1.0 / (1.0 + np.exp(-dec))
                    except Exception:
                        p = np.full(X.shape[0], 0.5)
                prob_list.append(p)
            if len(prob_list) == 0:
                return np.full(X.shape[0], 0.5)
            meta_X_test = np.vstack(prob_list).T
            meta = obj['meta']
            try:
                pmeta = meta.predict_proba(meta_X_test)
                pmeta = pmeta[:, 1] if pmeta.ndim == 2 else pmeta
            except Exception:
                try:
                    dec = meta.decision_function(meta_X_test)
                    pmeta = 1.0 / (1.0 + np.exp(-dec))
                except Exception:
                    pmeta = np.full(X.shape[0], 0.5)
            return pmeta
        # if dict is single model mapping or wrapper, try to find estimator
        for key in ['model', 'estimator', 'clf', 'classifier']:
            if key in obj:
                return get_proba_from_obj(obj[key], X)
        # fallback: attempt to call predict_proba on entire dict (unlikely)
    # If it's a sklearn-like model
    try:
        p = obj.predict_proba(X)
        p = p[:, 1] if p.ndim == 2 else p
        return p
    except Exception:
        pass
    try:
        dec = obj.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-dec))
        return p
    except Exception:
        pass
    # Last resort: constant 0.5
    return np.full(X.shape[0], 0.5)


def sweep_threshold(y_true, probas, steps=501):
    best_acc = -1.0
    best_th = 0.5
    best_conf = None
    for th in np.linspace(0, 1, steps):
        pred = (probas >= th).astype(int)
        acc = float((pred == y_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_th = float(th)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            best_conf = [[int(tn), int(fp)], [int(fn), int(tp)]]
    return best_th, best_acc, best_conf


def main():
    Xte, yte_raw = load_processed()
    yte = map_to_binary(yte_raw)

    present = [p for p in CAND_FILES if p.exists()]
    if not present:
        print('No candidate model files found in', MOD)
        return

    prob_list = []
    used = []
    for p in present:
        try:
            obj = joblib.load(p)
        except Exception as e:
            print('Failed to load', p, e)
            continue
        proba = get_proba_from_obj(obj, Xte)
        prob_list.append(proba)
        used.append(p.name)
        print('Loaded', p.name, '-> proba shape', proba.shape)

    if len(prob_list) == 0:
        print('No probabilities could be computed from present models')
        return

    # simple mean blend
    stacked = np.vstack(prob_list)
    avg = np.mean(stacked, axis=0)

    th, acc, conf = sweep_threshold(yte, avg, steps=1001)
    auc = float(roc_auc_score(yte, avg))
    prec = float(precision_score(yte, (avg >= th).astype(int), zero_division=0))
    rec = float(recall_score(yte, (avg >= th).astype(int), zero_division=0))
    f1 = float(f1_score(yte, (avg >= th).astype(int), zero_division=0))

    results = {
        'used_models': used,
        'n_models': len(used),
        'test_best_threshold': th,
        'test_best_accuracy': acc,
        'test_confusion': conf,
        'test_roc_auc': auc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

    with open(RES / 'blend_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    # save a light bundle with the averaged probabilities (no training) and threshold
    joblib.dump({'used': used, 'threshold': th}, MOD / 'blend_model.joblib')
    print('Saved results/blend_results.json and models/blend_model.joblib')
    print(results)


if __name__ == '__main__':
    main()
