"""Run ABA feature selection on processed NSL-KDD data using LightGBM CV fitness.

Outputs:
 - models/aba_best_mask.npy
 - models/aba_lgb_final.joblib
 - results/aba_history.csv  (iteration, best_score)
 - results/aba_results.csv  (final test metrics: accuracy, f1_macro, per-class recall/precision)

This script uses conservative defaults to limit runtime: pop_size=12, n_iter=12.
Adjust via CLI args if you want longer runs.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from src.feature_selection.aba import ArtificialButterfly


def load_processed_arrays(root: Path):
    ftr = root / 'data' / 'processed' / 'train_processed.npz'
    fte = root / 'data' / 'processed' / 'test_processed.npz'
    if not ftr.exists() or not fte.exists():
        raise FileNotFoundError('Processed arrays not found. Run src/preprocessing.py first.')
    dt = np.load(ftr)
    de = np.load(fte)
    X_train, y_train = dt['X'], dt['y']
    X_test, y_test = de['X'], de['y']
    return X_train, y_train, X_test, y_test


def make_fitness_lgb(X, y, cv=3, random_state=0):
    # create a cached fitness that uses LightGBM CV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    try:
        import lightgbm as lgb
    except Exception:
        raise RuntimeError('lightgbm not installed in venv')

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state, n_jobs=-1)

    cache = {}

    def fitness(X_sub, y_sub):
        # key by shape and hash of mean/std to be simple; mask caching is done by wrapper
        # Expect X_sub as view with reduced columns
        key = (X_sub.shape[1],)
        # if identical config already evaluated, this basic cache may help
        if key in cache:
            return cache[key]
        scores = cross_val_score(clf, X_sub, y_sub, cv=skf, scoring='f1_macro', n_jobs=1)
        val = float(np.mean(scores))
        cache[key] = val
        return val

    return fitness


def run_aba_and_evaluate(pop_size:int=12, n_iter:int=12, random_state:int=0):
    X_train, y_train, X_test, y_test = load_processed_arrays(ROOT)
    print('Loaded processed arrays. shapes:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    fitness = make_fitness_lgb(X_train, y_train, cv=3, random_state=random_state)

    # wrapper that accepts mask (boolean) but ABA expects fitness(X_sub, y)
    eval_cache = {}

    def fitness_mask_wrapper(X_sub, y_sub):
        # fitness receives X_sub directly; we use it as-is
        return fitness(X_sub, y_sub)

    aba = ArtificialButterfly(pop_size=pop_size, n_iter=n_iter, random_state=random_state)

    # run
    t0 = time.time()
    best_mask, best_score = aba.fit(X_train, y_train, fitness_mask_wrapper)
    t1 = time.time()
    print(f'ABA finished in {t1-t0:.1f}s, best_score={best_score:.6f}, n_features={int(best_mask.sum())}')

    # save mask and history
    models_dir = ROOT / 'models'
    results_dir = ROOT / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(models_dir / 'aba_best_mask.npy', best_mask)
    # history -> CSV
    import csv
    hist_f = results_dir / 'aba_history.csv'
    with open(hist_f, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['iteration','best_score'])
        for i, s in enumerate(aba.history_):
            writer.writerow([i, s])

    # retrain final LightGBM on selected features and evaluate on test
    sel_idx = best_mask.astype(bool)
    if not sel_idx.any():
        raise RuntimeError('ABA returned empty feature set')

    Xtr_sel = X_train[:, sel_idx]
    Xte_sel = X_test[:, sel_idx]

    try:
        import lightgbm as lgb
    except Exception:
        raise RuntimeError('lightgbm required for final training')

    clf_final = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=random_state, n_jobs=-1)
    clf_final.fit(Xtr_sel, y_train)

    # evaluate
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    y_pred = clf_final.predict(Xte_sel)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, output_dict=True)

    # save final model and results
    joblib.dump(clf_final, models_dir / 'aba_lgb_final.joblib')
    import json
    res = {'accuracy': acc, 'f1_macro': f1, 'best_score_cv': best_score, 'n_features': int(best_mask.sum()), 'time_s': t1-t0}
    res_f = results_dir / 'aba_results.csv'
    # write a small CSV + JSON report
    import pandas as pd
    df = pd.DataFrame([res])
    df.to_csv(res_f, index=False)
    with open(results_dir / 'aba_classification_report.json', 'w') as fh:
        json.dump(report, fh, indent=2)

    # save mask as human-readable indices
    sel_indices = np.flatnonzero(sel_idx)
    with open(results_dir / 'aba_selected_indices.txt', 'w') as fh:
        fh.write('\n'.join(map(str, map(int, sel_indices.tolist()))))

    print('Final results saved to', results_dir)
    print('Accuracy:', acc, 'f1_macro:', f1)

    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pop', type=int, default=12)
    p.add_argument('--iter', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_aba_and_evaluate(pop_size=args.pop, n_iter=args.iter, random_state=args.seed)
