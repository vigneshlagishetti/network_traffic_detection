"""Faster, chunked diagnostics to compute per-feature stats without heavy memory spikes.

This avoids forming large temporary arrays and prints a concise summary.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load():
    tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
    dt = np.load(tr)
    return dt['X'], dt['y']


def run(chunk_size=500):
    X, y = load()
    n_samples, n_features = X.shape
    print('train shape', X.shape)
    vars = np.empty(n_features, dtype=float)
    nzeros = np.empty(n_features, dtype=int)
    nnans = np.empty(n_features, dtype=int)

    for start in range(0, n_features, chunk_size):
        end = min(n_features, start + chunk_size)
        chunk = X[:, start:end]
        # variance with nan treated
        vars[start:end] = np.nanvar(chunk, axis=0)
        nzeros[start:end] = np.sum(chunk == 0, axis=0)
        nnans[start:end] = np.sum(np.isnan(chunk), axis=0)

    pct_zero = nzeros / n_samples
    pct_nan = nnans / n_samples

    n_var_zero = int((vars == 0).sum())
    n_var_near = int((vars <= 1e-12).sum())
    n_all_zero = int((nzeros == n_samples).sum())

    print('n_features', n_features)
    print('n_var_zero', n_var_zero)
    print('n_var_near<=1e-12', n_var_near)
    print('n_all_zero', n_all_zero)

    # print a few examples
    idx_zero = np.where(vars == 0)[0]
    print('example zero-var idx (first 20):', idx_zero[:20].tolist())

    # sample 10 features to show stats
    for j in range(min(10, n_features)):
        print(f'feat {j}: var={vars[j]:.3e}, pct_zero={pct_zero[j]:.4f}, pct_nan={pct_nan[j]:.4f}')

    # return arrays for further processing if called as module
    return vars, pct_zero, pct_nan


if __name__ == '__main__':
    run()
