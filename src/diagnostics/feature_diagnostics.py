"""Feature diagnostics for processed datasets.

Loads data/processed/*.npz and computes per-feature statistics:
- variance
- number of unique values
- percent zeros
- percent NaN

Saves results to results/feature_diagnostics.csv and prints a short summary.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_processed():
    tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
    te = ROOT / 'data' / 'processed' / 'test_processed.npz'
    if not tr.exists() or not te.exists():
        raise FileNotFoundError('Processed files not found; run preprocessing first')
    dt = np.load(tr)
    de = np.load(te)
    return dt['X'], dt['y'], de['X'], de['y']


def run():
    Xtr, ytr, Xte, yte = load_processed()
    n_samples, n_features = Xtr.shape
    print('train shape', Xtr.shape, 'test shape', Xte.shape)

    stats = []
    for j in range(n_features):
        col = Xtr[:, j]
        n_nan = np.isnan(col).sum()
        pct_nan = n_nan / n_samples
        n_zero = (col == 0).sum()
        pct_zero = n_zero / n_samples
        uniq = np.unique(col[~np.isnan(col)])
        n_uniq = uniq.size
        var = np.nanvar(col)
        stats.append({'feature_idx': j, 'variance': var, 'n_unique': int(n_uniq), 'pct_nan': float(pct_nan), 'pct_zero': float(pct_zero)})

    df = pd.DataFrame(stats)
    df = df.sort_values('variance', ascending=False).reset_index(drop=True)
    out_dir = ROOT / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / 'feature_diagnostics.csv'
    df.to_csv(out_f, index=False)

    const_feats = df[df['n_unique'] <= 1]
    near_zero = df[df['variance'] <= 1e-12]

    print('n_features', n_features)
    print('constant features (n_unique<=1):', len(const_feats))
    print('near-zero variance features (var<=1e-12):', len(near_zero))
    print('\nTop 10 features by variance:')
    print(df.head(10).to_string(index=False))
    print('\nBottom 10 features by variance:')
    print(df.tail(10).to_string(index=False))

    # label distribution
    import collections
    lab_counts = collections.Counter(ytr.tolist())
    print('\nLabel distribution (train):')
    for k, v in lab_counts.items():
        print(k, v)

    print('\nDiagnostics saved to', out_f)


if __name__ == '__main__':
    run()
