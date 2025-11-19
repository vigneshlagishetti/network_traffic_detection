import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

def load_processed():
    proc = ROOT / 'data' / 'processed'
    npz = proc / 'train_processed.npz'
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        X = d['X']
        y = d['y']
        return X, y
    return None, None

def load_csv():
    csvp = ROOT / 'combine.csv'
    if csvp.exists():
        df = pd.read_csv(csvp)
        # try to find label column heuristically
        label_cols = [c for c in df.columns if c.lower() in ('label','target','class')]
        if label_cols:
            lab = label_cols[0]
        else:
            lab = df.columns[-1]
        y = df[lab].values
        X = df.drop(columns=[lab]).values
        return X, y
    return None, None

def main():
    X, y = load_processed()
    src = 'processed'
    if X is None:
        X, y = load_csv()
        src = 'csv'
    if X is None:
        raise FileNotFoundError('No processed arrays or combine.csv found')

    out = {'source': src}
    out['n_rows'] = int(X.shape[0])
    out['n_features'] = int(X.shape[1]) if X.ndim==2 else None

    # label summary
    try:
        import numpy as _np
        uniq, counts = _np.unique(y, return_counts=True)
        out['label_counts'] = {str(int(u)) if float(u).is_integer() else str(u): int(c) for u,c in zip(uniq,counts)}
    except Exception:
        out['label_counts'] = {}

    if X.ndim==2:
        # compute per-feature stats (sample if many features)
        n_feat = X.shape[1]
        sample_feats = list(range(n_feat))
        # compute variance, zero pct, unique counts for all features
        variances = np.nanvar(X, axis=0)
        zeros_pct = (X==0).sum(axis=0)/X.shape[0]
        unique_counts = []
        for i in range(n_feat):
            try:
                unique_counts.append(int(np.unique(X[:,i]).shape[0]))
            except Exception:
                unique_counts.append(None)

        out['n_zero_variance'] = int((variances==0).sum())
        out['n_almost_zero_variance'] = int((variances<1e-12).sum())
        # top features by zero pct
        idx_sorted = np.argsort(-zeros_pct)
        top_sparse = []
        for i in idx_sorted[:50]:
            top_sparse.append({ 'feature_index': int(i), 'zeros_pct': float(zeros_pct[i]), 'unique_values': unique_counts[i], 'variance': float(variances[i]) })
        out['top_sparse_features'] = top_sparse

        # low variance list
        low_var_idx = np.where(variances==0)[0].tolist()
        # save csv of low-variance features (top 200)
        import csv
        low_csv = RES / 'low_variance_features.csv'
        with open(low_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['feature_index','variance','unique_values','zeros_pct'])
            for i in low_var_idx[:200]:
                w.writerow([int(i), float(variances[i]), unique_counts[i], float(zeros_pct[i])])
        out['low_variance_count'] = len(low_var_idx)

    # write json
    with open(RES / 'feature_diagnostics.json', 'w') as fh:
        json.dump(out, fh, indent=2)

    print('Wrote results/feature_diagnostics.json and low_variance_features.csv')

if __name__=='__main__':
    main()
