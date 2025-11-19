"""Quick binary (normal vs attack) LightGBM baseline.

This maps labels to 0 (normal) and 1 (attack) and trains on a stratified subsample
to get a fast estimate of binary detection performance (where 99%+ accuracy is
often achievable on NSL-KDD).
"""
from pathlib import Path
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[2]


def load_processed():
    tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
    te = ROOT / 'data' / 'processed' / 'test_processed.npz'
    dt = np.load(tr, allow_pickle=True)
    de = np.load(te, allow_pickle=True)
    return dt['X'], dt['y'], de['X'], de['y']


def to_binary(y):
    # treat string 'normal' as normal; any other label -> attack
    y = np.asarray(y)
    # some y may already be numeric; coerce to str then compare
    y_str = np.array([str(v).lower() for v in y])
    return (y_str != 'normal').astype(int)


def main():
    X_train, y_train_raw, X_test, y_test_raw = load_processed()
    print('Loaded shapes', X_train.shape, X_test.shape)
    y_train = to_binary(y_train_raw)
    y_test = to_binary(y_test_raw)

    # drop constant features
    var = np.nanvar(X_train, axis=0)
    keep = var > 1e-12
    Xtr = X_train[:, keep]
    Xte = X_test[:, keep]
    print('Kept features', Xtr.shape[1])

    # subsample for quick run
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=30000, random_state=42)
    idx_train, _ = next(sss.split(Xtr, y_train))
    Xtr_sub = Xtr[idx_train]
    ytr_sub = y_train[idx_train]
    print('Subsampled to', Xtr_sub.shape)

    try:
        import lightgbm as lgb
    except Exception:
        print('lightgbm not installed')
        return

    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=1)
    clf.fit(Xtr_sub, ytr_sub)

    y_pred = clf.predict(Xte)
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    print(f'Binary LightGBM -> acc: {acc:.4f}, f1: {f1:.4f}')

    models_dir = ROOT / 'models'
    results_dir = ROOT / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': clf, 'keep_mask': keep}, models_dir / 'lgb_binary_quick.joblib')
    import pandas as pd
    pd.DataFrame([{'model': 'lgb_binary_quick', 'accuracy': acc, 'f1_binary': f1, 'n_features': int(keep.sum())}]).to_csv(results_dir / 'binary_quick_results.csv', index=False)


if __name__ == '__main__':
    main()
