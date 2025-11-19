"""Quick LightGBM training with simple preprocessing fixes:
- drop constant features
- use sample weights = inverse class frequency

Saves model to models/lgb_quick_weighted.joblib and results to results/quick_results_weighted.csv
"""
from pathlib import Path
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[2]


def load_processed(path_train=None, path_test=None):
    if path_train is None:
        path_train = ROOT / 'data' / 'processed' / 'train_processed.npz'
    if path_test is None:
        path_test = ROOT / 'data' / 'processed' / 'test_processed.npz'
    dt = np.load(path_train)
    de = np.load(path_test)
    return dt['X'], dt['y'], de['X'], de['y']


def drop_constant(X_train, X_test):
    var = np.nanvar(X_train, axis=0)
    keep = var > 1e-12
    return X_train[:, keep], X_test[:, keep], keep


def main():
    X_train, y_train, X_test, y_test = load_processed()
    print('Loaded shapes', X_train.shape, X_test.shape)
    Xtr, Xte, keep = drop_constant(X_train, X_test)
    print('Kept features', Xtr.shape[1])

    # compute sample weights inversely proportional to class frequency
    import collections
    counts = collections.Counter(y_train.tolist())
    total = len(y_train)
    class_weight = {k: total / (len(counts) * v) for k, v in counts.items()}
    sample_weight = np.array([class_weight[int(l)] for l in y_train])

    # subsample training set for quick iteration (conservative default)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=30000, random_state=42)
    idx_train, _ = next(sss.split(Xtr, y_train))
    Xtr_sub = Xtr[idx_train]
    ytr_sub = y_train[idx_train]
    sw_sub = sample_weight[idx_train]
    print('Subsampled training to', Xtr_sub.shape)

    try:
        import lightgbm as lgb
    except Exception:
        print('lightgbm not installed')
        return

    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=1)
    clf.fit(Xtr_sub, ytr_sub, sample_weight=sw_sub)

    y_pred = clf.predict(Xte)
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'Weighted LightGBM -> acc: {acc:.4f}, f1_macro: {f1:.4f}')

    models_dir = ROOT / 'models'
    results_dir = ROOT / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': clf, 'keep_mask': keep}, models_dir / 'lgb_quick_weighted.joblib')
    import pandas as pd
    pd.DataFrame([{'model': 'lgb_quick_weighted', 'accuracy': acc, 'f1_macro': f1, 'n_features': int(keep.sum())}]).to_csv(results_dir / 'quick_results_weighted.csv', index=False)


if __name__ == '__main__':
    main()
