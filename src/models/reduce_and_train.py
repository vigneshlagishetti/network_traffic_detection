"""Select top-K features by mutual information and train LightGBM multiclass.

This is a fast filter approach to reduce dimensionality before running ABA or
more expensive hyperparameter searches.
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


def to_numeric_labels(y):
    # if labels are strings, convert to integers via factorize
    y = np.asarray(y)
    try:
        # check if already numeric
        _ = y.astype(float)
        return y.astype(int)
    except Exception:
        import pandas as pd
        return pd.factorize(y)[0]


def main(k=200, subsample=30000, mi_subsample=5000, random_state=42):
    X_train, y_train_raw, X_test, y_test_raw = load_processed()
    y_train = to_numeric_labels(y_train_raw)
    y_test = to_numeric_labels(y_test_raw)
    print('Loaded shapes', X_train.shape, X_test.shape)

    # drop constant features
    var = np.nanvar(X_train, axis=0)
    keep_var = var > 1e-12
    Xtr = X_train[:, keep_var]
    Xte = X_test[:, keep_var]
    print('After var filter, features:', Xtr.shape[1])

    # mutual information selection on a subsample for speed
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(mi_subsample, Xtr.shape[0]), random_state=random_state)
    idx, _ = next(sss.split(Xtr, y_train))
    Xs = Xtr[idx]
    ys = y_train[idx]

    from sklearn.feature_selection import f_classif
    print('Computing ANOVA F-score on subsample', Xs.shape)
    f_vals, p_vals = f_classif(Xs, ys)
    top_idx = np.argsort(f_vals)[-k:][::-1]
    # map back to original feature indices
    keep_indices = np.flatnonzero(keep_var)[top_idx]

    # select features and train on subsample
    Xtr_sel = Xtr[:, top_idx]
    Xte_sel = Xte[:, top_idx]

    try:
        import lightgbm as lgb
    except Exception:
        print('lightgbm not installed')
        return

    # further subsample for training speed
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=min(subsample, Xtr_sel.shape[0]), random_state=random_state)
    idx2, _ = next(sss2.split(Xtr_sel, y_train))
    Xtr_sub = Xtr_sel[idx2]
    ytr_sub = y_train[idx2]

    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state, n_jobs=1)
    clf.fit(Xtr_sub, ytr_sub)

    y_pred = clf.predict(Xte_sel)
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'Reduce+Train -> acc: {acc:.4f}, f1_macro: {f1:.4f}, selected_features: {k}')

    results_dir = ROOT / 'results'
    models_dir = ROOT / 'models'
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame([{'method': 'mutual_info_topk', 'k': k, 'accuracy': acc, 'f1_macro': f1}]).to_csv(results_dir / 'reduce_train_results.csv', index=False)
    joblib.dump({'model': clf, 'top_features_idx': keep_indices}, models_dir / 'lgb_reduce_topk.joblib')


if __name__ == '__main__':
    main()
