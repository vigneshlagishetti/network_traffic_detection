"""Train CatBoost on raw `combine.csv`.

This script is intended to be run in a conda environment where CatBoost is
available on Windows. It reads the raw CSV, detects likely categorical
columns, maps labels to binary, does a stratified numpy split, trains
CatBoost with early stopping, sweeps threshold for best accuracy on test,
and writes results + model to the repo.

Notes:
- This script does NOT attempt to install packages. Use the provided
  `environment-catboost.yml` to create a conda env with CatBoost.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'combine.csv'
MOD = ROOT / 'models'
RES = ROOT / 'results'
MOD.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

def map_to_binary(yarr):
    y = np.array(yarr)
    svals = [str(v).lower() for v in y[:1000]] if y.size>0 else []
    if any('normal' in s for s in svals):
        return np.array([0 if 'normal' in str(v).lower() else 1 for v in y], dtype=int)
    try:
        yi = y.astype(int)
        maj = int(np.argmax(np.bincount(yi)))
        return np.array([0 if int(v)==maj else 1 for v in yi], dtype=int)
    except Exception:
        try:
            yf = y.astype(float)
            return np.array([0 if float(v)==0.0 else 1 for v in yf], dtype=int)
        except Exception:
            first = y[0]
            return np.array([0 if v==first else 1 for v in y], dtype=int)

def stratified_split_numpy(y, test_size=0.15, seed=42):
    rng = np.random.RandomState(seed)
    y = np.array(y)
    classes = np.unique(y)
    train_idx = []
    test_idx = []
    for c in classes:
        idx = np.where(y==c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx)*test_size))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)

def detect_categorical(df, max_unique=100):
    cats = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            cats.append(col)
            continue
        # numeric but low cardinality -> categorical
        try:
            if df[col].nunique(dropna=False) <= max_unique:
                cats.append(col)
        except Exception:
            pass
    return cats

def sweep_threshold(y_true, proba):
    best_acc = -1.0; best_th = 0.5; best_conf=None
    y = np.array(y_true)
    for th in np.linspace(0,1,501):
        pred = (proba>=th).astype(int)
        acc = float((pred==y).mean())
        if acc>best_acc:
            best_acc=acc; best_th=float(th)
            tn = int(((y==0)&(pred==0)).sum())
            fp = int(((y==0)&(pred==1)).sum())
            fn = int(((y==1)&(pred==0)).sum())
            tp = int(((y==1)&(pred==1)).sum())
            best_conf = [[tn, fp],[fn, tp]]
    return best_th, best_acc, best_conf

def main():
    if not DATA.exists():
        raise FileNotFoundError(f'{DATA} not found')
    # read with low_memory=False to avoid mixed-dtype parsing warnings
    df = pd.read_csv(DATA, low_memory=False)
    # Expect there to be a label column; guess common names
    possible_labels = ['label','target','class','attack']
    label_col = None
    for c in possible_labels:
        if c in df.columns:
            label_col = c; break
    if label_col is None:
        # fallback: assume last column is label
        label_col = df.columns[-1]

    y_raw = df[label_col].values
    y = map_to_binary(y_raw)

    Xdf = df.drop(columns=[label_col])
    # detect categorical columns
    cats = detect_categorical(Xdf, max_unique=100)
    # convert categorical columns to string (CatBoost requires categorical values to be int or str)
    if len(cats) > 0:
        for c in cats:
            # replace NaN with a sentinel string and convert to str
            Xdf[c] = Xdf[c].fillna('nan').astype(str)

    # stratified split using indices, then slice the DataFrame to preserve dtypes
    tr_idx, te_idx = stratified_split_numpy(y, test_size=0.15, seed=42)
    X_tr_df = Xdf.iloc[tr_idx].reset_index(drop=True)
    X_te_df = Xdf.iloc[te_idx].reset_index(drop=True)
    # also keep numpy array views available for fallbacks
    X_tr = X_tr_df.values
    X_te = X_te_df.values
    # Ensure categorical columns remain strings after slicing (defensive)
    if len(cats) > 0:
        for c in cats:
            X_tr_df[c] = X_tr_df[c].fillna('nan').astype(str)
            X_te_df[c] = X_te_df[c].fillna('nan').astype(str)
    y_tr, y_te = y[tr_idx], y[te_idx]

    try:
        from catboost import CatBoostClassifier, Pool
    except Exception as e:
        raise RuntimeError('CatBoost is not installed in this environment. Use conda and the provided environment file.') from e

    clf = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=6, eval_metric='AUC', random_seed=42, early_stopping_rounds=50, verbose=100)
    # create Pools to pass categorical feature indices
    # create Pools from DataFrames and pass categorical column names
    try:
        # prefer passing DataFrames and column-names for categorical features
        if len(cats) > 0:
            pool_tr = Pool(X_tr_df, label=y_tr, cat_features=cats)
            pool_te = Pool(X_te_df, label=y_te, cat_features=cats)
        else:
            pool_tr = Pool(X_tr_df, label=y_tr)
            pool_te = Pool(X_te_df, label=y_te)
    except Exception:
        # fallback: convert DataFrames to numpy arrays (shouldn't happen since we converted cats to str)
        # fallback: convert to numpy arrays and pass integer indices for cat features
        X_tr = X_tr_df.values
        X_te = X_te_df.values
        if len(cats) > 0:
            cat_idx = [int(Xdf.columns.get_loc(c)) for c in cats]
            pool_tr = Pool(X_tr, label=y_tr, cat_features=cat_idx)
            pool_te = Pool(X_te, label=y_te, cat_features=cat_idx)
        else:
            pool_tr = Pool(X_tr, label=y_tr)
            pool_te = Pool(X_te, label=y_te)

    clf.fit(pool_tr, eval_set=pool_te)

    # predict using the Pool (recommended) to ensure categorical handling is consistent
    try:
        proba = clf.predict_proba(pool_te)[:,1]
    except Exception:
        # fallback to DataFrame/array if needed
        try:
            proba = clf.predict_proba(X_te_df)[:,1]
        except Exception:
            # use numpy array view (guaranteed defined above)
            proba = clf.predict_proba(X_te)[:,1]
    best_th, best_acc, best_conf = sweep_threshold(y_te, proba)

    # try compute ROC AUC
    try:
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(y_te, proba))
    except Exception:
        roc = None

    # ensure cat_idx exists for downstream recording
    try:
        cat_idx
    except NameError:
        cat_idx = [int(Xdf.columns.get_loc(c)) for c in cats] if len(cats)>0 else []

    results = {
        'n_rows': int(df.shape[0]),
        'n_features': int(Xdf.shape[1]),
        'categorical_columns': cats,
        'cat_idx': cat_idx,
        'best_threshold': best_th,
        'best_accuracy': best_acc,
        'confusion': best_conf,
        'roc_auc': roc,
        'model_info': clf.get_params()
    }

    with open(RES / 'catboost_raw_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    # save model
    joblib.dump({'catboost': clf, 'cat_features': cats, 'cat_idx': cat_idx, 'threshold': best_th}, MOD / 'catboost_raw.joblib')
    print('Saved results/catboost_raw_results.json and models/catboost_raw.joblib')

if __name__=='__main__':
    main()
