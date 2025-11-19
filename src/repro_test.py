"""Reproducibility test: reload final model, repeat deterministic split, score test set,
and compare metrics to results/catboost_raw_results.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'combine.csv'
MOD = ROOT / 'models'
RES = ROOT / 'results'


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
        try:
            if df[col].nunique(dropna=False) <= max_unique:
                cats.append(col)
        except Exception:
            pass
    return cats


def main():
    if not DATA.exists():
        raise SystemExit(f'{DATA} not found')

    df = pd.read_csv(DATA, low_memory=False)
    possible_labels = ['label','target','class','attack']
    label_col = None
    for c in possible_labels:
        if c in df.columns:
            label_col = c; break
    if label_col is None:
        label_col = df.columns[-1]

    y_raw = df[label_col].values
    y = map_to_binary(y_raw)

    Xdf = df.drop(columns=[label_col])
    cats = detect_categorical(Xdf, max_unique=100)
    if len(cats) > 0:
        for c in cats:
            Xdf[c] = Xdf[c].fillna('nan').astype(str)

    tr_idx, te_idx = stratified_split_numpy(y, test_size=0.15, seed=42)
    X_te_df = Xdf.iloc[te_idx].reset_index(drop=True)
    y_te = y[te_idx]

    # load model
    final_path = MOD / 'final_detector.joblib'
    if not final_path.exists():
        raise SystemExit(f'{final_path} not found')
    bundle = joblib.load(final_path)
    clf = bundle.get('catboost')

    # ensure cat columns are strings
    for c in cats:
        if c in X_te_df.columns:
            X_te_df[c] = X_te_df[c].fillna('nan').astype(str)

    try:
        proba = clf.predict_proba(X_te_df)[:,1]
    except Exception:
        # fallback to numpy
        proba = clf.predict_proba(X_te_df.values)[:,1]

    pred = (proba >= float(bundle.get('threshold', 0.5))).astype(int)

    acc = float(accuracy_score(y_te, pred))
    roc = float(roc_auc_score(y_te, proba))
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    prec = float(precision_score(y_te, pred))
    rec = float(recall_score(y_te, pred))
    f1 = float(f1_score(y_te, pred))

    print('Recomputed metrics on deterministic test split:')
    print(f'Accuracy: {acc:.10f}')
    print(f'ROC AUC:  {roc:.10f}')
    print(f'Confusion: [[{tn},{fp}],[{fn},{tp}]]')
    print(f'Precision: {prec:.6f} Recall: {rec:.6f} F1: {f1:.6f}')

    # compare to saved results
    res_path = RES / 'catboost_raw_results.json'
    if res_path.exists():
        with open(res_path, 'r') as fh:
            saved = json.load(fh)
        print('\nSaved results summary:')
        print(f"Saved accuracy: {saved.get('best_accuracy')}")
        print(f"Saved AUC: {saved.get('roc_auc')}")
        print(f"Saved confusion: {saved.get('confusion')}")


if __name__ == '__main__':
    main()
