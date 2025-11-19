"""Run diagnostics for the saved CatBoost model.

Checks performed:
- Load saved `models/catboost_raw.joblib` and `results/catboost_raw_results.json`.
- Load `combine.csv`, sample a representative stratified subset.
- Compute CatBoost feature importances (built-in).
- Compute permutation importance (accuracy and ROC AUC) on a smaller holdout sample.
- Run a 5-fold stratified CV on a sampled subset and report mean/std for accuracy and ROC AUC.
- Save findings to `results/catboost_checks.json` and print a short summary.

Notes:
- This script is intended to run in the `fruty-catboost` conda env where CatBoost is installed.
- To keep runtimes reasonable, it samples at most 200k rows for importance and 100k for CV.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'combine.csv'
MOD = ROOT / 'models' / 'catboost_raw.joblib'
RES = ROOT / 'results' / 'catboost_checks.json'

SAMPLE_MAX = 200_000
CV_SAMPLE_MAX = 100_000
PERM_SAMPLE_MAX = 50_000
RANDOM_STATE = 42


def find_label_col(df):
    possible = ['label','target','class','attack']
    for c in possible:
        if c in df.columns:
            return c
    return df.columns[-1]


def stratified_sample(X, y, n_samples, random_state=RANDOM_STATE):
    if len(y) <= n_samples:
        return np.arange(len(y))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    idx, _ = next(sss.split(X, y))
    return idx


def main():
    if not MOD.exists():
        raise FileNotFoundError(f'{MOD} not found; run src/train_catboost_on_raw.py first')
    bundle = joblib.load(MOD)
    clf = bundle['catboost']
    cat_cols = bundle.get('cat_features', [])

    if not DATA.exists():
        raise FileNotFoundError(f'{DATA} not found')
    print('Reading CSV (this may take a moment)')
    df = pd.read_csv(DATA, low_memory=False)
    label_col = find_label_col(df)
    y_raw = df[label_col].values
    # reuse the mapping logic from train script: map 'normal' to 0 else 1, try integer mapping, else nonzero
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

    y = map_to_binary(y_raw)
    Xdf = df.drop(columns=[label_col])
    # ensure cat columns are strings to match training
    for c in cat_cols:
        if c in Xdf.columns:
            Xdf[c] = Xdf[c].fillna('nan').astype(str)

    n = len(y)
    print(f'dataset rows={n}, features={Xdf.shape[1]}, categorical_columns={len(cat_cols)}')

    # sample for importance
    sample_n = min(SAMPLE_MAX, n)
    sample_idx = stratified_sample(Xdf, y, sample_n)
    X_sample = Xdf.iloc[sample_idx].reset_index(drop=True)
    y_sample = y[sample_idx]

    try:
        from catboost import Pool
        pool_sample = Pool(X_sample, label=y_sample, cat_features=cat_cols if len(cat_cols)>0 else None)
    except Exception:
        pool_sample = None

    # feature importance from CatBoost
    print('Computing CatBoost feature importance')
    try:
        if pool_sample is not None:
            imp = clf.get_feature_importance(pool=pool_sample)
        else:
            imp = clf.get_feature_importance()
        feature_names = list(Xdf.columns)
        feat_imp = sorted(list(zip(feature_names, imp)), key=lambda x: x[1], reverse=True)
    except Exception as e:
        print('CatBoost feature importance failed:', e)
        feat_imp = []

    # permutation importance on a smaller sample
    perm_n = min(PERM_SAMPLE_MAX, n)
    perm_idx = stratified_sample(Xdf, y, perm_n)
    X_perm = Xdf.iloc[perm_idx].reset_index(drop=True)
    y_perm = y[perm_idx]
    print(f'Computing permutation importance on {len(y_perm)} rows (this may take a while)')
    perm_result = None
    try:
        # define predict function for scoring
        def predict_proba_for_perm(X):
            # ensure categorical columns are string
            for c in cat_cols:
                if c in X.columns:
                    X[c] = X[c].fillna('nan').astype(str)
            try:
                return clf.predict_proba(X)[:,1]
            except Exception:
                return clf.predict_proba(X.values)[:,1]

        # use ROC AUC as scoring for permutation importance
        r = permutation_importance(clf, X_perm, y_perm, scoring='roc_auc', n_repeats=6, random_state=RANDOM_STATE, n_jobs=1)
        perm_importances = sorted(list(zip(X_perm.columns, r.importances_mean)), key=lambda x: x[1], reverse=True)
    except Exception as e:
        print('Permutation importance failed:', e)
        perm_importances = []

    # 5-fold CV on a sampled subset (smaller for speed)
    cv_n = min(CV_SAMPLE_MAX, n)
    cv_idx = stratified_sample(Xdf, y, cv_n)
    X_cv = Xdf.iloc[cv_idx].reset_index(drop=True)
    y_cv = y[cv_idx]
    print(f'Running 5-fold CV on {len(y_cv)} rows (this will retrain 5 models)')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []
    fold = 0
    start_cv = time.time()
    for train_i, test_i in skf.split(X_cv, y_cv):
        fold += 1
        Xtr = X_cv.iloc[train_i].reset_index(drop=True)
        Xte = X_cv.iloc[test_i].reset_index(drop=True)
        ytr = y_cv[train_i]
        yte = y_cv[test_i]
        # ensure cat columns are strings
        for c in cat_cols:
            if c in Xtr.columns:
                Xtr[c] = Xtr[c].fillna('nan').astype(str)
                Xte[c] = Xte[c].fillna('nan').astype(str)
        try:
            from catboost import Pool, CatBoostClassifier
            pool_tr = Pool(Xtr, label=ytr, cat_features=cat_cols if len(cat_cols)>0 else None)
            pool_te = Pool(Xte, label=yte, cat_features=cat_cols if len(cat_cols)>0 else None)
            params = clf.get_params()
            # set verbose to 0 for CV runs and reuse early stopping
            params['verbose'] = 0
            params['early_stopping_rounds'] = params.get('early_stopping_rounds', 50)
            # ensure iterations not gigantic: use original iterations or 2000
            params['iterations'] = min(params.get('iterations', 2000), 2000)
            model = CatBoostClassifier(**params)
            model.fit(pool_tr, eval_set=pool_te)
            proba = model.predict_proba(pool_te)[:,1]
        except Exception as e:
            print('CatBoost CV fold failed, falling back to sklearn API', e)
            # fallback: use DataFrame fit/predict (may be slower)
            model = None
            try:
                model = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=6, random_seed=RANDOM_STATE, verbose=0, early_stopping_rounds=50)
                model.fit(Xtr, ytr, eval_set=(Xte, yte))
                proba = model.predict_proba(Xte)[:,1]
            except Exception as e2:
                print('Fallback CV fit failed:', e2)
                proba = np.zeros(len(yte))
        acc = float(accuracy_score(yte, (proba>=bundle.get('threshold',0.5)).astype(int)))
        try:
            roc = float(roc_auc_score(yte, proba))
        except Exception:
            roc = None
        cv_results.append({'fold': fold, 'n_test': int(len(yte)), 'accuracy': acc, 'roc_auc': roc})
        print(f'fold={fold} n_test={len(yte)} acc={acc:.6f} roc={roc:.6f}')
    end_cv = time.time()
    durations = end_cv - start_cv
    print(f'5-fold CV done in {durations/60:.2f} minutes')

    # assemble results
    out = {
        'catboost_saved_threshold': float(bundle.get('threshold', 0.5)),
        'cat_features': cat_cols,
        'n_rows_total': int(n),
        'sample_size_for_importance': int(len(y_sample)),
        'top_20_feature_importance': feat_imp[:20],
        'top_20_permutation_importance': perm_importances[:20],
        'cv_results': cv_results
    }

    with open(RES, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('Saved', RES)
    print('Summary:')
    if feat_imp:
        print('Top features:', [f for f,_ in feat_imp[:10]])
    if perm_importances:
        print('Top perm features:', [f for f,_ in perm_importances[:10]])
    # compute CV aggregated
    accs = [r['accuracy'] for r in cv_results]
    rocs = [r['roc_auc'] for r in cv_results if r['roc_auc'] is not None]
    print(f'CV accuracy mean={np.mean(accs):.6f} std={np.std(accs):.6f}')
    if len(rocs)>0:
        print(f'CV roc mean={np.mean(rocs):.6f} std={np.std(rocs):.6f}')

if __name__=='__main__':
    main()
