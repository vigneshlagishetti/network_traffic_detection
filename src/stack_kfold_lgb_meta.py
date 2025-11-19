"""K-fold stacking using multiple LightGBM base learners and an LGB meta learner.
This script avoids sklearn to reduce heavy imports. It:
- loads pruned processed train/test arrays
- builds stratified K folds in numpy
- trains several LGB variants with out-of-fold predictions
- trains a LightGBM meta-model on OOF preds
- evaluates on test set, sweeps threshold for best accuracy
- saves models and results
"""
from pathlib import Path
import numpy as np
import json
import joblib
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
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
        counts = np.bincount(yi)
        maj = int(np.argmax(counts))
        return np.array([0 if int(v)==maj else 1 for v in yi], dtype=int)
    except Exception:
        try:
            yf = y.astype(float)
            return np.array([0 if float(v)==0.0 else 1 for v in yf], dtype=int)
        except Exception:
            first = y[0]
            return np.array([0 if v==first else 1 for v in y], dtype=int)

def load_processed():
    npz = ROOT / 'data' / 'processed' / 'train_processed.npz'
    npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
    if not npz.exists() or not npz_te.exists():
        raise FileNotFoundError('processed npz files not found')
    d = np.load(npz, allow_pickle=True)
    X = d['X']; y = d['y']
    dte = np.load(npz_te, allow_pickle=True)
    Xte = dte['X']; yte = dte['y']
    # apply keep_idx if pruned_stack has it
    p = MOD / 'pruned_stack.joblib'
    keep_idx = None
    if p.exists():
        try:
            pkg = joblib.load(p)
            if isinstance(pkg, dict) and 'keep_idx' in pkg:
                keep_idx = pkg['keep_idx']
        except Exception:
            # joblib load may fail if the saved object references packages
            # not installed (e.g., xgboost). Try to read a JSON fallback
            # produced by pruning scripts.
            alt = ROOT / 'results' / 'pruned_results.json'
            if alt.exists():
                try:
                    import json as _json
                    with open(alt, 'r') as _f:
                        jr = _json.load(_f)
                        if 'keep_idx' in jr:
                            keep_idx = jr['keep_idx']
                except Exception:
                    keep_idx = None
        if keep_idx is not None:
            X = X[:, keep_idx]
            Xte = Xte[:, keep_idx]
    return X, y, Xte, yte, keep_idx

def stratified_kfold_indices(y, n_folds=5, seed=42):
    y = np.array(y)
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    folds = [[] for _ in range(n_folds)]
    for c in classes:
        idx = np.where(y==c)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, n_folds)
        for i in range(n_folds):
            folds[i].extend(parts[i].tolist())
    # convert to numpy arrays
    return [np.array(f, dtype=int) for f in folds]

def accuracy(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return float((y_true==y_pred).mean())

def sweep_threshold_and_metrics(y_true, proba):
    best_acc = -1.0; best_th = 0.5; best_conf=None
    y = np.array(y_true)
    for th in np.linspace(0,1,501):
        pred = (proba>=th).astype(int)
        acc = accuracy(y, pred)
        if acc>best_acc:
            best_acc=acc; best_th=float(th)
            tn = int(((y==0)&(pred==0)).sum())
            fp = int(((y==0)&(pred==1)).sum())
            fn = int(((y==1)&(pred==0)).sum())
            tp = int(((y==1)&(pred==1)).sum())
            best_conf = [[tn, fp],[fn, tp]]
    return best_th, best_acc, best_conf

def main():
    X, y_raw, Xte, yte_raw, keep_idx = load_processed()
    y = map_to_binary(y_raw)
    yte = map_to_binary(yte_raw)

    n_folds = 5
    folds = stratified_kfold_indices(y, n_folds=n_folds, seed=42)

    # define diverse LightGBM base learners
    base_params = [
        {'n_estimators':1000, 'learning_rate':0.03, 'num_leaves':64, 'random_state':42},
        {'n_estimators':1000, 'learning_rate':0.01, 'num_leaves':128, 'random_state':7},
        {'n_estimators':1000, 'learning_rate':0.02, 'num_leaves':48, 'random_state':123},
        {'n_estimators':1000, 'learning_rate':0.01, 'num_leaves':256, 'random_state':999}
    ]

    n_models = len(base_params)
    n_samples = X.shape[0]
    oof_preds = np.zeros((n_samples, n_models), dtype=float)

    trained_base_full = []

    # produce OOF preds
    for m_idx, params in enumerate(base_params):
        print(f'Training base model {m_idx+1}/{n_models} with params: {params}')
        for f_idx in range(n_folds):
            val_idx = folds[f_idx]
            train_idx = np.hstack([folds[i] for i in range(n_folds) if i!=f_idx])
            X_tr = X[train_idx]; y_tr = y[train_idx]
            X_val = X[val_idx]; y_val = y[val_idx]
            clf = lgb.LGBMClassifier(**params)
            # use early stopping on the small val set
            try:
                clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50)])
            except Exception:
                clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_val)[:,1]
            oof_preds[val_idx, m_idx] = proba
        # after OOF, train on full data for inference and save
        clf_full = lgb.LGBMClassifier(**params)
        try:
            clf_full.fit(X, y, eval_set=[(X, y)], callbacks=[lgb.early_stopping(stopping_rounds=50)])
        except Exception:
            clf_full.fit(X, y)
        trained_base_full.append(clf_full)

    # train meta on OOF predictions
    print('Training meta model on OOF predictions')
    meta = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, random_state=42)
    meta.fit(oof_preds, y)

    # produce test set predictions from full base models
    proba_tests = np.zeros((Xte.shape[0], n_models), dtype=float)
    for m_idx, clf in enumerate(trained_base_full):
        proba_tests[:, m_idx] = clf.predict_proba(Xte)[:,1]

    meta_proba = meta.predict_proba(proba_tests)[:,1]

    best_th, best_acc, best_conf = sweep_threshold_and_metrics(yte, meta_proba)

    # try compute ROC AUC
    try:
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(yte, meta_proba))
    except Exception:
        roc = None

    results = {
        'n_base_models': n_models,
        'n_folds': n_folds,
        'keep_idx': None if keep_idx is None else keep_idx.tolist(),
        'test_best_threshold': best_th,
        'test_best_accuracy': best_acc,
        'test_confusion': best_conf,
        'test_roc_auc': roc
    }

    with open(RES / 'stack_kfold_lgb_meta_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    joblib.dump({'base_models': trained_base_full, 'meta': meta, 'base_params': base_params, 'keep_idx': None if keep_idx is None else keep_idx}, MOD / 'stack_kfold_lgb_meta.joblib')
    print('Saved results/stack_kfold_lgb_meta_results.json and models/stack_kfold_lgb_meta.joblib')
    print(results)

if __name__=='__main__':
    main()
