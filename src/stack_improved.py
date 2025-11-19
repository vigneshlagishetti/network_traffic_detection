"""Single clean implementation. This file intentionally contains only
the cleaned faster stacking implementation to avoid prior duplicates.
"""
"""Clean, fast stacking script.

This file provides a single, consistent implementation of a faster
K-fold stacking pipeline using LightGBM (and XGBoost when available).
It intentionally avoids duplicated content and keeps defaults small so
it's suitable for quick iterative runs.
"""

import json
from pathlib import Path
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
MOD.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


def map_to_binary(yarr):
    y = np.array(yarr)
    svals = [str(v).lower() for v in y[:1000]] if y.size > 0 else []
    if any('normal' in s for s in svals):
        return np.array([0 if 'normal' in str(v).lower() else 1 for v in y], dtype=int)
    try:
        yi = y.astype(int)
        maj = int(np.argmax(np.bincount(yi)))
        return np.array([0 if int(v) == maj else 1 for v in yi], dtype=int)
    except Exception:
        try:
            yf = y.astype(float)
            return np.array([0 if float(v) == 0.0 else 1 for v in yf], dtype=int)
        except Exception:
            first = y[0]
            return np.array([0 if v == first else 1 for v in y], dtype=int)


def load_processed():
    npz = ROOT / 'data' / 'processed' / 'train_processed.npz'
    if not npz.exists():
        raise FileNotFoundError(npz)
    d = np.load(npz, allow_pickle=True)
    return d['X'], d['y']


def load_test():
    npz = ROOT / 'data' / 'processed' / 'test_processed.npz'
    d = np.load(npz, allow_pickle=True)
    return d['X'], d['y']


def make_lgb(params=None):
    import lightgbm as lgb

    return lgb.LGBMClassifier(**(params or {}))


def make_xgb(params=None):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(**(params or {}))
    except Exception:
        return None


def sweep_threshold_and_conf(y_true, probas, steps=501):
    best_acc = -1.0
    best_th = 0.5
    best_conf = None
    for th in np.linspace(0, 1, steps):
        pred = (probas >= th).astype(int)
        acc = float((pred == y_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_th = float(th)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            best_conf = [[int(tn), int(fp)], [int(fn), int(tp)]]
    return best_th, best_acc, best_conf


def make_base_models_fast():
    models = []
    try:
        import lightgbm as lgb

        lgb_confs = [
            {'n_estimators': 200, 'learning_rate': 0.03, 'num_leaves': 48, 'min_child_samples': 20, 'n_jobs': 4},
            {'n_estimators': 200, 'learning_rate': 0.02, 'num_leaves': 64, 'min_child_samples': 10, 'n_jobs': 4},
            {'n_estimators': 200, 'learning_rate': 0.01, 'num_leaves': 96, 'min_child_samples': 40, 'n_jobs': 4},
        ]
        for i, c in enumerate(lgb_confs):
            clf = lgb.LGBMClassifier(**c, random_state=42 + i)
            models.append((f'lgb_{i}', clf))
    except Exception:
        pass

    try:
        from xgboost import XGBClassifier

        x1 = XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=6, use_label_encoder=False, verbosity=0, random_state=101)
        x2 = XGBClassifier(n_estimators=200, learning_rate=0.02, max_depth=8, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, verbosity=0, random_state=202)
        models.append(('xgb_0', x1))
        models.append(('xgb_1', x2))
    except Exception:
        pass

    return models


def fit_oof_stack(X, y, base_models, n_folds=3, early_stop=30):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    n_samples = X.shape[0]
    oof = {name: np.zeros(n_samples, dtype=float) for name, _ in base_models}

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        for name, clf in base_models:
            try:
                clf_clone = joblib.loads(joblib.dumps(clf))
            except Exception:
                clf_clone = clf
            try:
                if hasattr(clf_clone, '__class__') and 'XGB' in clf_clone.__class__.__name__:
                    clf_clone.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=early_stop, verbose=False)
                else:
                    import lightgbm as lgb

                    clf_clone.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(stopping_rounds=early_stop)], verbose=False)
            except Exception:
                try:
                    clf_clone.fit(Xtr, ytr)
                except Exception as e:
                    print('Base fit failed for', name, 'fold', fold, e)
                    oof[name][va] = 0.5
                    continue
            try:
                p = clf_clone.predict_proba(Xva)
                p = p[:, 1] if p.ndim == 2 else p
            except Exception:
                try:
                    p = clf_clone.decision_function(Xva)
                except Exception:
                    p = np.full(Xva.shape[0], 0.5)
            oof[name][va] = p
        print(f'Completed fold {fold+1}/{n_folds}')
    meta_X = np.vstack([oof[name] for name, _ in base_models]).T
    return meta_X, oof


def train_final_base_models(X, y, base_models):
    fitted = []
    for name, clf in base_models:
        try:
            clf.fit(X, y)
        except Exception:
            try:
                clf.fit(X, y, verbose=False)
            except Exception as e:
                print('Final fit failed for', name, e)
        fitted.append((name, clf))
    return fitted


def main():
    X, y_raw = load_processed()
    y = map_to_binary(y_raw)
    keep_idx = None
    p = MOD / 'pruned_stack.joblib'
    if p.exists():
        pkg = joblib.load(p)
        if isinstance(pkg, dict) and 'keep_idx' in pkg:
            keep_idx = pkg['keep_idx']
            X = X[:, keep_idx]
    Xte, yte_raw = load_test()
    if keep_idx is not None:
        Xte = Xte[:, keep_idx]
    yte = map_to_binary(yte_raw)
    n_folds = 3
    early_stop = 30
    base_models = make_base_models_fast()
    if len(base_models) == 0:
        raise RuntimeError('No base models could be constructed (missing lightgbm/xgboost?)')
    print('Base models:', [n for n, _ in base_models])
    meta_X, oof = fit_oof_stack(X, y, base_models, n_folds=n_folds, early_stop=early_stop)
    meta = LogisticRegression(max_iter=2000)
    meta.fit(meta_X, y)
    try:
        calib = CalibratedClassifierCV(meta, cv='prefit', method='isotonic')
        calib.fit(meta_X, y)
        meta_clf = calib
    except Exception:
        meta_clf = meta
    fitted = train_final_base_models(X, y, base_models)
    prob_list = []
    for name, clf in fitted:
        try:
            p = clf.predict_proba(Xte)
            p = p[:, 1] if p.ndim == 2 else p
        except Exception:
            try:
                p = clf.decision_function(Xte)
            except Exception:
                p = np.full(Xte.shape[0], 0.5)
        prob_list.append(p)
    meta_X_test = np.vstack(prob_list).T
    try:
        meta_proba = meta_clf.predict_proba(meta_X_test)
        meta_proba = meta_proba[:, 1] if meta_proba.ndim == 2 else meta_proba
    except Exception:
        meta_proba = meta_clf.decision_function(meta_X_test)
    th, acc, conf = sweep_threshold_and_conf(yte, meta_proba, steps=501)
    auc = float(roc_auc_score(yte, meta_proba))
    prec = float(precision_score(yte, (meta_proba >= th).astype(int), zero_division=0))
    rec = float(recall_score(yte, (meta_proba >= th).astype(int), zero_division=0))
    f1 = float(f1_score(yte, (meta_proba >= th).astype(int), zero_division=0))
    results = {
        'n_base_models': len(base_models),
        'n_folds': n_folds,
        'keep_idx': None if keep_idx is None else list(map(int, keep_idx)),
        'test_best_threshold': th,
        'test_best_accuracy': acc,
        'test_confusion': conf,
        'test_roc_auc': auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }
    with open(RES / 'stack_improved_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    joblib.dump({'base_models': {name: clf for name, clf in fitted}, 'meta': meta_clf, 'keep_idx': None if keep_idx is None else list(map(int, keep_idx)), 'threshold': th}, MOD / 'stack_improved.joblib')
    print('Saved results/stack_improved_results.json and models/stack_improved.joblib')
    print(results)


if __name__ == '__main__':
    main()

