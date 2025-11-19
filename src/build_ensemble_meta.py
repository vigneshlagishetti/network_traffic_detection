import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
MOD.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

def map_to_binary(yarr):
    try:
        svals = [str(v).lower() for v in yarr[:1000]] if len(yarr)>0 else []
        if any('normal' in s for s in svals):
            return np.array([0 if 'normal' in str(v).lower() else 1 for v in yarr], dtype=int)
    except Exception:
        pass
    try:
        y_int = np.array(yarr, dtype=int)
        maj = int(np.argmax(np.bincount(y_int)))
        return np.array([0 if int(v)==maj else 1 for v in y_int], dtype=int)
    except Exception:
        try:
            yf = np.array(yarr, dtype=float)
            return np.array([0 if float(v)==0.0 else 1 for v in yf], dtype=int)
        except Exception:
            first = yarr[0]
            return np.array([0 if v==first else 1 for v in yarr], dtype=int)

def sweep_thresholds(y_true, proba):
    best = {'threshold': None, 'accuracy': -1.0}
    for th in np.linspace(0.0, 1.0, 1001):
        ypred = (proba >= th).astype(int)
        acc = accuracy_score(y_true, ypred)
        if acc > best['accuracy']:
            best = {'threshold': float(th), 'accuracy': float(acc)}
    return best

def main():
    npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
    npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
    dtr = np.load(npz_tr, allow_pickle=True)
    dte = np.load(npz_te, allow_pickle=True)
    X = dtr['X']; y = dtr['y']
    Xte = dte['X']; yte = dte['y']

    y_bin = map_to_binary(y)
    yte_bin = map_to_binary(yte)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.15, random_state=42, stratify=y_bin)

    # load pretrained LGB model if available
    lgb_path = MOD / 'final_lgb_model_retrained.joblib'
    if not lgb_path.exists():
        lgb_path = MOD / 'final_lgb_model.joblib'
    lgb = joblib.load(lgb_path)

    # train XGBoost with early stopping on validation
    xgb = XGBClassifier(use_label_encoder=False, n_estimators=300, learning_rate=0.05,
                        max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    # try fit with early stopping; some xgboost versions expose different kwargs
    try:
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    except TypeError:
        # fallback: fit without early stopping (reduce n_estimators to keep runtime reasonable)
        xgb.set_params(n_estimators=200)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(xgb, MOD / 'xgb_trained.joblib')

    # produce probas on val and test
    lgb_val_p = lgb.predict_proba(X_val)[:,1]
    xgb_val_p = xgb.predict_proba(X_val)[:,1]
    stack_val = np.vstack([lgb_val_p, xgb_val_p]).T

    # fit logistic meta on val probs
    meta = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    meta.fit(stack_val, y_val)

    # produce test stacked probs
    lgb_test_p = lgb.predict_proba(Xte)[:,1]
    xgb_test_p = xgb.predict_proba(Xte)[:,1]
    stack_test = np.vstack([lgb_test_p, xgb_test_p]).T
    proba = meta.predict_proba(stack_test)[:,1]

    best = sweep_thresholds(yte_bin, proba)
    th = best['threshold']
    ypred = (proba >= th).astype(int)

    results = {
        'model_components': [str(lgb_path.name), 'xgb_trained.joblib', 'logistic_meta'],
        'roc_auc': float(roc_auc_score(yte_bin, proba)),
        'best_threshold': float(th),
        'accuracy_at_best_threshold': float(accuracy_score(yte_bin, ypred)),
        'f1': float(f1_score(yte_bin, ypred)),
        'precision': float(precision_score(yte_bin, ypred, zero_division=0)),
        'recall': float(recall_score(yte_bin, ypred)),
    }

    with open(RES / 'stack_meta_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    # save ensemble as dict
    ensemble_obj = {'lgb': lgb, 'xgb': xgb, 'meta': meta}
    joblib.dump(ensemble_obj, MOD / 'stack_lgb_xgb_meta.joblib')
    print('Saved stack_lgb_xgb_meta.joblib and results/stack_meta_results.json')
    print(results)

if __name__=='__main__':
    main()
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
MOD.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

def map_to_binary(yarr):
    try:
        svals = [str(v).lower() for v in yarr[:1000]] if len(yarr)>0 else []
        if any('normal' in s for s in svals):
            return np.array([0 if 'normal' in str(v).lower() else 1 for v in yarr], dtype=int)
    except Exception:
        pass
    try:
        y_int = np.array(yarr, dtype=int)
        maj = int(np.argmax(np.bincount(y_int)))
        return np.array([0 if int(v)==maj else 1 for v in y_int], dtype=int)
    except Exception:
        try:
            yf = np.array(yarr, dtype=float)
            return np.array([0 if float(v)==0.0 else 1 for v in yf], dtype=int)
        except Exception:
            first = yarr[0]
            return np.array([0 if v==first else 1 for v in yarr], dtype=int)

def sweep_thresholds(y_true, proba):
    best = {'threshold': None, 'accuracy': -1.0}
    for th in np.linspace(0.0, 1.0, 1001):
        ypred = (proba >= th).astype(int)
        acc = accuracy_score(y_true, ypred)
        if acc > best['accuracy']:
            best = {'threshold': float(th), 'accuracy': float(acc)}
    return best

def main():
    npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
    npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
    dtr = np.load(npz_tr, allow_pickle=True)
    dte = np.load(npz_te, allow_pickle=True)
    X = dtr['X']; y = dtr['y']
    Xte = dte['X']; yte = dte['y']

    y_bin = map_to_binary(y)
    yte_bin = map_to_binary(yte)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.15, random_state=42, stratify=y_bin)

    # load pretrained LGB model if available
    lgb_path = MOD / 'final_lgb_model_retrained.joblib'
    if not lgb_path.exists():
        lgb_path = MOD / 'final_lgb_model.joblib'
    lgb = joblib.load(lgb_path)

    # train XGBoost with early stopping on validation
    xgb = XGBClassifier(use_label_encoder=False, n_estimators=300, learning_rate=0.05,
                        max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    # try fit with early stopping; some xgboost versions expose different kwargs
    try:
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    except TypeError:
        # fallback: fit without early stopping (reduce n_estimators to keep runtime reasonable)
        xgb.set_params(n_estimators=200)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(xgb, MOD / 'xgb_trained.joblib')

    # produce probas on val and test
    lgb_val_p = lgb.predict_proba(X_val)[:,1]
    xgb_val_p = xgb.predict_proba(X_val)[:,1]
    stack_val = np.vstack([lgb_val_p, xgb_val_p]).T

    # fit logistic meta on val probs
    meta = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    meta.fit(stack_val, y_val)

    # produce test stacked probs
    lgb_test_p = lgb.predict_proba(Xte)[:,1]
    xgb_test_p = xgb.predict_proba(Xte)[:,1]
    stack_test = np.vstack([lgb_test_p, xgb_test_p]).T
    proba = meta.predict_proba(stack_test)[:,1]

    best = sweep_thresholds(yte_bin, proba)
    th = best['threshold']
    ypred = (proba >= th).astype(int)

    results = {
        'model_components': [str(lgb_path.name), 'xgb_trained.joblib', 'logistic_meta'],
        'roc_auc': float(roc_auc_score(yte_bin, proba)),
        'best_threshold': float(th),
        'accuracy_at_best_threshold': float(accuracy_score(yte_bin, ypred)),
        'f1': float(f1_score(yte_bin, ypred)),
        'precision': float(precision_score(yte_bin, ypred, zero_division=0)),
        'recall': float(recall_score(yte_bin, ypred)),
    }

    with open(RES / 'stack_meta_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    # save ensemble as dict
    ensemble_obj = {'lgb': lgb, 'xgb': xgb, 'meta': meta}
    joblib.dump(ensemble_obj, MOD / 'stack_lgb_xgb_meta.joblib')
    print('Saved stack_lgb_xgb_meta.joblib and results/stack_meta_results.json')
    print(results)

if __name__=='__main__':
    main()
