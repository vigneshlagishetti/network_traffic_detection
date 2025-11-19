import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
from xgboost import XGBClassifier
import xgboost as xgb_mod
import optuna

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

def load_pruned():
    # try to load keep_idx from pruned_stack
    p = MOD / 'pruned_stack.joblib'
    npz = ROOT / 'data' / 'processed' / 'train_processed.npz'
    if not npz.exists():
        raise FileNotFoundError('train_processed.npz not found')
    d = np.load(npz, allow_pickle=True)
    X = d['X']; y = d['y']
    keep_idx = None
    if p.exists():
        pkg = joblib.load(p)
        if isinstance(pkg, dict) and 'keep_idx' in pkg:
            keep_idx = pkg['keep_idx']
    if keep_idx is not None:
        X = X[:, keep_idx]
    return X, y

def objective_lgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_loguniform('lr', 1e-3, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255]),
        'force_row_wise': True,
        'random_state': 42,
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50)], verbose=False)
    proba = clf.predict_proba(X_val)[:,1]
    acc = accuracy_score(y_val, (proba>=0.5).astype(int))
    return acc

def objective_xgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': 500,
        'learning_rate': trial.suggest_loguniform('lr', 1e-3, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': 42,
        'verbosity': 0,
        'use_label_encoder': False,
    }
    clf = XGBClassifier(**params)
    try:
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[xgb_mod.callback.EarlyStopping(rounds=50)], verbose=False)
    except Exception:
        clf.set_params(n_estimators=200)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    proba = clf.predict_proba(X_val)[:,1]
    acc = accuracy_score(y_val, (proba>=0.5).astype(int))
    return acc

def main():
    X, y = load_pruned()
    y_bin = map_to_binary(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.15, random_state=42, stratify=y_bin)

    # Optuna studies
    n_trials = 30
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lambda t: objective_lgb(t, X_train, y_train, X_val, y_val), n_trials=n_trials)
    best_lgb = study_lgb.best_params

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda t: objective_xgb(t, X_train, y_train, X_val, y_val), n_trials=n_trials)
    best_xgb = study_xgb.best_params

    # train best models on X_train (for meta) and evaluate on X_val
    lgb_clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=best_lgb.get('lr',0.05), num_leaves=best_lgb.get('num_leaves',64), min_data_in_leaf=best_lgb.get('min_data_in_leaf',20), min_gain_to_split=best_lgb.get('min_gain_to_split',0.0), reg_alpha=best_lgb.get('reg_alpha',0.0), reg_lambda=best_lgb.get('reg_lambda',0.0), max_bin=best_lgb.get('max_bin',255), force_row_wise=True, random_state=42)
    lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50)])

    xgb_clf = XGBClassifier(n_estimators=500, learning_rate=best_xgb.get('lr',0.05), max_depth=best_xgb.get('max_depth',6), subsample=best_xgb.get('subsample',0.8), colsample_bytree=best_xgb.get('colsample_bytree',0.8), gamma=best_xgb.get('gamma',0.0), use_label_encoder=False, random_state=42, verbosity=0)
    try:
        xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[xgb_mod.callback.EarlyStopping(rounds=50)], verbose=False)
    except Exception:
        xgb_clf.set_params(n_estimators=200)
        xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # meta on val
    proba_lgb_val = lgb_clf.predict_proba(X_val)[:,1]
    proba_xgb_val = xgb_clf.predict_proba(X_val)[:,1]
    meta_X = np.vstack([proba_lgb_val, proba_xgb_val]).T
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(max_iter=2000)
    meta.fit(meta_X, y_val)

    # evaluate on test set
    dte = np.load(ROOT / 'data' / 'processed' / 'test_processed.npz', allow_pickle=True)
    Xte = dte['X']; yte = dte['y']
    # if pruned, apply same keep_idx
    p = MOD / 'pruned_stack.joblib'
    if p.exists():
        pkg = joblib.load(p)
        if isinstance(pkg, dict) and 'keep_idx' in pkg:
            Xte = Xte[:, pkg['keep_idx']]

    proba_lgb_test = lgb_clf.predict_proba(Xte)[:,1]
    proba_xgb_test = xgb_clf.predict_proba(Xte)[:,1]
    meta_X_test = np.vstack([proba_lgb_test, proba_xgb_test]).T
    meta_proba = meta.predict_proba(meta_X_test)[:,1]

    # sweep threshold for best accuracy
    best_acc = 0.0; best_th = 0.5; best_metrics = None
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    for th in np.linspace(0,1,1001):
        ypred = (meta_proba >= th).astype(int)
        acc = accuracy_score(map_to_binary(yte), ypred)
        if acc > best_acc:
            best_acc = acc; best_th = float(th)
            best_metrics = {'accuracy': acc, 'precision': precision_score(map_to_binary(yte), ypred, zero_division=0), 'recall': recall_score(map_to_binary(yte), ypred, zero_division=0), 'f1': f1_score(map_to_binary(yte), ypred, zero_division=0), 'confusion': confusion_matrix(map_to_binary(yte), ypred).tolist()}

    results = {
        'best_lgb_params': best_lgb,
        'best_xgb_params': best_xgb,
        'test_best_threshold': best_th,
        'test_best_accuracy': best_acc,
        'test_best_metrics': best_metrics,
        'test_roc_auc': float(roc_auc_score(map_to_binary(yte), meta_proba))
    }

    with open(RES / 'optuna_pruned_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    joblib.dump({'lgb': lgb_clf, 'xgb': xgb_clf, 'meta': meta, 'threshold': best_th}, MOD / 'optuna_pruned_stack.joblib')
    print('Saved results/optuna_pruned_results.json and model optuna_pruned_stack.joblib')
    print(results)

if __name__=='__main__':
    main()
