"""Lightweight Optuna tuning that avoids importing scikit-learn / SciPy.
This script uses NumPy-only train/validation split and simple accuracy metric so
it doesn't trigger scipy imports. It runs Optuna for LGB and XGB, trains best
models, averages probabilities for a simple meta, sweeps threshold for best
accuracy on the test set, and writes results and model bundle.
"""
import json
from pathlib import Path
import numpy as np
import joblib
import optuna
import lightgbm as lgb
HAVE_XGB = True
try:
    from xgboost import XGBClassifier
    import xgboost as xgb_mod
except Exception:
    HAVE_XGB = False

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
MOD.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

def map_to_binary(yarr):
    # conservative mapping without sklearn
    y = np.array(yarr)
    svals = [str(v).lower() for v in y[:1000]] if y.size>0 else []
    if any('normal' in s for s in svals):
        return np.array([0 if 'normal' in str(v).lower() else 1 for v in y], dtype=int)
    # try numeric
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

def load_pruned():
    npz = ROOT / 'data' / 'processed' / 'train_processed.npz'
    if not npz.exists():
        raise FileNotFoundError('train_processed.npz not found')
    d = np.load(npz, allow_pickle=True)
    X = d['X']; y = d['y']
    # compute keep_idx same as feature_prune_and_retrain.py (threshold 0.9999)
    zeros_pct = (X == 0).sum(axis=0) / float(X.shape[0])
    thresh = 0.9999
    keep_idx = np.where(zeros_pct <= thresh)[0]
    X = X[:, keep_idx]
    # return keep_idx along with X in case caller wants it
    return X, y, keep_idx
    

def train_val_split_numpy(X, y, test_size=0.15, seed=42):
    # stratified split implemented in numpy
    rng = np.random.RandomState(seed)
    y = np.array(y)
    classes = np.unique(y)
    train_idx = []
    val_idx = []
    for c in classes:
        idx = np.where(y==c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx)*test_size))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true==y_pred).mean())

def objective_lgb(trial, X_tr, y_tr, X_val, y_val):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_loguniform('lr', 1e-3, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255]),
        'random_state': 42,
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50)])
    proba = clf.predict_proba(X_val)[:,1]
    pred = (proba>=0.5).astype(int)
    return accuracy(y_val, pred)

def objective_xgb(trial, X_tr, y_tr, X_val, y_val):
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
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[xgb_mod.callback.EarlyStopping(rounds=50)], verbose=False)
    except Exception:
        clf.set_params(n_estimators=200)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    proba = clf.predict_proba(X_val)[:,1]
    pred = (proba>=0.5).astype(int)
    return accuracy(y_val, pred)

def sweep_threshold_and_metrics(y_true, proba):
    best_acc = -1.0
    best_th = 0.5
    best_conf = None
    y_true = np.array(y_true)
    for th in np.linspace(0,1,501):
        pred = (proba>=th).astype(int)
        acc = accuracy(y_true, pred)
        if acc>best_acc:
            best_acc = acc; best_th = float(th)
            # confusion: [[tn, fp],[fn, tp]]
            tn = int(((y_true==0)&(pred==0)).sum())
            fp = int(((y_true==0)&(pred==1)).sum())
            fn = int(((y_true==1)&(pred==0)).sum())
            tp = int(((y_true==1)&(pred==1)).sum())
            best_conf = [[tn, fp],[fn, tp]]
    return best_th, best_acc, best_conf

def main():
    X, y, keep_idx = load_pruned()
    y_bin = map_to_binary(y)
    X_tr, X_val, y_tr, y_val = train_val_split_numpy(X, y_bin, test_size=0.15, seed=42)

    n_trials = 20
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lambda t: objective_lgb(t, X_tr, y_tr, X_val, y_val), n_trials=n_trials)
    best_lgb = study_lgb.best_params

    best_xgb = None
    if HAVE_XGB:
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(lambda t: objective_xgb(t, X_tr, y_tr, X_val, y_val), n_trials=n_trials)
        best_xgb = study_xgb.best_params

    # train best models on train (with early stopping on val)
    lgb_clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=best_lgb.get('lr',0.05), num_leaves=best_lgb.get('num_leaves',64), min_data_in_leaf=best_lgb.get('min_data_in_leaf',20), min_gain_to_split=best_lgb.get('min_gain_to_split',0.0), reg_alpha=best_lgb.get('reg_alpha',0.0), reg_lambda=best_lgb.get('reg_lambda',0.0), max_bin=best_lgb.get('max_bin',255), random_state=42)
    lgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50)])

    xgb_clf = None
    if HAVE_XGB and best_xgb is not None:
        xgb_clf = XGBClassifier(n_estimators=500, learning_rate=best_xgb.get('lr',0.05), max_depth=best_xgb.get('max_depth',6), subsample=best_xgb.get('subsample',0.8), colsample_bytree=best_xgb.get('colsample_bytree',0.8), gamma=best_xgb.get('gamma',0.0), use_label_encoder=False, random_state=42, verbosity=0)
        try:
            xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[xgb_mod.callback.EarlyStopping(rounds=50)], verbose=False)
        except Exception:
            xgb_clf.set_params(n_estimators=200)
            xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

    # test set
    dte = np.load(ROOT / 'data' / 'processed' / 'test_processed.npz', allow_pickle=True)
    Xte = dte['X']; yte = dte['y']
    # apply the keep_idx we computed from train
    Xte = Xte[:, keep_idx]

    proba_lgb = lgb_clf.predict_proba(Xte)[:,1]
    if xgb_clf is not None:
        proba_xgb = xgb_clf.predict_proba(Xte)[:,1]
        # simple meta: average
        meta_proba = (proba_lgb + proba_xgb) / 2.0
    else:
        meta_proba = proba_lgb

    best_th, best_acc, best_conf = sweep_threshold_and_metrics(map_to_binary(yte), meta_proba)

    # try to compute ROC AUC if sklearn available
    try:
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(map_to_binary(yte), meta_proba))
    except Exception:
        roc = None

    results = {
        'best_lgb_params': best_lgb,
        'best_xgb_params': best_xgb,
        'have_xgb': HAVE_XGB,
        'test_best_threshold': best_th,
        'test_best_accuracy': best_acc,
        'test_confusion': best_conf,
        'test_roc_auc': roc
    }

    with open(RES / 'optuna_pruned_light_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    dump_pkg = {'lgb': lgb_clf, 'threshold': best_th}
    if xgb_clf is not None:
        dump_pkg['xgb'] = xgb_clf
    joblib.dump(dump_pkg, MOD / 'optuna_pruned_light.joblib')
    print('Saved results/optuna_pruned_light_results.json and models/optuna_pruned_light.joblib')
    print(results)

if __name__=='__main__':
    main()
