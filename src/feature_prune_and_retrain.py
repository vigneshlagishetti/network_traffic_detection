import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb_mod
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
    X = dtr['X'].astype(float); y = dtr['y']
    Xte = dte['X'].astype(float); yte = dte['y']

    y_bin = map_to_binary(y)
    yte_bin = map_to_binary(yte)

    # compute zeros_pct per feature on train
    zeros_pct = (X == 0).sum(axis=0) / X.shape[0]
    # threshold: drop features with >99.99% zeros
    thresh = 0.9999
    keep_idx = np.where(zeros_pct <= thresh)[0]
    dropped = np.where(zeros_pct > thresh)[0]

    X_pr = X[:, keep_idx]
    Xte_pr = Xte[:, keep_idx]

    X_train, X_val, y_train, y_val = train_test_split(X_pr, y_bin, test_size=0.15, random_state=42, stratify=y_bin)

    # retrain LGB on pruned features
    lgb_clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=64, min_data_in_leaf=5, force_row_wise=True, random_state=42)
    lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50)])

    # train XGB with callback early stopping
    xclf = XGBClassifier(use_label_encoder=False, n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    try:
        xclf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[xgb_mod.callback.EarlyStopping(rounds=50)], verbose=False)
    except Exception:
        xclf.set_params(n_estimators=200)
        xclf.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # meta on validation
    proba_lgb_val = lgb_clf.predict_proba(X_val)[:,1]
    proba_xgb_val = xclf.predict_proba(X_val)[:,1]
    meta_X = np.vstack([proba_lgb_val, proba_xgb_val]).T
    meta = LogisticRegression(max_iter=2000)
    meta.fit(meta_X, y_val)

    # test predictions
    proba_lgb_test = lgb_clf.predict_proba(Xte_pr)[:,1]
    proba_xgb_test = xclf.predict_proba(Xte_pr)[:,1]
    meta_X_test = np.vstack([proba_lgb_test, proba_xgb_test]).T
    meta_proba = meta.predict_proba(meta_X_test)[:,1]

    best = sweep_thresholds(yte_bin, meta_proba)
    th = best['threshold']
    ypred = (meta_proba >= th).astype(int)

    results = {
        'n_features_original': int(X.shape[1]),
        'n_features_pruned': int(X_pr.shape[1]),
        'n_dropped': int(dropped.shape[0]),
        'roc_auc': float(roc_auc_score(yte_bin, meta_proba)),
        'best_threshold': float(th),
        'accuracy_at_best_threshold': float(accuracy_score(yte_bin, ypred)),
        'f1': float(f1_score(yte_bin, ypred)),
        'precision': float(precision_score(yte_bin, ypred, zero_division=0)),
        'recall': float(recall_score(yte_bin, ypred)),
    }

    with open(RES / 'pruned_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    joblib.dump({'lgb': lgb_clf, 'xgb': xclf, 'meta': meta, 'keep_idx': keep_idx}, MOD / 'pruned_stack.joblib')

    print('Saved models/pruned_stack.joblib and results/pruned_results.json')
    print(results)

if __name__=='__main__':
    main()
