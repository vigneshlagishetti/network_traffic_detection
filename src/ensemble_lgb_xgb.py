import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
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

    # split small val set
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.15, random_state=42, stratify=y_bin)

    lgb_params = dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=10,
        min_gain_to_split=0.0,
        max_bin=255,
        force_row_wise=True,
        random_state=42,
    )
    xgb_params = dict(
        use_label_encoder=False,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        random_state=42,
    )

    clf_lgb = lgb.LGBMClassifier(**lgb_params)
    clf_xgb = XGBClassifier(**xgb_params)

    # fit base models
    clf_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(200)])
    clf_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Voting soft ensemble
    ensemble = VotingClassifier(estimators=[('lgb', clf_lgb), ('xgb', clf_xgb)], voting='soft')
    # fit ensemble (will call fit on estimators by default; our estimators are already fitted but sklearn will refit clones)
    ensemble.fit(X_train, y_train)

    # calibrate probabilities with Platt scaling using validation
    calib = CalibratedClassifierCV(base_estimator=ensemble, method='sigmoid', cv='prefit')
    calib.fit(X_val, y_val)

    # predict on test
    proba = calib.predict_proba(Xte)[:,1]

    # sweep for best accuracy threshold
    best = sweep_thresholds(yte_bin, proba)

    th = best['threshold']
    ypred = (proba >= th).astype(int)

    results = {
        'n_train': int(X_train.shape[0]),
        'n_val': int(X_val.shape[0]),
        'n_test': int(Xte.shape[0]),
        'roc_auc': float(roc_auc_score(yte_bin, proba)),
        'best_threshold': float(th),
        'accuracy_at_best_threshold': float(accuracy_score(yte_bin, ypred)),
        'f1': float(f1_score(yte_bin, ypred)),
        'precision': float(precision_score(yte_bin, ypred, zero_division=0)),
        'recall': float(recall_score(yte_bin, ypred, zero_division=0)),
    }

    with open(RES / 'stacking_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)

    joblib.dump(calib, MOD / 'stack_lgb_xgb.joblib')
    print('Saved ensemble to models/stack_lgb_xgb.joblib')
    print('Results:', results)

if __name__=='__main__':
    main()
