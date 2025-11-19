import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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

def main():
    npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
    npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
    if not npz_tr.exists() or not npz_te.exists():
        raise FileNotFoundError('Processed train/test npz not found')

    dtr = np.load(npz_tr, allow_pickle=True)
    X = dtr['X']; y = dtr['y']
    Xte = np.load(npz_te, allow_pickle=True)['X']; yte = np.load(npz_te, allow_pickle=True)['y']

    y_bin = map_to_binary(y)
    yte_bin = map_to_binary(yte)

    # split train->train/valid
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.15, random_state=42, stratify=y_bin)

    params = dict(
        objective='binary',
        boosting_type='gbdt',
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        min_data_in_leaf=10,
        min_gain_to_split=0.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        max_bin=255,
        force_row_wise=True,
        verbosity=-1,
        random_state=42,
    )

    clf = lgb.LGBMClassifier(n_estimators=2000, **params)
    # fit with callbacks
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )

    # save model
    out_model = MOD / 'final_lgb_model_retrained.joblib'
    joblib.dump(clf, out_model)

    # evaluate on test
    proba = clf.predict_proba(Xte)[:,1]
    th = 0.3506294007328901
    ypred = (proba >= th).astype(int)
    summary = {
        'model_path': str(out_model.name),
        'n_train': int(X_train.shape[0]),
        'n_val': int(X_val.shape[0]),
        'n_test': int(Xte.shape[0]),
        'accuracy': float(accuracy_score(yte_bin, ypred)),
        'f1': float(f1_score(yte_bin, ypred)),
        'precision': float(precision_score(yte_bin, ypred, zero_division=0)),
        'recall': float(recall_score(yte_bin, ypred, zero_division=0)),
        'roc_auc': float(roc_auc_score(yte_bin, proba)),
        'best_iteration': int(getattr(clf, 'best_iteration_', getattr(clf, 'best_iteration', -1)))
    }

    with open(RES / 'training_run_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    print('Retrained LGB saved to', out_model)
    print('Summary:', summary)

if __name__=='__main__':
    main()
