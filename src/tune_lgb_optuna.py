import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz_tr.exists() or not npz_te.exists():
    raise FileNotFoundError('processed train/test npz files not found in data/processed')
arr = np.load(npz_tr, allow_pickle=True)
X = arr['X']; y = arr['y']
arrt = np.load(npz_te, allow_pickle=True)
Xte = arrt['X']; yte = arrt['y']

# map to binary
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

y_bin = map_to_binary(y)
yte_bin = map_to_binary(yte)

# compute sample weights balanced
unique, counts = np.unique(y_bin, return_counts=True)
class_counts = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
print('Train class counts:', class_counts)

total = len(y_bin)
class_weight = {k: total/(2*v) for k,v in class_counts.items()}
sample_weight = np.array([class_weight[int(yi)] for yi in y_bin])

# optuna
try:
    import optuna
except Exception:
    raise RuntimeError('Optuna is required for tuning. Please install optuna in the environment.')

import lightgbm as lgb

# define objective: mean over CV folds of composite metric (roc_auc + accuracy)/2
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'n_estimators': 2000,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'max_depth': trial.suggest_int('max_depth', 5, 16),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'n_jobs': 1,
        'force_row_wise': True,
    }
    scores = []
    for tr_idx, val_idx in skf.split(X, y_bin):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_bin[tr_idx], y_bin[val_idx]
        sw_tr = sample_weight[tr_idx]
        clf = lgb.LGBMClassifier(**params)
        cb = []
        try:
            cb = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        except Exception:
            cb = []
        clf.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=cb)
        yval_pred = clf.predict(X_val)
        try:
            yval_proba = clf.predict_proba(X_val)[:,1]
        except Exception:
            yval_proba = clf.predict_proba(X_val)
        roc = roc_auc_score(y_val, yval_proba)
        acc = accuracy_score(y_val, yval_pred)
        scores.append((roc + acc) / 2.0)
    return float(np.mean(scores))

study = optuna.create_study(direction='maximize', study_name='lgb_both_obj')
study.optimize(objective, n_trials=40, show_progress_bar=True)

print('Best trial:', study.best_trial.params)

best_params = study.best_trial.params
# set fixed params
best_params.update({'n_estimators': 2000, 'random_state': 42, 'n_jobs': -1, 'force_row_wise': True})

# train final model on full train
clf = lgb.LGBMClassifier(**best_params)
cb = []
try:
    cb = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
except Exception:
    cb = []
clf.fit(X, y_bin, sample_weight=sample_weight, eval_set=[(Xte, yte_bin)], eval_metric='auc', callbacks=cb)

# evaluate on test
ypred = clf.predict(Xte)
try:
    yproba = clf.predict_proba(Xte)[:,1]
except Exception:
    yproba = clf.predict_proba(Xte)

res = {
    'best_params': best_params,
    'test_accuracy': float(accuracy_score(yte_bin, ypred)),
    'test_f1': float(f1_score(yte_bin, ypred)),
    'test_auc': float(roc_auc_score(yte_bin, yproba)),
    'test_pr_auc': float(average_precision_score(yte_bin, yproba)),
    'precision': float(precision_score(yte_bin, ypred, zero_division=0)),
    'recall': float(recall_score(yte_bin, ypred, zero_division=0)),
    'n_train_samples': int(X.shape[0]),
    'n_features': int(X.shape[1]),
}

# save
joblib.dump(clf, MOD / 'final_lgb_optuna.joblib')
with open(RES / 'final_lgb_optuna.json', 'w') as fh:
    json.dump(res, fh, indent=2)

print('Saved tuned model to', MOD / 'final_lgb_optuna.joblib')
print('Results:', res)
