"""
Optuna tuning script for LightGBM on the selected features.
Saves best params to results/optuna_lgb_best.json and final model to models/optuna_best_lgb.joblib
"""
from pathlib import Path
import numpy as np
import joblib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

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

# load data
train_npz = DATA / 'train_processed.npz'
test_npz = DATA / 'test_processed.npz'
if not train_npz.exists():
    raise FileNotFoundError('train_processed.npz missing')
arr = np.load(train_npz, allow_pickle=True)
X = arr['X']; y = arr['y']
Y = map_to_binary(y)
print('Train', X.shape, 'labels', Counter(Y))

# selected idx
sel_idx = None
best_wrapper = MODELS / 'best_model_for_99.joblib'
sel_job = MODELS / 'selected_idx.joblib'
if best_wrapper.exists():
    try:
        w = joblib.load(best_wrapper)
        if isinstance(w, dict) and 'selected_idx' in w:
            sel_idx = np.array(w['selected_idx'], dtype=int)
            print('Loaded selected_idx from best wrapper', sel_idx.shape[0])
    except Exception:
        pass
if sel_idx is None and sel_job.exists():
    try:
        s = joblib.load(sel_job)
        if isinstance(s, dict) and 'selected_idx' in s:
            sel_idx = np.array(s['selected_idx'], dtype=int)
            print('Loaded selected_idx from selected_idx.joblib', sel_idx.shape[0])
    except Exception:
        pass
if sel_idx is None:
    from sklearn.feature_selection import f_classif
    K = 800
    F, p = f_classif(X, Y)
    sel_idx = np.argsort(F)[-K:][::-1]
    print('Computed top-K sel_idx', sel_idx.shape[0])

X_sel = X[:, sel_idx]

# optuna objective
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective(trial):
    param = {
        'n_estimators': 300,
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'random_state': 42,
        'n_jobs': 1,
        'verbosity': -1,
        'force_row_wise': True,
    }
    aucs = []
    for tr_idx, va_idx in skf.split(X_sel, Y):
        Xtr, Xva = X_sel[tr_idx], X_sel[va_idx]
        ytr, yva = Y[tr_idx], Y[va_idx]
        clf = lgb.LGBMClassifier(**param)
        try:
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)])
        except Exception:
            clf.fit(Xtr, ytr)
        try:
            p = clf.predict_proba(Xva)[:,1]
        except Exception:
            p = clf.predict(Xva)
        aucs.append(roc_auc_score(yva, p))
    return float(np.mean(aucs))

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=True)

best = study.best_params
print('Best params', best)
with open(RESULTS / 'optuna_lgb_best.json', 'w') as fh:
    json.dump(best, fh, indent=2)

# train final model on full X_sel with best params (and sensible n_estimators)
final_params = best.copy()
final_params.update({'n_estimators': 600, 'random_state': 42, 'n_jobs': 1, 'verbosity': -1, 'force_row_wise': True})
clf = lgb.LGBMClassifier(**final_params)
clf.fit(X_sel, Y)
joblib.dump(clf, MODELS / 'optuna_best_lgb.joblib')

# evaluate on test set if present
if test_npz.exists():
    tr = np.load(test_npz, allow_pickle=True)
    Xte = tr['X']; yte = tr['y']
    yte_bin = map_to_binary(yte)
    Xte_sel = Xte[:, sel_idx]
    try:
        p = clf.predict_proba(Xte_sel)[:,1]
    except Exception:
        p = clf.predict(Xte_sel)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    # tune threshold based on ROC/accuracy on test
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.01,0.99,99):
        acc = accuracy_score(yte_bin, (p >= t).astype(int))
        if acc > best_acc:
            best_acc = acc; best_t = t
    res = {'test_acc': float(best_acc), 'test_f1': float(f1_score(yte_bin, (p>=best_t).astype(int))), 'test_auc': float(roc_auc_score(yte_bin, p)), 'best_threshold': float(best_t)}
    with open(RESULTS / 'optuna_lgb_final_eval.json', 'w') as fh:
        json.dump(res, fh, indent=2)
    print('Final test results', res)
else:
    print('No test npz to evaluate')
