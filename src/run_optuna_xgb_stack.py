import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import optuna
import lightgbm as lgb
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz_tr.exists() or not npz_te.exists():
    raise FileNotFoundError('processed train/test npz not found')
arr = np.load(npz_tr, allow_pickle=True)
X = arr['X']; y = arr['y']
arrt = np.load(npz_te, allow_pickle=True)
Xte = arrt['X']; yte = arrt['y']

# robust binary mapping

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

# sample weights
unique, counts = np.unique(y_bin, return_counts=True)
class_counts = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
print('Train class counts:', class_counts)
total = len(y_bin)
class_weight = {k: total/(2*v) for k,v in class_counts.items()}
sample_weight = np.array([class_weight[int(yi)] for yi in y_bin])

# Optuna objective for LightGBM

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 4, 16),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        # use LightGBM-friendly names to avoid warnings
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
    }
    # small internal split for tuning
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(X, y_bin, sample_weight, test_size=0.2, stratify=y_bin, random_state=42)
    clf = lgb.LGBMClassifier(n_estimators=2000, random_state=42, n_jobs=-1, **params)
    # use callback API for early stopping to be compatible with LightGBM versions
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        sample_weight=w_tr,
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)],
    )
    proba = clf.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, proba)
    return auc

study = optuna.create_study(direction='maximize')
print('Starting Optuna study (50 trials) for LightGBM AUC')
study.optimize(objective, n_trials=50)
print('Best study params:', study.best_params)

# train final tuned LGB on full train
best_params = study.best_params
best_lgb = lgb.LGBMClassifier(n_estimators=2000, random_state=42, n_jobs=-1, **best_params)
best_lgb.fit(X, y_bin, sample_weight=sample_weight, eval_metric='auc')
joblib.dump(best_lgb, MOD / 'lgb_optuna_tuned.joblib')

# weighted lgb baseline
scale_pos = class_counts.get(0,1)/class_counts.get(1,1)
weighted_lgb = lgb.LGBMClassifier(n_estimators=1000, random_state=42, scale_pos_weight=scale_pos, n_jobs=-1)
weighted_lgb.fit(X, y_bin)
joblib.dump(weighted_lgb, MOD / 'lgb_weighted_baseline.joblib')

# train XGBoost baseline
xgb = XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42, n_estimators=1000, verbosity=0, n_jobs=-1)
xgb.fit(X, y_bin)
joblib.dump(xgb, MOD / 'xgb_baseline.joblib')

# stacking: tuned lgb + xgb + weighted lgb
estimators = [('lgb_tuned', best_lgb), ('xgb', xgb), ('lgb_weighted', weighted_lgb)]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=-1, passthrough=False)
print('Fitting stacking classifier on full train')
stack.fit(X, y_bin)
joblib.dump(stack, MOD / 'stack_lgb_xgb.joblib')

# evaluate all candidates on test
candidates = {'lgb_tuned': best_lgb, 'lgb_weighted': weighted_lgb, 'xgb': xgb, 'stack': stack}
summary = {}
for name, mdl in candidates.items():
    try:
        proba = mdl.predict_proba(Xte)[:,1]
    except Exception:
        try:
            dec = mdl.decision_function(Xte)
            proba = 1/(1+np.exp(-dec))
        except Exception:
            summary[name] = {'error': 'no proba/decision_function'}
            continue
    preds = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(yte_bin, preds))
    f1 = float(f1_score(yte_bin, preds))
    prec = float(precision_score(yte_bin, preds, zero_division=0))
    rec = float(recall_score(yte_bin, preds, zero_division=0))
    auc = float(roc_auc_score(yte_bin, proba))
    tn, fp, fn, tp = confusion_matrix(yte_bin, preds).ravel()
    spec = float(tn/(tn+fp)) if (tn+fp)>0 else 0.0
    summary[name] = {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'roc_auc': auc, 'specificity': spec}

# pick best by composite
best_name = None; best_comp = -1
for n, m in summary.items():
    comp = 0.5*(m['roc_auc'] + m['accuracy'])
    if comp > best_comp:
        best_comp = comp; best_name = n

# save results
with open(RES / 'optuna_xgb_stack_results.json', 'w') as fh:
    json.dump({'summary': summary, 'best_model': best_name, 'composite': best_comp, 'study_params': study.best_params}, fh, indent=2)

# copy best model to final
import shutil
src = MOD / (best_name + '.joblib' if best_name!='stack' else 'stack_lgb_xgb.joblib')
shutil.copy(src, MOD / 'final_model_best.joblib')
print('Done. Best model:', best_name)
print('Saved results to', RES / 'optuna_xgb_stack_results.json')
