"""
Ensemble workflow using the original `combine.csv`.

Steps:
- load `combine.csv` (expects a `label` column)
- apply saved `preprocessing_pipeline.joblib` to obtain transformed features
- prefilter top-K features by f_classif (K=500)
- run ABA on the top-K subset to find compact feature mask (pop=12, n_iter=20)
- train LightGBM and XGBoost on selected features with OOF stacking (5-fold)
- evaluate on a holdout split and save models/results

Outputs:
- models/ensemble_lgb.joblib
- models/ensemble_xgb.joblib (if xgboost available)
- models/ensemble_stack_meta.joblib
- results/ensemble_results.json
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
print('ensemble_from_original: starting (cwd=', Path.cwd(), ')')

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

csvp = ROOT / 'combine.csv'
if not csvp.exists():
    raise FileNotFoundError(f"combine.csv not found at {csvp}")

# use the NSL-KDD loader which assigns the canonical column names
from src.preprocessing import load_nsl_kdd
df = load_nsl_kdd(str(csvp))
if 'label' not in df.columns:
    raise RuntimeError('Loaded dataframe does not contain "label" column')
y = df['label'].values
X_df = df.drop(columns=['label']).copy()

# coerce numeric columns except categorical ones
categorical_features = ["protocol_type", "service", "flag"]
for col in X_df.columns:
    if col not in categorical_features:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

# load saved preprocessing pipeline
pipe_path = ROOT / 'data' / 'processed' / 'preprocessing_pipeline.joblib'
if not pipe_path.exists():
    raise FileNotFoundError('Preprocessing pipeline not found. Run src/preprocessing.py first or ensure pipeline exists in data/processed')
pipeline = joblib.load(pipe_path)

X = pipeline.transform(X_df)
print('Transformed feature matrix shape:', X.shape)

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
print('Binary counts:', dict(Counter(y_bin)))

from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedShuffleSplit

# prefilter top-K
K = 500
F, pvals = f_classif(X, y_bin)
topk_idx = np.argsort(F)[-K:][::-1]
X_topk = X[:, topk_idx]
print('Top-K shape:', X_topk.shape)

# run ABA on top-K
from src.feature_selection.aba import ArtificialButterfly

pop_size = 12
n_iter = 20
aba = ArtificialButterfly(pop_size=pop_size, n_iter=n_iter, random_state=1)

def fitness_wrapper(X_sub, y_sub):
    # quick LightGBM CV fitness
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    try:
        import lightgbm as lgb
    except Exception:
        raise RuntimeError('lightgbm required for ABA fitness')
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=1, n_jobs=1)
    scores = cross_val_score(clf, X_sub, y_sub, cv=skf, scoring='f1_macro', n_jobs=1)
    return float(np.mean(scores))

print('Running ABA (this may take a while)...')
best_mask_topk, best_score = aba.fit(X_topk, y_bin, fitness_wrapper)
print('ABA done. best_score=', best_score, 'n_features=', int(best_mask_topk.sum()))

# map mask back to original transformed feature indices
selected_topk_indices = topk_idx[best_mask_topk.astype(bool)]
print('Selected feature count (global):', len(selected_topk_indices))

X_selected = X[:, selected_topk_indices]

# train/test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X_selected, y_bin))
X_tr, X_te = X_selected[train_idx], X_selected[test_idx]
y_tr, y_te = y_bin[train_idx], y_bin[test_idx]

# class weights
cnt = Counter(y_tr.tolist()); total = len(y_tr)
class_w = {k: total/(len(cnt)*v) for k,v in cnt.items()}
sw_tr = np.array([class_w[int(yy)] for yy in y_tr])

# train LightGBM
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
clf_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, random_state=42, n_jobs=1)
clf_lgb.fit(X_tr, y_tr, sample_weight=sw_tr)
joblib.dump(clf_lgb, MOD / 'ensemble_lgb.joblib')

# train XGBoost if available
has_xgb = False
try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    has_xgb = False

if has_xgb:
    clf_xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf_xgb.fit(X_tr, y_tr)
    joblib.dump(clf_xgb, MOD / 'ensemble_xgb.joblib')

# stacking using out-of-fold predictions
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

oof_preds = np.zeros((X_tr.shape[0], 0))
test_preds = np.zeros((X_te.shape[0], 0))

# first base: lgb
oof = np.zeros(X_tr.shape[0])
test_fold_preds = np.zeros((X_te.shape[0], skf.n_splits))
for i, (tr, val) in enumerate(skf.split(X_tr, y_tr)):
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, random_state=42, n_jobs=1)
    clf.fit(X_tr[tr], y_tr[tr], sample_weight=np.array([class_w[int(yy)] for yy in y_tr[tr]]))
    oof[val] = clf.predict_proba(X_tr[val])[:,1]
    test_fold_preds[:, i] = clf.predict_proba(X_te)[:,1]
oof_preds = np.column_stack([oof_preds, oof])
test_preds = np.column_stack([test_preds, test_fold_preds.mean(axis=1)])

if has_xgb:
    oof = np.zeros(X_tr.shape[0])
    test_fold_preds = np.zeros((X_te.shape[0], skf.n_splits))
    for i, (tr, val) in enumerate(skf.split(X_tr, y_tr)):
        clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
        clf.fit(X_tr[tr], y_tr[tr])
        oof[val] = clf.predict_proba(X_tr[val])[:,1]
        test_fold_preds[:, i] = clf.predict_proba(X_te)[:,1]
    oof_preds = np.column_stack([oof_preds, oof])
    test_preds = np.column_stack([test_preds, test_fold_preds.mean(axis=1)])

# meta learner
meta = LogisticRegression(max_iter=400)
meta.fit(oof_preds, y_tr)
joblib.dump(meta, MOD / 'ensemble_stack_meta.joblib')

# evaluate ensemble
meta_test_proba = meta.predict_proba(test_preds)[:,1]
meta_test_pred = (meta_test_proba >= 0.5).astype(int)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
res = {
    'lgb_test_acc': float(accuracy_score(y_te, (test_preds[:,0] >= 0.5).astype(int))),
    'lgb_test_f1': float(f1_score(y_te, (test_preds[:,0] >= 0.5).astype(int))),
}
if has_xgb:
    res.update({'xgb_test_acc': float(accuracy_score(y_te, (test_preds[:,1] >= 0.5).astype(int))),
                'xgb_test_f1': float(f1_score(y_te, (test_preds[:,1] >= 0.5).astype(int)))})
res.update({'stack_test_acc': float(accuracy_score(y_te, meta_test_pred)), 'stack_test_f1': float(f1_score(y_te, meta_test_pred)), 'stack_test_auc': float(roc_auc_score(y_te, meta_test_proba)), 'aba_best_score': float(best_score), 'n_selected_features': int(len(selected_topk_indices))})

with open(RES / 'ensemble_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)

print('Ensemble results:', res)
print('Saved models in', MOD)
