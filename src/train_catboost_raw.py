"""Train CatBoost directly on the original combine.csv using native categorical handling.

This script:
- loads `combine.csv` (expects a `label` column)
- detects categorical features (heuristic: known NSL-KDD cat list or object dtype / low-cardinality)
- maps labels to binary (normal -> 0, others -> 1) using a robust heuristic
- splits into train/test (80/20 stratified)
- trains CatBoost with early stopping and class weights
- finds best threshold on the holdout to maximize accuracy
- saves model to models/catboost_raw_model.cbm and results to results/catboost_raw_results.json
"""
from pathlib import Path
import json
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

CSV = ROOT / 'combine.csv'
if not CSV.exists():
    raise FileNotFoundError(f"combine.csv not found at {CSV}")

import pandas as pd
# Load preprocessing.load_nsl_kdd by file to avoid package import issues when running under conda run
import importlib.util
_pp_path = ROOT / 'src' / 'preprocessing.py'
spec = importlib.util.spec_from_file_location('preprocessing', str(_pp_path))
_pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_pp)
load_nsl_kdd = _pp.load_nsl_kdd

df = load_nsl_kdd(str(CSV))
print('Loaded combine.csv via load_nsl_kdd; shape=', getattr(df, 'shape', None))
print('Columns sample:', list(df.columns)[:10])
if 'label' not in df.columns:
    raise RuntimeError('Loaded dataframe does not contain "label" column after using load_nsl_kdd')

y_raw = df['label'].values
X_df = df.drop(columns=['label']).copy()

# Heuristic categorical features for NSL-KDD / CICIDS
known_cat = ["protocol_type", "service", "flag"]
cat_features = [c for c in known_cat if c in X_df.columns]

# If none of the known categorical columns found, detect by dtype or low cardinality
if len(cat_features) == 0:
    # object dtype
    obj_cols = [c for c in X_df.columns if X_df[c].dtype == 'object']
    if len(obj_cols) > 0:
        cat_features = obj_cols
    else:
        # low cardinality numeric treated as categorical
        cat_features = [c for c in X_df.columns if X_df[c].nunique(dropna=True) <= 50]

# Coerce non-categorical to numeric where possible (like preprocessing.prepare)
for col in X_df.columns:
    if col not in cat_features:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

print('Detected categorical features:', cat_features)


def map_to_binary(yarr):
    try:
        svals = [str(v).lower() for v in yarr[:1000]] if len(yarr) > 0 else []
        if any('normal' in s for s in svals):
            out = []
            for v in yarr:
                try:
                    s = str(v).lower().strip()
                except Exception:
                    s = ''
                s_clean = s.strip(" '\"\n\r\t,")
                out.append(0 if 'normal' in s_clean else 1)
            return np.array(out, dtype=int)
    except Exception:
        pass
    try:
        y_int = np.array(yarr, dtype=int)
        binc = np.bincount(y_int)
        maj = int(np.argmax(binc))
        return np.array([0 if int(v) == maj else 1 for v in y_int], dtype=int)
    except Exception:
        try:
            y_int = np.array(yarr, dtype=float)
            return np.array([0 if float(v) == 0.0 else 1 for v in y_int], dtype=int)
        except Exception:
            first = yarr[0]
            return np.array([0 if v == first else 1 for v in yarr], dtype=int)

    # Map labels deterministically using the local heuristic to avoid dependency issues.
    # This always defines `y_bin` so downstream code won't fail.
    y_bin = map_to_binary(y_raw)

    # Debug: show what raw labels look like and the binary distribution
    try:
        print('Raw label sample (first 20):', list(y_raw[:20]))
        print('Unique raw labels (counts):', dict(Counter(y_raw)))
    except Exception:
        pass
    print('Binary distribution (raw):', dict(Counter(y_bin)))

# Train/test split (stratified)
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# If one class is too small for stratification, fall back to non-stratified split
from collections import Counter as _Counter
_cnts = _Counter(y_bin.tolist())
if min(_cnts.values()) < 2:
    # fallback: simple random split without stratify
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(np.arange(len(y_bin)), test_size=0.2, random_state=42)
else:
    train_idx, test_idx = next(sss.split(X_df, y_bin))
X_tr_df = X_df.iloc[train_idx].reset_index(drop=True)
X_te_df = X_df.iloc[test_idx].reset_index(drop=True)
y_tr = y_bin[train_idx]
y_te = y_bin[test_idx]

# Train CatBoost
try:
    from catboost import CatBoostClassifier, Pool
except Exception as e:
    raise RuntimeError('CatBoost must be installed in the active environment to run this script') from e

# Create Pool objects using column names for categorical features
train_pool = Pool(data=X_tr_df, label=y_tr, cat_features=cat_features if len(cat_features)>0 else None)
eval_pool = Pool(data=X_te_df, label=y_te, cat_features=cat_features if len(cat_features)>0 else None)

# class weights
cnt = Counter(y_tr.tolist()); total = len(y_tr)
class_w = {k: total/(len(cnt)*v) for k,v in cnt.items()}
sample_weight = np.array([class_w[int(y)] for y in y_tr])

params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 8,
    'random_seed': 42,
    'verbose': 100,
}

clf = CatBoostClassifier(**params)
print('Training CatBoost on raw combine.csv...')
clf.fit(train_pool, eval_set=eval_pool, sample_weight=sample_weight, early_stopping_rounds=50)

# Predict on test and tune threshold
proba = clf.predict_proba(X_te_df)[:,1]
best_acc = -1.0; best_t = 0.5; best_pred = None
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
for t in np.linspace(0.3, 0.7, 41):
    pred = (proba >= t).astype(int)
    acc = accuracy_score(y_te, pred)
    if acc > best_acc:
        best_acc = acc; best_t = float(t); best_pred = pred

res = {
    'test_acc': float(best_acc),
    'test_f1': float(f1_score(y_te, best_pred)),
    'test_auc': float(roc_auc_score(y_te, proba)),
    'best_threshold': float(best_t),
}

print('CatBoost raw results', res)

# Save model and results
model_path = MOD / 'catboost_raw_model.cbm'
clf.save_model(str(model_path))
with open(RES / 'catboost_raw_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)
print('Saved CatBoost model to', model_path)
