"""
Run binary (normal vs attack) validation using processed NSL-KDD arrays.
Saves results to results/binary_cv_results.json and final model to models/lgb_binary_full.joblib
"""
import json
from pathlib import Path
import numpy as np
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

npz = ROOT / 'data' / 'processed' / 'train_processed.npz'
if not npz.exists():
    raise FileNotFoundError(f"Processed train file not found: {npz}\nRun src/preprocessing.py first")
arr = np.load(npz, allow_pickle=True)
X = arr['X']
y = arr['y']

# convert labels to binary: normal -> 0, attack -> 1
# handle bytes / numbers / strings
def to_binary_labels(yarr):
    # Robust conversion: handle bytes, whitespace, and labels that contain 'normal' as substring.
    out = []
    for v in yarr:
        try:
            s = str(v).lower().strip()
        except Exception:
            s = ''
        # strip surrounding punctuation
        s_clean = s.strip(' \"\'\n\r\t,')
        if 'normal' in s_clean:
            out.append(0)
        else:
            out.append(1)
    return np.array(out, dtype=int)

y_bin = to_binary_labels(y)

print('Loaded X,y:', X.shape, np.bincount(y_bin))

# LightGBM CV
try:
    import lightgbm as lgb
    has_lgb = True
except Exception:
    has_lgb = False

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import joblib

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
accs=[]
f1s=[]
aucs=[]
precs=[]
recs=[]
fold=0

# To keep CV runs tractable, run CV on a stratified subsample if dataset is large
CV_SUBSAMPLE = 20000
if X.shape[0] > CV_SUBSAMPLE:
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=CV_SUBSAMPLE, random_state=1)
    sub_idx, _ = next(sss.split(X, y_bin))
    X_cv = X[sub_idx]; y_cv = y_bin[sub_idx]
else:
    X_cv = X; y_cv = y_bin

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
accs=[]
f1s=[]
aucs=[]
precs=[]
recs=[]
fold=0

for train_idx, val_idx in skf.split(X_cv, y_cv):
    fold+=1
    Xtr, Xv = X_cv[train_idx], X_cv[val_idx]
    ytr, yv = y_cv[train_idx], y_cv[val_idx]
    # sample weighting simple: inverse frequency
    cnt = Counter(ytr.tolist())
    total = len(ytr)
    class_w = {k: total / (len(cnt) * v) for k,v in cnt.items()}
    sw = np.array([class_w[int(yy)] for yy in ytr])

    if has_lgb:
        # use fewer estimators for faster CV runs; this is a quick validation run
        clf = lgb.LGBMClassifier(n_estimators=30, learning_rate=0.05, random_state=1, n_jobs=1)
        clf.fit(Xtr, ytr, sample_weight=sw)
    else:
        # fallback
        clf = LogisticRegression(max_iter=200)
        clf.fit(Xtr, ytr)

    yp = clf.predict(Xv)
    try:
        yp_proba = clf.predict_proba(Xv)[:,1]
    except Exception:
        yp_proba = None

    acc = accuracy_score(yv, yp)
    f1 = f1_score(yv, yp)
    auc = roc_auc_score(yv, yp_proba) if yp_proba is not None else None
    prec = precision_score(yv, yp)
    rec = recall_score(yv, yp)

    print(f'Fold {fold}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}' if auc is not None else f'Fold {fold}: acc={acc:.4f} f1={f1:.4f}')
    accs.append(acc); f1s.append(f1); aucs.append(auc); precs.append(prec); recs.append(rec)

# aggregate
res = {
    'cv_acc_mean': float(np.mean(accs)),
    'cv_acc_std': float(np.std(accs)),
    'cv_f1_mean': float(np.mean(f1s)),
    'cv_f1_std': float(np.std(f1s)),
    'cv_auc_mean': float(np.nanmean(aucs)) if aucs else None,
    'cv_auc_std': float(np.nanstd(aucs)) if aucs else None,
    'cv_precision_mean': float(np.mean(precs)),
    'cv_recall_mean': float(np.mean(recs)),
    'n_samples': int(X.shape[0]),
    'n_features': int(X.shape[1])
}

# fit on full train and evaluate on processed test if exists
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if npz_te.exists():
    arrt = np.load(npz_te, allow_pickle=True)
    Xte = arrt['X']; yte = arrt['y']
    yte_bin = to_binary_labels(yte)
    # refit on full train
    if has_lgb:
        # final full-model uses moderate number of trees to keep runtime reasonable here
        clf_full = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=1, n_jobs=1)
        cnt = Counter(y_bin.tolist()); total = len(y_bin)
        class_w = {k: total / (len(cnt) * v) for k,v in cnt.items()}
        sw_full = np.array([class_w[int(yy)] for yy in y_bin])
        clf_full.fit(X, y_bin, sample_weight=sw_full)
    else:
        clf_full = LogisticRegression(max_iter=400)
        clf_full.fit(X, y_bin)
    ypred = clf_full.predict(Xte)
    try:
        yproba = clf_full.predict_proba(Xte)[:,1]
        auc_full = roc_auc_score(yte_bin, yproba)
    except Exception:
        auc_full = None
    acc_full = accuracy_score(yte_bin, ypred)
    f1_full = f1_score(yte_bin, ypred)
    res.update({'test_accuracy': float(acc_full), 'test_f1': float(f1_full), 'test_auc': float(auc_full) if auc_full is not None else None})
    # save model
    joblib.dump(clf_full, MOD / 'lgb_binary_full.joblib')
    print('Evaluation on test set -> acc:', acc_full, 'f1:', f1_full, 'auc:', auc_full)
else:
    print('No processed test file found; skipping test eval')

# save results
with open(RES / 'binary_cv_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)
print('Saved results/binary_cv_results.json')
print('Summary:', res)
