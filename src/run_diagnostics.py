import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, precision_score, recall_score
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)
MOD = ROOT / 'models'

# load model
clf = joblib.load(MOD / 'final_lgb_model.joblib')
# load processed test
npz = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz.exists():
    raise FileNotFoundError(npz)
arr = np.load(npz, allow_pickle=True)
X = arr['X']
y = arr['y']

# map y to binary using same heuristics as training
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

# predict proba
proba = clf.predict_proba(X)
classes = getattr(clf, 'classes_', None)
print('model.classes_:', classes)
print('proba shape:', proba.shape)

# determine positive column: choose the column corresponding to label 1 if present, else assume column 1
pos_idx = None
if classes is not None:
    # try to find numeric 1
    for i,c in enumerate(classes):
        try:
            if int(c) == 1:
                pos_idx = i
                break
        except Exception:
            pass
if pos_idx is None:
    pos_idx = 1 if proba.shape[1] > 1 else 0

proba_pos = proba[:, pos_idx]

# basic stats
stats = {
    'n_samples': int(len(y_bin)),
    'label_counts': {str(int(k)): int(v) for k,v in zip(*np.unique(y_bin, return_counts=True))},
    'proba_pos_mean': float(proba_pos.mean()),
    'proba_pos_std': float(proba_pos.std()),
}

# metrics
roc_auc = float(roc_auc_score(y_bin, proba_pos))
pr_auc = float(average_precision_score(y_bin, proba_pos))

fpr, tpr, roc_th = roc_curve(y_bin, proba_pos)
prec, rec, pr_th = precision_recall_curve(y_bin, proba_pos)

# find thresholds achieving recall targets
def find_threshold_for_recall(recalls, thresholds, target):
    # precision_recall_curve returns thresholds aligned with precision/rec arrays: len(thresholds)=len(prec)-1
    # we want smallest threshold that gives recall >= target
    idxs = [i for i,r in enumerate(recalls[:-1]) if r >= target]
    if not idxs:
        return None
    i = idxs[0]
    # threshold corresponds to thresholds[i]
    return float(thresholds[i])

# use PR curve arrays
th_for_rec90 = find_threshold_for_recall(rec, pr_th, 0.90)
th_for_rec95 = find_threshold_for_recall(rec, pr_th, 0.95)

# helper to compute confusion metrics at threshold
from sklearn.metrics import confusion_matrix

def metrics_at_threshold(y_true, prob_pos, thresh):
    ypred = (prob_pos >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, ypred, labels=[0,1]).ravel()
    prec = precision_score(y_true, ypred, zero_division=0)
    rec = recall_score(y_true, ypred, zero_division=0)
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else None
    return {
        'threshold': float(thresh),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'precision': float(prec), 'recall': float(rec), 'fpr': (None if fpr_val is None else float(fpr_val))
    }

candidates = {}
# default 0.5
candidates['0.5'] = metrics_at_threshold(y_bin, proba_pos, 0.5)
# default 0.0 (always positive)
candidates['0.0'] = metrics_at_threshold(y_bin, proba_pos, 0.0)
# threshold that maximizes f1
from sklearn.metrics import f1_score
best_f1 = -1
best_thresh = None
# examine thresholds from PR curve
for t in np.unique(np.concatenate(([0.0], pr_th, [1.0]))):
    # compute f1
    ypred = (proba_pos >= t).astype(int)
    f1 = f1_score(y_bin, ypred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1; best_thresh = float(t)
candidates['best_f1'] = metrics_at_threshold(y_bin, proba_pos, best_thresh)

# thresholds for rec targets
if th_for_rec90 is not None:
    candidates['recall_0.90'] = metrics_at_threshold(y_bin, proba_pos, th_for_rec90)
else:
    candidates['recall_0.90'] = None
if th_for_rec95 is not None:
    candidates['recall_0.95'] = metrics_at_threshold(y_bin, proba_pos, th_for_rec95)
else:
    candidates['recall_0.95'] = None

out = {
    'stats': stats,
    'roc_auc': roc_auc,
    'pr_auc': pr_auc,
    'n_roc_thresholds': int(len(roc_th)),
    'n_pr_thresholds': int(len(pr_th)),
    'suggested_thresholds': candidates,
}

# save results
with open(RES / 'diagnostics_thresholds.json', 'w') as fh:
    json.dump(out, fh, indent=2)

# print concise summary
print('n_samples:', out['stats']['n_samples'])
print('label_counts:', out['stats']['label_counts'])
print(f"ROC AUC: {roc_auc:.6f}, PR AUC: {pr_auc:.6f}")
print('Suggested thresholds and metrics:')
for k,v in out['suggested_thresholds'].items():
    print('-', k, '->', v)

print('\nSaved results to', RES / 'diagnostics_thresholds.json')
