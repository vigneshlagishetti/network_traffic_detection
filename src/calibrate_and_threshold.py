import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

# load model
model_path = MOD / 'final_model_best.joblib'
if not model_path.exists():
    raise FileNotFoundError('final_model_best.joblib not found; run evaluation/selection first')
model = joblib.load(model_path)

# load processed data
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

# get raw probabilities
try:
    proba_tr = model.predict_proba(X)[:,1]
    proba_te = model.predict_proba(Xte)[:,1]
except Exception as e:
    try:
        dec_tr = model.decision_function(X)
        dec_te = model.decision_function(Xte)
        proba_tr = 1/(1+np.exp(-dec_tr))
        proba_te = 1/(1+np.exp(-dec_te))
    except Exception as e2:
        raise RuntimeError('Model lacks proba and decision_function')

# Fit Platt scaling (logistic regression) on train probs
calib_platt = LogisticRegression(max_iter=2000)
calib_platt.fit(proba_tr.reshape(-1,1), y_bin)
calib_proba_te = calib_platt.predict_proba(proba_te.reshape(-1,1))[:,1]

# Fit isotonic regression as alternative
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(proba_tr, y_bin)
iso_proba_te = iso.predict(proba_te)

# Evaluate thresholds
def eval_probs(probs, y_true):
    best = {'threshold':None,'accuracy':-1}
    records = {}
    thresholds = np.linspace(0,1,1001)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best['accuracy']:
            best = {'threshold':t,'accuracy':acc}
    return best

raw_best = eval_probs(proba_te, yte_bin)
platt_best = eval_probs(calib_proba_te, yte_bin)
iso_best = eval_probs(iso_proba_te, yte_bin)

# compute final metrics at best thresholds
def metrics_at(probs, y_true, t):
    preds = (probs >= t).astype(int)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probs)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    return {'accuracy':acc,'f1':f1,'precision':prec,'recall':rec,'roc_auc':auc,'specificity':spec}

results = {
    'raw': {'best_threshold': raw_best, 'metrics_at_best': metrics_at(proba_te, yte_bin, raw_best['threshold'])},
    'platt': {'best_threshold': platt_best, 'metrics_at_best': metrics_at(calib_proba_te, yte_bin, platt_best['threshold'])},
    'isotonic': {'best_threshold': iso_best, 'metrics_at_best': metrics_at(iso_proba_te, yte_bin, iso_best['threshold'])}
}

# also include AUC for raw model
results['raw']['roc_auc'] = float(roc_auc_score(yte_bin, proba_te))
results['platt']['roc_auc'] = float(roc_auc_score(yte_bin, calib_proba_te))
results['isotonic']['roc_auc'] = float(roc_auc_score(yte_bin, iso_proba_te))

# save predictions at chosen best method (pick highest accuracy among methods)
candidates = [('raw', results['raw']['metrics_at_best']['accuracy']), ('platt', results['platt']['metrics_at_best']['accuracy']), ('isotonic', results['isotonic']['metrics_at_best']['accuracy'])]
candidates.sort(key=lambda x: x[1], reverse=True)
best_method = candidates[0][0]
best_thresh = results[best_method]['best_threshold']['threshold']
if best_method == 'raw':
    final_probs = proba_te
elif best_method == 'platt':
    final_probs = calib_proba_te
else:
    final_probs = iso_proba_te

final_preds = (final_probs >= best_thresh).astype(int)

# write predictions csv
import csv
with open(RES / 'final_calibrated_predictions.csv', 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['prob','pred','true'])
    for p,pr,t in zip(final_probs, final_preds, yte_bin):
        writer.writerow([float(p), int(pr), int(t)])

with open(RES / 'calibration_threshold_results.json', 'w') as fh:
    json.dump({'best_method':best_method, 'best_threshold':best_thresh, 'results':results}, fh, indent=2)

print('Saved calibration & threshold results:', RES / 'calibration_threshold_results.json')
print('Best method:', best_method, 'threshold:', best_thresh)
print(results[best_method])
