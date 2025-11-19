from pathlib import Path
import numpy as np
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'

pred_csv = RES / 'final_predictions.csv'
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'

if not pred_csv.exists():
    raise FileNotFoundError(pred_csv)
if not npz_te.exists():
    raise FileNotFoundError(npz_te)

# load test true labels
d = np.load(npz_te, allow_pickle=True)
Xte = d['X']; yte = d['y']

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

yte_bin = map_to_binary(yte)

# load predictions
probas = []
preds = []
with open(pred_csv,'r',newline='') as fh:
    rdr = csv.DictReader(fh)
    for row in rdr:
        prob = float(row['prob_positive'])
        pred = int(row['pred_label'])
        probas.append(prob)
        preds.append(pred)

probas = np.array(probas)
preds = np.array(preds)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
acc = accuracy_score(yte_bin, preds)
prec = precision_score(yte_bin, preds, zero_division=0)
rec = recall_score(yte_bin, preds, zero_division=0)
f1 = f1_score(yte_bin, preds, zero_division=0)
auc = roc_auc_score(yte_bin, probas)
tn,fp,fn,tp = confusion_matrix(yte_bin, preds).ravel()
spec = tn / (tn+fp)

out = {
    'accuracy': acc,
    'precision': prec,
    'recall': rec,
    'sensitivity': rec,
    'specificity': spec,
    'f1': f1,
    'roc_auc': auc,
    'confusion': [[int(tn),int(fp)],[int(fn),int(tp)]]
}

print(json.dumps(out, indent=2))
with open(RES / 'final_full_metrics.json','w') as fh:
    json.dump(out, fh, indent=2)
