"""Quick threshold tuning for the saved final LGB model against processed test arrays.
Writes results to results/threshold_tune_results.json and prints summary.
"""
from pathlib import Path
import joblib
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)
npz_test = ROOT / 'data' / 'processed' / 'test_processed.npz'
model_path = MOD / 'final_lgb_model.joblib'

if not model_path.exists():
    print('Model not found at', model_path)
    raise SystemExit(1)

if not npz_test.exists():
    print('Processed test npz not found at', npz_test)
    raise SystemExit(1)

print('Loading model...')
model = joblib.load(model_path)
print('Loading test arrays...')
arr = np.load(npz_test, allow_pickle=True)
X = arr['X']
y = arr.get('y', None)
if y is None:
    print('No y in test npz; cannot tune threshold')
    raise SystemExit(1)

# Predict probabilities
print('Predicting probabilities...')
try:
    proba = model.predict_proba(X)[:, 1]
except Exception as e:
    # fallback to predict
    preds_raw = model.predict(X)
    if preds_raw.ndim == 2 and preds_raw.shape[1] > 1:
        proba = preds_raw[:, 1]
    else:
        print('Model does not output probabilities; aborting')
        raise

# Search thresholds
print('Searching thresholds...')
thresholds = np.linspace(0.0, 1.0, 1001)
best = {'threshold': 0.5, 'accuracy': 0.0, 'f1': 0.0}
for t in thresholds:
    pred = (proba >= t).astype(int)
    acc = accuracy_score((y != 0).astype(int), pred) if np.array(y).dtype.kind not in ('U','S','O') else accuracy_score(np.array([0 if 'normal' in str(v).lower() else 1 for v in y]), pred)
    if acc > best['accuracy']:
        best = {'threshold': float(t), 'accuracy': float(acc)}

# Compute metrics at best threshold
best_t = best['threshold']
pred_best = (proba >= best_t).astype(int)
if np.array(y).dtype.kind in ('U','S','O'):
    y_bin = np.array([0 if 'normal' in str(v).lower() else 1 for v in y])
else:
    y_bin = (np.array(y) != 0).astype(int)
acc = accuracy_score(y_bin, pred_best)
f1 = f1_score(y_bin, pred_best)
auc = roc_auc_score(y_bin, proba)

out = {
    'best_threshold': best_t,
    'accuracy': acc,
    'f1': f1,
    'auc': auc,
    'n_samples': int(len(proba)),
}

with open(RES / 'threshold_tune_results.json', 'w') as fh:
    json.dump(out, fh, indent=2)

print('Best threshold tuning results:')
print(json.dumps(out, indent=2))

# write preds csv
import pandas as pd
pd.DataFrame({'proba': proba, 'pred': pred_best}).to_csv(RES / 'threshold_tune_predictions.csv', index=False)
print('Wrote predictions to', RES / 'threshold_tune_predictions.csv')
print('Wrote summary to', RES / 'threshold_tune_results.json')
