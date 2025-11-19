"""Inspect test y distribution and confusion matrices at candidate thresholds."""
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
ROOT = Path(__file__).resolve().parents[1]
npz_test = ROOT / 'data' / 'processed' / 'test_processed.npz'
model_path = ROOT / 'models' / 'final_lgb_model.joblib'
print('Loading...')
arr = np.load(npz_test, allow_pickle=True)
X = arr['X']
y = arr['y']
print('y shape:', y.shape)
# map y to binary same as earlier
if np.array(y).dtype.kind in ('U','S','O'):
    y_bin = np.array([0 if 'normal' in str(v).lower() else 1 for v in y])
else:
    y_bin = (np.array(y) != 0).astype(int)
unique, counts = np.unique(y_bin, return_counts=True)
print('Label distribution (binary):', dict(zip(unique.tolist(), counts.tolist())))

model = joblib.load(model_path)
proba = model.predict_proba(X)[:,1]

for t in [0.0, 0.01, 0.1, 0.3, 0.5, 0.7]:
    pred = (proba >= t).astype(int)
    cm = confusion_matrix(y_bin, pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp+tn)/cm.sum()
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
    print(f"threshold={t:.2f}: acc={acc:.6f}, precision={precision:.6f}, recall={recall:.6f}, TP={tp},FP={fp},TN={tn},FN={fn}")

print('\nTop 5 prob positives with label:')
idx = np.argsort(-proba)[:10]
for i in idx[:10]:
    print(i, proba[i], y_bin[i])
