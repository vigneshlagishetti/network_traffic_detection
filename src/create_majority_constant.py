"""Create a constant classifier that always predicts label 1 (positive) and evaluate on test set.
Saves model to models/majority_always_positive.joblib and results to results/majority_always_positive_results.json
"""
from pathlib import Path
import joblib
import numpy as np
import json
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
MOD.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

test_npz = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not test_npz.exists():
    print('Test npz missing')
    raise SystemExit(1)
arr = np.load(test_npz, allow_pickle=True)
X_test = arr['X']
y_test = arr['y']

# Build constant classifier predicting '1'
clf = DummyClassifier(strategy='constant', constant=1)
# DummyClassifier doesn't need fit but for API we call fit
clf.fit(X_test, y_test)

preds = clf.predict(X_test)
proba = None
try:
    proba = clf.predict_proba(X_test)[:,1]
except Exception:
    proba = None

# Map y to binary as in threshold_tune/inspect
import numpy as _np
if _np.array(y_test).dtype.kind in ('U','S','O'):
    y_bin = _np.array([0 if 'normal' in str(v).lower() else 1 for v in y_test])
else:
    y_bin = (_np.array(y_test) != 0).astype(int)

acc = float(accuracy_score(y_bin, preds))
try:
    f1 = float(f1_score(y_bin, preds))
except Exception:
    f1 = None
try:
    auc = float(roc_auc_score(y_bin, proba)) if proba is not None else None
except Exception:
    auc = None

out = {
    'accuracy': acc,
    'f1': f1,
    'auc': auc,
    'n_test': int(len(y_test)),
}

joblib.dump(clf, MOD / 'majority_always_positive.joblib')
with open(RES / 'majority_always_positive_results.json','w') as fh:
    json.dump(out, fh, indent=2)

print('Saved constant-positive model to', MOD / 'majority_always_positive.joblib')
print('Results:', out)
