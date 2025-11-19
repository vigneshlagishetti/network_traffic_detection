"""Train and save a majority-class baseline (DummyClassifier) using processed arrays.
Saves model to models/majority_baseline.joblib and results to results/majority_baseline_results.json
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

train_npz = ROOT / 'data' / 'processed' / 'train_processed.npz'
test_npz = ROOT / 'data' / 'processed' / 'test_processed.npz'

if not train_npz.exists() or not test_npz.exists():
    print('Processed train/test npz files not found. Aborting.')
    raise SystemExit(1)

train = np.load(train_npz, allow_pickle=True)
X_train = train['X']
y_train = train['y']

test = np.load(test_npz, allow_pickle=True)
X_test = test['X']
y_test = test['y']

clf = DummyClassifier(strategy='most_frequent')
clf.fit(X_train, y_train)

# predict on test
proba = None
try:
    proba = clf.predict_proba(X_test)[:,1]
except Exception:
    pass
preds = clf.predict(X_test)

# map y to binary if needed
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

joblib.dump(clf, MOD / 'majority_baseline.joblib')
with open(RES / 'majority_baseline_results.json','w') as fh:
    json.dump(out, fh, indent=2)

print('Saved majority baseline model to', MOD / 'majority_baseline.joblib')
print('Results:', out)
