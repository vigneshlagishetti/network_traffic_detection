"""
Load the saved best model wrapper and evaluate on processed test set.
Prints model metadata and evaluation metrics, and writes results to results/best_for_99_results.json.
"""
from pathlib import Path
import joblib
import json
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

wrapper_path = MOD / 'best_model_for_99.joblib'
if not wrapper_path.exists():
    print('No wrapper model found at', wrapper_path)
    raise SystemExit(1)

print('Loading', wrapper_path)
wrapper = joblib.load(wrapper_path)
print('Loaded wrapper type:', type(wrapper))

model = None
selected_idx = None
threshold = None
params = None
if isinstance(wrapper, dict):
    model = wrapper.get('model')
    selected_idx = wrapper.get('selected_idx', None)
    threshold = wrapper.get('threshold', wrapper.get('thr', wrapper.get('best_threshold', None)))
    params = wrapper.get('params', None)
else:
    # wrapper may directly be an estimator
    model = wrapper

print('Model type:', type(model))
print('Params:', params)
print('Threshold:', threshold)

# Load test processed
test_npz = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not test_npz.exists():
    print('No test_processed.npz found at', test_npz)
    raise SystemExit(1)

arr = np.load(test_npz, allow_pickle=True)
Xte = arr['X']; yte = arr['y']
print('Loaded test X shape:', Xte.shape)

# map to binary (reuse mapping heuristic)
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
print('Test binary counts:', dict(Counter(yte_bin)))

if selected_idx is not None:
    try:
        Xte = Xte[:, selected_idx]
        print('Sliced test to selected indices ->', Xte.shape)
    except Exception as e:
        print('Failed to slice test with selected_idx:', e)

res = {'model_type': type(model).__name__, 'params': params}

try:
    proba = model.predict_proba(Xte)[:,1]
    pred = (proba >= (threshold if threshold is not None else 0.5)).astype(int)
    res.update({
        'test_acc': float(accuracy_score(yte_bin, pred)),
        'test_f1': float(f1_score(yte_bin, pred)),
        'test_auc': float(roc_auc_score(yte_bin, proba)),
        'threshold': float(threshold) if threshold is not None else 0.5,
    })
    print('Eval -> acc: {:.6f}, f1: {:.6f}, auc: {:.6f}'.format(res['test_acc'], res['test_f1'], res['test_auc']))
except Exception as e:
    print('Model evaluation failed:', e)
    res['eval_error'] = str(e)

outp = RES / 'best_for_99_results.json'
with open(outp, 'w') as fh:
    json.dump(res, fh, indent=2)
print('Wrote results to', outp)
print('Done')
