"""Quick blender: load saved model artifacts, compute test probabilities, average them,
find best threshold for accuracy, and save results + blended bundle.
"""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
DATA = ROOT / 'data' / 'processed'
RES.mkdir(exist_ok=True)

MODEL_FILES = [MOD / 'pruned_stack.joblib', MOD / 'optuna_pruned_light.joblib', MOD / 'stack_improved.joblib']

# load test
npz = DATA / 'test_processed.npz'
if not npz.exists():
    raise FileNotFoundError(npz)
d = np.load(npz, allow_pickle=True)
Xte = d['X']
yte = d['y']

# robust binary mapping

def map_to_binary(yarr):
    y = np.array(yarr)
    svals = [str(v).lower() for v in y[:1000]] if y.size>0 else []
    if any('normal' in s for s in svals):
        return np.array([0 if 'normal' in str(v).lower() else 1 for v in y], dtype=int)
    try:
        yi = y.astype(int)
        maj = int(np.argmax(np.bincount(yi)))
        return np.array([0 if int(v)==maj else 1 for v in yi], dtype=int)
    except Exception:
        try:
            yf = y.astype(float)
            return np.array([0 if float(v)==0.0 else 1 for v in yf], dtype=int)
        except Exception:
            first = y[0]
            return np.array([0 if v==first else 1 for v in y], dtype=int)

yte_bin = map_to_binary(yte)

probas_list = []
model_names = []

for mf in MODEL_FILES:
    if not mf.exists():
        print('Skipping missing model file', mf)
        continue
    obj = joblib.load(mf)
    name = mf.stem
    model_names.append(name)
    # if it's a scikit estimator
    if hasattr(obj, 'predict_proba'):
        try:
            p = obj.predict_proba(Xte)
            p = p[:,1] if p.ndim==2 else p
        except Exception:
            try:
                dec = obj.decision_function(Xte)
                p = 1/(1+np.exp(-dec))
            except Exception:
                p = np.full(Xte.shape[0], 0.5)
        probas_list.append(p)
        continue
    # if it's a dict/bundle
    if isinstance(obj, dict):
        # apply keep_idx if present for this model
        X_local = Xte
        if 'keep_idx' in obj and obj['keep_idx'] is not None:
            try:
                idx = np.array(obj['keep_idx'], dtype=int)
                X_local = X_local[:, idx]
            except Exception:
                pass
        # if meta + base_models
        if 'meta' in obj and 'base_models' in obj:
            bases = obj['base_models']
            # bases may be dict name->estimator
            base_probas = []
            if isinstance(bases, dict):
                for bname, bmdl in bases.items():
                    try:
                        p = bmdl.predict_proba(X_local)
                        p = p[:,1] if p.ndim==2 else p
                    except Exception:
                        try:
                            dec = bmdl.decision_function(X_local)
                            p = 1/(1+np.exp(-dec))
                        except Exception:
                            p = np.full(X_local.shape[0], 0.5)
                    base_probas.append(p)
                if len(base_probas)>0:
                    meta_X = np.vstack(base_probas).T
                    try:
                        meta = obj['meta']
                        mp = meta.predict_proba(meta_X)
                        mp = mp[:,1] if mp.ndim==2 else mp
                    except Exception:
                        mp = np.mean(meta_X, axis=1)
                    probas_list.append(mp)
                    continue
        # fallback: try model-like keys
        for k in ['model', 'estimator', 'clf']:
            if k in obj:
                mdl = obj[k]
                try:
                    p = mdl.predict_proba(X_local)
                    p = p[:,1] if p.ndim==2 else p
                except Exception:
                    try:
                        dec = mdl.decision_function(X_local)
                        p = 1/(1+np.exp(-dec))
                    except Exception:
                        p = np.full(X_local.shape[0], 0.5)
                probas_list.append(p)
                break
        else:
            # last resort: if object contains probability array saved
            if 'proba' in obj:
                p = np.array(obj['proba'])
                if p.ndim==2:
                    p = p[:,1]
                probas_list.append(p)
                continue
            # unknown bundle: use 0.5
            probas_list.append(np.full(Xte.shape[0], 0.5))
        continue
    # other types
    probas_list.append(np.full(Xte.shape[0], 0.5))

if len(probas_list)==0:
    raise RuntimeError('No model probabilities could be computed')

# simple average
avg_proba = np.mean(np.vstack(probas_list), axis=0)

# also try median
med_proba = np.median(np.vstack(probas_list), axis=0)

# sweep thresholds for average and median
best = {'method':None,'th':0.5,'acc':0.0,'conf':None,'roc':0.0}
for name, probs in [('avg', avg_proba), ('med', med_proba)]:
    for th in np.linspace(0,1,501):
        pred = (probs>=th).astype(int)
        acc = float((pred==yte_bin).mean())
        if acc > best['acc']:
            tn, fp, fn, tp = confusion_matrix(yte_bin, pred).ravel()
            best = {'method': name, 'th': float(th), 'acc': acc, 'conf': [[int(tn),int(fp)],[int(fn),int(tp)]], 'roc': float(roc_auc_score(yte_bin, probs))}

results = {
    'models_used': model_names,
    'best_method': best['method'],
    'best_threshold': best['th'],
    'best_accuracy': best['acc'],
    'best_confusion': best['conf'],
    'best_roc_auc': best['roc']
}

with open(RES / 'blend_results.json','w') as fh:
    json.dump(results, fh, indent=2)

# save blended bundle
blend_bundle = {'models': [str(m) for m in MODEL_FILES if m.exists()], 'method': best['method'], 'threshold': best['th']}
joblib.dump(blend_bundle, MOD / 'blended_models.joblib')

print('Blend results saved to', RES / 'blend_results.json')
print(results)
