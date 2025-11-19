import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz_te.exists():
    raise FileNotFoundError('processed test npz not found')
arrt = np.load(npz_te, allow_pickle=True)
Xte = arrt['X']; yte = arrt['y']

# robust label mapping
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

models = sorted(MOD.glob('*.joblib'))
summary = {}
best_score = -1
best_model_path = None
best_metrics = None

for mpath in models:
    name = mpath.name
    try:
        model = joblib.load(mpath)
    except Exception as e:
        summary[name] = {'error': f'load error: {repr(e)}'}
        continue
    # try to get proba
    try:
        proba = model.predict_proba(Xte)
        if proba.ndim==1:
            probs = proba
        else:
            probs = proba[:,1]
    except Exception as e:
        # try decision_function
        try:
            dec = model.decision_function(Xte)
            # scale to 0-1 via sigmoid
            import numpy as _np
            probs = 1/(1+_np.exp(-dec))
        except Exception as e2:
            summary[name] = {'error': f'no proba/decision_function: {repr(e)}; {repr(e2)}'}
            continue
    # predictions at 0.5
    preds = (probs >= 0.5).astype(int)
    try:
        acc = float(accuracy_score(yte_bin, preds))
        f1 = float(f1_score(yte_bin, preds))
        prec = float(precision_score(yte_bin, preds, zero_division=0))
        rec = float(recall_score(yte_bin, preds, zero_division=0))
        auc = float(roc_auc_score(yte_bin, probs))
        tn, fp, fn, tp = confusion_matrix(yte_bin, preds).ravel()
        spec = float(tn / (tn+fp)) if (tn+fp)>0 else 0.0
    except Exception as e:
        summary[name] = {'error': f'metric error: {repr(e)}'}
        continue
    summary[name] = {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'roc_auc': auc,
        'specificity': spec
    }
    # composite
    composite = 0.5 * (auc + acc)
    if composite > best_score:
        best_score = composite
        best_model_path = mpath
        best_metrics = summary[name]

# save summary
with open(RES / 'model_catalog_eval_auto.json', 'w') as fh:
    json.dump(summary, fh, indent=2)

if best_model_path:
    # copy best model to canonical final
    import shutil
    shutil.copy(best_model_path, MOD / 'final_model_best.joblib')
    out = {'selected_model': best_model_path.name, 'metrics': best_metrics, 'composite': best_score}
    with open(RES / 'model_selection_summary.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('Selected', best_model_path.name)
else:
    out = {'selected_model': None}
    with open(RES / 'model_selection_summary.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('No selectable model found')

print('Wrote results to', RES)
