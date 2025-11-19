import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

candidates = [
    MOD / 'final_lgb_model.joblib',
    MOD / 'final_lgb_model_weighted.joblib',
    MOD / 'final_lgb_optuna.joblib'
]
# also look for any other models starting with final_lgb
for p in MOD.glob('final_lgb*.joblib'):
    if p not in candidates:
        candidates.append(p)

npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz_te.exists():
    raise FileNotFoundError('test_processed.npz not found')
arr = np.load(npz_te, allow_pickle=True)
Xte = arr['X']; yte = arr['y']

# map to binary (same as others)
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

results = {}
best_score = -1
best_model = None
for mpath in candidates:
    if not mpath.exists():
        continue
    try:
        clf = joblib.load(mpath)
    except Exception as e:
        results[str(mpath.name)] = {'error': str(e)}
        continue
    try:
        proba = clf.predict_proba(Xte)
        if proba.ndim==2:
            # choose prob of class 1
            classes = getattr(clf, 'classes_', None)
            pos_idx = 1
            if classes is not None:
                for i,c in enumerate(classes):
                    try:
                        if int(c)==1:
                            pos_idx = i
                            break
                    except Exception:
                        pass
            proba_pos = proba[:, pos_idx]
        else:
            proba_pos = proba
    except Exception:
        # fallback to decision_function
        try:
            proba_pos = clf.decision_function(Xte)
        except Exception as e:
            results[str(mpath.name)] = {'error': 'no proba/decision_function: '+str(e)}
            continue
    ypred = (proba_pos >= 0.5).astype(int)
    acc = float(accuracy_score(yte_bin, ypred))
    f1 = float(f1_score(yte_bin, ypred))
    prec = float(precision_score(yte_bin, ypred, zero_division=0))
    rec = float(recall_score(yte_bin, ypred, zero_division=0))
    try:
        auc = float(roc_auc_score(yte_bin, proba_pos))
    except Exception:
        auc = None
    try:
        pr_auc = float(average_precision_score(yte_bin, proba_pos))
    except Exception:
        pr_auc = None
    # composite score: average of auc (if present) and accuracy
    comp = None
    if auc is not None:
        comp = (auc + acc) / 2.0
    else:
        comp = acc
    results[str(mpath.name)] = {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'roc_auc': auc,
        'pr_auc': pr_auc,
        'composite': comp
    }
    if comp is not None and comp > best_score:
        best_score = comp
        best_model = mpath

# write results
with open(RES / 'model_comparison.json', 'w') as fh:
    json.dump(results, fh, indent=2)

if best_model is not None:
    # copy best model to final name
    final_path = MOD / 'final_model_best.joblib'
    joblib.dump(joblib.load(best_model), final_path)
    summary = {'best_model': str(best_model.name), 'best_score': best_score}
    with open(RES / 'final_model_selection.json', 'w') as fh:
        json.dump({'selected': str(best_model.name), 'best_score': best_score, 'results': results}, fh, indent=2)
    print('Selected best model:', best_model.name)
    print('Saved final model as', final_path)
else:
    print('No candidate models found or all failed')

print('Wrote model comparison to', RES / 'model_comparison.json')
