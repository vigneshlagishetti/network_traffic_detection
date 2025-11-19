import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

# candidate pattern: choose a curated list
candidate_names = [
    'final_lgb_model.joblib',
    'final_lgb_model_weighted.joblib',
    'best_binary_model.joblib',
    'best_model_for_99.joblib',
    'ensemble_meta.joblib',
    'catboost_wrapper.joblib',
    'lgb_binary_full.joblib'
]

npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
if not npz_te.exists():
    raise FileNotFoundError('test_processed.npz not found')
arr = np.load(npz_te, allow_pickle=True)
Xte = arr['X']; yte = arr['y']

# mapping
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
for name in candidate_names:
    path = MOD / name
    if not path.exists():
        continue
    try:
        clf = joblib.load(path)
    except Exception as e:
        results[name] = {'error': str(e)}
        continue
    # get proba
    try:
        proba = clf.predict_proba(Xte)
        if proba.ndim==2:
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
        try:
            proba_pos = clf.decision_function(Xte)
        except Exception as e:
            results[name] = {'error': 'no proba/decision_function: '+str(e)}
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
    results[name] = {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'roc_auc': auc,
        'pr_auc': pr_auc
    }

with open(RES / 'model_catalog_eval.json', 'w') as fh:
    json.dump(results, fh, indent=2)

print('Wrote results to', RES / 'model_catalog_eval.json')
