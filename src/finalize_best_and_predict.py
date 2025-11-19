import json
from pathlib import Path
import joblib
import numpy as np
from shutil import copyfile
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

# choose best model (from prior evaluation): final_lgb_model.joblib
best_name = 'final_lgb_model.joblib'
best_path = MOD / best_name
if not best_path.exists():
    raise FileNotFoundError(best_path)
# copy to canonical final name
final_path = MOD / 'final_model_best.joblib'
copyfile(best_path, final_path)

# load test
npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
arr = np.load(npz_te, allow_pickle=True)
Xte = arr['X']; yte = arr['y']

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

# default threshold (can be overridden by saved model bundle)
th = 0.3506294007328901

loaded = joblib.load(final_path)
# handle cases where the saved object is a dict with components
clf = None
keep_idx = None
if isinstance(loaded, dict):
    # prefer meta (stack) if present, otherwise lgb, otherwise first estimator
    if 'meta' in loaded and loaded['meta'] is not None:
        clf = loaded['meta']
    elif 'lgb' in loaded and loaded['lgb'] is not None:
        clf = loaded['lgb']
    elif 'model' in loaded and loaded['model'] is not None:
        clf = loaded['model']
    else:
        # pick first estimator-like value
        for v in loaded.values():
            if hasattr(v, 'predict_proba') or hasattr(v, 'decision_function'):
                clf = v
                break
    keep_idx = loaded.get('keep_idx', None)
    # if a threshold was saved, prefer it
    th = float(loaded.get('threshold', th)) if loaded.get('threshold', None) is not None else th
else:
    clf = loaded

# apply keep_idx if present
if keep_idx is not None:
    try:
        Xte = Xte[:, keep_idx]
    except Exception:
        pass

# predict proba
proba_pos = None
# If the loaded object was a dict with base models + meta, build meta features
if isinstance(loaded, dict) and ('meta' in loaded) and (('lgb' in loaded) or ('xgb' in loaded) or ('base_models' in loaded)):
    prob_list = []
    # base LGB
    if 'lgb' in loaded and loaded['lgb'] is not None:
        try:
            p = loaded['lgb'].predict_proba(Xte)
            prob_list.append(p[:,1] if p.ndim==2 else p)
        except Exception:
            pass
    # base XGB
    if 'xgb' in loaded and loaded['xgb'] is not None:
        try:
            p = loaded['xgb'].predict_proba(Xte)
            prob_list.append(p[:,1] if p.ndim==2 else p)
        except Exception:
            pass
    # base_models list
    if 'base_models' in loaded and loaded['base_models'] is not None:
        for bm in loaded['base_models']:
            try:
                p = bm.predict_proba(Xte)
                prob_list.append(p[:,1] if p.ndim==2 else p)
            except Exception:
                try:
                    p = bm.decision_function(Xte)
                    prob_list.append(p)
                except Exception:
                    pass
    if len(prob_list)==0:
        # fallback to trying meta directly
        try:
            proba = loaded['meta'].predict_proba(Xte)
            proba_pos = proba[:,1] if proba.ndim==2 else proba
        except Exception:
            pass
    else:
        meta_X = np.vstack(prob_list).T
        meta = loaded['meta']
        try:
            pmeta = meta.predict_proba(meta_X)
            proba_pos = pmeta[:,1] if pmeta.ndim==2 else pmeta
        except Exception:
            try:
                proba_pos = meta.decision_function(meta_X)
            except Exception as e:
                raise RuntimeError('Meta model cannot predict on stacked features: '+str(e))
else:
    # single estimator case
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
            raise RuntimeError('Model has no proba/decision_function: '+str(e))

ypred = (proba_pos >= th).astype(int)
acc = float(accuracy_score(yte_bin, ypred))
f1 = float(f1_score(yte_bin, ypred))
prec = float(precision_score(yte_bin, ypred, zero_division=0))
rec = float(recall_score(yte_bin, ypred, zero_division=0))
auc = float(roc_auc_score(yte_bin, proba_pos))

# save predictions
import csv
out_csv = RES / 'final_predictions.csv'
with open(out_csv, 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['index','prob_positive','pred_label'])
    for i,p in enumerate(proba_pos):
        writer.writerow([i, float(p), int(p>=th)])

summary = {
    'selected_model': str(best_name),
    'final_model_path': str(final_path.name),
    'threshold_used': th,
    'accuracy_at_threshold': acc,
    'f1_at_threshold': f1,
    'precision_at_threshold': prec,
    'recall_at_threshold': rec,
    'roc_auc': auc
}
with open(RES / 'final_selection_and_predictions.json', 'w') as fh:
    json.dump(summary, fh, indent=2)

print('Selected and saved final model as', final_path.name)
print('Wrote predictions to', out_csv)
print('Summary:', summary)
