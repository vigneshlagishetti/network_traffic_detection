"""Run the finalized best detector and save predictions.

This script loads `models/final_lgb_model.joblib` and applies it to either:
- `combine.csv` (if present) using the saved preprocessing pipeline `data/processed/preprocessing_pipeline.joblib`,
  or
- `data/processed/test_processed.npz` (fallback).

It writes predictions to `results/final_predictions.csv` and a short summary JSON.

Notes:
- Loading the saved joblib model requires LightGBM to be installed in the environment
  where this script runs (joblib unpickles LightGBM objects). If you see
  ModuleNotFoundError: No module named 'lightgbm', install it with
    pip install lightgbm
  or
    conda install -c conda-forge lightgbm
"""
from pathlib import Path
import joblib
import json
import numpy as np
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

model_path = MOD / 'final_lgb_model.joblib'
pipe_path = ROOT / 'data' / 'processed' / 'preprocessing_pipeline.joblib'
npz_test = ROOT / 'data' / 'processed' / 'test_processed.npz'
csv_raw = ROOT / 'combine.csv'

if not model_path.exists():
    print('Model not found at', model_path)
    sys.exit(1)

try:
    model = joblib.load(model_path)
except ModuleNotFoundError as e:
    print('ERROR: needed package not installed to unpickle model:', e)
    print('Install LightGBM in this environment, for example:')
    print('  pip install lightgbm')
    print('or with conda:')
    print('  conda install -c conda-forge lightgbm')
    raise

# Load input features
X = None
y = None
feature_source = None
if csv_raw.exists() and pipe_path.exists():
    # Use a streaming approach: read `combine.csv` in chunks and apply the saved pipeline
    # to each chunk. This avoids building the full dense OHE matrix in memory which
    # previously caused MemoryError and a feature-count mismatch with the trained model.
    import importlib.util
    spec = importlib.util.spec_from_file_location('preprocessing', str(ROOT / 'src' / 'preprocessing.py'))
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)

    # Load pipeline and prepare streaming predictor
    try:
        pipe = joblib.load(pipe_path)
    except Exception as e:
        print('Failed to load preprocessing pipeline, falling back to raw matrix:', repr(e))
        df = pp.load_nsl_kdd(str(csv_raw))
        if 'label' in df.columns:
            y = df['label'].values
        X_df = df.drop(columns=['label']) if 'label' in df.columns else df
        X = X_df.values
        feature_source = 'combine.csv (raw->values)'
    else:
        def stream_predict_from_csv(model, pipeline, csv_path, out_csv_path, summary_path, chunksize=20000):
            import pandas as _pd
            import numpy as _np

            # determine threshold from results
            thresh = 0.5
            try:
                jf = RES / 'final_lgb_results.json'
                if jf.exists():
                    info = json.load(open(jf))
                    thresh = info.get('threshold', thresh)
            except Exception:
                pass

            write_header = True
            total = 0
            proba_list = []
            pred_list = []
            y_list = []

            colnames = getattr(pp, 'NSL_KDD_COLUMNS', None)
            reader = _pd.read_csv(csv_path, header=None, names=colnames, na_values=[''], skipinitialspace=True, low_memory=False, chunksize=chunksize)
            for i, chunk in enumerate(reader):
                print(f'Processing chunk {i} rows {len(chunk)}')
                if 'label' in chunk.columns:
                    y_chunk = chunk['label'].values
                    X_chunk = chunk.drop(columns=['label']).copy()
                else:
                    y_chunk = None
                    X_chunk = chunk.copy()

                # Coerce numeric columns (except categorical features) to numeric
                categorical_features = ['protocol_type', 'service', 'flag']
                for col in X_chunk.columns:
                    if col not in categorical_features:
                        X_chunk[col] = _pd.to_numeric(X_chunk[col], errors='coerce')

                # Apply pipeline transform on chunk
                X_t = pipeline.transform(X_chunk)

                # Ensure feature count matches model expectation
                expected = getattr(model, 'n_features_in_', None)
                if expected is not None and X_t.shape[1] != expected:
                    raise ValueError(f"Transformed chunk has {X_t.shape[1]} features but model expects {expected}")

                # Predict probabilities and labels
                proba_chunk = model.predict_proba(X_t)[:, 1]
                pred_chunk = (proba_chunk >= thresh).astype(int)

                # Append results to CSV incrementally
                df_out_chunk = _pd.DataFrame({'proba': proba_chunk, 'pred': pred_chunk})
                if write_header:
                    df_out_chunk.to_csv(out_csv_path, index=False, mode='w')
                    write_header = False
                else:
                    df_out_chunk.to_csv(out_csv_path, index=False, header=False, mode='a')

                total += len(proba_chunk)
                proba_list.append(proba_chunk)
                pred_list.append(pred_chunk)
                if y_chunk is not None:
                    y_list.append(y_chunk)

            # build summary
            summary = {
                'model_path': str(model_path),
                'feature_source': f'streamed:{csv_path}',
                'n_samples': int(total),
                'threshold': float(thresh),
            }
            if len(y_list) > 0:
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                y_all = _np.concatenate(y_list)
                if y_all.dtype.kind in ('U', 'S', 'O'):
                    y_bin = _np.array([0 if 'normal' in str(v).lower() else 1 for v in y_all])
                else:
                    y_bin = (y_all != 0).astype(int)
                preds_all = _np.concatenate(pred_list)
                prob_all = _np.concatenate(proba_list)
                summary.update({
                    'accuracy': float(accuracy_score(y_bin, preds_all)),
                    'f1': float(f1_score(y_bin, preds_all)),
                    'auc': float(roc_auc_score(y_bin, prob_all)),
                })

            with open(summary_path, 'w') as fh:
                json.dump(summary, fh, indent=2)

            print('Streamed predictions written to', out_csv_path)
            print('Stream summary written to', summary_path)

        # run streaming predictor
        try:
            out_csv = RES / 'final_predictions.csv'
            summary_path = RES / 'final_predictions_summary.json'
            stream_predict_from_csv(model, pipe, csv_raw, out_csv, summary_path)
            # streaming completes and writes outputs
            sys.exit(0)
        except Exception as e:
            print('Streaming prediction failed:', repr(e))
            # Fall back to loading entire CSV into memory (best-effort)
            df = pp.load_nsl_kdd(str(csv_raw))
            if 'label' in df.columns:
                y = df['label'].values
            X_df = df.drop(columns=['label']) if 'label' in df.columns else df
            try:
                X = pipe.transform(X_df)
                feature_source = 'pipeline(combine.csv)'
            except Exception as e2:
                print('Full pipeline.transform fallback failed:', repr(e2))
                X = X_df.values
                feature_source = 'combine.csv (raw->values)'
    

elif npz_test.exists():
    arr = np.load(npz_test, allow_pickle=True)
    X = arr['X']
    y = arr.get('y', None)
    feature_source = 'data/processed/test_processed.npz'
else:
    print('No input data found (need combine.csv+pipeline or data/processed/test_processed.npz)')
    sys.exit(1)

print('Loaded model from', model_path)
print('Feature source:', feature_source)

# Predict
def _ensure_numeric_matrix(X, X_df_fallback=None):
    """Return a numeric 2D numpy array suitable for model.predict.
    Attempts to coerce strings to numbers, factorize object columns, and fill NaNs.
    """
    import pandas as _pd
    import numpy as _np

    # If it's a DataFrame, operate directly
    if isinstance(X, _pd.DataFrame):
        Xc = X.copy()
        for col in Xc.columns:
            coerced = _pd.to_numeric(Xc[col], errors='coerce')
            non_na_ratio = coerced.notna().mean()
            if non_na_ratio >= 0.5:
                Xc[col] = coerced
            else:
                Xc[col] = Xc[col].astype(str)
        for col in Xc.columns:
            if Xc[col].dtype == object or str(Xc[col].dtype).startswith('category'):
                codes, _ = _pd.factorize(Xc[col].astype(str), sort=True)
                Xc[col] = codes.astype(float)
        for col in Xc.columns:
            if _pd.api.types.is_numeric_dtype(Xc[col]):
                med = _pd.to_numeric(Xc[col], errors='coerce').median()
                if _np.isnan(med):
                    med = 0.0
                Xc[col] = Xc[col].fillna(med)
        try:
            return Xc.values.astype(float)
        except Exception:
            return Xc.values

    # If it's a numpy array
    if isinstance(X, (list, tuple)):
        X = _np.array(X)
    if isinstance(X, _np.ndarray):
        if X.dtype.kind in ('U', 'S', 'O'):
            # try to build DF and coerce
            try:
                df_tmp = _pd.DataFrame(X)
                return _ensure_numeric_matrix(df_tmp)
            except Exception:
                # last resort: try astype float
                try:
                    return X.astype(float)
                except Exception:
                    raise
        else:
            return X.astype(float)

    # fallback: if X_df_fallback provided, try coercion on that
    if X_df_fallback is not None:
        return _ensure_numeric_matrix(X_df_fallback)
    # otherwise try convert
    import numpy as _np
    return _np.asarray(X, dtype=float)

try:
    # make sure X is numeric
    X = _ensure_numeric_matrix(X, X_df_fallback=locals().get('X_df', None))
    proba = model.predict_proba(X)[:,1]
except Exception:
    # some wrappers use predict; fall back
    preds = model.predict(X)
    # if preds are probabilities
    if preds.ndim == 2 and preds.shape[1] > 1:
        proba = preds[:,1]
    else:
        proba = np.array(preds, dtype=float)

# choose threshold: prefer final_lgb_results.json if present
threshold = 0.5
try:
    jf = RES / 'final_lgb_results.json'
    if jf.exists():
        info = json.load(open(jf))
        # try known keys
        threshold = info.get('threshold', threshold)
except Exception:
    pass

pred = (proba >= threshold).astype(int)

# Save predictions CSV
out_csv = RES / 'final_predictions.csv'
import pandas as pd
df_out = pd.DataFrame({'proba': proba, 'pred': pred})
df_out.to_csv(out_csv, index=False)

# Summary
summary = {
    'model_path': str(model_path),
    'feature_source': feature_source,
    'n_samples': int(len(proba)),
    'threshold': float(threshold),
}
if y is not None:
    try:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        # map y to binary if needed
        import numpy as _np
        y_arr = _np.array(y)
        # simple mapping: if strings containing 'normal' -> 0 else 1
        if y_arr.dtype.kind in ('U', 'S', 'O'):
            y_bin = _np.array([0 if 'normal' in str(v).lower() else 1 for v in y_arr])
        else:
            # numeric
            y_bin = (y_arr != 0).astype(int)
        summary.update({
            'accuracy': float(accuracy_score(y_bin, pred)),
            'f1': float(f1_score(y_bin, pred)),
            'auc': float(roc_auc_score(y_bin, proba)),
        })
    except Exception as e:
        summary['metrics_error'] = str(e)

with open(RES / 'final_predictions_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)

print('Wrote predictions to', out_csv)
print('Summary:', summary)
