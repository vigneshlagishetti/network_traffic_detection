import os
import json
from collections import OrderedDict
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

ROOT = r"c:\ECHO\Projects\Personal_Projects\Fruty"
CSV = os.path.join(ROOT, "combine.csv")
CHECKS = os.path.join(ROOT, "results", "catboost_checks.json")
OUT = os.path.join(ROOT, "results", "leakage_scan.json")

# Force target label name (strip/ignore whitespace)
TARGET_OVERRIDE = 'Label'

with open(CHECKS, 'r', encoding='utf-8') as f:
    checks = json.load(f)

perm = checks.get('top_20_permutation_importance') or checks.get('top_permutation_importance') or []
if not perm:
    raise SystemExit('No permutation importance found in checks file')

top_feats = [p[0] for p in perm][:10]

# Read a sample for speed (200k rows if possible)
SAMPLE_N = 200_000
chunks = pd.read_csv(CSV, chunksize=100_000, low_memory=False)
try:
    df_sample = next(chunks)
    if len(df_sample) < SAMPLE_N:
        try:
            df_sample = pd.concat([df_sample, next(chunks)])
        except StopIteration:
            pass
except StopIteration:
    df_sample = pd.read_csv(CSV, low_memory=False)

nrows = len(df_sample)
print(f'Using sample of {nrows} rows for targeted leakage scan w/ MI')

# Find the actual column name in df_sample matching TARGET_OVERRIDE (strip spaces)
found_target = None
for c in df_sample.columns:
    if c.strip().lower() == TARGET_OVERRIDE.strip().lower():
        found_target = c
        break

if found_target is None:
    raise SystemExit(f'Could not find target column matching {TARGET_OVERRIDE!r} in CSV header')

print(f'Using target column: {found_target!r} (matched to override {TARGET_OVERRIDE!r})')

target_col = found_target

# Coerce label to integer codes (safe: factorize)
y_raw = df_sample[target_col]
codes, uniques = pd.factorize(y_raw)
if len(uniques) > 2:
    print(f'Warning: target has {len(uniques)} distinct values in sample; factorizing to integers')

# Map to 0/1-like integers via factorize
y = codes.astype(int)

report = OrderedDict()
report['sample_rows'] = int(nrows)
report['target_col'] = target_col
report['target_cardinality_sample'] = int(len(uniques))
report['features'] = []

for raw in top_feats:
    feat_name = raw
    # locate matching column ignoring leading/trailing whitespace
    col = None
    if feat_name in df_sample.columns:
        col = feat_name
    else:
        s = feat_name.strip()
        for c in df_sample.columns:
            if c.strip().lower() == s.lower():
                col = c
                break
    if col is None:
        msg = dict(feature=feat_name, present=False, note='column not found in sample')
        report['features'].append(msg)
        print(f'Feature {feat_name!r}: not found in sample (skipped)')
        continue

    ser = df_sample[col]
    nuniq = int(ser.nunique(dropna=False))
    prop_unique = nuniq / float(nrows)

    stats = dict(feature=feat_name, present=True, col_name_in_sample=col,
                 n_unique=nuniq, prop_unique=float(prop_unique))

    vc = ser.value_counts(dropna=False).head(10).to_dict()
    try:
        grp = df_sample.groupby(col)[target_col].agg(['count','mean'])
    except Exception:
        grp = None

    deterministic_vals = 0
    rows_covered = 0
    det_examples = []
    if grp is not None:
        det_mask = (grp['mean'] <= 0.001) | (grp['mean'] >= 0.999)
        det = grp[det_mask]
        deterministic_vals = int(len(det))
        rows_covered = int(det['count'].sum())
        for val, r in det.sort_values('count', ascending=False).head(5).iterrows():
            det_examples.append({'value': str(val), 'count': int(r['count']), 'label_mean': float(r['mean'])})

    stats.update(dict(top_value_counts={str(k): int(v) for k,v in vc.items()},
                      deterministic_value_count=deterministic_vals,
                      rows_covered_by_deterministic_values=rows_covered,
                      rows_covered_pct=rows_covered / float(nrows) if nrows>0 else None,
                      deterministic_examples=det_examples))

    # prepare X column for MI
    try:
        X = ser.fillna(-999)
        if X.dtype.name == 'category' or X.dtype == object:
            X_proc = X.astype('category').cat.codes.values.reshape(-1,1)
            discrete = True
        else:
            X_proc = X.values.reshape(-1,1)
            discrete = False
        mi = float(mutual_info_classif(X_proc, y, discrete_features=discrete, random_state=0)[0])
    except Exception as e:
        mi = None

    # numeric correlation using factorized y
    try:
        if pd.api.types.is_numeric_dtype(ser):
            corr = float(ser.corr(pd.Series(y).astype(float)))
        else:
            corr = None
    except Exception:
        corr = None

    stats.update(dict(mutual_information=mi, correlation_with_target=corr))

    report['features'].append(stats)
    print(f"Scanned {feat_name!r}: n_unique={nuniq}, deterministic_values={deterministic_vals}, rows_covered={rows_covered} ({stats['rows_covered_pct']:.4f}) , mi={mi}, corr={corr}")

os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

print('\nTargeted leakage+MI scan saved to', OUT)
print('Done')
