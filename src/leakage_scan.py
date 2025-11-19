import os
import json
import math
from collections import OrderedDict
import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif

ROOT = r"c:\ECHO\Projects\Personal_Projects\Fruty"
CSV = os.path.join(ROOT, "combine.csv")
CHECKS = os.path.join(ROOT, "results", "catboost_checks.json")
OUT = os.path.join(ROOT, "results", "leakage_scan.json")

# defensive: load checks file
with open(CHECKS, 'r', encoding='utf-8') as f:
    checks = json.load(f)

# extract top features from permutation importance
perm = checks.get('top_20_permutation_importance') or checks.get('top_permutation_importance') or []
if not perm:
    raise SystemExit('No permutation importance found in checks file')

top_feats = [p[0] for p in perm][:10]
# strip column names for matching
top_feats_stripped = [f.strip() for f in top_feats]

# Read a sample for speed (200k rows if possible)
SAMPLE_N = 200_000
chunks = pd.read_csv(CSV, chunksize=100_000, low_memory=False)
try:
    df_sample = next(chunks)
    # if need more rows, concat next chunk
    if len(df_sample) < SAMPLE_N:
        try:
            df_sample = pd.concat([df_sample, next(chunks)])
        except StopIteration:
            pass
except StopIteration:
    # tiny file
    df_sample = pd.read_csv(CSV, low_memory=False)

nrows = len(df_sample)
print(f'Using sample of {nrows} rows for quick leakage scan')

# detect target column
possible_targets = ['Label','label','target','Target','Class','class','is_attack','attack','y','Y']
target_col = None
for t in possible_targets:
    if t in df_sample.columns:
        target_col = t
        break
# fallback: find binary column with two unique values and name contains 'label' or 'class'
if target_col is None:
    for c in df_sample.columns:
        nunique = df_sample[c].dropna().unique()
        if len(nunique) <= 2 and set(map(str, nunique)).issubset({'0','1','True','False','true','false'}):
            target_col = c
            break

if target_col is None:
    raise SystemExit('Could not detect target column in sample (tried common names)')

print(f'Detected target column: {target_col}')

report = OrderedDict()
report['sample_rows'] = int(nrows)
report['target_col'] = target_col
report['features'] = []

for raw in top_feats:
    feat_name = raw
    # find matching column in dataframe (try exact, then stripped)
    col = None
    if feat_name in df_sample.columns:
        col = feat_name
    else:
        # try stripped
        s = feat_name.strip()
        if s in df_sample.columns:
            col = s
        else:
            # try case-insensitive
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

    # basic stats
    stats = dict(feature=feat_name, present=True, col_name_in_sample=col,
                 n_unique=nuniq, prop_unique=float(prop_unique))

    # value counts (top 5)
    vc = ser.value_counts(dropna=False).head(5).to_dict()
    # group by value -> mean target and counts
    try:
        grp = df_sample.groupby(col)[target_col].agg(['count','mean'])
    except Exception as e:
        grp = None

    deterministic_vals = 0
    rows_covered = 0
    det_examples = []
    if grp is not None:
        # deterministic if mean very near 0 or 1
        det_mask = (grp['mean'] <= 0.001) | (grp['mean'] >= 0.999)
        det = grp[det_mask]
        deterministic_vals = int(len(det))
        rows_covered = int(det['count'].sum())
        # top deterministic examples
        for val, r in det.sort_values('count', ascending=False).head(5).iterrows():
            det_examples.append({'value': str(val), 'count': int(r['count']), 'label_mean': float(r['mean'])})

    stats.update(dict(top_value_counts={str(k): int(v) for k,v in vc.items()},
                      deterministic_value_count=deterministic_vals,
                      rows_covered_by_deterministic_values=rows_covered,
                      rows_covered_pct=rows_covered / float(nrows) if nrows>0 else None,
                      deterministic_examples=det_examples))

    # mutual information (safe for numeric or categorical)
    try:
        X = ser.fillna(-999)
        # convert categorical to codes
        if X.dtype.name == 'category' or X.dtype == object:
            X_proc = X.astype('category').cat.codes.values.reshape(-1,1)
            discrete = True
        else:
            X_proc = X.values.reshape(-1,1)
            discrete = False
        y = df_sample[target_col].astype(int).values
        mi = float(mutual_info_classif(X_proc, y, discrete_features=discrete, random_state=0)[0])
    except Exception as e:
        mi = None

    # numeric correlation
    try:
        if pd.api.types.is_numeric_dtype(ser):
            corr = float(ser.corr(df_sample[target_col].astype(float)))
        else:
            corr = None
    except Exception:
        corr = None

    stats.update(dict(mutual_information=mi, correlation_with_target=corr))

    report['features'].append(stats)
    print(f"Scanned {feat_name!r}: n_unique={nuniq}, deterministic_values={deterministic_vals}, rows_covered={rows_covered} ({stats['rows_covered_pct']:.4f}) , mi={mi}, corr={corr}")

# write out the report
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

print('\nLeakage scan saved to', OUT)
print('Summary:')
for f in report['features']:
    if not f.get('present'):
        print(' -', f['feature'], 'NOT FOUND')
    else:
        print(' -', f['feature'].strip(),
              f"| unique={f['n_unique']}",
              f"| det_vals={f['deterministic_value_count']}",
              f"| rows_cov={f['rows_covered_by_deterministic_values']}",
              f"| rows_pct={(f['rows_covered_pct'] if f['rows_covered_pct'] is not None else 'NA')}")

# Also print any features with >0 deterministic values
bad = [f for f in report['features'] if f.get('present') and f.get('deterministic_value_count',0) > 0]
if bad:
    print('\nPotential leakage suspects (deterministic values found):')
    for f in bad:
        print(' -', f['feature'].strip(), '->', f['deterministic_value_count'], 'values covering', f['rows_covered_by_deterministic_values'], 'rows')
else:
    print('\nNo deterministic value->label mappings found in the sampled rows for top features')

print('Done')
