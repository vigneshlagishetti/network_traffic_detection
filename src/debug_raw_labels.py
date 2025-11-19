"""Quick diagnostic: load combine.csv and print label samples and counts.
Runs fast and avoids training heavy models.
"""
from pathlib import Path
from collections import Counter
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'combine.csv'
if not CSV.exists():
    raise FileNotFoundError(CSV)

# load repo loader
spec = importlib.util.spec_from_file_location('preprocessing', str(ROOT / 'src' / 'preprocessing.py'))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
df = pp.load_nsl_kdd(str(CSV))
print('Loaded shape:', df.shape)
if 'label' not in df.columns:
    print('No label column found; columns:', list(df.columns)[:10])
else:
    vals = df['label'].values
    from itertools import islice
    print('Label sample (first 50):', list(islice(vals, 50)))
    cnt = Counter(vals)
    print('Unique label counts (top 20):')
    for k,v in list(cnt.items())[:20]:
        print(' ', k, v)
    print('Total rows:', len(vals))
