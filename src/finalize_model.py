"""Finalize the selected model for production.

This script copies `models/catboost_raw.joblib` to
`models/final_detector.joblib` and writes `results/final_selection.json`
containing the metrics and rationale for selection.
"""
from pathlib import Path
import joblib
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
RES = ROOT / 'results'

src = MOD / 'catboost_raw.joblib'
dst = MOD / 'final_detector.joblib'
res_in = RES / 'catboost_raw_results.json'
res_out = RES / 'final_selection.json'

if not src.exists():
    raise SystemExit(f'{src} not found')

bundle = joblib.load(src)
# write a copy as the final detector bundle
joblib.dump(bundle, dst)

metrics = {}
if res_in.exists():
    with open(res_in, 'r') as fh:
        metrics = json.load(fh)

final = {
    'selected_model': str(dst.as_posix()),
    'selected_at': datetime.utcnow().isoformat() + 'Z',
    'reason': 'Best single-model performance (accuracy and ROC AUC) from CatBoost training on raw combine.csv; validated by permutation importance and sampled 5-fold CV.',
    'metrics': metrics,
    'inference_cli': 'src/predict_with_catboost.py',
    'threshold': metrics.get('best_threshold', None)
}

with open(res_out, 'w') as fh:
    json.dump(final, fh, indent=2)

print(f'Wrote {dst} and {res_out}')
