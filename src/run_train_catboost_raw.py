"""Wrapper to run train_catboost_raw.py and capture errors to results/ for debugging.
This helps when terminal output may be suppressed; the wrapper writes a JSON error file if an exception occurs.
"""
from pathlib import Path
import runpy, json, traceback

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

try:
    runpy.run_path(str(ROOT / 'src' / 'train_catboost_raw.py'), run_name='__main__')
    (RES / 'catboost_raw_ok.txt').write_text('ok')
except Exception as e:
    info = {'error': str(e), 'traceback': traceback.format_exc()}
    with open(RES / 'catboost_raw_error.json', 'w') as fh:
        json.dump(info, fh, indent=2)
    # re-raise so conda run shows exit code
    raise
