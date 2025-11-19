"""Test loading saved model joblib files and report exceptions.
Run this in the same environment used for `mixed_ensemble.py` to diagnose load failures.
"""
from pathlib import Path
import joblib
from traceback import format_exc

MOD = Path(__file__).resolve().parents[1] / 'models'
files = sorted(MOD.glob('ensemble_base_lgb_fold*.joblib')) + sorted(MOD.glob('bag_lgb_*.joblib'))
print('Found files:', files)
for f in files:
    try:
        print('\nLoading', f)
        obj = joblib.load(f)
        print('Loaded OK; type:', type(obj))
        # try to call predict_proba if present (quick smoke test)
        if hasattr(obj, 'predict_proba'):
            print('Has predict_proba, OK')
        else:
            print('No predict_proba attribute')
    except Exception as e:
        print('ERROR loading', f)
        print(format_exc())
