"""Smoke loader to verify the saved CatBoost bundle loads and prints basic info.
"""
import joblib
from pathlib import Path

path = Path('models/catboost_raw.joblib')
if not path.exists():
    raise SystemExit(f'{path} not found')

bundle = joblib.load(path)
clf = bundle.get('catboost')
cats = bundle.get('cat_features', [])
threshold = bundle.get('threshold', None)

print('Loaded bundle:', path)
print('Model type:', type(clf).__name__)
print('Num cat features:', len(cats))
print('Threshold:', threshold)
