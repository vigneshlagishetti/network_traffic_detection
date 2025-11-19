import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models'
files = [
    'pruned_stack.joblib',
    'stack_kfold_lgb_meta.joblib',
    'final_lgb_model.joblib',
    'optuna_pruned_light.joblib'
]

for fn in files:
    p = MOD / fn
    print('---', fn, 'exists:', p.exists())
    if not p.exists():
        continue
    try:
        m = joblib.load(p)
        print(' type:', type(m))
        if isinstance(m, dict):
            print(' keys:', list(m.keys()))
            # also print nested types
            for k,v in m.items():
                print('  ', k, '->', type(v))
        else:
            print(' class:', m.__class__)
            print(' has_predict_proba:', hasattr(m, 'predict_proba'))
            print(' has_decision_function:', hasattr(m, 'decision_function'))
    except Exception as e:
        print(' ERROR loading:', e)

print('done')
