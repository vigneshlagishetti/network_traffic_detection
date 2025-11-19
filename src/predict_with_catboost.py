"""Simple inference wrapper for the saved CatBoost model bundle.

Usage examples:
  python src/predict_with_catboost.py --input data/sample_to_score.csv --output results/catboost_predictions.csv

The script expects the model bundle created by `src/train_catboost_on_raw.py` at
`models/catboost_raw.joblib` by default. It will coerce configured categorical
columns to strings, predict probabilities, apply the saved threshold, and write
predictions with probability to the output CSV.
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/catboost_raw.joblib')
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='results/catboost_predictions.csv')
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'{model_path} not found')

    bundle = joblib.load(model_path)
    clf = bundle.get('catboost')
    cat_cols = bundle.get('cat_features', [])
    threshold = float(bundle.get('threshold', 0.5))

    X = pd.read_csv(args.input, low_memory=False)

    # ensure columns exist and cast cats to string
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna('nan').astype(str)

    try:
        proba = clf.predict_proba(X)[:, 1]
    except Exception:
        # try passing DataFrame via Pool handled internally by CatBoost
        proba = clf.predict_proba(X)[:, 1]

    pred = (proba >= threshold).astype(int)

    out = X.copy()
    out['catboost_proba'] = proba
    out['catboost_pred'] = pred
    out.to_csv(args.output, index=False)

    tn = int(((pred == 0)).sum())
    tp = int(((pred == 1)).sum())
    print(f'Wrote {args.output} rows={len(out)} pred_1={tp} pred_0={tn} threshold={threshold}')


if __name__ == '__main__':
    main()
