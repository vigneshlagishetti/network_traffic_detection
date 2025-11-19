# Final comparison and recommendation

Date: 2025-11-01

Summary
-------

We compared multiple approaches trained from the original `combine.csv`.

- Best saved non-CatBoost (stacking / blend):
  - Accuracy: ~0.97596
  - ROC AUC: ~0.99543

- Blend of multiple models:
  - Accuracy: ~0.97583
  - ROC AUC: ~0.99599

- CatBoost (trained on raw `combine.csv` in conda env):
  - Best threshold: 0.496
  - Accuracy: 0.9990456665
  - ROC AUC: 0.9999598440
  - Confusion: [[250691,234],[83,81161]]

Recommendation
--------------

The CatBoost model in `models/catboost_raw.joblib` materially outperforms the
previous stack and blend models on both accuracy and ROC AUC. I recommend
promoting CatBoost as the single best detector, subject to the safety checks
below.

Safety checks performed
-----------------------

1. Permutation importance on a 50k sample — top features look plausible
   (ports, window sizes, packet sizes, IATs), not an obvious ``label`` column.
2. 5-fold stratified CV on a 100k sample — mean accuracy 0.99872 (std 0.00019),
   mean ROC AUC 0.999932 (std 0.000031).

Next recommended actions (before deployment)
-------------------------------------------

1. Inspect distribution of the top permuted features (Destination Port, etc.)
   across classes to ensure no single-value leakage. (Quick check recommended.)
2. Compute SHAP (TreeSHAP) summary on a 50k sample to validate contribution
   directions and detect possible artifacts.
3. If dataset is time-ordered, test time-split generalization (train on past,
   test on future).
4. Optionally run a full K-fold CV on the entire dataset for the most robust
   estimate (long-running).

How to run inference
--------------------

Use the included inference script:

  python src/predict_with_catboost.py --input path/to/input.csv --output results/preds.csv

The script expects column names and will coerce configured categorical columns
to strings before prediction. The saved bundle contains the threshold to use.

Artifacts
---------

- `models/catboost_raw.joblib` — model bundle (keys: 'catboost', 'cat_features', 'cat_idx', 'threshold')
- `results/catboost_raw_results.json` — result metrics from the training run
- `results/catboost_checks.json` — safety check outputs (permutation importance and sampled CV)
- `src/predict_with_catboost.py` — inference wrapper

If you want, I will run the quick distribution checks for the top 10 permutation
features now (fast) and then produce a SHAP summary plot (needs shap installed).

Signed-off-by: automated agent
