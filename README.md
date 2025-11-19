# EmberEye

A production-ready binary detector and analysis toolkit for network/traffic data trained from `combine.csv`.

This repository contains training code, reproducibility checks, diagnostics, and deployment utilities for a CatBoost-based binary detector promoted to production, plus experiment artifacts for other candidate models and ensembles.

## 🔥 Project overview

EmberEye (also referenced historically as Fruty / EmbeReye) packages a single, high-performing CatBoost model and the supporting scripts/notebooks used to train, validate and diagnose it. The repository emphasizes reproducibility and clear experiment artifacts:

- Trained model bundles in `models/`
- Canonical experiment outputs in `results/` (JSON + figures)
- Training & validation scripts in `src/`
- Interactive EDA and diagnostics in `notebooks/`

The dataset used for development is `combine.csv` (root). Many heavy operations use sampling (e.g., 100k–200k rows); full-dataset runs are possible but long.

## 📊 Dataset

- File: `combine.csv` (primary training + validation data)
- Size: ~2.2M rows, ~78 features (mix of numeric and categorical). Target column in the CSV header is `Label`.

Always confirm the header names (some CSVs include leading/trailing whitespace in headers); the `src/` scripts attempt to be robust but explicit checks are recommended.

## 🎯 Objectives

- Produce a single, production-ready binary detector (CatBoost) and bundle it as `models/final_detector.joblib`.
- Validate robustness with sampled CV, permutation importance, leakage scans and reproducibility tests.
- Provide simple inference tooling and a compact reproducibility checklist for auditors.

## 🛠️ Technologies

- Python 3.x
- CatBoost (final model), scikit-learn
- pandas, numpy
- joblib for model persistence
- matplotlib, seaborn, plotly for visualization
- Optional: shap for explanation cells in notebooks

Environment artifacts:

- `environment-catboost.yml` — recommended conda env for CatBoost on Windows
- `requirements.txt` — pip-installable deps for notebooks and utility scripts

## 🔁 How to run (Windows — quick)

Create and activate the conda environment (recommended for CatBoost on Windows):

```cmd
conda env create -f environment-catboost.yml -n embereye
conda activate embereye
```

Install pip deps (optional / for notebooks):

```cmd
pip install -r requirements.txt
```

Train CatBoost (example; long-running):

```cmd
python src\train_catboost_on_raw.py --data combine.csv --out_dir results --model_out models/catboost_raw.joblib
```

Run diagnostics (permutation importance + sampled CV):

```cmd
python src\catboost_checks.py --model models/catboost_raw.joblib --data combine.csv --out results/catboost_checks.json
```

Finalize the production bundle:

```cmd
python src\finalize_model.py --src models/catboost_raw.joblib --dst models/final_detector.joblib --out results/final_selection.json
```

Smoke-load the final model:

```cmd
python src\_smoke_load_catboost.py --model models/final_detector.joblib
```

Run inference:

```cmd
python src\predict_with_catboost.py --model models/final_detector.joblib --input sample_input.csv --output results\preds.csv --threshold 0.496
```

See `--help` on each script for sampling, seed and other useful flags.

## 📁 Project structure (short)

```
combine.csv
models/
notebooks/
results/
src/
README.md
README_FULL.md
environment-catboost.yml
requirements.txt
```

Key scripts in `src/`:

- `train_catboost_on_raw.py` — train and save CatBoost artifact
- `catboost_checks.py` — permutation importance and sampled CV
- `predict_with_catboost.py` — inference CLI
- `_smoke_load_catboost.py` — quick model load test
- `repro_test.py` — reproducibility verification
- `finalize_model.py` — create `models/final_detector.joblib`
- `leakage_scan*` — leakage detection and MI analysis

<!-- Center credit moved to end -->

## 🔍 Diagnostics & results

Canonical result files (examples):

- `results/catboost_raw_results.json` — final training metrics (best_threshold, accuracy, roc_auc)
- `results/catboost_checks.json` — permutation importance and sampled CV folds
- `results/leakage_scan.json` — leakage and mutual information outputs

Key reported snapshot (from results):

- best_threshold: 0.496
- best_accuracy: 0.9990456665131303 (≈ 99.9046%)
- roc_auc: 0.9999598440131221 (≈ 0.99996)
- confusion matrix (at best threshold): [[250691, 234], [83, 81161]]
- CV (sampled 5-fold on 100k rows): mean accuracy = 0.99872 (std = 0.00019)

Other candidate models / ensembles (examples):
- Best saved non-CatBoost stack/blend: accuracy ≈ 0.9758250532292406 (≈ 97.58%)

Sources (ground truth): `results/catboost_raw_results.json`, `results/catboost_checks.json`, `results/final_comparison_report.md`, `results/blend_results.json`

Use the JSON files in `results/` as the ground truth for reported numbers and audits.

## 📈 Visualizations

This repository includes an interactive notebook and repeatable scripts to produce visualizations used during model analysis and validation. Use the notebook for exploration and the scripts for CI-friendly exports.

- `notebooks/visualization.ipynb` — interactive notebook containing cells to:

  - load a sampled subset and the model artifact (`models/final_detector.joblib`),
  - plot class balance and feature distributions by class,
  - render correlation heatmaps and 2D projections (PCA/UMAP),
  - compute and plot permutation importance (ranked bar chart),
  - compute SHAP (TreeSHAP) summaries and beeswarm plots for feature-level explanations,
  - draw ROC and Precision-Recall curves for the holdout or sampled CV folds.
- Script-driven exports: when you run the diagnostics scripts (for example `src/catboost_checks.py` and `src/threshold_tune.py`) they write JSON summaries under `results/` and can be paired with the notebook cells to export PNG/HTML figures into `results/figs/`.

Quick commands (Windows cmd) to reproduce key visuals non-interactively:

```cmd
REM execute the notebook headlessly and save outputs (requires jupyter and notebook deps)
jupyter nbconvert --to notebook --execute notebooks/visualization.ipynb --ExecutePreprocessor.timeout=600 --output notebooks/visualization_executed.ipynb
```

Interpretation tips:

- Permutation importance highlights features the model relies on; validate top features for leakage or domain plausibility.
- SHAP summary and beeswarm show global and per-sample feature impacts and directions — useful for debugging model decisions and explaining edge cases.
- Use PR curves when class imbalance is high; ROC AUC is useful for ranking but can be optimistic on imbalanced sets.

## 🧪 Models evaluated (concise)

- CatBoost — final single-model detector (promoted to `models/final_detector.joblib`). Best operating-point accuracy with very high AUC.
- LightGBM / XGBoost — used in Optuna tuning and stacking experiments; strong AUC in trials but did not outperform CatBoost at the chosen operating threshold in our runs.
- Stacked / blended ensembles — improved AUC in some experiments; retained in `results/stack_improved_results.json` and `results/blend_results.json` for follow-up research.
- RandomForest — used for quick diagnostic importances.

Recommendation: CatBoost selected for production due to reproducible metrics, ease of packaging, and inference efficiency. If the priority is pure research AUC, re-running ensemble CV is recommended.

## 🔧 Notes & gotchas

- CatBoost on Windows: prefer conda (`environment-catboost.yml`) to avoid pip build issues.
- CSV headers: verify `Label` column naming and whitespace; leakage scans assume correct target mapping.
- Large dataset: many heavy ops use sampling; full K-fold CV is long-running but included as an advised next step.

## 📈 Next steps (optional)

- Run SHAP TreeSHAP on a 10–20k sample and save plots to `results/figs/`.
- Produce a per-model comparison table by extracting metrics from `results/*.json` and add it to `README_FULL.md`.
- Add CI/GitHub Actions smoke test to validate model loading and a tiny inference check.

## 🗂️ Git LFS (large files)

This repository tracks large assets with Git LFS (dataset and model bundles):

- Tracked via LFS: `combine.csv`, `models/*.joblib`, `data/processed/*.npz`

Collaborators — do this once after cloning:

```cmd
git lfs install
git lfs pull
```

Notes

- GitHub enforces LFS storage/bandwidth quotas. Monitor usage in the repository’s Settings → Packages/LFS. If you exceed quota, pushes/pulls of LFS objects may be blocked until you reduce usage or add capacity.
- If you see “file exceeds 100 MB” or “LFS bandwidth exceeded” errors, ensure LFS is installed, consider pruning old large artifacts, or move heavy assets to external storage.

## 🤝 Contributing

Contributions welcome — open an issue or submit a PR. Please include reproducible tests for code changes.

## 📄 License

Educational / research use. See `LICENSE` if included.

<!-- dmlnbmVzaA== -->

| 🌟 Made with ❤️ by**Sai Meghana** 🌟 |
| :------------------------------------------: |
