"""Exploratory Data Analysis script for NSL-KDD processed data.

Generates and saves plots to results/figures and writes a short summary to results/eda_summary.txt.
Run with the project's virtual environment Python, e.g.:

.venv\Scripts\python src\eda.py
"""
import os
import sys
from pathlib import Path

# ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_SUM = ROOT / "results" / "eda_summary.txt"


def load_processed():
    train_f = ROOT / "data" / "processed" / "train_processed.npz"
    test_f = ROOT / "data" / "processed" / "test_processed.npz"
    if train_f.exists() and test_f.exists():
        data_train = np.load(train_f)
        data_test = np.load(test_f)
        X_train = data_train["X"]
        y_train = data_train["y"]
        X_test = data_test["X"]
        y_test = data_test["y"]
        return X_train, y_train, X_test, y_test
    else:
        raise FileNotFoundError("Processed train/test not found in data/processed. Run src/preprocessing.py first.")


def write_summary(text: str):
    with open(OUT_SUM, "w", encoding="utf8") as f:
        f.write(text)


def plot_class_distribution(y, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(8, 4))
    plt.bar(unique.astype(str), counts)
    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_missingness(X, out_path, max_cols=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # X is numeric array; check for NaNs per column
    nan_counts = np.isnan(X).sum(axis=0)
    cols = min(len(nan_counts), max_cols)
    plt.figure(figsize=(10, 4))
    plt.plot(nan_counts[:cols], marker="o")
    plt.title("Missing values per feature (first %d columns)" % cols)
    plt.xlabel("Feature index")
    plt.ylabel("# missing")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_distributions(X, out_dir, sample_features=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_features = X.shape[1]
    idxs = np.linspace(0, n_features - 1, min(sample_features, n_features), dtype=int)
    for i in idxs:
        plt.figure(figsize=(6, 3))
        col = X[:, i]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        plt.hist(col, bins=50)
        plt.title(f"Feature {i} distribution (non-NaN)")
        plt.tight_layout()
        plt.savefig(out_dir / f"feature_{i:03d}_hist.png")
        plt.close()


def main():
    try:
        X_train, y_train, X_test, y_test = load_processed()
    except Exception as e:
        print("Error loading processed data:", e)
        return

    summary_lines = []
    summary_lines.append(f"train shape: {X_train.shape}, y shape: {y_train.shape}")
    summary_lines.append(f"test shape: {X_test.shape}, y shape: {y_test.shape}")

    # class distribution
    plot_class_distribution(y_train, FIG_DIR / "class_distribution_train.png")
    summary_lines.append("Saved class distribution plot: results/figures/class_distribution_train.png")

    # missingness plot
    plot_missingness(X_train, FIG_DIR / "missingness_train.png")
    summary_lines.append("Saved missingness plot: results/figures/missingness_train.png")

    # feature distributions (sample)
    plot_feature_distributions(X_train, FIG_DIR, sample_features=6)
    summary_lines.append("Saved sample feature histograms to results/figures/")

    write_summary("\n".join(summary_lines))
    print("EDA complete. Summary written to results/eda_summary.txt")


if __name__ == "__main__":
    main()
