"""Train baseline classifiers on processed NSL-KDD data and evaluate.

Trains a small set of models (RandomForest, HistGradientBoosting, LogisticRegression),
does a brief randomized search, builds a stacking classifier, and saves results.
"""
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def fit_and_eval():
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    X_train, y_train, X_test, y_test = load_processed()

    results = []

    # Define estimators
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    hgb = HistGradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=200, n_jobs=-1)

    # Lightweight hyperparameter grids
    rf_params = {"n_estimators": [100, 200], "max_depth": [None, 16, 32]}
    hgb_params = {"learning_rate": [0.05, 0.1], "max_iter": [100, 200]}
    lr_params = {"C": [0.1, 1.0, 10.0]}

    def run_search(est, params, name):
        search = RandomizedSearchCV(est, params, n_iter=3, cv=3, scoring="f1_macro", n_jobs=1, random_state=42)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"{name} -> acc: {acc:.4f}, f1_macro: {f1:.4f}")
        joblib.dump(best, MODEL_DIR / f"{name}.joblib")
        results.append({"model": name, "accuracy": acc, "f1_macro": f1})
        return best

    best_rf = run_search(rf, rf_params, "random_forest")
    best_hgb = run_search(hgb, hgb_params, "hist_gb")
    best_lr = run_search(lr, lr_params, "logistic")

    # Stacking classifier
    estimators = [("rf", best_rf), ("hgb", best_hgb), ("lr", best_lr)]
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=200), n_jobs=-1)
    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"stacking -> acc: {acc:.4f}, f1_macro: {f1:.4f}")
    joblib.dump(stack, MODEL_DIR / "stacking.joblib")
    results.append({"model": "stacking", "accuracy": acc, "f1_macro": f1})

    # Save results table
    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
    print("Saved baseline results to results/baseline_results.csv")


if __name__ == "__main__":
    fit_and_eval()
