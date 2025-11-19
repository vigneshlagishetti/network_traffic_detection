"""Quick baseline training using LightGBM to get a fast accuracy estimate.

This trains a single LightGBM classifier with modest settings to produce a fast baseline.
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import joblib
import pandas as pd

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


def main():
    try:
        X_train, y_train, X_test, y_test = load_processed()
    except Exception as e:
        print("Error loading processed data:", e)
        return

    try:
        import lightgbm as lgb
    except Exception as e:
        print("LightGBM not installed. Install with: pip install lightgbm")
        return

    # small fast model
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"LightGBM quick baseline -> acc: {acc:.4f}, f1_macro: {f1:.4f}")

    joblib.dump(clf, MODEL_DIR / "lgb_quick.joblib")
    pd.DataFrame([{"model":"lgb_quick","accuracy":acc,"f1_macro":f1}]).to_csv(RESULTS_DIR / "quick_results.csv", index=False)
    print("Saved quick model and results")


if __name__ == '__main__':
    main()
