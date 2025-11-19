"""Quick test: load NSL-KDD raw files and print basic info without heavy deps.

This avoids importing scikit-learn/scipy to confirm the raw data is present and parsable.
"""
import os
import sys

# Ensure project root is on sys.path so we can import the local `src` package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_nsl_kdd


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(root, "data", "raw")
    train = os.path.join(raw_dir, "KDDTrain+.txt")
    test = os.path.join(raw_dir, "KDDTest+.txt")

    print("Train file:", train)
    print("Test file:", test)

    if not os.path.exists(train) or not os.path.exists(test):
        print("Missing NSL-KDD raw files in data/raw. Run src/download_data.py first.")
        return

    print("Loading train (this should be quick)...")
    df_train = load_nsl_kdd(train)
    print("Train shape:", df_train.shape)
    print("Train label distribution:\n", df_train['label'].value_counts().head(10))

    print("Loading test (this should be quick)...")
    df_test = load_nsl_kdd(test)
    print("Test shape:", df_test.shape)
    print("Test label distribution:\n", df_test['label'].value_counts().head(10))

    sample_out = os.path.join(root, 'data', 'processed', 'sample_train_head.csv')
    os.makedirs(os.path.dirname(sample_out), exist_ok=True)
    df_train.head(100).to_csv(sample_out, index=False)
    print(f"Wrote sample head to {sample_out}")


if __name__ == '__main__':
    main()
