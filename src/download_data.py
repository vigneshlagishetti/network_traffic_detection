"""Download script for NSL-KDD datasets (KDDTrain+.txt and KDDTest+.txt)

This script downloads the commonly used NSL-KDD training and test files
from a canonical GitHub mirror. It saves files to `data/raw/`.

Run:
    python src\download_data.py
"""
import os
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")

FILES = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def download_file(url: str, dest: str):
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f"Saved {dest} ({size:,} bytes)")


def main():
    ensure_dir(RAW_DIR)
    for name, url in FILES.items():
        dest = os.path.join(RAW_DIR, name)
        if os.path.exists(dest):
            print(f"Already exists: {dest} (skipping)")
            continue
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    main()
