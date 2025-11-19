"""Quick smoke test for ABA module.
Run this inside the project venv to validate import and basic behavior.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.feature_selection.aba import ArtificialButterfly


def fitness(Xs, ys):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=200)
    return cross_val_score(clf, Xs, ys, cv=3, scoring='f1_macro').mean()


def main():
    X = np.random.RandomState(0).rand(100, 20)
    y = (X[:, 0] > 0.5).astype(int)
    aba = ArtificialButterfly(pop_size=8, n_iter=4, random_state=1)
    mask, score = aba.fit(X, y, fitness)
    print('smoke ok, best score', score, 'n_features', int(mask.sum()))


if __name__ == '__main__':
    main()
