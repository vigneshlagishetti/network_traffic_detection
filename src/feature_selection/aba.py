"""Artificial Butterfly Algorithm (ABA) — lightweight binary feature selector.

This file provides a simple, well-documented implementation of a binary
Artificial Butterfly Algorithm suitable as a feature-selection wrapper.

API:
 - ArtificialButterfly(pop_size=20, n_iter=30, p_switch=0.8, random_state=None)
 - fit(self, X, y, fitness_fn)
   where fitness_fn(X_sub, y) -> float (higher is better)

The implementation uses continuous encodings in [0,1] and thresholds at 0.5
to produce binary masks. It's intentionally simple and fast so it can be used
as a surrogate optimizer (e.g., using a LightGBM quick model as the fitness).

This is a deterministic, local implementation for experimentation and will be
extended later with hybrid PSO/GA refinement and logging utilities.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, List, Optional


class ArtificialButterfly:
    """Simple Artificial Butterfly Algorithm for binary feature selection.

    Parameters
    ----------
    pop_size : int
        Number of butterflies (candidate solutions).
    n_iter : int
        Number of iterations to run.
    p_switch : float
        Probability of global search (move toward global best). Lower means
        more local search.
    random_state : Optional[int]
        Seed for reproducibility.
    """

    def __init__(self, pop_size: int = 20, n_iter: int = 30, p_switch: float = 0.8, random_state: Optional[int] = None):
        assert pop_size > 1
        assert 0.0 <= p_switch <= 1.0
        self.pop_size = int(pop_size)
        self.n_iter = int(n_iter)
        self.p_switch = float(p_switch)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # to be filled after fit
        self.best_mask_: Optional[np.ndarray] = None
        self.best_score_: Optional[float] = None
        self.history_: List[float] = []

    def _threshold(self, pop: np.ndarray) -> np.ndarray:
        # continuous values in [0,1] -> binary mask
        return (pop >= 0.5).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray, fitness_fn: Callable[[np.ndarray, np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """Run ABA to select features.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        fitness_fn : callable
            Function that takes (X_selected, y) and returns a scalar score (higher is better).

        Returns
        -------
        best_mask, best_score
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # initialize population in continuous space [0,1]
        pop = self.rng.rand(self.pop_size, n_features)

        # evaluate initial population
        masks = self._threshold(pop)
        scores = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            mask = masks[i].astype(bool)
            if not mask.any():
                # ensure at least one feature
                mask[self.rng.randint(0, n_features)] = True
            X_sub = X[:, mask]
            try:
                scores[i] = float(fitness_fn(X_sub, y))
            except Exception:
                scores[i] = -np.inf

        best_idx = int(np.argmax(scores))
        best = pop[best_idx].copy()
        best_score = float(scores[best_idx])
        self.history_.append(best_score)

        # main loop
        for t in range(self.n_iter):
            for i in range(self.pop_size):
                r = self.rng.rand()
                # decide global or local search
                if r < self.p_switch:
                    # global: move toward global best
                    step = self.rng.rand(n_features) * (best - pop[i])
                else:
                    # local: move using two random butterflies
                    a, b = self.rng.choice(self.pop_size, size=2, replace=False)
                    step = self.rng.rand(n_features) * (pop[a] - pop[b])

                # update with a small scaling factor
                new = pop[i] + 0.1 * step
                # simple random perturbation (exploration)
                eps = 0.01 * self.rng.randn(n_features)
                new = np.clip(new + eps, 0.0, 1.0)

                # evaluate new candidate
                mask_new = (new >= 0.5)
                if not mask_new.any():
                    mask_new[self.rng.randint(0, n_features)] = True
                X_sub = X[:, mask_new]
                try:
                    new_score = float(fitness_fn(X_sub, y))
                except Exception:
                    new_score = -np.inf

                # replace if better
                if new_score > scores[i]:
                    pop[i] = new
                    scores[i] = new_score

                    # update global best
                    if new_score > best_score:
                        best_score = new_score
                        best = new.copy()

            self.history_.append(best_score)

        # finalize
        best_mask = (best >= 0.5).astype(int)
        # ensure at least one feature
        if not best_mask.any():
            best_mask[self.rng.randint(0, n_features)] = 1

        self.best_mask_ = best_mask
        self.best_score_ = float(best_score)
        return best_mask, best_score


def example_usage():
    """Very small smoke example using a quick sklearn estimator as fitness.

    This function is only for local testing and will not be executed automatically.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # synthetic data
    X = np.random.RandomState(0).rand(200, 30)
    y = (X[:, 0] + X[:, 1] * 0.5 > 0.9).astype(int)

    def fitness(X_sub, y):
        if X_sub.shape[1] == 0:
            return 0.0
        clf = LogisticRegression(max_iter=200)
        return cross_val_score(clf, X_sub, y, cv=3, scoring='f1_macro').mean()

    aba = ArtificialButterfly(pop_size=10, n_iter=5, random_state=0)
    mask, score = aba.fit(X, y, fitness)
    print('example best score', score, 'n_features', mask.sum())


if __name__ == '__main__':
    example_usage()
