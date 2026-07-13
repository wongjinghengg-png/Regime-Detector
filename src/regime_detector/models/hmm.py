"""Gaussian Hidden Markov Model regime detector.

An HMM is the canonical tool for regime detection: it models the market as a
sequence of *hidden* states, each emitting returns from its own Gaussian, with
a transition matrix capturing how sticky regimes are. Unlike the baseline and
the clustering model, it explicitly rewards temporal persistence, so it tends
to produce cleaner, less jittery regime sequences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import order_labels_by_volatility


class HMMRegimeDetector:
    def __init__(self, n_regimes: int = 3, n_iter: int = 200, seed: int = 42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.seed = seed
        self.model = None

    def fit_predict(self, features: pd.DataFrame, X: np.ndarray) -> np.ndarray:
        from hmmlearn.hmm import GaussianHMM

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.seed,
        )
        self.model.fit(X)
        raw = self.model.predict(X)
        return order_labels_by_volatility(raw, features["realized_vol"])

    @property
    def transition_matrix(self) -> np.ndarray | None:
        return None if self.model is None else self.model.transmat_

    def expected_durations(self) -> np.ndarray | None:
        """Expected persistence (in days) of each raw hidden state = 1/(1-p_ii)."""
        if self.model is None:
            return None
        diag = np.clip(np.diag(self.model.transmat_), 0, 1 - 1e-9)
        return 1.0 / (1.0 - diag)


def fit_predict(features: pd.DataFrame, X: np.ndarray, n_regimes: int = 3) -> np.ndarray:
    return HMMRegimeDetector(n_regimes=n_regimes).fit_predict(features, X)
