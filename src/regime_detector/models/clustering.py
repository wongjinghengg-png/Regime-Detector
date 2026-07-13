"""Clustering-based regime detectors (KMeans and Gaussian Mixture).

These treat each day as an independent point in feature space and cluster them.
They ignore time ordering entirely, which makes them a useful middle ground
between the naive threshold baseline and the sequence-aware HMM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import order_labels_by_volatility


def kmeans_fit_predict(
    features: pd.DataFrame, X: np.ndarray, n_regimes: int = 3, seed: int = 42
) -> np.ndarray:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_regimes, n_init=10, random_state=seed)
    raw = km.fit_predict(X)
    return order_labels_by_volatility(raw, features["realized_vol"])


def gmm_fit_predict(
    features: pd.DataFrame, X: np.ndarray, n_regimes: int = 3, seed: int = 42
) -> np.ndarray:
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_regimes, covariance_type="full", random_state=seed
    )
    raw = gmm.fit_predict(X)
    return order_labels_by_volatility(raw, features["realized_vol"])
