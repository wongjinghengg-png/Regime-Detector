"""Shared helpers for regime models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def order_labels_by_volatility(
    labels: np.ndarray, realized_vol: pd.Series | np.ndarray
) -> np.ndarray:
    """Remap arbitrary cluster ids to a stable, interpretable ordering.

    Unsupervised models assign regime ids arbitrarily (run twice, get the same
    clusters with swapped numbers). We relabel so that regime 0 is always the
    lowest-volatility state and the highest id is the most volatile, which makes
    plots and metrics comparable across models and runs.
    """
    labels = np.asarray(labels)
    vol = np.asarray(realized_vol, dtype=float)
    uniq = np.unique(labels)
    mean_vol = {k: vol[labels == k].mean() for k in uniq}
    order = sorted(uniq, key=lambda k: mean_vol[k])
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels])
