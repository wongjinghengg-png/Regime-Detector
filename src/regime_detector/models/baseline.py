"""Rule-based volatility-threshold baseline.

The simplest possible regime detector: bucket each day by where its realized
volatility falls relative to historical quantiles. It has no notion of state
persistence, which is exactly why the HMM (which models transitions) should
beat it — having this baseline is what makes that comparison meaningful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_predict(
    features: pd.DataFrame,
    n_regimes: int = 3,
    quantiles: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Assign each row a regime 0..n_regimes-1 by realized-vol quantile bucket.

    Regime 0 = calmest. Labels are already vol-ordered by construction.
    """
    vol = features["realized_vol"].to_numpy(dtype=float)
    if quantiles is None:
        quantiles = tuple(np.linspace(0, 1, n_regimes + 1)[1:-1])
    edges = np.quantile(vol, quantiles)
    return np.digitize(vol, edges).astype(int)
