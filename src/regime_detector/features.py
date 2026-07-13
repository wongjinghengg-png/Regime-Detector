"""Feature engineering for regime detection.

All features are derived from the price series and are strictly causal (they
use only past and current data at each point) *except* where noted. Regime
models are unsupervised, so we do not have a leakage problem in the usual
supervised sense, but keeping features causal makes the exercise closer to a
realistic setup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def build_features(prices: pd.DataFrame, vol_window: int = 21) -> pd.DataFrame:
    """Return a feature frame aligned to ``prices`` (rows with NaNs dropped).

    Features
    --------
    log_return       : daily log return
    realized_vol     : rolling annualized std of returns (``vol_window`` days)
    downside_vol     : rolling annualized std of negative returns only
    drawdown         : current drawdown from the running peak (<= 0)
    return_zscore    : return standardized by its rolling mean/std
    """
    close = prices["Close"].astype(float)
    log_ret = np.log(close / close.shift(1))

    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)

    downside = log_ret.where(log_ret < 0, 0.0)
    downside_vol = downside.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)

    running_peak = close.cummax()
    drawdown = close / running_peak - 1.0

    roll_mean = log_ret.rolling(vol_window).mean()
    roll_std = log_ret.rolling(vol_window).std()
    return_zscore = (log_ret - roll_mean) / roll_std.replace(0.0, np.nan)

    feats = pd.DataFrame(
        {
            "log_return": log_ret,
            "realized_vol": realized_vol,
            "downside_vol": downside_vol,
            "drawdown": drawdown,
            "return_zscore": return_zscore,
        }
    )
    return feats.dropna()


def model_matrix(features: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    """Select and standardize the columns fed to a model.

    Standardization uses the full-sample mean/std. This is a deliberate,
    documented simplification: it introduces mild lookahead and is one of the
    reasons this project is a research prototype, not an execution system.
    """
    if columns is None:
        columns = ["log_return", "realized_vol", "drawdown"]
    x = features[columns].to_numpy(dtype=float)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (x - mu) / sigma
