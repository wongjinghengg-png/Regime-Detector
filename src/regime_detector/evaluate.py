"""Regime-conditional evaluation.

Since regimes are unsupervised, we do not have ground-truth labels to score
against. Instead we characterize what each detected regime *looks like* —
average return, volatility, drawdown, frequency — and measure how persistent
and how well-separated the regimes are. Good regimes are economically distinct
(calm regimes have high Sharpe, crisis regimes have deep drawdowns) and sticky
(few switches).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import TRADING_DAYS


def regime_summary(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-regime economic statistics."""
    df = features.copy()
    df["regime"] = labels
    rows = []
    for r, g in df.groupby("regime"):
        ret = g["log_return"]
        ann_ret = ret.mean() * TRADING_DAYS
        ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        rows.append(
            {
                "regime": int(r),
                "n_days": len(g),
                "pct_time": len(g) / len(df),
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "avg_drawdown": g["drawdown"].mean(),
                "worst_drawdown": g["drawdown"].min(),
            }
        )
    return pd.DataFrame(rows).set_index("regime").sort_index()


def n_switches(labels: np.ndarray) -> int:
    """Number of times the regime changes across the series (lower = stickier)."""
    labels = np.asarray(labels)
    return int((labels[1:] != labels[:-1]).sum())


def average_run_length(labels: np.ndarray) -> float:
    """Mean number of consecutive days spent in a regime before switching."""
    return len(labels) / (n_switches(labels) + 1)


def stability_report(labels_by_model: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare persistence across models."""
    rows = []
    for name, labels in labels_by_model.items():
        rows.append(
            {
                "model": name,
                "n_regimes": len(np.unique(labels)),
                "n_switches": n_switches(labels),
                "avg_run_length_days": average_run_length(labels),
            }
        )
    return pd.DataFrame(rows).set_index("model")
