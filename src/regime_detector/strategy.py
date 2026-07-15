"""A regime-conditioned strategy backtest — built only on out-of-sample signals.

This is the "so what?" of the project: if the detector is any good, tilting
exposure by the *online* regime call should improve risk-adjusted returns versus
just holding the asset.

Two rules keep it honest (an interviewer will check both):

1. **No lookahead in the signal.** It uses the walk-forward online regimes from
   ``walkforward.py``, each of which is computed from data up to that day only.
2. **No lookahead in execution.** The regime known at the close of day *t* sets
   the position held over day *t+1* (``shift(1)``). You never trade on a return
   you haven't observed yet.

The regime→exposure map is fixed and intentionally un-optimized (100% in calm,
50% in choppy, 0% in crisis). Tuning those weights on the same data would be
overfitting; the point is to test the *signal*, not to curve-fit a strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import TRADING_DAYS

# Exposure to the asset per regime id (0 = calmest ... higher = more stressed).
DEFAULT_WEIGHTS = {0: 1.0, 1: 0.5, 2: 0.0}


def backtest(
    features: pd.DataFrame,
    online: pd.Series,
    weights: dict[int, float] | None = None,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Run the regime-tilted strategy vs. buy-and-hold over the online window.

    Parameters
    ----------
    online   : out-of-sample regime labels (``-1`` during warm-up) from
               ``walkforward.walk_forward_hmm``.
    weights  : regime id -> target exposure (0..1). Missing regimes default flat.
    cost_bps : round-trip-agnostic transaction cost, in basis points per unit of
               turnover (|Δweight|), charged when the position changes.

    Returns a DataFrame indexed by date with columns: ``regime`` (the signal
    driving that day's position), ``weight``, ``asset_return``, ``strategy_return``,
    ``buyhold_return``, ``strategy_equity``, ``buyhold_equity``.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Restrict to days that have a real online signal (drop the warm-up).
    valid = online[online >= 0]
    idx = valid.index
    if len(idx) < 2:
        raise ValueError("Not enough out-of-sample signal to backtest.")

    log_ret = features.loc[idx, "log_return"]
    simple_ret = np.expm1(log_ret)  # scale-correct returns for partial exposure

    # Signal known at close of t drives the position over t+1: shift by one day.
    target_w = valid.map(lambda r: weights.get(int(r), 0.0)).astype(float)
    held_w = target_w.shift(1).fillna(0.0)

    turnover = held_w.diff().abs().fillna(held_w.abs())
    cost = turnover * (cost_bps / 1e4)

    strat_ret = held_w * simple_ret - cost
    buyhold_ret = simple_ret

    out = pd.DataFrame(
        {
            "regime": valid.astype(int),
            "weight": held_w,
            "asset_return": simple_ret,
            "strategy_return": strat_ret,
            "buyhold_return": buyhold_ret,
        }
    )
    out["strategy_equity"] = (1.0 + out["strategy_return"]).cumprod()
    out["buyhold_equity"] = (1.0 + out["buyhold_return"]).cumprod()
    return out


def performance_stats(returns: pd.Series) -> dict:
    """Standard risk/return summary for a daily simple-return series."""
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    n_years = len(returns) / TRADING_DAYS
    total_return = equity.iloc[-1] - 1.0
    cagr = equity.iloc[-1] ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)
              if returns.std() > 0 else np.nan)
    drawdown = equity / equity.cummax() - 1.0
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
    }


def compare(bt: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side stats table for strategy vs. buy-and-hold."""
    rows = {
        "Regime strategy": performance_stats(bt["strategy_return"]),
        "Buy & hold": performance_stats(bt["buyhold_return"]),
    }
    return pd.DataFrame(rows).T
