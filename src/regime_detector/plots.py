"""Plotting utilities — the money shot of the project.

The headline chart is the price line shaded by detected regime. It makes the
model's output legible at a glance and is the single most useful thing to drop
into a README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Colorblind-friendly ordering: calm -> stressed.
REGIME_COLORS = ["#2ca25f", "#fdae61", "#d7191c", "#6a51a3", "#2c7fb8"]
REGIME_NAMES = ["Calm", "Choppy", "Crisis", "Regime 3", "Regime 4"]


def _labels_to_spans(index: pd.DatetimeIndex, labels: np.ndarray):
    """Yield (start, end, regime) contiguous spans for axvspan shading."""
    labels = np.asarray(labels)
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            yield index[start], index[i], labels[start]
            start = i
    yield index[start], index[-1], labels[start]


def plot_regimes(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    labels: np.ndarray,
    title: str = "Detected market regimes",
    ax=None,
):
    """Plot the close price shaded by regime. Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 5))

    idx = features.index
    close = prices.loc[idx, "Close"]
    ax.plot(idx, close, color="black", linewidth=1.0, zorder=3)

    n_reg = int(np.max(labels)) + 1
    for start, end, r in _labels_to_spans(idx, labels):
        ax.axvspan(start, end, color=REGIME_COLORS[r % len(REGIME_COLORS)], alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS[r % len(REGIME_COLORS)], alpha=0.4)
        for r in range(n_reg)
    ]
    ax.legend(handles, REGIME_NAMES[:n_reg], loc="upper left", framealpha=0.9)
    ax.set_title(title)
    ax.set_ylabel("Close price")
    ax.margins(x=0)
    return ax


def plot_online_vs_hindsight(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    online: pd.Series,
    hindsight: np.ndarray,
    savepath: str | None = None,
):
    """Two stacked panels: real-time (online) vs. hindsight regime calls.

    The visual point is the lag — where the online panel's colored blocks start
    *after* the hindsight panel's, that's the detection delay a live system pays.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True)

    # Online panel only shades days that actually have a call (skip warm-up).
    on = online.to_numpy()
    warm = on >= 0
    plot_regimes(
        prices,
        features.loc[online.index[warm]],
        on[warm],
        title="Online (walk-forward, out-of-sample)",
        ax=axes[0],
    )
    plot_regimes(
        prices, features, hindsight, title="Hindsight (full-sample)", ax=axes[1]
    )
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
    return fig


def plot_backtest(bt, title: str = "Regime strategy vs. buy & hold", savepath=None):
    """Equity curves (top) and strategy exposure over time (bottom).

    Uses only the out-of-sample window. The exposure panel makes the mechanism
    obvious: the strategy cuts risk when the online detector flags stress.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(bt.index, bt["buyhold_equity"], color="#888888", lw=1.3, label="Buy & hold")
    ax1.plot(bt.index, bt["strategy_equity"], color="#2c7fb8", lw=1.5, label="Regime strategy")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.margins(x=0)

    ax2.fill_between(bt.index, 0, bt["weight"], color="#2c7fb8", alpha=0.35, step="pre")
    ax2.set_ylabel("Exposure")
    ax2.set_ylim(-0.05, 1.05)
    ax2.margins(x=0)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
    return fig


def plot_model_comparison(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    labels_by_model: dict[str, np.ndarray],
    savepath: str | None = None,
):
    """Stacked panels, one per model, sharing the x-axis."""
    import matplotlib.pyplot as plt

    n = len(labels_by_model)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (name, labels) in zip(axes, labels_by_model.items()):
        plot_regimes(prices, features, labels, title=name, ax=ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
    return fig
