"""Data loading for the regime detector.

Downloads daily OHLCV data from Yahoo Finance and caches it locally so runs
are reproducible and offline-friendly. When no network is available (or the
download fails) a deterministic synthetic series is generated instead, so the
rest of the pipeline — and the test suite — always has something to work with.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(os.environ.get("REGIME_DATA_DIR", "data"))


def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("^", "").replace("/", "-")
    return CACHE_DIR / f"{safe}_{start}_{end}.csv"


def load_prices(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str = "2024-01-01",
    use_cache: bool = True,
    allow_synthetic: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame indexed by date with at least a ``Close`` column.

    Resolution order: local cache -> Yahoo Finance -> synthetic fallback.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, start, end)

    if use_cache and path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    df = _download(ticker, start, end)
    if df is not None and not df.empty:
        df.to_csv(path)
        return df

    if not allow_synthetic:
        raise RuntimeError(
            f"Could not download {ticker} and synthetic fallback is disabled."
        )
    df = synthetic_prices(start, end)
    df.to_csv(path)
    return df


def _download(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        raw = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=True
        )
        if raw is None or raw.empty:
            return None
        # yfinance may return a MultiIndex column frame for a single ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index.name = "Date"
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw]
        return raw[keep]
    except Exception:
        return None


def synthetic_prices(start: str, end: str, seed: int = 7) -> pd.DataFrame:
    """Generate a deterministic price series with three baked-in regimes.

    The series switches between a calm uptrend, a choppy sideways stretch, and
    a high-volatility drawdown so that any regime model has real structure to
    recover. Used as an offline fallback and in tests.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Per-regime (daily drift, daily vol).
    regimes = {
        0: (0.0006, 0.007),   # calm bull
        1: (0.0000, 0.012),   # choppy / sideways
        2: (-0.0012, 0.028),  # crisis / high-vol drawdown
    }
    # Simple 3-state Markov chain for the hidden regime path.
    trans = np.array(
        [
            [0.985, 0.013, 0.002],
            [0.020, 0.965, 0.015],
            [0.030, 0.040, 0.930],
        ]
    )
    state = 0
    rets = np.empty(n)
    for i in range(n):
        mu, sigma = regimes[state]
        rets[i] = rng.normal(mu, sigma)
        state = rng.choice(3, p=trans[state])

    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"Close": close}, index=dates)
    df.index.name = "Date"
    # Rough OHLC/Volume so downstream code that expects them still works.
    df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0])
    df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + rng.uniform(0, 0.004, n))
    df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - rng.uniform(0, 0.004, n))
    df["Volume"] = rng.integers(5_000_000, 50_000_000, n)
    return df[["Open", "High", "Low", "Close", "Volume"]]
