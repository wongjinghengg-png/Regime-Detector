"""Walk-forward (out-of-sample) regime detection.

The rest of the project fits every model on the *entire* history at once and
reads regimes off with hindsight. That is fine for characterizing regimes but
is not how a real system works: on any given day you only have the past.

This module fixes that for the HMM. It refits the model on an expanding window
and, at each step, assigns the current day a regime using **only data available
up to that day** (filtered / online inference — no future leakage). Comparing
the online call against the full-sample hindsight call reveals the detection
*lag*: how many days it takes to recognize a regime shift in real time.

This is the single most important step from "research prototype" toward
something defensible, so it is kept deliberately explicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import model_matrix
from .models.base import order_labels_by_volatility


def walk_forward_hmm(
    features: pd.DataFrame,
    n_regimes: int = 3,
    min_train: int = 504,
    refit_every: int = 63,
    n_iter: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> pd.Series:
    """Return online regime labels indexed by date.

    Parameters
    ----------
    min_train   : days of history required before the first call (~2y default).
    refit_every : refit the HMM every N days; between refits the current model
                  filters new observations. Larger = faster, staler.

    At each evaluated day ``t`` we fit/reuse an HMM on ``features[:t+1]`` and take
    the regime of the *last* day. Because that day has no future inside its
    window, its smoothed estimate equals the filtered (online) estimate — so the
    call is genuinely out-of-sample. Labels are volatility-ordered **using only
    in-window data**, keeping regime ids consistent across refits without peeking
    ahead.
    """
    from hmmlearn.hmm import GaussianHMM

    n = len(features)
    if min_train >= n:
        raise ValueError(f"min_train={min_train} >= available history ({n}).")

    online = np.full(n, -1, dtype=int)
    model = None

    for t in range(min_train, n):
        window = features.iloc[: t + 1]
        Xw = model_matrix(window)  # standardized on in-window data only

        if (t - min_train) % refit_every == 0 or model is None:
            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=n_iter,
                random_state=seed,
            )
            try:
                model.fit(Xw)
            except Exception:
                # Degenerate fit (rare, early windows): carry the last regime.
                online[t] = online[t - 1] if t > min_train else 0
                if verbose:
                    print(f"[t={t}] fit failed, carrying forward")
                continue

        raw = model.predict(Xw)
        ordered = order_labels_by_volatility(raw, window["realized_vol"])
        online[t] = int(ordered[-1])

    return pd.Series(online, index=features.index, name="online_regime")


def detection_lag(
    online: pd.Series, hindsight: np.ndarray, target_regime: int
) -> dict:
    """Measure how late the online detector is, for one regime.

    For every contiguous stretch of ``target_regime`` in the hindsight labels,
    find the first day the online detector also called that regime, and report
    the lag in days. Stretches the online model never catches count as misses.
    """
    hindsight = np.asarray(hindsight)
    on = online.to_numpy()
    valid = on >= 0  # skip the warm-up period with no online call

    lags = []
    misses = 0
    in_event = False
    start = 0
    for i in range(len(hindsight)):
        is_target = hindsight[i] == target_regime
        if is_target and not in_event:
            in_event, start = True, i
        elif not is_target and in_event:
            in_event = False
            lag = _first_agreement_lag(on, valid, start, i, target_regime)
            (lags.append(lag) if lag is not None else None)
            misses += lag is None
    if in_event:
        lag = _first_agreement_lag(on, valid, start, len(hindsight), target_regime)
        (lags.append(lag) if lag is not None else None)
        misses += lag is None

    return {
        "target_regime": target_regime,
        "n_events": len(lags) + misses,
        "n_detected": len(lags),
        "n_missed": misses,
        "mean_lag_days": float(np.mean(lags)) if lags else float("nan"),
        "median_lag_days": float(np.median(lags)) if lags else float("nan"),
    }


def _first_agreement_lag(on, valid, start, end, target_regime):
    for j in range(start, end):
        if valid[j] and on[j] == target_regime:
            return j - start
    return None


def agreement_rate(online: pd.Series, hindsight: np.ndarray) -> float:
    """Fraction of evaluated days where the online call matches hindsight."""
    on = online.to_numpy()
    hindsight = np.asarray(hindsight)
    mask = on >= 0
    if mask.sum() == 0:
        return float("nan")
    return float((on[mask] == hindsight[mask]).mean())
