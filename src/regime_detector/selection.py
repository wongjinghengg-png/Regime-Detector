"""Automatic regime-count selection via information criteria.

The rest of the project takes the number of regimes as a given (``--regimes 3``).
That is a hidden modelling choice: too few regimes blur distinct states together,
too many split noise into spurious ones. Information criteria (BIC / AIC) score
each count by trading goodness-of-fit against complexity.

A subtlety worth knowing: on financial data, **raw BIC/AIC minimization tends to
over-select HMM states** — the Gaussian-emission assumption is imperfect, so each
extra state keeps improving the likelihood faster than the penalty discourages,
and the criterion runs to the largest candidate (or wanders). So the default here
is the **elbow**: the count past which adding a regime stops meaningfully
improving the fit. That is more robust and interpretable than the naive minimum,
and picking the elbow over the raw optimum is itself the point — it shows the
criterion is a guide, not an oracle. Raw ``bic``/``aic`` minimization remain
available for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _n_free_params(n_states: int, n_features: int) -> int:
    """Free parameters of a full-covariance Gaussian HMM (fallback for BIC/AIC)."""
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    covars = n_states * n_features * (n_features + 1) // 2
    return start + trans + means + covars


def score_n_regimes(
    X: np.ndarray,
    candidates=range(2, 7),
    n_iter: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Fit an HMM for each candidate count and return a scoring table.

    Columns: ``n_regimes``, ``log_likelihood``, ``n_params``, ``bic``, ``aic``.
    Rows where fitting fails are skipped.
    """
    from hmmlearn.hmm import GaussianHMM

    n_features = X.shape[1]
    rows = []
    for n in candidates:
        model = GaussianHMM(
            n_components=n,
            covariance_type="full",
            n_iter=n_iter,
            random_state=seed,
        )
        try:
            model.fit(X)
            ll = model.score(X)
        except Exception:
            continue

        # Prefer hmmlearn's built-ins; fall back to the manual formula.
        try:
            bic = model.bic(X)
            aic = model.aic(X)
        except AttributeError:
            k = _n_free_params(n, n_features)
            bic = -2 * ll + k * np.log(len(X))
            aic = -2 * ll + 2 * k

        rows.append(
            {
                "n_regimes": n,
                "log_likelihood": ll,
                "n_params": _n_free_params(n, n_features),
                "bic": bic,
                "aic": aic,
            }
        )
    if not rows:
        raise RuntimeError("No candidate HMM could be fit.")
    table = pd.DataFrame(rows).set_index("n_regimes")
    # Marginal BIC improvement from adding the n-th regime (higher = more useful).
    table["bic_improvement"] = -table["bic"].diff()
    return table


def _elbow(table: pd.DataFrame) -> int:
    """Pick the count where BIC improvement decelerates most (the elbow).

    ``deceleration[n] = improvement[n] - improvement[n+1]``. The elbow is the
    ``n`` after which the marginal gain drops the hardest — i.e. the last count
    that still buys a big improvement. Falls back to the raw BIC minimum if the
    improvement curve is too short to have an interior elbow.
    """
    imp = table["bic_improvement"]
    ns = list(table.index)
    if len(ns) < 3:
        return int(table["bic"].idxmin())
    best_n, best_decel = None, -np.inf
    for i in range(1, len(ns) - 1):  # need imp[n] and imp[n+1]
        n = ns[i]
        decel = imp.iloc[i] - imp.iloc[i + 1]
        if decel > best_decel:
            best_decel, best_n = decel, n
    return int(best_n) if best_n is not None else int(table["bic"].idxmin())


def select_n_regimes(
    X: np.ndarray,
    candidates=range(2, 7),
    criterion: str = "elbow",
    n_iter: int = 200,
    seed: int = 42,
) -> tuple[int, pd.DataFrame]:
    """Return ``(best_n, scoring_table)`` for the chosen selection rule.

    criterion:
      - ``"elbow"`` (default): BIC diminishing-returns elbow — robust to the
        over-selection that plagues raw minimization on financial data.
      - ``"bic"`` / ``"aic"``: the raw information-criterion minimum.
    """
    if criterion not in ("elbow", "bic", "aic"):
        raise ValueError("criterion must be 'elbow', 'bic', or 'aic'")
    table = score_n_regimes(X, candidates, n_iter=n_iter, seed=seed)
    if criterion == "elbow":
        best_n = _elbow(table)
    else:
        best_n = int(table[criterion].idxmin())
    return best_n, table
