"""Tests run fully offline against the synthetic data fallback."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime_detector import (  # noqa: E402
    data,
    evaluate,
    features,
    selection,
    strategy,
    walkforward,
)
from regime_detector.models import baseline, clustering, hmm  # noqa: E402


@pytest.fixture(scope="module")
def prepared():
    prices = data.synthetic_prices("2010-01-01", "2020-01-01")
    feats = features.build_features(prices)
    X = features.model_matrix(feats)
    return prices, feats, X


def test_synthetic_prices_are_deterministic():
    a = data.synthetic_prices("2010-01-01", "2012-01-01")
    b = data.synthetic_prices("2010-01-01", "2012-01-01")
    assert a["Close"].equals(b["Close"])
    assert (a["Close"] > 0).all()


def test_features_are_causal_and_clean(prepared):
    _, feats, _ = prepared
    assert not feats.isna().any().any()
    assert (feats["drawdown"] <= 1e-9).all()  # drawdown is never positive
    assert (feats["realized_vol"] >= 0).all()


def test_labels_are_volatility_ordered(prepared):
    _, feats, X = prepared
    labels = hmm.fit_predict(feats, X, n_regimes=3)
    summary = evaluate.regime_summary(feats, labels)
    vols = summary["ann_vol"].to_numpy()
    assert np.all(np.diff(vols) >= -1e-9)  # regime 0 calmest -> monotonic vol


@pytest.mark.parametrize("n", [2, 3, 4])
def test_all_models_produce_valid_labels(prepared, n):
    _, feats, X = prepared
    models = {
        "baseline": baseline.fit_predict(feats, n_regimes=n),
        "kmeans": clustering.kmeans_fit_predict(feats, X, n_regimes=n),
        "gmm": clustering.gmm_fit_predict(feats, X, n_regimes=n),
        "hmm": hmm.fit_predict(feats, X, n_regimes=n),
    }
    for name, labels in models.items():
        assert len(labels) == len(feats), name
        assert set(np.unique(labels)).issubset(set(range(n))), name


def test_hmm_is_stickier_than_baseline(prepared):
    """The HMM models transitions, so it should switch regimes less often."""
    _, feats, X = prepared
    hmm_labels = hmm.fit_predict(feats, X, n_regimes=3)
    base_labels = baseline.fit_predict(feats, n_regimes=3)
    assert evaluate.n_switches(hmm_labels) < evaluate.n_switches(base_labels)


def test_crisis_regime_has_negative_returns(prepared):
    """The highest-vol regime should look like a drawdown, not a calm uptrend."""
    _, feats, X = prepared
    labels = hmm.fit_predict(feats, X, n_regimes=3)
    summary = evaluate.regime_summary(feats, labels)
    crisis = summary.index.max()
    calm = summary.index.min()
    assert summary.loc[crisis, "worst_drawdown"] < summary.loc[calm, "worst_drawdown"]


@pytest.fixture(scope="module")
def walk_forward_result(prepared):
    _, feats, _ = prepared
    # Small/fast settings keep the test quick while exercising the real path.
    online = walkforward.walk_forward_hmm(
        feats, min_train=252, refit_every=126, n_iter=50
    )
    return feats, online


def test_walk_forward_has_no_lookahead_warmup(walk_forward_result):
    """No online call should exist before min_train days have elapsed."""
    feats, online = walk_forward_result
    assert len(online) == len(feats)
    assert (online.iloc[:252] == -1).all()      # warm-up: no call yet
    assert (online.iloc[252:] >= 0).all()       # every later day has a call


def test_walk_forward_labels_are_valid(walk_forward_result):
    _, online = walk_forward_result
    called = online[online >= 0]
    assert set(called.unique()).issubset({0, 1, 2})


def test_online_is_less_accurate_than_hindsight_is_to_itself(walk_forward_result):
    """Out-of-sample agreement is imperfect — that's the whole point (lag/noise)."""
    feats, online = walk_forward_result
    X = features.model_matrix(feats)
    hindsight = hmm.fit_predict(feats, X, n_regimes=3)
    agree = walkforward.agreement_rate(online, hindsight)
    assert 0.3 < agree < 1.0  # correlated with hindsight, but not identical


def test_detection_lag_is_nonnegative(walk_forward_result):
    feats, online = walk_forward_result
    X = features.model_matrix(feats)
    hindsight = hmm.fit_predict(feats, X, n_regimes=3)
    lag = walkforward.detection_lag(online, hindsight, target_regime=2)
    assert lag["n_events"] >= lag["n_detected"] >= 0
    if lag["n_detected"] > 0:
        assert lag["mean_lag_days"] >= 0


def test_backtest_uses_lagged_signal_no_lookahead(walk_forward_result):
    """Position on day t must come from the regime known on day t-1."""
    feats, online = walk_forward_result
    bt = strategy.backtest(feats, online, cost_bps=0.0)
    valid = online[online >= 0]
    expected_target = valid.map(strategy.DEFAULT_WEIGHTS).astype(float)
    # Held weight is the target shifted forward one day (first day starts flat).
    assert bt["weight"].iloc[0] == 0.0
    pd.testing.assert_series_equal(
        bt["weight"].iloc[1:],
        expected_target.shift(1).iloc[1:],
        check_names=False,
    )


def test_backtest_reduces_drawdown(walk_forward_result):
    """De-risking in crisis should cut max drawdown vs. buy-and-hold."""
    feats, online = walk_forward_result
    bt = strategy.backtest(feats, online)
    stats = strategy.compare(bt)
    # Max drawdown is negative; strategy's should be shallower (closer to 0).
    assert stats.loc["Regime strategy", "max_drawdown"] >= stats.loc[
        "Buy & hold", "max_drawdown"
    ]


def test_transaction_costs_reduce_return(walk_forward_result):
    feats, online = walk_forward_result
    free = strategy.backtest(feats, online, cost_bps=0.0)
    costly = strategy.backtest(feats, online, cost_bps=10.0)
    assert (
        costly["strategy_equity"].iloc[-1] < free["strategy_equity"].iloc[-1]
    )


def test_zero_exposure_regime_earns_nothing(walk_forward_result):
    """With all weights zero the strategy return is exactly zero (minus costs)."""
    feats, online = walk_forward_result
    bt = strategy.backtest(feats, online, weights={0: 0.0, 1: 0.0, 2: 0.0}, cost_bps=0.0)
    assert np.allclose(bt["strategy_return"], 0.0)


@pytest.fixture(scope="module")
def selection_table(prepared):
    _, _, X = prepared
    return selection.score_n_regimes(X, candidates=range(2, 6), n_iter=50)


def test_selection_table_shape(selection_table):
    assert list(selection_table.index) == [2, 3, 4, 5]
    for col in ["log_likelihood", "n_params", "bic", "aic", "bic_improvement"]:
        assert col in selection_table.columns
    # More states => more free parameters, always.
    assert selection_table["n_params"].is_monotonic_increasing


def test_raw_bic_selection_returns_candidate(prepared):
    _, _, X = prepared
    best_n, table = selection.select_n_regimes(X, candidates=range(2, 6), criterion="bic")
    assert best_n in table.index
    assert best_n == int(table["bic"].idxmin())


def test_elbow_is_more_parsimonious_than_raw_bic(prepared):
    """The elbow exists to avoid the raw criterion's over-selection of states.

    On financial-style data BIC keeps falling, so its raw minimum sits at the
    largest candidate; the elbow should pick a smaller, interior count.
    """
    _, _, X = prepared
    cands = range(2, 7)
    elbow_n, table = selection.select_n_regimes(X, candidates=cands, criterion="elbow")
    raw_bic_n = int(table["bic"].idxmin())
    assert elbow_n in table.index
    assert elbow_n <= raw_bic_n
    assert elbow_n != max(cands)  # never just runs to the largest candidate


def test_free_param_count_formula():
    # 3-state, 3-feature full-covariance HMM:
    # start 2 + trans 6 + means 9 + covars 3*6 = 35
    assert selection._n_free_params(3, 3) == 35
