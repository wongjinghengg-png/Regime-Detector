"""Tests run fully offline against the synthetic data fallback."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime_detector import data, evaluate, features  # noqa: E402
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
