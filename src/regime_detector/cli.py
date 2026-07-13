"""Command-line entry point.

    python -m regime_detector --ticker SPY --regimes 3

Runs the full pipeline (download -> features -> all models -> evaluation ->
plots) and writes a comparison figure plus a per-regime stats table to reports/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import data, evaluate, features
from .models import baseline, clustering, hmm


def run(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str = "2024-01-01",
    n_regimes: int = 3,
    outdir: str = "reports",
    make_plots: bool = True,
) -> dict:
    prices = data.load_prices(ticker, start, end)
    feats = features.build_features(prices)
    X = features.model_matrix(feats)

    labels = {
        "Threshold baseline": baseline.fit_predict(feats, n_regimes=n_regimes),
        "KMeans": clustering.kmeans_fit_predict(feats, X, n_regimes=n_regimes),
        "GMM": clustering.gmm_fit_predict(feats, X, n_regimes=n_regimes),
        "HMM": hmm.fit_predict(feats, X, n_regimes=n_regimes),
    }

    Path(outdir).mkdir(parents=True, exist_ok=True)
    stability = evaluate.stability_report(labels)
    stability.to_csv(Path(outdir) / "stability.csv")

    print(f"\n=== Regime detection on {ticker} ({start} -> {end}) ===")
    print(f"Observations: {len(feats)}\n")
    print("Persistence by model:")
    print(stability.round(2).to_string())

    print("\nHMM per-regime economics:")
    summary = evaluate.regime_summary(feats, labels["HMM"])
    print(summary.round(3).to_string())
    summary.to_csv(Path(outdir) / "hmm_regime_summary.csv")

    if make_plots:
        try:
            from . import plots

            fig_path = Path(outdir) / "regime_comparison.png"
            plots.plot_model_comparison(prices, feats, labels, savepath=str(fig_path))
            print(f"\nSaved comparison figure -> {fig_path}")
        except Exception as exc:  # plotting is optional / headless-safe
            print(f"\n[plot skipped: {exc}]")

    return {"prices": prices, "features": feats, "labels": labels}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Market regime detector (research prototype)")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--regimes", type=int, default=3)
    p.add_argument("--outdir", default="reports")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    run(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        n_regimes=args.regimes,
        outdir=args.outdir,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
