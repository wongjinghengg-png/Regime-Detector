"""Command-line entry point.

    python -m regime_detector --ticker SPY --regimes 3

Runs the full pipeline (download -> features -> all models -> evaluation ->
plots) and writes a comparison figure plus a per-regime stats table to reports/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import data, evaluate, features, selection, strategy, walkforward
from .models import baseline, clustering, hmm


def run(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str = "2024-01-01",
    n_regimes: int = 3,
    outdir: str = "reports",
    make_plots: bool = True,
    walk_forward: bool = False,
    backtest: bool = False,
    auto_regimes: bool = False,
) -> dict:
    prices = data.load_prices(ticker, start, end)
    feats = features.build_features(prices)
    X = features.model_matrix(feats)

    selection_table = None
    if auto_regimes:
        best_n, selection_table = selection.select_n_regimes(X, criterion="elbow")
        raw_bic_min = int(selection_table["bic"].idxmin())
        print("=== Regime-count selection ===")
        print(selection_table.round(1).to_string())
        print(
            f"\nElbow selects {best_n} regimes (overriding --regimes). "
            f"Raw BIC minimum is {raw_bic_min} — information criteria tend to "
            f"over-select HMM states on financial data, so the elbow is used."
        )
        n_regimes = best_n

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

    result = {"prices": prices, "features": feats, "labels": labels}

    if selection_table is not None:
        selection_table.to_csv(Path(outdir) / "regime_selection.csv")
        result["selection"] = selection_table

    if walk_forward or backtest:
        print("\n=== Walk-forward (out-of-sample) HMM ===")
        online = walkforward.walk_forward_hmm(feats, n_regimes=n_regimes)
        hindsight = labels["HMM"]
        agree = walkforward.agreement_rate(online, hindsight)
        crisis = int(max(labels["HMM"]))
        lag = walkforward.detection_lag(online, hindsight, crisis)
        print(f"Online vs hindsight agreement: {agree:.1%}")
        print(
            f"Crisis regime (id {crisis}): detected "
            f"{lag['n_detected']}/{lag['n_events']} events, "
            f"mean lag {lag['mean_lag_days']:.1f} days"
        )
        result["online"] = online

    if backtest:
        print("\n=== Regime-conditioned backtest (out-of-sample signal) ===")
        bt = strategy.backtest(feats, result["online"])
        comparison = strategy.compare(bt)
        print(comparison.round(3).to_string())
        comparison.to_csv(Path(outdir).joinpath("backtest_stats.csv"))
        result["backtest"] = bt

    if make_plots:
        try:
            from . import plots

            fig_path = Path(outdir) / "regime_comparison.png"
            plots.plot_model_comparison(prices, feats, labels, savepath=str(fig_path))
            print(f"\nSaved comparison figure -> {fig_path}")

            if selection_table is not None:
                sel_path = Path(outdir) / "regime_selection.png"
                plots.plot_selection(selection_table, n_regimes, savepath=str(sel_path))
                print(f"Saved selection figure -> {sel_path}")

            if walk_forward or backtest:
                wf_path = Path(outdir) / "walk_forward.png"
                plots.plot_online_vs_hindsight(
                    prices, feats, result["online"], labels["HMM"], savepath=str(wf_path)
                )
                print(f"Saved walk-forward figure -> {wf_path}")

            if backtest:
                bt_path = Path(outdir) / "backtest.png"
                plots.plot_backtest(result["backtest"], savepath=str(bt_path))
                print(f"Saved backtest figure -> {bt_path}")
        except Exception as exc:  # plotting is optional / headless-safe
            print(f"\n[plot skipped: {exc}]")

    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Market regime detector (research prototype)")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--regimes", type=int, default=3)
    p.add_argument("--outdir", default="reports")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--walk-forward",
        action="store_true",
        help="also run out-of-sample walk-forward HMM (slower, ~40s)",
    )
    p.add_argument(
        "--backtest",
        action="store_true",
        help="run the regime-conditioned strategy backtest (implies --walk-forward)",
    )
    p.add_argument(
        "--auto-regimes",
        action="store_true",
        help="select the number of regimes by BIC (overrides --regimes)",
    )
    args = p.parse_args(argv)
    run(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        n_regimes=args.regimes,
        outdir=args.outdir,
        make_plots=not args.no_plots,
        walk_forward=args.walk_forward,
        backtest=args.backtest,
        auto_regimes=args.auto_regimes,
    )


if __name__ == "__main__":
    main()
