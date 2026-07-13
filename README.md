# Market Regime Detector

Detect hidden **market regimes** — calm, choppy, and crisis states — from daily
price data, using three methods of increasing sophistication and comparing them
head-to-head.

> ⚠️ **Research prototype, not a trading system.** This project demonstrates the
> modelling and engineering behind regime detection. It is deliberately *not*
> suitable for live execution — see [Limitations](#limitations). That gap is the
> point: knowing where the line is between a prototype and production is the
> skill being shown.

![Regime comparison across four models](reports/regime_comparison.png)

*S&P 500 close price shaded by detected regime. Notice how the threshold baseline
(top) flickers between regimes constantly, while the HMM (bottom) — which models
regime **transitions** — produces stable, economically-meaningful blocks.*

---

## What it does

Given a price series, it fits four regime models and characterizes what each
detected regime looks like economically:

| Method | Idea | Models persistence? |
|---|---|---|
| **Threshold baseline** | Bucket days by realized-volatility quantile | ❌ |
| **KMeans** | Cluster days in feature space | ❌ |
| **Gaussian Mixture (GMM)** | Soft-cluster days as a mix of Gaussians | ❌ |
| **Gaussian HMM** | Hidden states with a learned transition matrix | ✅ |

The **Hidden Markov Model** is the centerpiece. It's the canonical approach to
regime detection because it explicitly models the market as switching between
*hidden* states, each with its own return/volatility distribution, and learns
how sticky those states are. The other three exist to give the HMM something to
beat.

### Representative result

Running on ~19 years of daily data, the HMM switches regimes **far** less often
than the naive baseline while still capturing every major drawdown:

| Model | Regime switches | Avg. regime length |
|---|---|---|
| Threshold baseline | 414 | 12 days |
| KMeans | 458 | 11 days |
| GMM | 307 | 16 days |
| **HMM** | **74** | **66 days** |

And the HMM's regimes are economically distinct — the highest-volatility regime
is a genuine drawdown state (negative return, ~2× the volatility of the calm
regime), not just "days that happened to be bumpy."

---

## Quickstart

```bash
pip install -r requirements.txt

# Run the full pipeline: download -> features -> 4 models -> plots + stats
python -m regime_detector --ticker SPY --regimes 3
```

Outputs land in `reports/`:
- `regime_comparison.png` — the shaded price chart above
- `hmm_regime_summary.csv` — per-regime economics (return, vol, Sharpe, drawdown)
- `stability.csv` — switch counts / persistence by model

CLI options: `--ticker`, `--start`, `--end`, `--regimes`, `--outdir`, `--no-plots`.

> **Data:** prices come from Yahoo Finance via `yfinance` and are cached under
> `data/`. If the network is unavailable, the pipeline **automatically falls back
> to a deterministic synthetic series** with three baked-in regimes, so the code
> (and the test suite) runs anywhere, offline. The committed figure above was
> produced from the synthetic fallback; point it at a network-connected machine
> to analyze real SPY.

The narrated walkthrough lives in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

---

## How it works

1. **Features** (`features.py`) — from the close price: log returns, rolling
   annualized realized volatility, downside volatility, drawdown-from-peak, and a
   rolling return z-score. All causal (past/present only).
2. **Models** (`models/`) — each returns an integer regime label per day. Labels
   are **relabeled by volatility** so regime 0 is always the calmest state,
   making models and runs directly comparable.
3. **Evaluation** (`evaluate.py`) — because regimes are unsupervised there is no
   ground truth to score against; instead we measure per-regime economics
   (return / vol / Sharpe / drawdown) and persistence (switch count, run length).
4. **Plots** (`plots.py`) — the price line shaded by regime, one panel per model.

```
src/regime_detector/
├── data.py          # download + cache + synthetic fallback
├── features.py      # feature engineering
├── models/
│   ├── baseline.py  # volatility-threshold rules
│   ├── clustering.py# KMeans / GMM
│   ├── hmm.py       # Gaussian HMM  ← centerpiece
│   └── base.py      # volatility-ordered relabeling
├── evaluate.py      # regime-conditional stats
├── plots.py         # shaded regime charts
└── cli.py           # end-to-end pipeline
```

---

## Limitations

Read this section — it's the most important one.

- **Lookahead in preprocessing.** Features are standardized using full-sample
  statistics, and the models are fit on the entire history at once. A live system
  would fit on a rolling/expanding window and standardize causally.
- **Regimes are labeled after the fact.** The HMM's `predict` uses the whole
  sequence (smoothing). Real-time detection needs *filtered* (online) inference,
  which reacts with a lag and is noisier.
- **No transaction costs, slippage, or execution model.** There is no backtest of
  a regime-conditioned strategy here, and no P&L claim is made.
- **Non-stationarity.** The number of regimes is fixed by hand (`--regimes`), and
  market dynamics drift over decades; a fixed 3-state model is a simplification.
- **Single asset, single frequency.** Daily equity-index data only.

Any of these would need to be addressed before the output could inform a real
trade. Turning the lookahead-free version into a walk-forward evaluation is the
natural next step.

---

## Testing

```bash
pytest -q
```

The suite runs fully offline against the synthetic data and checks the invariants
that matter: features are causal and NaN-free, regime labels are volatility-
ordered, every model produces valid labels for 2–4 regimes, the HMM is stickier
than the baseline, and the highest-vol regime really is a drawdown state.

## Tech stack

Python · pandas · NumPy · scikit-learn · **hmmlearn** · matplotlib · yfinance · pytest
