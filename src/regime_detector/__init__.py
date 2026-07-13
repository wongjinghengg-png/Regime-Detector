"""Market regime detector — a research prototype.

Detects hidden market regimes (calm / choppy / crisis) from price data using
three methods of increasing sophistication: a volatility-threshold baseline,
clustering (KMeans / GMM), and a Gaussian Hidden Markov Model.

Not intended for live trading — see the "Limitations" section of the README.
"""
from __future__ import annotations

__version__ = "0.1.0"
