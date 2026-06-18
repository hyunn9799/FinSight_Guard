"""Tests for market regime classification and per-regime performance summaries."""
import pytest
import numpy as np
import pandas as pd
from tests.fixtures.optimization_data import synthetic_prices
from src.backtest.regime import classify_regime_periods, compute_regime_performance
from src.backtest.robust import CostAssumptions


def _make_bull_prices(n: int = 200) -> pd.Series:
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(100.0 + np.arange(n) * 0.5, index=idx, name="Close")


def _make_bear_prices(n: int = 200) -> pd.Series:
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(200.0 - np.arange(n) * 0.5, index=idx, name="Close")


def _make_sideways_prices(n: int = 200, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(100.0 + rng.normal(0, 0.5, n).cumsum(), index=idx, name="Close")


def test_classify_rising_series_as_bull():
    labels = classify_regime_periods(_make_bull_prices(), lookback_days=30)
    bull_fraction = (labels == "bull").mean()
    assert bull_fraction > 0.5


def test_classify_falling_series_as_bear():
    labels = classify_regime_periods(_make_bear_prices(), lookback_days=30)
    bear_fraction = (labels == "bear").mean()
    assert bear_fraction > 0.5


def test_classify_flat_series_as_sideways():
    labels = classify_regime_periods(_make_sideways_prices(), lookback_days=30)
    sideways_fraction = (labels == "sideways").mean()
    assert sideways_fraction > 0.3


def test_regime_labels_are_valid_strings():
    prices = synthetic_prices(n=300)["Close"]
    labels = classify_regime_periods(prices, lookback_days=30)
    valid = {"bull", "bear", "sideways", "high_volatility", "low_volatility"}
    assert set(labels.dropna().unique()).issubset(valid)


def test_regime_performance_low_confidence_below_10_trades():
    from tests.fixtures.optimization_data import synthetic_trades_few
    trades_df = synthetic_trades_few(n_sells=3)
    prices = synthetic_prices(n=300)["Close"]
    labels = classify_regime_periods(prices, lookback_days=30)
    summaries = compute_regime_performance(trades_df, labels, initial_balance=10_000,
                                           cost=CostAssumptions())
    if summaries:
        for s in summaries:
            if s["completed_trades"] < 10:
                assert s["confidence"] == "low"


def test_regime_performance_low_confidence_below_60_trading_days():
    prices = pd.Series(
        100.0 + np.arange(50) * 0.2,
        index=pd.bdate_range("2022-01-03", periods=50),
        name="Close"
    )
    labels = classify_regime_periods(prices, lookback_days=20)
    from tests.fixtures.optimization_data import synthetic_trades
    trades_df = synthetic_trades(n_sells=5, seed=1)
    summaries = compute_regime_performance(trades_df, labels, initial_balance=10_000,
                                           cost=CostAssumptions())
    for s in summaries:
        if s["trading_days"] < 60:
            assert s["confidence"] == "low"
