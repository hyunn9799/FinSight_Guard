"""Tests for the ported backtest strategy (no live API calls)."""

import numpy as np
import pandas as pd

from src.backtest.strategy import BacktestParams, BacktestResult, run_backtest


def _synthetic_prices(n: int = 120) -> pd.DataFrame:
    # Deterministic oscillating series so kernel regression / divergences have
    # something to act on without relying on randomness.
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    t = np.arange(n)
    close = 100 + 10 * np.sin(t / 6.0) + t * 0.1
    frame = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 1_000),
        },
        index=idx,
    )
    return frame


def test_run_backtest_returns_result_with_expected_shape() -> None:
    result = run_backtest(_synthetic_prices(), BacktestParams(), initial_balance=10_000)

    assert isinstance(result, BacktestResult)
    assert isinstance(result.profit_pct, float)
    assert isinstance(result.trades, pd.DataFrame)
    assert "signal" in result.enriched.columns
    assert "RSI" in result.enriched.columns


def test_run_backtest_accepts_dict_params() -> None:
    params = {"rsi_period": 14, "kr_window": 20, "kr_bandwidth": 4.0, "bb_k": 2.0,
              "extrema_order": 5, "rsi_oversold": 30, "rsi_overbought": 70}
    result = run_backtest(_synthetic_prices(), params, initial_balance=10_000)

    assert isinstance(result, BacktestResult)


def test_run_backtest_handles_window_larger_than_history() -> None:
    short = _synthetic_prices(n=10)
    result = run_backtest(short, BacktestParams(kr_window=50), initial_balance=10_000)

    assert result.profit_pct == -100.0
    assert result.trades.empty


def test_params_from_dict_ignores_unknown_keys() -> None:
    params = BacktestParams.from_dict({"rsi_period": 9, "unknown_key": 123})

    assert params.rsi_period == 9
    assert params.kr_window == BacktestParams().kr_window
