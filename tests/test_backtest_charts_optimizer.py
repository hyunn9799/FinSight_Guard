"""Tests for backtest charting and Optuna optimization (no live API calls)."""

import matplotlib
import numpy as np
import pandas as pd

from src.backtest.charts import build_backtest_figure, configure_korean_font
from src.backtest.optimizer import OptimizationResult, SearchSpace, optimize_backtest
from src.backtest.strategy import BacktestParams, run_backtest


def _synthetic_prices(n: int = 140) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    t = np.arange(n)
    close = 100 + 11 * np.sin(t / 6.5) + t * 0.08
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 1_000),
        },
        index=idx,
    )


def test_configure_korean_font_is_safe_and_idempotent() -> None:
    first = configure_korean_font()
    second = configure_korean_font()

    assert first == second  # cached
    assert first is None or isinstance(first, str)


def test_build_backtest_figure_returns_two_panel_figure() -> None:
    result = run_backtest(_synthetic_prices(), BacktestParams(), initial_balance=10_000)

    fig = build_backtest_figure(
        result.enriched,
        result.divergences,
        ticker="TEST",
        rsi_oversold=30,
        rsi_overbought=70,
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2
    matplotlib.pyplot.close(fig)


def test_optimize_backtest_runs_and_reports_progress() -> None:
    progress: list[tuple[int, int]] = []

    result = optimize_backtest(
        _synthetic_prices(),
        initial_balance=10_000,
        n_trials=3,
        search_space=SearchSpace(),
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert isinstance(result, OptimizationResult)
    assert result.completed_trials == 3
    assert progress[-1] == (3, 3)
    assert set(result.best_params) >= {"kr_window", "rsi_period", "bb_k"}
    assert isinstance(result.best_profit_pct, float)
