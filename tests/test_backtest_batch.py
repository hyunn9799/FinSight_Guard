"""Tests for multi-ticker batch backtesting (injected loader, no live API)."""

import numpy as np
import pandas as pd

from src.backtest.batch import BatchResult, run_batch_backtest
from src.backtest.strategy import BacktestParams


def _synthetic_prices(n: int = 130) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    t = np.arange(n)
    close = 100 + 9 * np.sin(t / 6.0) + t * 0.07
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


def test_run_batch_backtest_aggregates_successes_and_records_failures() -> None:
    progress: list[int] = []

    def fake_loader(ticker: str, start, end) -> pd.DataFrame:
        if ticker == "EMPTY":
            return pd.DataFrame()
        if ticker == "BOOM":
            raise ValueError("delisted")
        return _synthetic_prices()

    stocks = [("AAA", "에이"), ("BBB", "비"), ("EMPTY", "없음"), ("BOOM", "오류")]

    result = run_batch_backtest(
        stocks,
        start="2025-01-01",
        end="2025-06-01",
        params=BacktestParams(bb_k=0.7),
        initial_balance=10_000,
        loader=fake_loader,
        progress_callback=lambda done, total, item: progress.append(done),
    )

    assert isinstance(result, BatchResult)
    assert result.successful_count == 2
    assert result.average_profit_pct is not None
    statuses = {item.ticker: item.status for item in result.results}
    assert statuses == {"AAA": "success", "BBB": "success", "EMPTY": "no_data", "BOOM": "error"}
    assert progress == [1, 2, 3, 4]
    assert list(result.as_dataframe()["티커"]) == ["AAA", "BBB"]


def test_run_batch_backtest_all_failures_yields_none_average() -> None:
    def empty_loader(ticker: str, start, end) -> pd.DataFrame:
        return pd.DataFrame()

    result = run_batch_backtest(
        [("X", "엑스"), ("Y", "와이")],
        start="2025-01-01",
        end="2025-06-01",
        params=BacktestParams(),
        initial_balance=10_000,
        loader=empty_loader,
    )

    assert result.successful_count == 0
    assert result.average_profit_pct is None
    assert result.as_dataframe().empty
