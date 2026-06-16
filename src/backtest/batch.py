"""Multi-ticker batch backtesting (Streamlit-independent).

Ported from the standalone project's "average profit calculator" page. Runs a
fixed parameter set across many tickers and aggregates the historical-simulation
returns. These are past simulations for comparison only, never investment advice.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.backtest.data_loader import load_price_history
from src.backtest.strategy import BacktestParams, run_backtest

Loader = Callable[[str, object, object], pd.DataFrame]


@dataclass
class StockResult:
    """Per-ticker outcome of a batch backtest run."""

    ticker: str
    label: str
    profit_pct: float | None
    status: str  # "success" | "no_data" | "error"
    note: str = ""


@dataclass
class BatchResult:
    """Aggregate outcome across all tickers in a batch."""

    results: list[StockResult] = field(default_factory=list)
    average_profit_pct: float | None = None
    successful_count: int = 0

    def as_dataframe(self) -> pd.DataFrame:
        """Return successful rows as a tidy DataFrame for display."""
        rows = [
            {"종목": item.label, "티커": item.ticker, "수익률 (%)": item.profit_pct}
            for item in self.results
            if item.status == "success"
        ]
        return pd.DataFrame(rows)


def run_batch_backtest(
    stocks: Sequence[tuple[str, str]],
    *,
    start: object,
    end: object,
    params: BacktestParams | dict,
    initial_balance: float,
    fee: float = 0.001,
    loader: Loader = load_price_history,
    progress_callback: Callable[[int, int, StockResult], None] | None = None,
) -> BatchResult:
    """Run a fixed-parameter backtest across many ``(ticker, label)`` pairs.

    Each ticker is loaded and simulated independently; a load failure or empty
    frame is recorded as ``no_data``/``error`` and skipped from the average
    rather than aborting the whole batch.
    """
    resolved_params = (
        params if isinstance(params, BacktestParams) else BacktestParams.from_dict(params)
    )
    results: list[StockResult] = []
    total = len(stocks)

    for index, (ticker, label) in enumerate(stocks, start=1):
        try:
            frame = loader(ticker, start, end)
            if frame is None or frame.empty:
                item = StockResult(ticker, label, None, "no_data", "데이터가 없어 건너뜁니다.")
            else:
                result = run_backtest(frame, resolved_params, initial_balance, fee)
                item = StockResult(ticker, label, round(result.profit_pct, 2), "success")
        except Exception as exc:  # noqa: BLE001 - isolate per-ticker failures
            item = StockResult(ticker, label, None, "error", str(exc))

        results.append(item)
        if progress_callback is not None:
            progress_callback(index, total, item)

    successes = [item.profit_pct for item in results if item.status == "success"]
    average = round(sum(successes) / len(successes), 2) if successes else None
    return BatchResult(
        results=results,
        average_profit_pct=average,
        successful_count=len(successes),
    )
