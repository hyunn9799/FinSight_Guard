"""Date-range price loader for backtesting.

Separate from ``src.tools.market_data`` (which is period-based and returns a
``Date`` column for the research agents). The backtest needs a contiguous OHLC
frame indexed by date. Shares the same NaN-bar guard: yfinance can return the
current in-progress trading day with NaN OHLC, which would corrupt the
simulation.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

OHLC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns (single-ticker download) to names."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def load_price_history(
    ticker: str,
    start: str | date,
    end: str | date,
) -> pd.DataFrame:
    """Download adjusted OHLC history for ``ticker`` between ``start`` and ``end``.

    Returns a DataFrame indexed by date with ``Open/High/Low/Close/Volume``
    columns and incomplete (NaN OHLC) bars removed. Raises ``ValueError`` when
    no usable rows are returned.
    """
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        raise ValueError("Ticker must not be empty.")

    raw = yf.download(clean_ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No price data returned for {clean_ticker}.")

    frame = _flatten_columns(raw)
    missing = [column for column in ["Open", "High", "Low", "Close"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Price data missing required columns: {', '.join(missing)}")

    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    if frame.empty:
        raise ValueError(f"Price data for {clean_ticker} has no complete bars.")

    keep = [column for column in OHLC_COLUMNS if column in frame.columns]
    return frame[keep].copy()
