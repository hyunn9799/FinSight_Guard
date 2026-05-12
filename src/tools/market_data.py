"""Market data provider tools."""

from typing import Any

import pandas as pd
import yfinance as yf

from src.tools.retry import retry


PRICE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def validate_ticker_data(df: pd.DataFrame) -> None:
    """Validate that a price dataframe contains usable market data."""
    if df is None or df.empty:
        raise ValueError("No price data returned for ticker.")

    missing_columns = [
        column for column in ["Open", "High", "Low", "Close", "Volume"] if column not in df.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Price data is missing required columns: {missing}")


def normalize_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance history output to Date/Open/High/Low/Close/Volume."""
    validate_ticker_data(df)

    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [
            column[0] if isinstance(column, tuple) else column for column in normalized.columns
        ]

    if "Date" not in normalized.columns:
        normalized = normalized.reset_index()

    if "Date" not in normalized.columns:
        first_column = normalized.columns[0]
        normalized = normalized.rename(columns={first_column: "Date"})

    normalized = normalized[PRICE_COLUMNS].copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"]).dt.date
    return normalized


@retry(max_attempts=2, delay_seconds=0.5)
def fetch_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch normalized price history from yfinance."""
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        raise ValueError("Ticker must not be empty.")

    try:
        raw_data: Any = yf.Ticker(clean_ticker).history(period=period, interval=interval)
    except Exception as exc:  # yfinance can raise transport or parser exceptions.
        raise ValueError(f"Failed to fetch price data for {clean_ticker}: {exc}") from exc

    try:
        return normalize_price_dataframe(raw_data)
    except ValueError as exc:
        raise ValueError(f"Invalid price data for {clean_ticker}: {exc}") from exc
