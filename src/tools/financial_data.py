"""Financial and fundamental data provider tools."""

from typing import Any

import yfinance as yf

from src.tools.retry import retry


FINANCIAL_FIELDS = [
    "longName",
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "priceToBook",
    "returnOnEquity",
    "profitMargins",
    "debtToEquity",
    "freeCashflow",
]


def _none_if_missing(value: Any) -> Any:
    if value in ("", "None", "nan"):
        return None
    return value


@retry(max_attempts=2, delay_seconds=0.5)
def fetch_basic_financials(ticker: str) -> dict[str, Any]:
    """Fetch basic company financial fields from yfinance."""
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        raise ValueError("Ticker must not be empty.")

    try:
        info = yf.Ticker(clean_ticker).info or {}
    except Exception as exc:
        raise ValueError(f"Failed to fetch financial data for {clean_ticker}: {exc}") from exc

    result = {"ticker": clean_ticker}
    for field in FINANCIAL_FIELDS:
        result[field] = _none_if_missing(info.get(field))
    return result
