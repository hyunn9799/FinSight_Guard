"""Tests for data provider tools without live API calls."""

from datetime import date

import pandas as pd
import pytest

from src.tools.financial_data import fetch_basic_financials
from src.tools.market_data import (
    fetch_price_history,
    normalize_price_dataframe,
    validate_ticker_data,
)
from src.tools.news_search import search_recent_news
from src.tools.retry import retry


class FakeMarketTicker:
    """Fake yfinance ticker object for price history tests."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, period: str, interval: str) -> pd.DataFrame:
        assert period == "1y"
        assert interval == "1d"
        return pd.DataFrame(
            {
                "Open": [10.0],
                "High": [12.0],
                "Low": [9.5],
                "Close": [11.0],
                "Volume": [1000],
            },
            index=pd.to_datetime(["2026-05-11"]),
        )


class EmptyMarketTicker:
    """Fake yfinance ticker object that returns no data."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, period: str, interval: str) -> pd.DataFrame:
        return pd.DataFrame()


class FakeFinancialTicker:
    """Fake yfinance ticker object for financial info tests."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.info = {
            "longName": "Example Inc.",
            "sector": "Technology",
            "marketCap": 1000,
            "trailingPE": 20.5,
        }


def test_normalize_price_dataframe_returns_expected_columns() -> None:
    raw = pd.DataFrame(
        {
            "Open": [1],
            "High": [2],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [100],
        },
        index=pd.to_datetime(["2026-05-11"]),
    )

    normalized = normalize_price_dataframe(raw)

    assert list(normalized.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert normalized.loc[0, "Date"] == date(2026, 5, 11)


def test_validate_ticker_data_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="No price data"):
        validate_ticker_data(pd.DataFrame())


def test_fetch_price_history_uses_yfinance_and_normalizes(monkeypatch) -> None:
    import src.tools.market_data as market_data

    monkeypatch.setattr(market_data.yf, "Ticker", FakeMarketTicker)

    result = fetch_price_history("aapl")

    assert result.loc[0, "Close"] == 11.0
    assert result.loc[0, "Volume"] == 1000


def test_fetch_price_history_raises_clear_value_error(monkeypatch) -> None:
    import src.tools.market_data as market_data

    monkeypatch.setattr(market_data.yf, "Ticker", EmptyMarketTicker)

    with pytest.raises(ValueError, match="Invalid price data for AAPL"):
        fetch_price_history("AAPL")


def test_fetch_basic_financials_handles_missing_fields(monkeypatch) -> None:
    import src.tools.financial_data as financial_data

    monkeypatch.setattr(financial_data.yf, "Ticker", FakeFinancialTicker)

    result = fetch_basic_financials("msft")

    assert result["ticker"] == "MSFT"
    assert result["longName"] == "Example Inc."
    assert result["industry"] is None
    assert result["freeCashflow"] is None


def test_search_recent_news_uses_mock_when_keys_missing(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

    results = search_recent_news("nvda", company_name="NVIDIA", max_results=1)

    assert len(results) == 1
    assert results[0]["source"] == "mock"
    assert "NVIDIA" in results[0]["title"]


def test_search_recent_news_falls_back_when_live_provider_fails(monkeypatch) -> None:
    import src.tools.news_search as news_search

    def fail_tavily(ticker: str, company_name: str | None, max_results: int):
        raise RuntimeError("provider unavailable")

    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr(news_search, "_search_with_tavily", fail_tavily)

    results = search_recent_news("tsla", company_name="Tesla", max_results=2)

    assert results
    assert all(item["source"] == "mock" for item in results)


def test_search_recent_news_uses_tavily_when_configured(monkeypatch) -> None:
    import src.tools.news_search as news_search

    def fake_post_json(url: str, api_key: str, payload: dict):
        assert url == news_search.TAVILY_SEARCH_URL
        assert api_key == "fake-tavily-key"
        assert payload["topic"] == "news"
        return {
            "results": [
                {
                    "title": "Apple announces product update",
                    "content": "Apple reported details relevant to investors.",
                    "url": "https://example.com/apple-news",
                    "published_date": "2026-05-12",
                }
            ]
        }

    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr(news_search, "_post_json", fake_post_json)

    results = search_recent_news("aapl", company_name="Apple Inc.", max_results=1)

    assert results == [
        {
            "title": "Apple announces product update",
            "summary": "Apple reported details relevant to investors.",
            "url": "https://example.com/apple-news",
            "published_at": "2026-05-12",
            "source": "tavily",
        }
    ]


def test_search_recent_news_falls_back_from_tavily_to_firecrawl(monkeypatch) -> None:
    import src.tools.news_search as news_search

    def fake_post_json(url: str, api_key: str, payload: dict):
        if url == news_search.TAVILY_SEARCH_URL:
            raise RuntimeError("tavily unavailable")
        assert url == news_search.FIRECRAWL_SEARCH_URL
        assert api_key == "fake-firecrawl-key"
        assert payload["sources"] == ["web"]
        return {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Tesla event risk update",
                        "description": "Tesla faces a monitored event risk.",
                        "url": "https://example.com/tesla-risk",
                    }
                ]
            },
        }

    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fake-firecrawl-key")
    monkeypatch.setattr(news_search, "_post_json", fake_post_json)

    results = search_recent_news("tsla", company_name="Tesla", max_results=1)

    assert results[0]["source"] == "firecrawl"
    assert results[0]["url"] == "https://example.com/tesla-risk"


def test_retry_reraises_final_exception() -> None:
    calls = {"count": 0}

    @retry(max_attempts=2, delay_seconds=0)
    def always_fails() -> None:
        calls["count"] += 1
        raise RuntimeError("still failing")

    with pytest.raises(RuntimeError, match="still failing"):
        always_fails()

    assert calls["count"] == 2
