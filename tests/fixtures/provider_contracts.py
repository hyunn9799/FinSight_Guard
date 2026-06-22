"""Deterministic provider-contract fixtures.

No live calls. Every builder returns fixed, hand-authored data so contract
tests are reproducible. Real raw shapes for two providers (A/B) plus
normalized-record and degraded-scenario builders are filled in by later tasks.
"""

from __future__ import annotations

from src.providers.normalization import RawCompanyPayload, RawFinancialRow, RawMarketData, RawNewsItem


def raw_news_provider_a() -> list[RawNewsItem]:
    """Provider A spelling: {title, content, url}."""
    return [
        RawNewsItem(
            title="Acme beats Q2 earnings",
            content="Acme reported revenue above consensus.",
            url="https://news.example.com/acme-q2",
            source="Example Wire",
            published="2026-06-01T13:00:00Z",
        )
    ]


def raw_news_provider_b() -> list[RawNewsItem]:
    """Provider B spelling: {headline, summary_text, source_url}."""
    return [
        RawNewsItem(
            headline="Acme beats Q2 earnings",
            summary_text="Acme reported revenue above consensus.",
            source_url="https://news.example.com/acme-q2",
            source="Example Wire",
            published="2026-06-01T13:00:00Z",
        )
    ]


def raw_company_payload() -> RawCompanyPayload:
    return RawCompanyPayload(
        name="Acme Corp", sector="Technology", industry="Software",
        country="US", exchange="NASDAQ", currency="USD", about="Maker of widgets.",
    )


def raw_financial_rows() -> list[RawFinancialRow]:
    return [
        RawFinancialRow(name="revenue", value=1234.5, period="FY2025", currency="USD", unit="millions"),
        RawFinancialRow(metric="net_income", value=210.0, period="FY2025", currency="USD", unit="millions"),
    ]


def raw_market_data() -> RawMarketData:
    return RawMarketData(
        ticker="ACME",
        candles=[{"t": "2026-06-01", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 1000}],
    )


# Builders are added incrementally:
#   T009/T012 -> raw_news_provider_a / raw_news_provider_b
#   T013      -> raw_company_payload / raw_financial_rows
#   T014      -> raw_market_data
#   T018+     -> normalized record builders for persistence
#   T029+     -> scenario_report_input builder


def scenario_inputs_complete():
    """Normalized records for a fully-populated scenario (deterministic)."""
    common = dict(request_id="req1", ticker_id="tk1", raw_response_id="raw1")
    from src.providers.normalization import (
        normalize_company, normalize_financials, normalize_news,
    )
    cp = normalize_company(raw=raw_company_payload(), **common).records[0]
    news = normalize_news(raw_items=raw_news_provider_a(), **common).records
    metrics = normalize_financials(raw_rows=raw_financial_rows(), **common).records
    return cp, news, metrics
