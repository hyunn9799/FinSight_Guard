"""Deterministic provider-contract fixtures.

No live calls. Every builder returns fixed, hand-authored data so contract
tests are reproducible. Real raw shapes for two providers (A/B) plus
normalized-record and degraded-scenario builders are filled in by later tasks.
"""

from __future__ import annotations

from src.providers.normalization import RawNewsItem


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


# Builders are added incrementally:
#   T009/T012 -> raw_news_provider_a / raw_news_provider_b
#   T013      -> raw_company_payload / raw_financial_rows
#   T014      -> raw_market_data
#   T018+     -> normalized record builders for persistence
#   T029+     -> scenario_report_input builder
