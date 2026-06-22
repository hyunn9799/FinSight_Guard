"""Provider-agnostic interfaces (006).

Providers return normalized objects + lineage + status, never raw payloads.
These Protocols define the boundary that future MCP adapters must satisfy.
No adapter is implemented in this feature.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from src.providers.entities import CompanyProfile, FinancialMetric, NewsEvent
from src.providers.enums import (
    NormalizationStatus,
    ProviderError,
    Warning,
    _Contract,
)


class NewsProviderRequest(_Contract):
    ticker: str
    company_hint: str | None = None
    as_of_date: date | None = None
    max_results: int = 20


class FinancialProviderRequest(_Contract):
    ticker: str
    company_hint: str | None = None
    as_of_date: date | None = None
    requested_metrics: list[str] = []


class MarketDataProviderRequest(_Contract):
    ticker: str
    period: str | None = None
    interval: str | None = None
    as_of_date: date | None = None


class NewsProviderResult(_Contract):
    raw_response_ref: str
    normalization_status: NormalizationStatus
    news_events: list[NewsEvent] = []
    warnings: list[Warning] = []
    errors: list[ProviderError] = []


class FinancialProviderResult(_Contract):
    raw_response_ref: str
    normalization_status: NormalizationStatus
    company_profile: CompanyProfile | None = None
    financial_metrics: list[FinancialMetric] = []
    warnings: list[Warning] = []
    errors: list[ProviderError] = []


class MarketDataProviderResult(_Contract):
    raw_response_ref: str
    normalization_status: NormalizationStatus
    normalized_market_data_ref: str | None = None
    warnings: list[Warning] = []
    errors: list[ProviderError] = []


@runtime_checkable
class NewsProvider(Protocol):
    def fetch_news(self, request: NewsProviderRequest) -> NewsProviderResult: ...


@runtime_checkable
class FinancialProvider(Protocol):
    def fetch_financials(
        self, request: FinancialProviderRequest
    ) -> FinancialProviderResult: ...


@runtime_checkable
class MarketDataProvider(Protocol):
    def fetch_market_data(
        self, request: MarketDataProviderRequest
    ) -> MarketDataProviderResult: ...
