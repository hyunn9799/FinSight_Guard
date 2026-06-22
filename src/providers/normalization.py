"""Raw-fixture shapes, normalization result containers, and helper seams (006).

Raw* models intentionally use LOOSE provider-specific field names (e.g. two
providers spell the same news field differently). Normalizers translate those
into the stable entity contracts. Bodies are filled in US1 (T012-T014).
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from src.providers.entities import (
    CompanyProfile,
    FinancialMetric,
    NewsEvent,
)
from src.providers.enums import (
    NormalizationStatus,
    ProviderError,
    Warning,
    _Contract,
)


# --- Raw provider shapes (deliberately permissive: extra allowed) -----------
class _Raw(_Contract):
    # Raw payloads vary by provider; allow unknown keys so fixtures stay honest.
    model_config = ConfigDict(extra="allow")


class RawNewsItem(_Raw):
    # Provider A: {title, content, url}; Provider B: {headline, summary_text, source_url}
    title: str | None = None
    headline: str | None = None
    content: str | None = None
    summary_text: str | None = None
    url: str | None = None
    source_url: str | None = None
    source: str | None = None
    published: str | None = None


class RawCompanyPayload(_Raw):
    name: str | None = None
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    exchange: str | None = None
    currency: str | None = None
    about: str | None = None


class RawFinancialRow(_Raw):
    name: str | None = None
    metric: str | None = None
    value: Any = None
    period: str | None = None
    currency: str | None = None
    unit: str | None = None


class RawMarketData(_Raw):
    ticker: str | None = None
    candles: list[dict[str, Any]] = Field(default_factory=list)


# --- Normalization result container -----------------------------------------
class NormalizationResult(_Contract):
    status: NormalizationStatus
    records: list[Any] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)
    errors: list[ProviderError] = Field(default_factory=list)
    normalized_market_data_ref: str | None = None


# --- Helper seams (implemented in US1) --------------------------------------
from datetime import datetime


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_news(
    *, raw_items: list[RawNewsItem], request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    records: list[NewsEvent] = []
    warnings: list[Warning] = []
    for item in raw_items:
        title = item.title or item.headline
        summary = item.content or item.summary_text
        source_url = item.url or item.source_url
        status = NormalizationStatus.SUCCESS
        if title is None:
            # cannot build a meaningful event without a title
            warnings.append(Warning(code="missing_title", message="news item missing title"))
            status = NormalizationStatus.PARTIAL_SUCCESS
            continue
        if source_url is None:
            warnings.append(Warning(code="missing_url", message="news item missing source url", field="source_url"))
            status = NormalizationStatus.PARTIAL_SUCCESS
        records.append(
            NewsEvent(
                request_id=request_id,
                ticker_id=ticker_id,
                raw_response_id=raw_response_id,
                title=title,
                summary=summary,
                source_name=item.source,
                source_url=source_url,
                published_at=_parse_dt(item.published),
                normalization_status=status,
                warnings=[w for w in warnings if w.field == "source_url"],
            )
        )
    overall = NormalizationStatus.SUCCESS if not warnings else NormalizationStatus.PARTIAL_SUCCESS
    if not records:
        overall = NormalizationStatus.INSUFFICIENT_DATA
    return NormalizationResult(status=overall, records=records, warnings=warnings)


def normalize_company(
    *, raw: RawCompanyPayload, request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    raise NotImplementedError("Implemented in US1 / T013")


def normalize_financials(
    *, raw_rows: list[RawFinancialRow], request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    raise NotImplementedError("Implemented in US1 / T013")


def normalize_market_data(
    *, raw: RawMarketData, request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    raise NotImplementedError("Implemented in US1 / T014")
