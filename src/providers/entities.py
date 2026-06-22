"""Normalized provider contracts and internal derived analysis contracts (006).

Field names follow specs/006-provider-mcp-contracts/data-model.md exactly.
Normalized contracts (CompanyProfile/NewsEvent/FinancialMetric) trace to a
RawProviderResponse via raw_response_id. Derived contracts
(TechnicalAnalysisResult/WaveAnalysisResult) trace to normalized market data,
evidence, and rule refs — NOT to raw provider payloads.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from src.providers.enums import NormalizationStatus, RuleStatus, Warning, _Contract


class CompanyProfile(_Contract):
    company_profile_id: str | None = None
    request_id: str
    ticker_id: str
    raw_response_id: str
    company_name: str
    legal_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    exchange: str | None = None
    currency: str | None = None
    description: str | None = None
    normalization_status: NormalizationStatus
    warnings: list[Warning] = Field(default_factory=list)
    evidence_id: str | None = None


class NewsEvent(_Contract):
    news_event_id: str | None = None
    request_id: str
    ticker_id: str
    raw_response_id: str
    title: str
    summary: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    published_at: datetime | None = None
    collected_at: datetime | None = None
    event_type: str | None = None
    sentiment_label: str | None = None
    risk_tags: list[str] = Field(default_factory=list)
    normalization_status: NormalizationStatus
    warnings: list[Warning] = Field(default_factory=list)
    evidence_id: str | None = None


class FinancialMetric(_Contract):
    financial_metric_id: str | None = None
    request_id: str
    ticker_id: str
    raw_response_id: str
    metric_name: str
    metric_value: float | str | None = None
    period: str | None = None
    currency: str | None = None
    unit: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    collected_at: datetime | None = None
    normalization_status: NormalizationStatus
    warnings: list[Warning] = Field(default_factory=list)
    evidence_id: str | None = None


class TechnicalAnalysisResult(_Contract):
    technical_analysis_result_id: str | None = None
    request_id: str
    ticker_id: str
    source_market_data_refs: list[str] = Field(default_factory=list)
    indicator_values: dict[str, float] = Field(default_factory=dict)
    trend_state: str | None = None
    momentum_state: str | None = None
    volatility_state: str | None = None
    normalization_or_derivation_status: NormalizationStatus
    warnings: list[Warning] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class WaveAnalysisResult(_Contract):
    wave_analysis_result_id: str | None = None
    request_id: str
    ticker_id: str
    source_market_data_refs: list[str] = Field(default_factory=list)
    rule_refs: list[str] = Field(default_factory=list)
    candidate_summary: str | None = None
    rule_statuses: dict[str, RuleStatus] = Field(default_factory=dict)
    confirmation_conditions: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    uncertainty_notes: str | None = None
    warnings: list[Warning] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
