"""ScenarioReportInput + VectorReference (006). No raw payloads, no trading fields."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from src.providers.entities import (
    CompanyProfile, FinancialMetric, NewsEvent,
    TechnicalAnalysisResult, WaveAnalysisResult,
)
from src.providers.enums import DegradationStatus, Warning, _Contract


class VectorReference(_Contract):
    source_kind: Literal["news_original", "neowave_rule_explanation", "report_chunk"]
    canonical_ref_id: str
    source_uri: str | None = None
    chunk_id: str | None = None

    @field_validator("canonical_ref_id")
    @classmethod
    def _ref_must_resolve(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("VectorReference.canonical_ref_id must resolve to a canonical PG reference")
        return v


class ScenarioReportInput(_Contract):
    request_id: str
    ticker: str
    company_profile: CompanyProfile | None = None
    news_events: list[NewsEvent] = Field(default_factory=list)
    financial_metrics: list[FinancialMetric] = Field(default_factory=list)
    technical_analysis_results: list[TechnicalAnalysisResult] = Field(default_factory=list)
    wave_analysis_results: list[WaveAnalysisResult] = Field(default_factory=list)
    graph_context: dict = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    vector_references: list[VectorReference] = Field(default_factory=list)
    missing_data_notes: list[str] = Field(default_factory=list)
    degradation_status: DegradationStatus
    warnings: list[Warning] = Field(default_factory=list)
