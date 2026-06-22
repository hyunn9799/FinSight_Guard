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


def build_scenario_report_input(
    *, request_id, ticker, company_profile, news_events, financial_metrics,
    technical_analysis_results, wave_analysis_results, graph_context, vector_references,
):
    notes: list[str] = []
    warnings: list[Warning] = []

    def _check(present: bool, category: str) -> None:
        if not present:
            notes.append(f"missing {category}")
            warnings.append(Warning(code=f"missing_{category}", message=f"no {category} available"))

    _check(company_profile is not None, "company_profile")
    _check(bool(news_events), "news_events")
    _check(bool(financial_metrics), "financial_metrics")
    _check(bool(technical_analysis_results), "technical_analysis")
    if not graph_context:
        notes.append("missing graph_context")
        warnings.append(Warning(code="missing_graph_context", message="graph context missing or stale"))

    have_any = any([company_profile, news_events, financial_metrics, technical_analysis_results])
    required_core = company_profile is not None and bool(financial_metrics)
    if not have_any:
        status = DegradationStatus.INSUFFICIENT_DATA
    elif not required_core:
        status = DegradationStatus.PARTIAL_PROVIDER_FAILURE
    elif not graph_context:
        status = DegradationStatus.GRAPH_MAPPING_DEGRADED
    elif notes:
        status = DegradationStatus.PARTIAL_PROVIDER_FAILURE
    else:
        status = DegradationStatus.COMPLETE

    evidence_ids: list[str] = []
    if company_profile and company_profile.evidence_id:
        evidence_ids.append(company_profile.evidence_id)
    for e in news_events:
        if e.evidence_id:
            evidence_ids.append(e.evidence_id)
    for m in financial_metrics:
        if m.evidence_id:
            evidence_ids.append(m.evidence_id)

    return ScenarioReportInput(
        request_id=request_id, ticker=ticker, company_profile=company_profile,
        news_events=list(news_events), financial_metrics=list(financial_metrics),
        technical_analysis_results=list(technical_analysis_results),
        wave_analysis_results=list(wave_analysis_results),
        graph_context=graph_context, evidence_ids=evidence_ids,
        vector_references=list(vector_references), missing_data_notes=notes,
        degradation_status=status, warnings=warnings,
    )
