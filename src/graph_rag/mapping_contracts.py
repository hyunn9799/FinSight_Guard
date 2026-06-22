"""006 graph-mapping eligibility contract. 005 owns the graph model itself."""

from __future__ import annotations

from src.providers.entities import (
    CompanyProfile, FinancialMetric, NewsEvent,
    TechnicalAnalysisResult, WaveAnalysisResult,
)
from src.providers.enums import _Contract

_NODE_TYPE_BY_CLASS = {
    CompanyProfile: "Company",
    NewsEvent: "NewsEvent",
    FinancialMetric: "FinancialMetric",
    TechnicalAnalysisResult: "TechnicalAnalysisResult",
    WaveAnalysisResult: "WaveAnalysisResult",
}

ELIGIBLE_NODE_TYPES = frozenset(
    {"Company", "Ticker", "NewsEvent", "FinancialMetric",
     "TechnicalAnalysisResult", "WaveAnalysisResult", "Risk", "Evidence"}
)


class GraphMappingRule(_Contract):
    source_contract_type: str
    graph_node_type: str
    relationship_type: str | None = None
    required_fields: list[str] = []
    evidence_id_required: bool = False
    projection_scope: str = "company_ticker_centered"


class GraphEligibleSpec(_Contract):
    source_contract_type: str
    source_canonical_ref: str
    graph_node_type: str
    required_fields: list[str] = []
    evidence_ref: str | None = None


def is_graph_eligible(record) -> bool:
    return type(record) in _NODE_TYPE_BY_CLASS


def _canonical_ref(record) -> str:
    # Prefer the record's own canonical id field; fall back to request/ticker.
    for attr in ("company_profile_id", "news_event_id", "financial_metric_id",
                 "technical_analysis_result_id", "wave_analysis_result_id"):
        val = getattr(record, attr, None)
        if val:
            return val
    return f"{record.request_id}:{record.ticker_id}"


def build_eligible_specs(records: list) -> list[GraphEligibleSpec]:
    specs: list[GraphEligibleSpec] = []
    for rec in records:
        node_type = _NODE_TYPE_BY_CLASS.get(type(rec))
        if node_type is None:
            continue  # raw payloads, candles, rows, dicts -> never projected
        evidence_ref = getattr(rec, "evidence_id", None)
        if evidence_ref is None:
            ev_ids = getattr(rec, "evidence_ids", []) or []
            evidence_ref = ev_ids[0] if ev_ids else None
        specs.append(GraphEligibleSpec(
            source_contract_type=type(rec).__name__,
            source_canonical_ref=_canonical_ref(rec),
            graph_node_type=node_type,
            evidence_ref=evidence_ref,
        ))
    return specs
