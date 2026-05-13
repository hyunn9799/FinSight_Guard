"""Rewrite agent for evaluator-driven report revision."""

import re

from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import AgentName, GraphState, ResearchReport, WorkflowError
from src.graph_rag.graph_schema import GraphContext, GraphEdge
from src.safety.forbidden_phrases import FORBIDDEN_PHRASES, REQUIRED_DISCLAIMER


REWRITE_NODE = "rewrite_agent"
EVIDENCE_ID_PATTERN = re.compile(
    r"(?P<prefix>근거:\s*|evidence_id=)(?P<evidence_id>(?:market|fundamental|news|system)_[A-Za-z0-9_]+)"
)
AGENT_LABELS: dict[AgentName, str] = {
    "market": "시장",
    "fundamental": "재무",
    "news": "뉴스",
}
AGENT_SECTION_NAMES: dict[AgentName, str] = {
    "market": "시장 분석",
    "fundamental": "재무 분석",
    "news": "뉴스 분석",
}

SAFE_REPLACEMENTS = {
    "무조건 매수": "추가 근거를 검토할 수 있는 시나리오",
    "강력 매수": "긍정 요인을 참고할 수 있는 시나리오",
    "반드시 매수": "조건을 나누어 검토할 수 있는 시나리오",
    "지금 사야 합니다": "현재 데이터는 참고할 수 있습니다",
    "매도해야 합니다": "리스크 회피 시나리오를 검토할 수 있습니다",
    "수익 보장": "성과를 보장하지 않는 참고 정보",
    "손실 없음": "손실 가능성을 함께 고려해야 하는 정보",
    "확실한 수익": "불확실성을 포함한 참고 정보",
    "원금 보장": "원금 손실 가능성을 고려해야 하는 정보",
    "목표가 보장": "목표 가격을 보장하지 않는 참고 정보",
    "매수 추천": "검토할 수 있는 시나리오",
    "매도 추천": "리스크를 확인해야 하는 시나리오",
    "반드시 사야": "추가 근거를 확인해야",
    "무위험": "위험 요인을 함께 고려해야 하는",
    "반드시 수익": "성과를 보장하지 않는 참고",
}


def _report_from_state(state: GraphState) -> ResearchReport | None:
    return state.get("draft_report") or state.get("final_report")


def _safe_text(text: str) -> str:
    rewritten = text or ""
    for phrase in [*FORBIDDEN_PHRASES, *SAFE_REPLACEMENTS.keys()]:
        rewritten = rewritten.replace(phrase, SAFE_REPLACEMENTS[phrase])
    return rewritten


def _dedupe_evidence(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    seen_ids: set[str] = set()
    deduped: list[EvidenceItem] = []
    for item in evidence:
        if item.evidence_id in seen_ids:
            continue
        seen_ids.add(item.evidence_id)
        deduped.append(item)
    return deduped


def _collect_evidence(state: GraphState) -> list[EvidenceItem]:
    evidence = list(state.get("evidence", []))
    for analysis_key in ("market_analysis", "fundamental_analysis", "news_analysis"):
        analysis = state.get(analysis_key)
        if analysis is not None:
            evidence.extend(analysis.evidence)
    return _dedupe_evidence(evidence)


def _allowed_evidence_ids(state: GraphState) -> set[str]:
    evidence_ids = {item.evidence_id for item in _collect_evidence(state) if item.evidence_id}
    graph_context = state.get("graph_context")
    if graph_context is not None:
        evidence_ids.update(evidence_id for evidence_id in graph_context.evidence_ids if evidence_id)
        for edge in [*graph_context.edges, *graph_context.risk_relations, *graph_context.positive_relations]:
            if edge.evidence_id:
                evidence_ids.add(edge.evidence_id)
    return evidence_ids


def _strip_unknown_evidence_refs(text: str, allowed_ids: set[str]) -> str:
    rewritten = _safe_text(text)

    def replace(match: re.Match[str]) -> str:
        evidence_id = match.group("evidence_id")
        if evidence_id in allowed_ids:
            return match.group(0)
        return f"{match.group('prefix')}사용 가능한 근거 데이터 제한"

    return EVIDENCE_ID_PATTERN.sub(replace, rewritten)


def _edge_label(edge: GraphEdge) -> str:
    evidence = f", evidence_id={edge.evidence_id}" if edge.evidence_id else ""
    description = f" - {edge.description}" if edge.description else ""
    return f"{edge.source_id} -> {edge.target_id}: {edge.relation_type}{evidence}{description}"


def _graph_context_section(graph_context: GraphContext | None) -> str:
    title = "관계 기반 리스크 및 근거 요약"
    if graph_context is None:
        return ""

    lines = [title]
    if graph_context.key_relations_summary:
        lines.append("핵심 관계:")
        lines.extend(f"- {summary}" for summary in graph_context.key_relations_summary[:5])
    else:
        lines.append("핵심 관계: 현재 사용 가능한 EvidenceItem만으로는 별도 관계 요약이 제한됩니다.")

    if graph_context.risk_relations:
        lines.append("리스크 관계:")
        lines.extend(f"- {_edge_label(edge)}" for edge in graph_context.risk_relations[:5])
    else:
        lines.append("리스크 관계: 명시적으로 추출된 리스크 관계가 없습니다.")

    if graph_context.positive_relations:
        lines.append("긍정 관계:")
        lines.extend(f"- {_edge_label(edge)}" for edge in graph_context.positive_relations[:5])
    else:
        lines.append("긍정 관계: 명시적으로 추출된 긍정 관계가 없습니다.")

    if graph_context.evidence_ids:
        lines.append("연결 EvidenceItem ID: " + ", ".join(graph_context.evidence_ids[:10]))
    else:
        lines.append("연결 EvidenceItem ID: 없음")
    return "\n".join(lines)


def _ensure_graph_context_section(report: ResearchReport, state: GraphState) -> str:
    existing = _strip_unknown_evidence_refs(report.graph_context_section, _allowed_evidence_ids(state)).strip()
    graph_context = state.get("graph_context")
    if graph_context is None:
        return existing
    generated = _graph_context_section(graph_context)
    if not existing or "관계 기반 리스크 및 근거 요약" not in existing:
        return generated
    missing_risk = [
        edge
        for edge in graph_context.risk_relations
        if not any(
            marker and marker in existing
            for marker in (edge.source_id, edge.target_id, edge.relation_type, edge.description or "")
        )
    ]
    if missing_risk:
        return existing + "\n" + "\n".join(f"- {_edge_label(edge)}" for edge in missing_risk[:5])
    return existing


def _ensure_risk_factors(report: ResearchReport, state: GraphState) -> str:
    risk_factors = _safe_text(report.risk_factors).strip()
    if not risk_factors:
        risk_factors = (
            "가격 변동성, 실적 발표, 거시 환경 변화, 뉴스 이벤트 해석 차이, 데이터 지연 또는 누락이 "
            "보고서 해석에 영향을 줄 수 있습니다. 손실 가능성을 포함한 리스크를 검토할 수 있습니다."
        )

    graph_context = state.get("graph_context")
    if graph_context is not None and graph_context.risk_relations:
        existing = risk_factors
        additions = []
        for edge in graph_context.risk_relations[:5]:
            if any(
                marker and marker in existing
                for marker in (edge.source_id, edge.target_id, edge.relation_type, edge.description or "")
            ):
                continue
            additions.append(f"GraphContext 리스크 관계도 함께 확인해야 합니다: {_edge_label(edge)}")
        if additions:
            risk_factors = risk_factors + "\n" + "\n".join(additions)
    return risk_factors


def _supervisor_skipped_agents(state: GraphState) -> set[AgentName]:
    plan = state.get("supervisor_plan")
    skipped = set(state.get("skipped_agents", []))
    if plan is not None:
        skipped.update(plan.skipped_agents)
    return skipped


def _supervisor_failed_agents(state: GraphState) -> set[AgentName]:
    plan = state.get("supervisor_plan")
    failed = set(state.get("failed_agents", []))
    if plan is not None:
        failed.update(plan.failed_agents)
    return failed


def _planned_agents(state: GraphState) -> list[AgentName]:
    plan = state.get("supervisor_plan")
    if plan is None:
        return []
    return list(plan.planned_agent_order or plan.required_agents)


def _ensure_limitations(report: ResearchReport, state: GraphState) -> str:
    limitations = _safe_text(report.limitations).strip()
    if not limitations:
        limitations = (
            "본 분석은 공개 데이터, 수집 가능한 EvidenceItem, 단순 규칙 기반 분류에 의존합니다. "
            "데이터 지연, 누락 필드, mock 뉴스 fallback, 모델 한계로 인해 실제 상황을 완전히 반영하지 못할 수 있습니다."
        )

    notes = []
    if not _collect_evidence(state):
        notes.append("사용 가능한 근거 데이터가 제한적입니다.")
        notes.append("해당 주장에 대한 충분한 근거 데이터가 없어 제한적으로 해석해야 합니다.")

    plan = state.get("supervisor_plan")
    if plan is not None and plan.execution_mode == "selective":
        notes.append("Supervisor 계획에 따라 일부 분석만 수행되어 근거 범위가 제한적입니다.")

    for agent in sorted(_supervisor_skipped_agents(state)):
        section = AGENT_SECTION_NAMES[agent]
        notes.append(f"이번 질문은 선택 분석 중심으로 분류되어 {section}은 상세 실행 대상에서 제외되었습니다.")

    for agent in sorted(_supervisor_failed_agents(state)):
        section = AGENT_SECTION_NAMES[agent]
        notes.append(f"{section} 데이터 수집이 제한되어 해당 섹션은 제한적으로 해석해야 합니다.")

    for agent in _planned_agents(state):
        analysis_key = f"{agent}_analysis"
        if agent in _supervisor_skipped_agents(state) or agent in _supervisor_failed_agents(state):
            continue
        if analysis_key not in state:
            label = AGENT_LABELS[agent]
            notes.append(f"계획된 {label} 결과가 누락되어 limitations에 명시하며 제한적으로 해석해야 합니다.")

    for note in notes:
        if note not in limitations:
            limitations = limitations + "\n" + note
    return limitations


def _ensure_evidence_summary(report: ResearchReport, state: GraphState) -> str:
    allowed_ids = _allowed_evidence_ids(state)
    original_ids = {
        match.group("evidence_id")
        for match in EVIDENCE_ID_PATTERN.finditer(report.evidence_summary or "")
    }
    evidence_summary = _strip_unknown_evidence_refs(report.evidence_summary, allowed_ids).strip()
    unknown_ids = original_ids - allowed_ids
    if evidence_summary and not unknown_ids:
        return evidence_summary

    evidence = _collect_evidence(state)
    if not evidence:
        return (
            "사용 가능한 근거 데이터가 제한적입니다. "
            "해당 주장에 대한 충분한 근거 데이터가 없어 제한적으로 해석해야 합니다."
        )

    lines = []
    for index, item in enumerate(evidence[:10], start=1):
        lines.append(
            f"{index}. evidence_id={item.evidence_id}; source_type={item.source_type}; "
            f"metric_name={item.metric_name}; metric_value={item.metric_value}; "
            f"description={item.description}"
        )

    graph_context = state.get("graph_context")
    if graph_context is not None and graph_context.evidence_ids:
        allowed_ids = _allowed_evidence_ids(state)
        graph_ids = [evidence_id for evidence_id in graph_context.evidence_ids if evidence_id in allowed_ids]
        if graph_ids:
            lines.append("GraphContext 연결 EvidenceItem ID: " + ", ".join(graph_ids[:10]))
    return "\n".join(lines)


def _revision_note(state: GraphState) -> str:
    evaluation = state.get("evaluation_result")
    if evaluation is None:
        return "평가 결과가 없어 기본 안전 보정만 적용했습니다."

    issues = _safe_text("; ".join(evaluation.issues)) if evaluation.issues else "명시된 이슈 없음"
    suggestions = (
        _safe_text("; ".join(evaluation.revision_suggestions))
        if evaluation.revision_suggestions
        else "명시된 수정 제안 없음"
    )
    return f"평가 이슈: {issues}\n수정 제안 반영: {suggestions}"


def run_rewrite_agent(state: GraphState) -> dict:
    """Revise a report based on evaluator feedback while preserving evidence."""
    report = _report_from_state(state)
    if report is None:
        error = WorkflowError(
            node=REWRITE_NODE,
            message="No report is available for rewrite.",
            error_type="rewrite_input_error",
            recoverable=False,
        )
        return {"status": "failed", "errors": [*state.get("errors", []), error]}

    rewrite_attempts = state.get("rewrite_attempts", 0) + 1
    allowed_ids = _allowed_evidence_ids(state)
    revised_report = report.model_copy(
        update={
            "executive_summary": _strip_unknown_evidence_refs(report.executive_summary, allowed_ids),
            "market_section": _strip_unknown_evidence_refs(report.market_section, allowed_ids),
            "fundamental_section": _strip_unknown_evidence_refs(report.fundamental_section, allowed_ids),
            "news_section": _strip_unknown_evidence_refs(report.news_section, allowed_ids),
            "graph_context_section": _strip_unknown_evidence_refs(
                _ensure_graph_context_section(report, state),
                allowed_ids,
            ),
            "scenario_analysis": _strip_unknown_evidence_refs(report.scenario_analysis, allowed_ids),
            "risk_factors": _strip_unknown_evidence_refs(_ensure_risk_factors(report, state), allowed_ids),
            "limitations": _strip_unknown_evidence_refs(_ensure_limitations(report, state), allowed_ids),
            "evidence_summary": _ensure_evidence_summary(report, state),
            "disclaimer": REQUIRED_DISCLAIMER,
        }
    )

    revision_note = _revision_note(state)
    if revision_note not in revised_report.limitations:
        revised_report = revised_report.model_copy(
            update={
                "limitations": revised_report.limitations + "\n" + revision_note,
            }
        )

    return {
        "status": "success",
        "draft_report": revised_report,
        "rewrite_attempts": rewrite_attempts,
    }
