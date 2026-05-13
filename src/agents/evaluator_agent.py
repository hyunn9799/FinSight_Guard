"""Evaluator agent for grounding and safety review."""

from datetime import date
import re
from typing import Any

from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import AgentName, EvaluationResult, GraphState, ResearchReport, WorkflowError
from src.safety.safety_checker import (
    find_forbidden_phrases,
    has_limitations,
    has_required_disclaimer,
    has_risk_disclosure,
)


EVALUATOR_NODE = "evaluator_agent"
EVIDENCE_ID_PATTERN = re.compile(
    r"(?:근거:\s*|evidence_id=)\s*((?:market|fundamental|news|system)_[A-Za-z0-9_]+)"
)
CERTAINTY_PHRASES = [
    "수익 보장",
    "손실 없음",
    "확실한 수익",
    "원금 보장",
    "목표가 보장",
    "반드시 수익",
    "무위험",
]
AGENT_ANALYSIS_KEYS: dict[AgentName, str] = {
    "market": "market_analysis",
    "fundamental": "fundamental_analysis",
    "news": "news_analysis",
}
AGENT_LABELS: dict[AgentName, str] = {
    "market": "시장",
    "fundamental": "재무",
    "news": "뉴스",
}


def _report_from_state(state: GraphState) -> ResearchReport | None:
    return state.get("draft_report") or state.get("final_report")


def _report_text(report: Any) -> str:
    if report is None:
        return ""
    if hasattr(report, "model_dump"):
        return "\n".join(str(value) for value in report.model_dump().values() if value is not None)
    if isinstance(report, dict):
        return "\n".join(str(value) for value in report.values() if value is not None)
    return str(report)


def _has_evidence_summary(report: ResearchReport) -> bool:
    return bool(report.evidence_summary and report.evidence_summary.strip())


def _clamp_score(score: float) -> float:
    return round(max(0.0, min(1.0, score)), 2)


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


def _referenced_evidence_ids(text: str) -> set[str]:
    return set(EVIDENCE_ID_PATTERN.findall(text or ""))


def _planned_agents(state: GraphState) -> list[AgentName]:
    plan = state.get("supervisor_plan")
    if plan is None:
        return []
    return list(plan.planned_agent_order or plan.required_agents)


def _skipped_agents(state: GraphState) -> set[AgentName]:
    plan = state.get("supervisor_plan")
    skipped = set(state.get("skipped_agents", []))
    if plan is not None:
        skipped.update(plan.skipped_agents)
    return skipped


def _failed_agents(state: GraphState) -> set[AgentName]:
    plan = state.get("supervisor_plan")
    failed = set(state.get("failed_agents", []))
    if plan is not None:
        failed.update(plan.failed_agents)
    return failed


def _limitation_mentions_agent(report: ResearchReport, agent: AgentName) -> bool:
    text = f"{report.limitations}\n{report.evidence_summary}"
    label = AGENT_LABELS[agent]
    return label in text and any(keyword in text for keyword in ("제한", "누락", "실패", "제외"))


def _graph_risk_is_reflected(state: GraphState, report: ResearchReport) -> bool:
    graph_context = state.get("graph_context")
    if graph_context is None or not graph_context.risk_relations:
        return True
    graph_text = "\n".join(
        [
            report.risk_factors,
            report.graph_context_section,
            report.evidence_summary,
        ]
    )
    if not report.graph_context_section.strip():
        return False
    for edge in graph_context.risk_relations:
        markers = [
            edge.source_id,
            edge.target_id,
            edge.relation_type,
            edge.description or "",
        ]
        if any(marker and marker in graph_text for marker in markers):
            return True
    return False


def _source_grounding_score(
    *,
    has_evidence: bool,
    available_evidence_ids: set[str],
    referenced_evidence_ids: set[str],
    unknown_evidence_ids: set[str],
    has_evidence_summary: bool,
    graph_context_ignored: bool,
    planned_missing_hidden: bool,
) -> float:
    score = 1.0
    if not has_evidence_summary:
        score -= 0.5
    if has_evidence and not referenced_evidence_ids:
        score -= 0.4
    if not has_evidence:
        score -= 0.4
    if unknown_evidence_ids:
        score -= 0.8
    if graph_context_ignored:
        score -= 0.4
    if planned_missing_hidden:
        score -= 0.3
    if available_evidence_ids and referenced_evidence_ids & available_evidence_ids:
        score += 0.1
    return _clamp_score(score)


def _numeric_consistency_score(report: ResearchReport) -> float:
    """Placeholder numeric consistency check for MVP.

    The full implementation can cross-check numeric claims against EvidenceItem
    values. For now, reports with an evidence summary get a neutral-pass score.
    """
    return 1.0 if _has_evidence_summary(report) else 0.5


def _freshness_score(report: ResearchReport) -> float:
    age_days = (date.today() - report.data_date).days
    if age_days < 0:
        return 0.7
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.8
    if age_days <= 90:
        return 0.5
    return 0.2


def _append_issue(
    condition: bool,
    *,
    issue: str,
    suggestion: str,
    issues: list[str],
    revision_suggestions: list[str],
) -> None:
    if condition:
        issues.append(issue)
        revision_suggestions.append(suggestion)


def run_evaluator_agent(state: GraphState) -> dict:
    """Evaluate report safety, grounding, and required responsible-AI sections."""
    report = _report_from_state(state)
    if report is None:
        error = WorkflowError(
            node=EVALUATOR_NODE,
            message="No report is available for evaluation.",
            error_type="evaluation_input_error",
            recoverable=False,
        )
        failed_result = EvaluationResult(
            overall_pass=False,
            source_grounding_score=0.0,
            numeric_consistency_score=0.0,
            safety_score=0.0,
            risk_disclosure_score=0.0,
            freshness_score=0.0,
            issues=["평가할 보고서가 없습니다."],
            revision_suggestions=["Coordinator Agent가 초안 보고서를 생성한 뒤 평가를 다시 실행하세요."],
        )
        return {
            "status": "failed",
            "evaluation_result": failed_result,
            "errors": [*state.get("errors", []), error],
        }

    text = _report_text(report)
    forbidden_matches = find_forbidden_phrases(text)
    certainty_matches = [phrase for phrase in CERTAINTY_PHRASES if phrase in text]
    has_disclaimer = has_required_disclaimer(report)
    has_risks = has_risk_disclosure(report)
    has_limits = has_limitations(report)
    has_evidence = _has_evidence_summary(report)
    evidence_items = _collect_evidence(state)
    available_evidence_ids = _allowed_evidence_ids(state)
    referenced_evidence_ids = _referenced_evidence_ids(text)
    unknown_evidence_ids = referenced_evidence_ids - available_evidence_ids
    graph_context = state.get("graph_context")
    graph_context_ignored = graph_context is not None and not _graph_risk_is_reflected(state, report)
    skipped_agents = _skipped_agents(state)
    failed_agents = _failed_agents(state)
    planned_agents = _planned_agents(state)
    missing_planned_agents = [
        agent
        for agent in planned_agents
        if agent not in skipped_agents
        and state.get(AGENT_ANALYSIS_KEYS[agent]) is None
    ]
    hidden_missing_or_failed_agents = [
        agent
        for agent in [*missing_planned_agents, *failed_agents]
        if agent not in skipped_agents and not _limitation_mentions_agent(report, agent)
    ]

    issues: list[str] = []
    revision_suggestions: list[str] = []
    critical_issues: list[str] = []
    if forbidden_matches:
        issue = f"금지된 투자 권유 문구가 포함되어 있습니다: {', '.join(forbidden_matches)}"
        issues.append(issue)
        critical_issues.append(issue)
        revision_suggestions.append("직접적인 매수/매도/수익 보장 표현을 제거하고 시나리오 기반 문장으로 바꾸세요.")
    if certainty_matches:
        issue = f"확정적 수익 또는 원금 보장 표현이 포함되어 있습니다: {', '.join(certainty_matches)}"
        issues.append(issue)
        critical_issues.append(issue)
        revision_suggestions.append("성과를 보장하거나 확실성을 암시하는 표현을 제거하세요.")

    _append_issue(
        not has_disclaimer,
        issue="필수 고지문이 누락되었습니다.",
        suggestion="정확한 필수 고지문을 disclaimer 섹션에 추가하세요.",
        issues=issues,
        revision_suggestions=revision_suggestions,
    )
    if not has_disclaimer:
        critical_issues.append("필수 고지문이 누락되었습니다.")
    _append_issue(
        not has_risks,
        issue="리스크 공시가 부족합니다.",
        suggestion="가격 변동성, 데이터 지연, 기업/거시 이벤트 등 주요 리스크를 추가하세요.",
        issues=issues,
        revision_suggestions=revision_suggestions,
    )
    if not has_risks:
        critical_issues.append("리스크 공시가 부족합니다.")
    _append_issue(
        not has_limits,
        issue="분석 한계가 누락되었습니다.",
        suggestion="데이터 범위, 모델 한계, 뉴스 검색 한계 등 제한 사항을 명시하세요.",
        issues=issues,
        revision_suggestions=revision_suggestions,
    )
    if not has_limits:
        critical_issues.append("분석 한계가 누락되었습니다.")
    _append_issue(
        not has_evidence,
        issue="근거 요약이 누락되었습니다.",
        suggestion="시장, 재무, 뉴스 근거를 요약한 evidence_summary 섹션을 추가하세요.",
        issues=issues,
        revision_suggestions=revision_suggestions,
    )
    if referenced_evidence_ids and unknown_evidence_ids:
        issue = "보고서가 state에 없는 EvidenceItem ID를 참조합니다: " + ", ".join(
            sorted(unknown_evidence_ids)
        )
        issues.append(issue)
        critical_issues.append(issue)
        revision_suggestions.append("보고서의 근거 표기는 state에 존재하는 EvidenceItem ID만 사용하세요.")
    if evidence_items and not (referenced_evidence_ids & available_evidence_ids):
        issues.append("사용 가능한 EvidenceItem ID가 보고서 본문에 참조되지 않았습니다.")
        revision_suggestions.append("중요 주장 옆에 [근거: evidence_id] 형식으로 출처를 표시하세요.")
    for agent in missing_planned_agents:
        issue = f"계획된 {AGENT_LABELS[agent]} 결과가 누락되었습니다."
        issues.append(issue)
        revision_suggestions.append(f"{AGENT_LABELS[agent]} 누락 사유를 한계 섹션에 명시하세요.")
    for agent in hidden_missing_or_failed_agents:
        issue = f"계획된 {AGENT_LABELS[agent]} 누락 또는 실패가 limitations에 명시되지 않았습니다."
        issues.append(issue)
        critical_issues.append(issue)
        revision_suggestions.append(f"{AGENT_LABELS[agent]} 제한 사항을 limitations에 추가하세요.")
    if graph_context is not None and not report.graph_context_section.strip():
        issues.append("GraphContext가 있지만 graph_context_section이 비어 있습니다.")
        revision_suggestions.append("관계 기반 리스크 및 근거 요약 섹션을 추가하세요.")
    if graph_context_ignored:
        issue = "GraphContext 리스크 관계가 risk_factors 또는 graph context 요약에 반영되지 않았습니다."
        issues.append(issue)
        critical_issues.append(issue)
        revision_suggestions.append("graph_context.risk_relations를 리스크 또는 관계 요약 섹션에 반영하세요.")

    source_grounding_score = _source_grounding_score(
        has_evidence=bool(evidence_items or available_evidence_ids),
        available_evidence_ids=available_evidence_ids,
        referenced_evidence_ids=referenced_evidence_ids,
        unknown_evidence_ids=unknown_evidence_ids,
        has_evidence_summary=has_evidence,
        graph_context_ignored=graph_context_ignored,
        planned_missing_hidden=bool(hidden_missing_or_failed_agents),
    )
    numeric_consistency_score = _numeric_consistency_score(report)
    safety_score = 1.0
    if forbidden_matches:
        safety_score -= 1.0
    if certainty_matches:
        safety_score -= 0.7
    if not has_disclaimer:
        safety_score -= 0.5
    safety_score = _clamp_score(safety_score)
    risk_disclosure_score = 1.0
    if not has_risks:
        risk_disclosure_score -= 0.7
    if not has_limits:
        risk_disclosure_score -= 0.5
    if graph_context_ignored:
        risk_disclosure_score -= 0.3
    risk_disclosure_score = _clamp_score(risk_disclosure_score)
    freshness_score = _freshness_score(report)

    if source_grounding_score < 0.7:
        issues.append("EvidenceItem 기반 근거가 부족합니다.")
        revision_suggestions.append("보고서의 핵심 주장과 연결된 EvidenceItem을 추가하세요.")

    critical_checks_pass = not critical_issues and all(
        [
            has_disclaimer,
            has_risks,
            has_limits,
            has_evidence,
            source_grounding_score >= 0.7,
            safety_score >= 0.7,
            risk_disclosure_score >= 0.7,
        ]
    )
    result = EvaluationResult(
        overall_pass=critical_checks_pass,
        source_grounding_score=source_grounding_score,
        numeric_consistency_score=numeric_consistency_score,
        safety_score=safety_score,
        risk_disclosure_score=risk_disclosure_score,
        freshness_score=freshness_score,
        issues=issues,
        revision_suggestions=revision_suggestions,
    )
    return {
        "status": "success" if result.overall_pass else "degraded",
        "evaluation_result": result,
    }
