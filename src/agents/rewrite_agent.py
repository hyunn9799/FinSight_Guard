"""Rewrite agent for evaluator-driven report revision."""

from src.graph.state import GraphState, ResearchReport, WorkflowError
from src.safety.forbidden_phrases import FORBIDDEN_PHRASES, REQUIRED_DISCLAIMER


REWRITE_NODE = "rewrite_agent"

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
}


def _report_from_state(state: GraphState) -> ResearchReport | None:
    return state.get("draft_report") or state.get("final_report")


def _safe_text(text: str) -> str:
    rewritten = text or ""
    for phrase in FORBIDDEN_PHRASES:
        rewritten = rewritten.replace(phrase, SAFE_REPLACEMENTS[phrase])
    return rewritten


def _ensure_risk_factors(report: ResearchReport) -> str:
    risk_factors = _safe_text(report.risk_factors).strip()
    if risk_factors:
        return risk_factors
    return (
        "가격 변동성, 실적 발표, 거시 환경 변화, 뉴스 이벤트 해석 차이, 데이터 지연 또는 누락이 "
        "보고서 해석에 영향을 줄 수 있습니다. 손실 가능성을 포함한 리스크를 검토할 수 있습니다."
    )


def _ensure_limitations(report: ResearchReport) -> str:
    limitations = _safe_text(report.limitations).strip()
    if limitations:
        return limitations
    return (
        "본 분석은 공개 데이터, 수집 가능한 EvidenceItem, 단순 규칙 기반 분류에 의존합니다. "
        "데이터 지연, 누락 필드, mock 뉴스 fallback, 모델 한계로 인해 실제 상황을 완전히 반영하지 못할 수 있습니다."
    )


def _ensure_evidence_summary(report: ResearchReport, state: GraphState) -> str:
    evidence_summary = _safe_text(report.evidence_summary).strip()
    if evidence_summary:
        return evidence_summary

    evidence = state.get("evidence", [])
    if not evidence:
        return (
            "현재 보고서에 연결된 EvidenceItem이 부족합니다. 시장, 재무, 뉴스 근거가 추가되면 "
            "핵심 주장과 수치의 출처를 더 명확히 검토할 수 있습니다."
        )

    lines = []
    for index, item in enumerate(evidence[:10], start=1):
        value = "" if item.metric_value is None else f" 값={item.metric_value}"
        lines.append(f"{index}. [{item.source_type}] {item.metric_name}{value} - {item.description}")
    return "\n".join(lines)


def _revision_note(state: GraphState) -> str:
    evaluation = state.get("evaluation_result")
    if evaluation is None:
        return "평가 결과가 없어 기본 안전 보정만 적용했습니다."

    issues = "; ".join(evaluation.issues) if evaluation.issues else "명시된 이슈 없음"
    suggestions = (
        "; ".join(evaluation.revision_suggestions)
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
    revised_report = report.model_copy(
        update={
            "executive_summary": _safe_text(report.executive_summary),
            "market_section": _safe_text(report.market_section),
            "fundamental_section": _safe_text(report.fundamental_section),
            "news_section": _safe_text(report.news_section),
            "scenario_analysis": _safe_text(report.scenario_analysis),
            "risk_factors": _ensure_risk_factors(report),
            "limitations": _ensure_limitations(report),
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
