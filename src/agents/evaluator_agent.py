"""Evaluator agent for grounding and safety review."""

from datetime import date
from typing import Any

from src.graph.state import EvaluationResult, GraphState, ResearchReport, WorkflowError
from src.safety.safety_checker import (
    find_forbidden_phrases,
    has_limitations,
    has_required_disclaimer,
    has_risk_disclosure,
)


EVALUATOR_NODE = "evaluator_agent"


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


def _source_grounding_score(state: GraphState, report: ResearchReport) -> float:
    evidence_count = len(state.get("evidence", []))
    if not _has_evidence_summary(report):
        return 0.0
    if evidence_count >= 5:
        return 1.0
    if evidence_count >= 3:
        return 0.8
    if evidence_count >= 1:
        return 0.5
    return 0.0


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
    has_disclaimer = has_required_disclaimer(report)
    has_risks = has_risk_disclosure(report)
    has_limits = has_limitations(report)
    has_evidence = _has_evidence_summary(report)

    issues: list[str] = []
    revision_suggestions: list[str] = []
    if forbidden_matches:
        issues.append(f"금지된 투자 권유 문구가 포함되어 있습니다: {', '.join(forbidden_matches)}")
        revision_suggestions.append("직접적인 매수/매도/수익 보장 표현을 제거하고 시나리오 기반 문장으로 바꾸세요.")

    _append_issue(
        not has_disclaimer,
        issue="필수 고지문이 누락되었습니다.",
        suggestion="정확한 필수 고지문을 disclaimer 섹션에 추가하세요.",
        issues=issues,
        revision_suggestions=revision_suggestions,
    )
    _append_issue(
        not has_risks,
        issue="리스크 공시가 부족합니다.",
        suggestion="가격 변동성, 데이터 지연, 기업/거시 이벤트 등 주요 리스크를 추가하세요.",
        issues=issues,
        revision_suggestions=revision_suggestions,
    )
    _append_issue(
        not has_limits,
        issue="분석 한계가 누락되었습니다.",
        suggestion="데이터 범위, 모델 한계, 뉴스 검색 한계 등 제한 사항을 명시하세요.",
        issues=issues,
        revision_suggestions=revision_suggestions,
    )
    _append_issue(
        not has_evidence,
        issue="근거 요약이 누락되었습니다.",
        suggestion="시장, 재무, 뉴스 근거를 요약한 evidence_summary 섹션을 추가하세요.",
        issues=issues,
        revision_suggestions=revision_suggestions,
    )

    source_grounding_score = _source_grounding_score(state, report)
    numeric_consistency_score = _numeric_consistency_score(report)
    safety_score = 0.0 if forbidden_matches else (1.0 if has_disclaimer else 0.5)
    risk_disclosure_score = 1.0 if has_risks and has_limits else (0.5 if has_risks or has_limits else 0.0)
    freshness_score = _freshness_score(report)

    if source_grounding_score < 0.5:
        issues.append("EvidenceItem 기반 근거가 부족합니다.")
        revision_suggestions.append("보고서의 핵심 주장과 연결된 EvidenceItem을 추가하세요.")

    critical_checks_pass = all(
        [
            not forbidden_matches,
            has_disclaimer,
            has_risks,
            has_limits,
            has_evidence,
            source_grounding_score >= 0.5,
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
