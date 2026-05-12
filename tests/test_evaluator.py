"""Tests for evaluator agent behavior."""

from datetime import UTC, date, datetime

from src.agents.evaluator_agent import run_evaluator_agent
from src.evidence.evidence_builder import build_market_evidence
from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import GraphState, ResearchReport
from src.safety.forbidden_phrases import REQUIRED_DISCLAIMER


def _evidence_items(count: int = 5) -> list[EvidenceItem]:
    return [
        build_market_evidence(
            ticker="AAPL",
            metric_name=f"metric_{index}",
            metric_value=float(index),
            description=f"Evidence item {index}",
            collected_at=datetime(2026, 5, 12, tzinfo=UTC),
        )
        for index in range(count)
    ]


def _report(
    *,
    executive_summary: str = "교육 목적의 리서치 요약입니다.",
    risk_factors: str = "가격 변동성, 실적 발표, 거시 환경 변화 리스크가 있습니다.",
    limitations: str = "본 분석은 공개 데이터와 제한된 뉴스 검색 결과에 기반합니다.",
    evidence_summary: str = "시장, 재무, 뉴스 EvidenceItem을 기준으로 작성했습니다.",
    disclaimer: str = REQUIRED_DISCLAIMER,
) -> ResearchReport:
    return ResearchReport(
        title="AAPL Research Report",
        ticker="AAPL",
        data_date=date.today(),
        executive_summary=executive_summary,
        market_section="MA20, RSI 등 시장 지표를 요약했습니다.",
        fundamental_section="PER, ROE 등 재무 지표를 요약했습니다.",
        news_section="뉴스 이벤트를 요약했습니다.",
        scenario_analysis="관망 시나리오, 분할 접근 시나리오, 리스크 회피 시나리오를 비교합니다.",
        risk_factors=risk_factors,
        limitations=limitations,
        evidence_summary=evidence_summary,
        disclaimer=disclaimer,
    )


def _state(report: ResearchReport, evidence_count: int = 5) -> GraphState:
    return {
        "ticker": report.ticker,
        "draft_report": report,
        "evidence": _evidence_items(evidence_count),
    }


def test_unsafe_report_with_forbidden_phrase_fails() -> None:
    result = run_evaluator_agent(
        _state(_report(executive_summary="이 종목은 무조건 매수 관점으로 봐야 합니다."))
    )

    evaluation = result["evaluation_result"]

    assert evaluation.overall_pass is False
    assert evaluation.safety_score == 0.0
    assert any("금지된 투자 권유 문구" in issue for issue in evaluation.issues)


def test_report_without_disclaimer_fails() -> None:
    result = run_evaluator_agent(_state(_report(disclaimer="")))

    evaluation = result["evaluation_result"]

    assert evaluation.overall_pass is False
    assert evaluation.safety_score < 1.0
    assert "필수 고지문이 누락되었습니다." in evaluation.issues


def test_report_without_risks_fails() -> None:
    result = run_evaluator_agent(_state(_report(risk_factors="")))

    evaluation = result["evaluation_result"]

    assert evaluation.overall_pass is False
    assert evaluation.risk_disclosure_score < 1.0
    assert "리스크 공시가 부족합니다." in evaluation.issues


def test_safe_report_passes() -> None:
    result = run_evaluator_agent(_state(_report()))

    evaluation = result["evaluation_result"]

    assert result["status"] == "success"
    assert evaluation.overall_pass is True
    assert evaluation.source_grounding_score == 1.0
    assert evaluation.numeric_consistency_score == 1.0
    assert evaluation.safety_score == 1.0
    assert evaluation.risk_disclosure_score == 1.0
    assert evaluation.issues == []
