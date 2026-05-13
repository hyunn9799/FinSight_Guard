"""Tests for evaluator agent behavior."""

from datetime import UTC, date, datetime

from src.agents.evaluator_agent import run_evaluator_agent
from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import GraphState, MarketAnalysis, ResearchReport, SupervisorPlan
from src.graph_rag.graph_schema import GraphContext, GraphEdge, GraphNode
from src.safety.forbidden_phrases import REQUIRED_DISCLAIMER


def _evidence_items(count: int = 5) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            evidence_id=f"market_{index:03d}",
            source_type="market",
            source_name="test",
            source_url=None,
            collected_at=datetime(2026, 5, 12, tzinfo=UTC),
            ticker="AAPL",
            metric_name=f"metric_{index}",
            metric_value=float(index),
            description=f"Evidence item {index}",
        )
        for index in range(count)
    ]


def _report(
    *,
    executive_summary: str = "교육 목적의 리서치 요약입니다.",
    market_section: str = "MA20, RSI 등 시장 지표를 요약했습니다. [근거: market_000]",
    risk_factors: str = "가격 변동성, 실적 발표, 거시 환경 변화 리스크가 있습니다.",
    limitations: str = "본 분석은 공개 데이터와 제한된 뉴스 검색 결과에 기반합니다.",
    evidence_summary: str = (
        "1. evidence_id=market_000; source_type=market; metric_name=metric_0; "
        "metric_value=0.0; description=Evidence item 0"
    ),
    disclaimer: str = REQUIRED_DISCLAIMER,
    graph_context_section: str = "",
) -> ResearchReport:
    return ResearchReport(
        title="AAPL Research Report",
        ticker="AAPL",
        data_date=date.today(),
        executive_summary=executive_summary,
        market_section=market_section,
        fundamental_section="PER, ROE 등 재무 지표를 요약했습니다.",
        news_section="뉴스 이벤트를 요약했습니다.",
        graph_context_section=graph_context_section,
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


def test_report_without_evidence_summary_fails_source_grounding() -> None:
    result = run_evaluator_agent(_state(_report(evidence_summary=""), evidence_count=5))

    evaluation = result["evaluation_result"]

    assert result["status"] == "degraded"
    assert evaluation.overall_pass is False
    assert evaluation.source_grounding_score < 0.7
    assert "근거 요약이 누락되었습니다." in evaluation.issues
    assert "EvidenceItem 기반 근거가 부족합니다." in evaluation.issues


def test_report_with_too_little_evidence_fails_grounding() -> None:
    result = run_evaluator_agent(_state(_report(), evidence_count=0))

    evaluation = result["evaluation_result"]

    assert evaluation.overall_pass is False
    assert evaluation.source_grounding_score < 0.7
    assert "EvidenceItem 기반 근거가 부족합니다." in evaluation.issues


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


def test_unknown_evidence_id_fails() -> None:
    report = _report(
        market_section="RSI를 요약했습니다. [근거: market_999]",
        evidence_summary=(
            "1. evidence_id=market_999; source_type=market; metric_name=RSI; "
            "metric_value=50; description=unknown"
        ),
    )

    result = run_evaluator_agent(_state(report))
    evaluation = result["evaluation_result"]

    assert evaluation.overall_pass is False
    assert any("state에 없는 EvidenceItem ID" in issue for issue in evaluation.issues)


def test_skipped_fundamental_analysis_does_not_fail() -> None:
    state = _state(_report(limitations="기술적 분석 중심으로 재무 분석은 상세 실행 대상에서 제외되었습니다."))
    state["supervisor_plan"] = SupervisorPlan(
        next_node="coordinator_node",
        question_type="technical_analysis",
        planned_agent_order=["market"],
        skipped_agents=["fundamental"],
    )
    state["skipped_agents"] = ["fundamental"]
    state["market_analysis"] = MarketAnalysis(
        ticker="AAPL",
        summary="시장 요약",
        evidence=state["evidence"],
    )

    result = run_evaluator_agent(state)
    evaluation = result["evaluation_result"]

    assert evaluation.overall_pass is True
    assert not any("재무" in issue and "누락" in issue for issue in evaluation.issues)


def test_planned_market_analysis_missing_creates_failure_when_hidden() -> None:
    state = _state(_report())
    state["supervisor_plan"] = SupervisorPlan(
        next_node="coordinator_node",
        question_type="technical_analysis",
        planned_agent_order=["market"],
    )

    result = run_evaluator_agent(state)
    evaluation = result["evaluation_result"]

    assert evaluation.overall_pass is False
    assert any("계획된 시장 결과가 누락" in issue for issue in evaluation.issues)
    assert any("limitations에 명시되지 않았습니다" in issue for issue in evaluation.issues)


def _graph_context() -> GraphContext:
    edge = GraphEdge(
        source_id="risk:규제",
        target_id="company:AAPL",
        relation_type="negative_risk",
        evidence_id="market_000",
        description="규제 리스크가 언급되었습니다.",
    )
    return GraphContext(
        ticker="AAPL",
        focus="news_risk",
        nodes=[
            GraphNode(node_id="company:AAPL", node_type="company", name="AAPL"),
            GraphNode(node_id="risk:규제", node_type="risk", name="규제"),
        ],
        edges=[edge],
        key_relations_summary=["규제 -> AAPL: negative_risk"],
        risk_relations=[edge],
        positive_relations=[],
        evidence_ids=["market_000"],
    )


def test_graph_context_risk_not_reflected_fails() -> None:
    state = _state(_report(graph_context_section="관계 기반 리스크 및 근거 요약"))
    state["graph_context"] = _graph_context()

    result = run_evaluator_agent(state)
    evaluation = result["evaluation_result"]

    assert evaluation.overall_pass is False
    assert any("GraphContext 리스크 관계" in issue for issue in evaluation.issues)


def test_report_with_graph_context_risk_reflected_passes() -> None:
    state = _state(
        _report(
            risk_factors="가격 변동성과 규제 리스크가 있습니다. [근거: market_000]",
            graph_context_section=(
                "관계 기반 리스크 및 근거 요약\n"
                "리스크 관계:\n"
                "- risk:규제 -> company:AAPL: negative_risk, evidence_id=market_000"
            ),
        )
    )
    state["graph_context"] = _graph_context()

    result = run_evaluator_agent(state)
    evaluation = result["evaluation_result"]

    assert evaluation.overall_pass is True


def test_evaluator_returns_failed_result_when_report_is_missing() -> None:
    result = run_evaluator_agent({"ticker": "AAPL", "evidence": _evidence_items()})

    evaluation = result["evaluation_result"]

    assert result["status"] == "failed"
    assert evaluation.overall_pass is False
    assert result["errors"][0].error_type == "evaluation_input_error"
