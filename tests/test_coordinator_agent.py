"""Tests for Coordinator Agent report generation."""

from datetime import UTC, datetime

from src.agents.coordinator_agent import run_coordinator_agent
from src.evidence.evidence_builder import build_market_evidence
from src.graph.state import MarketAnalysis, NewsAnalysis, SupervisorPlan, UserInput
from src.graph_rag.graph_schema import GraphContext, GraphEdge, GraphNode
from src.safety.forbidden_phrases import FORBIDDEN_PHRASES, REQUIRED_DISCLAIMER


EXTRA_FORBIDDEN_PHRASES = [
    "매수 추천",
    "매도 추천",
    "강력 매수",
    "반드시 사야",
    "수익 보장",
    "손실 없음",
    "확실한 수익",
]


def _collected_at() -> datetime:
    return datetime(2026, 5, 12, tzinfo=UTC)


def _market_evidence():
    return build_market_evidence(
        ticker="AAPL",
        metric_name="RSI",
        metric_value=54.2,
        description="AAPL RSI 지표는 단기 추세 확인에 사용됩니다.",
        collected_at=_collected_at(),
    )


def _graph_context(evidence_id: str) -> GraphContext:
    risk_edge = GraphEdge(
        source_id="risk:규제",
        target_id="company:AAPL",
        relation_type="negative_risk",
        evidence_id=evidence_id,
        description="규제 요인이 AAPL 리스크로 언급되었습니다.",
    )
    positive_edge = GraphEdge(
        source_id="event:positive_signal",
        target_id="company:AAPL",
        relation_type="positive_driver",
        evidence_id=evidence_id,
        description="긍정 요인이 AAPL 근거로 언급되었습니다.",
    )
    return GraphContext(
        ticker="AAPL",
        focus="technical",
        nodes=[
            GraphNode(node_id="company:AAPL", node_type="company", name="AAPL"),
            GraphNode(node_id="risk:규제", node_type="risk", name="규제"),
        ],
        edges=[risk_edge, positive_edge],
        key_relations_summary=["규제 -> AAPL: negative_risk"],
        risk_relations=[risk_edge],
        positive_relations=[positive_edge],
        evidence_ids=[evidence_id],
    )


def _report_text(report) -> str:
    return "\n".join(str(value) for value in report.model_dump().values())


def test_coordinator_handles_missing_fundamental_analysis() -> None:
    evidence = _market_evidence()

    result = run_coordinator_agent(
        {
            "ticker": "AAPL",
            "user_input": UserInput(ticker="AAPL", user_query="차트상 단기 진입 괜찮아?"),
            "market_analysis": MarketAnalysis(
                ticker="AAPL",
                summary="시장 요약",
                trend_summary="RSI 기준 단기 추세를 참고할 수 있습니다.",
                evidence=[evidence],
            ),
            "news_analysis": NewsAnalysis(ticker="AAPL", summary="뉴스 요약"),
            "supervisor_plan": SupervisorPlan(
                next_node="coordinator_node",
                question_type="technical_analysis",
                skipped_agents=["fundamental"],
                planned_agent_order=["market", "news"],
            ),
            "skipped_agents": ["fundamental"],
            "evidence": [evidence],
        }
    )

    report = result["draft_report"]

    assert result["status"] == "success"
    assert "재무 분석은 상세 실행 대상에서 제외되었습니다" in report.fundamental_section
    assert "기술적 분석 중심" in report.scenario_analysis
    assert "재무 분석은 상세 실행 대상에서 제외되었습니다" in report.limitations


def test_coordinator_handles_missing_news_analysis() -> None:
    evidence = _market_evidence()

    result = run_coordinator_agent(
        {
            "ticker": "AAPL",
            "market_analysis": MarketAnalysis(ticker="AAPL", summary="시장 요약", evidence=[evidence]),
            "supervisor_plan": SupervisorPlan(
                next_node="coordinator_node",
                question_type="technical_analysis",
                skipped_agents=["news"],
                planned_agent_order=["market"],
            ),
            "skipped_agents": ["news"],
            "evidence": [evidence],
        }
    )

    report = result["draft_report"]

    assert "뉴스 분석은 상세 실행 대상에서 제외되었습니다" in report.news_section
    assert "뉴스 분석은 상세 실행 대상에서 제외되었습니다" in report.limitations


def test_coordinator_includes_graph_context_section() -> None:
    evidence = _market_evidence()
    context = _graph_context(evidence.evidence_id)

    result = run_coordinator_agent(
        {
            "ticker": "AAPL",
            "market_analysis": MarketAnalysis(ticker="AAPL", summary="시장 요약", evidence=[evidence]),
            "graph_context": context,
            "evidence": [evidence],
        }
    )

    report = result["draft_report"]

    assert "관계 기반 리스크 및 근거 요약" in report.graph_context_section
    assert "규제 -> AAPL: negative_risk" in report.graph_context_section
    assert evidence.evidence_id in report.graph_context_section
    assert "관계 기반 리스크 및 근거 요약" in report.evidence_summary


def test_report_contains_evidence_id_when_evidence_exists() -> None:
    evidence = _market_evidence()

    result = run_coordinator_agent(
        {
            "ticker": "AAPL",
            "market_analysis": MarketAnalysis(
                ticker="AAPL",
                summary="시장 요약",
                trend_summary="RSI는 54.2로 중립 구간 참고 지표입니다.",
                evidence=[evidence],
            ),
            "evidence": [evidence],
        }
    )

    report = result["draft_report"]
    text = _report_text(report)

    assert evidence.evidence_id in text
    assert f"[근거: {evidence.evidence_id}]" in report.market_section
    assert f"evidence_id={evidence.evidence_id}" in report.evidence_summary
    assert "source_type=market" in report.evidence_summary
    assert "metric_name=RSI" in report.evidence_summary
    assert "metric_value=54.2" in report.evidence_summary


def test_report_does_not_contain_fake_evidence_ids() -> None:
    evidence = _market_evidence()
    fake_evidence_id = "fake_999"

    result = run_coordinator_agent(
        {
            "ticker": "AAPL",
            "market_analysis": MarketAnalysis(
                ticker="AAPL",
                summary="시장 요약",
                evidence=[evidence],
            ),
            "evidence": [evidence],
        }
    )

    assert fake_evidence_id not in _report_text(result["draft_report"])


def test_limitation_is_added_when_no_evidence_exists() -> None:
    result = run_coordinator_agent(
        {
            "ticker": "AAPL",
            "supervisor_plan": SupervisorPlan(
                next_node="coordinator_node",
                question_type="safety_or_unclear",
            ),
        }
    )

    report = result["draft_report"]

    assert "사용 가능한 근거 데이터가 제한적입니다." in report.limitations
    assert "사용 가능한 근거 데이터가 제한적입니다." in report.evidence_summary
    assert "EvidenceItem ID가 없습니다" in report.evidence_summary


def test_partial_evidence_case_is_handled_gracefully() -> None:
    evidence = _market_evidence()

    result = run_coordinator_agent(
        {
            "ticker": "AAPL",
            "market_analysis": MarketAnalysis(
                ticker="AAPL",
                summary="시장 요약",
                evidence=[evidence],
            ),
            "supervisor_plan": SupervisorPlan(
                next_node="coordinator_node",
                question_type="technical_analysis",
                execution_mode="selective",
                skipped_agents=["fundamental"],
                planned_agent_order=["market", "news"],
            ),
            "skipped_agents": ["fundamental"],
            "evidence": [evidence],
        }
    )

    report = result["draft_report"]

    assert result["status"] == "success"
    assert evidence.evidence_id in _report_text(report)
    assert "Supervisor 계획에 따라 일부 분석만 수행되어 근거 범위가 제한적입니다." in report.limitations
    assert "재무 분석은 상세 실행 대상에서 제외되었습니다" in report.limitations


def test_coordinator_mentions_failed_agent_as_degraded_data() -> None:
    result = run_coordinator_agent(
        {
            "ticker": "AAPL",
            "supervisor_plan": SupervisorPlan(
                next_node="coordinator_node",
                question_type="fundamental_analysis",
                failed_agents=["fundamental"],
            ),
            "failed_agents": ["fundamental"],
        }
    )

    report = result["draft_report"]

    assert "재무 분석 데이터 수집이 제한되어 해당 섹션은 제한적으로 해석해야 합니다." in report.fundamental_section
    assert "재무 분석 데이터 수집이 제한되어 해당 섹션은 제한적으로 해석해야 합니다." in report.limitations


def test_coordinator_includes_disclaimer_and_avoids_forbidden_phrases() -> None:
    result = run_coordinator_agent(
        {
            "ticker": "AAPL",
            "supervisor_plan": SupervisorPlan(
                next_node="coordinator_node",
                question_type="safety_or_unclear",
            ),
        }
    )

    report = result["draft_report"]
    text = _report_text(report)

    assert report.disclaimer == REQUIRED_DISCLAIMER
    for phrase in [*FORBIDDEN_PHRASES, *EXTRA_FORBIDDEN_PHRASES]:
        assert phrase not in text
