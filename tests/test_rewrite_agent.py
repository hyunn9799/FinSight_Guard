"""Tests for evaluator-driven rewrite behavior."""

from datetime import UTC, date, datetime

from src.agents.rewrite_agent import run_rewrite_agent
from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import EvaluationResult, GraphState, ResearchReport, SupervisorPlan
from src.graph_rag.graph_schema import GraphContext, GraphEdge, GraphNode
from src.safety.forbidden_phrases import REQUIRED_DISCLAIMER


def _evidence(evidence_id: str = "market_001") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_type="market",
        source_name="test",
        source_url=None,
        collected_at=datetime(2026, 5, 12, tzinfo=UTC),
        ticker="AAPL",
        metric_name="RSI",
        metric_value=58.2,
        description="RSI는 중립 구간입니다.",
    )


def _evaluation(issues: list[str] | None = None) -> EvaluationResult:
    return EvaluationResult(
        overall_pass=False,
        source_grounding_score=0.4,
        numeric_consistency_score=0.8,
        safety_score=0.4,
        risk_disclosure_score=0.4,
        freshness_score=1.0,
        issues=issues or ["수정 필요"],
        revision_suggestions=["안전 문구와 근거 요약을 보강하세요."],
    )


def _report(**overrides: str) -> ResearchReport:
    values = {
        "title": "AAPL Research Report",
        "ticker": "AAPL",
        "data_date": date(2026, 5, 12),
        "executive_summary": "교육 목적의 리서치 요약입니다.",
        "market_section": "RSI는 58.2입니다. [근거: market_001]",
        "fundamental_section": "재무 분석은 제한적으로 참고할 수 있습니다.",
        "news_section": "뉴스 분석은 제한적으로 참고할 수 있습니다.",
        "graph_context_section": "",
        "scenario_analysis": "관망 시나리오, 분할 접근 시나리오, 리스크 회피 시나리오를 비교합니다.",
        "risk_factors": "가격 변동성과 데이터 지연 리스크가 있습니다.",
        "limitations": "공개 데이터 기반 분석입니다.",
        "evidence_summary": (
            "1. evidence_id=market_001; source_type=market; metric_name=RSI; "
            "metric_value=58.2; description=RSI는 중립 구간입니다."
        ),
        "disclaimer": REQUIRED_DISCLAIMER,
    }
    values.update(overrides)
    return ResearchReport(**values)


def _state(report: ResearchReport, *, evidence: list[EvidenceItem] | None = None) -> GraphState:
    return {
        "ticker": "AAPL",
        "draft_report": report,
        "evaluation_result": _evaluation(),
        "evidence": [] if evidence is None else evidence,
    }


def _report_text(report: ResearchReport) -> str:
    return "\n".join(str(value) for value in report.model_dump().values())


def _graph_context() -> GraphContext:
    edge = GraphEdge(
        source_id="risk:규제",
        target_id="company:AAPL",
        relation_type="negative_risk",
        evidence_id="market_001",
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
        evidence_ids=["market_001"],
    )


def test_rewrite_removes_forbidden_phrase() -> None:
    state = _state(
        _report(executive_summary="이 종목은 무조건 매수 관점입니다."),
        evidence=[_evidence()],
    )

    result = run_rewrite_agent(state)
    report = result["draft_report"]

    assert "무조건 매수" not in _report_text(report)
    assert "검토할 수 있는 시나리오" in report.executive_summary


def test_rewrite_adds_required_disclaimer_when_missing() -> None:
    state = _state(_report(disclaimer=""), evidence=[_evidence()])

    result = run_rewrite_agent(state)

    assert result["draft_report"].disclaimer == REQUIRED_DISCLAIMER


def test_rewrite_adds_limitation_when_evidence_is_missing() -> None:
    state = _state(_report(evidence_summary=""), evidence=[])

    result = run_rewrite_agent(state)
    report = result["draft_report"]

    assert "사용 가능한 근거 데이터가 제한적입니다." in report.limitations
    assert "충분한 근거 데이터가 없어 제한적으로 해석해야 합니다" in report.evidence_summary


def test_rewrite_adds_graph_context_summary_when_graph_context_exists() -> None:
    state = _state(
        _report(graph_context_section="", risk_factors="가격 변동성 리스크가 있습니다."),
        evidence=[_evidence()],
    )
    state["graph_context"] = _graph_context()

    result = run_rewrite_agent(state)
    report = result["draft_report"]

    assert "관계 기반 리스크 및 근거 요약" in report.graph_context_section
    assert "risk:규제 -> company:AAPL: negative_risk" in report.graph_context_section
    assert "market_001" in report.graph_context_section
    assert "GraphContext 리스크 관계도 함께 확인해야 합니다" in report.risk_factors


def test_rewrite_adds_skipped_agent_limitation() -> None:
    state = _state(_report(), evidence=[_evidence()])
    state["supervisor_plan"] = SupervisorPlan(
        next_node="coordinator_node",
        question_type="technical_analysis",
        execution_mode="selective",
        planned_agent_order=["market", "news"],
        skipped_agents=["fundamental"],
    )
    state["skipped_agents"] = ["fundamental"]

    result = run_rewrite_agent(state)
    report = result["draft_report"]

    assert "Supervisor 계획에 따라 일부 분석만 수행되어 근거 범위가 제한적입니다." in report.limitations
    assert "재무 분석은 상세 실행 대상에서 제외되었습니다" in report.limitations


def test_rewrite_increments_rewrite_attempts() -> None:
    state = _state(_report(), evidence=[_evidence()])
    state["rewrite_attempts"] = 1

    result = run_rewrite_agent(state)

    assert result["rewrite_attempts"] == 2


def test_rewrite_does_not_keep_unknown_evidence_id() -> None:
    state = _state(
        _report(
            market_section="RSI는 58.2입니다. [근거: market_999]",
            evidence_summary=(
                "1. evidence_id=market_999; source_type=market; metric_name=RSI; "
                "metric_value=58.2; description=존재하지 않는 근거"
            ),
        ),
        evidence=[_evidence("market_001")],
    )

    result = run_rewrite_agent(state)
    report = result["draft_report"]
    text = _report_text(report)

    assert "market_999" not in text
    assert "market_001" in report.evidence_summary
