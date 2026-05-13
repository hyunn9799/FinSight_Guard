"""Tests for LangGraph routing behavior."""

from datetime import UTC, date, datetime
from pathlib import Path

from src.agents.rewrite_agent import run_rewrite_agent
from src.evidence.evidence_builder import (
    build_fundamental_evidence,
    build_market_evidence,
    build_news_evidence,
)
from src.graph.routing import route_after_evaluation, route_after_supervisor
from src.graph.state import SupervisorPlan
from src.graph.state import (
    EvaluationResult,
    FundamentalAnalysis,
    MarketAnalysis,
    NewsAnalysis,
    ResearchReport,
)
from src.graph.workflow import run_research_workflow
from src.safety.forbidden_phrases import REQUIRED_DISCLAIMER


def _evaluation(overall_pass: bool) -> EvaluationResult:
    return EvaluationResult(
        overall_pass=overall_pass,
        source_grounding_score=1.0 if overall_pass else 0.2,
        numeric_consistency_score=1.0,
        safety_score=1.0 if overall_pass else 0.0,
        risk_disclosure_score=1.0,
        freshness_score=1.0,
        issues=[] if overall_pass else ["failed"],
        revision_suggestions=[] if overall_pass else ["rewrite"],
    )


def _unsafe_report() -> ResearchReport:
    return ResearchReport(
        title="AAPL Research Report",
        ticker="AAPL",
        data_date=date.today(),
        executive_summary="지금 사야 합니다.",
        market_section="시장",
        fundamental_section="재무",
        news_section="뉴스",
        scenario_analysis="관망 시나리오",
        risk_factors="",
        limitations="",
        evidence_summary="",
        disclaimer="",
    )


def _collected_at() -> datetime:
    return datetime(2026, 5, 12, tzinfo=UTC)


def test_evaluator_pass_routes_to_save() -> None:
    state = {"evaluation_result": _evaluation(True), "rewrite_attempts": 0}

    assert route_after_evaluation(state) == "save_report_node"


def test_evaluator_fail_routes_to_rewrite() -> None:
    state = {"evaluation_result": _evaluation(False), "rewrite_attempts": 1}

    assert route_after_evaluation(state) == "rewrite_node"


def test_rewrite_routing_still_retries_until_max_attempts() -> None:
    state = {"evaluation_result": _evaluation(False), "rewrite_attempts": 0}

    assert route_after_evaluation(state) == "rewrite_node"


def test_max_rewrite_count_routes_to_save_failed_report() -> None:
    state = {"evaluation_result": _evaluation(False), "rewrite_attempts": 2}

    assert route_after_evaluation(state) == "save_report_node"


def test_missing_supervisor_plan_routes_to_coordinator_fallback() -> None:
    assert route_after_supervisor({}) == "graph_context_node"


def test_supervisor_plan_routes_to_selected_node() -> None:
    state = {
        "supervisor_plan": SupervisorPlan(
            next_node="market_node",
            rationale="market_analysis가 없어 Market Agent를 먼저 실행합니다.",
        )
    }

    assert route_after_supervisor(state) == "market_node"


def test_completed_supervisor_plan_routes_to_graph_context_node() -> None:
    state = {
        "supervisor_plan": SupervisorPlan(
            next_node="coordinator_node",
            rationale="계획된 분석이 모두 완료되어 Coordinator Agent로 이동합니다.",
        )
    }

    assert route_after_supervisor(state) == "graph_context_node"


def test_rewrite_agent_removes_unsafe_language_and_adds_required_sections() -> None:
    state = {
        "draft_report": _unsafe_report(),
        "evaluation_result": _evaluation(False),
        "rewrite_attempts": 1,
        "evidence": [],
    }

    result = run_rewrite_agent(state)
    report = result["draft_report"]

    assert result["rewrite_attempts"] == 2
    assert "지금 사야 합니다" not in report.executive_summary
    assert report.disclaimer == REQUIRED_DISCLAIMER
    assert report.risk_factors
    assert report.limitations
    assert report.evidence_summary


def test_invalid_ticker_returns_validation_error(monkeypatch, tmp_path) -> None:
    import src.observability.logger as project_logger

    monkeypatch.setattr(project_logger, "LOG_DIR", tmp_path)
    monkeypatch.setattr(project_logger, "_CONFIGURED", False)

    result = run_research_workflow("", "중기", "중립형")

    assert result["run_id"]
    assert result["final_report"] is None
    assert result["report_path"] is None
    assert result["errors"]
    assert result["errors"][0].error_type == "validation_error"


def test_full_workflow_success_path_with_mocked_agents(monkeypatch, tmp_path) -> None:
    import src.graph.workflow as workflow
    import src.observability.logger as project_logger

    calls: list[str] = []
    market_evidence = build_market_evidence(
        ticker="AAPL",
        metric_name="MA20",
        metric_value=180.5,
        description="20일 이동평균",
        collected_at=_collected_at(),
    )
    fundamental_evidence = build_fundamental_evidence(
        ticker="AAPL",
        metric_name="trailingPE",
        metric_value=28.1,
        description="후행 PER",
        collected_at=_collected_at(),
    )
    news_evidence = build_news_evidence(
        ticker="AAPL",
        source_name="Mock News",
        source_url="https://example.com/aapl",
        description="제품 이벤트 리스크 기사",
        collected_at=_collected_at(),
    )

    def fake_market_agent(state: dict) -> dict:
        calls.append("market")
        assert state["ticker"] == "AAPL"
        return {
            "status": "success",
            "evidence": [market_evidence],
            "market_analysis": MarketAnalysis(
                ticker="AAPL",
                summary="시장 지표 요약",
                trend_summary="MA20 기준 추세를 점검했습니다.",
                momentum_summary="RSI 기준 과열 신호는 제한적입니다.",
                volatility_summary="ATR 기준 변동성은 관찰 대상입니다.",
                evidence=[market_evidence],
            ),
        }

    def fake_fundamental_agent(state: dict) -> dict:
        calls.append("fundamental")
        assert state["market_analysis"].ticker == "AAPL"
        evidence = [*state.get("evidence", []), fundamental_evidence]
        return {
            "status": "success",
            "evidence": evidence,
            "fundamental_analysis": FundamentalAnalysis(
                ticker="AAPL",
                summary="펀더멘털 요약",
                valuation_summary="PER을 기준으로 밸류에이션을 비교했습니다.",
                profitability_summary="수익성 지표를 점검했습니다.",
                stability_summary="재무 안정성 항목을 점검했습니다.",
                evidence=[fundamental_evidence],
            ),
        }

    def fake_news_agent(state: dict) -> dict:
        calls.append("news")
        assert state["fundamental_analysis"].ticker == "AAPL"
        evidence = [*state.get("evidence", []), news_evidence]
        return {
            "status": "success",
            "evidence": evidence,
            "news_analysis": NewsAnalysis(
                ticker="AAPL",
                summary="뉴스 요약",
                positive_factors=["제품 수요 관련 긍정 요인"],
                negative_factors=["규제 및 경쟁 리스크"],
                event_risks=["실적 발표 변동성"],
                evidence=[news_evidence],
            ),
        }

    original_coordinator_agent = workflow.run_coordinator_agent
    original_graph_context_builder_node = workflow.run_graph_context_builder_node

    def fake_graph_context_builder_node(state: dict) -> dict:
        calls.append("graph_context")
        return original_graph_context_builder_node(state)

    def fake_coordinator_agent(state: dict) -> dict:
        calls.append("coordinator")
        assert state.get("graph_context") is not None
        return original_coordinator_agent(state)

    def fake_save_report_json(run_id: str, payload: dict) -> str:
        report_path = tmp_path / f"{run_id}_AAPL_report.json"
        report_path.write_text("{}", encoding="utf-8")
        return str(report_path)

    def fake_save_report_markdown(run_id: str, report: ResearchReport) -> str:
        markdown_path = tmp_path / f"{run_id}_AAPL_report.md"
        markdown_path.write_text(report.title, encoding="utf-8")
        return str(markdown_path)

    monkeypatch.setattr(project_logger, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(project_logger, "_CONFIGURED", False)
    monkeypatch.setattr(workflow, "run_market_agent", fake_market_agent)
    monkeypatch.setattr(workflow, "run_fundamental_agent", fake_fundamental_agent)
    monkeypatch.setattr(workflow, "run_news_agent", fake_news_agent)
    monkeypatch.setattr(workflow, "run_graph_context_builder_node", fake_graph_context_builder_node)
    monkeypatch.setattr(workflow, "run_coordinator_agent", fake_coordinator_agent)
    monkeypatch.setattr(workflow, "save_report_json", fake_save_report_json)
    monkeypatch.setattr(workflow, "save_report_markdown", fake_save_report_markdown)

    result = run_research_workflow("aapl", "장기", "중립형")

    assert result["run_id"]
    assert result["user_input"]["ticker"] == "AAPL"
    assert result["errors"] == []
    assert result["final_report"].ticker == "AAPL"
    assert "관망 시나리오" in result["final_report"].scenario_analysis
    assert result["final_report"].disclaimer == REQUIRED_DISCLAIMER
    assert result["evaluation_result"].overall_pass is True
    assert result["evaluation_result"].source_grounding_score >= 0.8
    assert result["report_path"].endswith("_AAPL_report.json")
    assert Path(result["report_path"]).exists()
    assert calls == ["market", "fundamental", "news", "graph_context", "coordinator"]
