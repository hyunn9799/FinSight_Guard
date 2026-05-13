"""End-to-end workflow tests with mocked data agents and no live APIs."""

from datetime import UTC, date, datetime

from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import (
    EvaluationResult,
    FundamentalAnalysis,
    MarketAnalysis,
    NewsAnalysis,
    ResearchReport,
)
from src.graph.workflow import run_research_workflow
from src.safety.forbidden_phrases import REQUIRED_DISCLAIMER


def _evidence(
    evidence_id: str,
    source_type: str,
    metric_name: str,
    metric_value: float | str | None,
    description: str,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_type=source_type,  # type: ignore[arg-type]
        source_name="mock",
        source_url=None,
        collected_at=datetime(2026, 5, 12, tzinfo=UTC),
        ticker="AAPL",
        metric_name=metric_name,
        metric_value=metric_value,
        description=description,
    )


MARKET_EVIDENCE = _evidence(
    "market_001",
    "market",
    "RSI",
    58.2,
    "RSI는 중립 구간이며 단기 변동성 점검이 필요합니다.",
)
FUNDAMENTAL_EVIDENCE = _evidence(
    "fundamental_001",
    "fundamental",
    "PER",
    28.1,
    "PER 기준 밸류에이션 비교가 필요합니다.",
)
NEWS_EVIDENCE = _evidence(
    "news_001",
    "news",
    "news_item",
    None,
    "최근 악재와 규제 리스크가 언급되었습니다.",
)


def _patch_storage(monkeypatch, tmp_path) -> None:
    import src.graph.workflow as workflow
    import src.observability.logger as project_logger

    monkeypatch.setattr(project_logger, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(project_logger, "_CONFIGURED", False)

    def fake_save_report_json(run_id: str, payload: dict) -> str:
        path = tmp_path / f"{run_id}_AAPL_report.json"
        path.write_text("{}", encoding="utf-8")
        return str(path)

    def fake_save_report_markdown(run_id: str, report: ResearchReport) -> str:
        path = tmp_path / f"{run_id}_AAPL_report.md"
        path.write_text(report.title, encoding="utf-8")
        return str(path)

    def fake_save_run(run_id: str, payload: dict) -> str:
        path = tmp_path / f"{run_id}_run.json"
        path.write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr(workflow, "save_report_json", fake_save_report_json)
    monkeypatch.setattr(workflow, "save_report_markdown", fake_save_report_markdown)
    monkeypatch.setattr(workflow, "save_run", fake_save_run)


def _patch_mock_agents(monkeypatch, calls: list[str] | None = None) -> None:
    import src.graph.workflow as workflow

    def fake_market_agent(state: dict) -> dict:
        if calls is not None:
            calls.append("market")
        evidence = [*state.get("evidence", []), MARKET_EVIDENCE]
        return {
            "status": "success",
            "evidence": evidence,
            "market_analysis": MarketAnalysis(
                ticker="AAPL",
                summary="시장 지표 요약",
                trend_summary="RSI와 이동평균으로 단기 추세를 점검했습니다.",
                momentum_summary="RSI는 중립 구간입니다.",
                volatility_summary="단기 변동성은 리스크로 확인해야 합니다.",
                evidence=[MARKET_EVIDENCE],
            ),
        }

    def fake_fundamental_agent(state: dict) -> dict:
        if calls is not None:
            calls.append("fundamental")
        evidence = [*state.get("evidence", []), FUNDAMENTAL_EVIDENCE]
        return {
            "status": "success",
            "evidence": evidence,
            "fundamental_analysis": FundamentalAnalysis(
                ticker="AAPL",
                summary="펀더멘털 요약",
                valuation_summary="PER 기준 밸류에이션을 비교했습니다.",
                profitability_summary="수익성은 추가 확인이 필요합니다.",
                stability_summary="재무 안정성은 제한적으로 해석합니다.",
                evidence=[FUNDAMENTAL_EVIDENCE],
            ),
        }

    def fake_news_agent(state: dict) -> dict:
        if calls is not None:
            calls.append("news")
        evidence = [*state.get("evidence", []), NEWS_EVIDENCE]
        return {
            "status": "success",
            "evidence": evidence,
            "news_analysis": NewsAnalysis(
                ticker="AAPL",
                summary="최근 악재와 규제 리스크를 점검했습니다.",
                positive_factors=["수요 개선 가능성"],
                negative_factors=["규제 리스크"],
                event_risks=["악재에 따른 단기 변동성"],
                evidence=[NEWS_EVIDENCE],
            ),
        }

    monkeypatch.setattr(workflow, "run_market_agent", fake_market_agent)
    monkeypatch.setattr(workflow, "run_fundamental_agent", fake_fundamental_agent)
    monkeypatch.setattr(workflow, "run_news_agent", fake_news_agent)


def _run_with_mocks(monkeypatch, tmp_path, query: str, calls: list[str] | None = None) -> dict:
    _patch_storage(monkeypatch, tmp_path)
    _patch_mock_agents(monkeypatch, calls)
    return run_research_workflow(
        "aapl",
        "장기",
        "중립형",
        user_query=query,
    )


def _report_text(report: ResearchReport) -> str:
    return "\n".join(str(value) for value in report.model_dump().values() if value is not None)


def test_technical_analysis_route(monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    result = _run_with_mocks(monkeypatch, tmp_path, "차트상 단기 진입 괜찮아?", calls)

    plan = result["supervisor_plan"]

    assert plan.question_type == "technical_analysis"
    assert plan.planned_agent_order == ["market", "news"]
    assert result["market_analysis"] is not None
    assert result["final_report"] is not None
    assert result["evaluation_result"] is not None
    assert calls == ["market", "news"]


def test_fundamental_analysis_route(monkeypatch, tmp_path) -> None:
    result = _run_with_mocks(monkeypatch, tmp_path, "장기적으로 저평가야?")

    plan = result["supervisor_plan"]
    report = result["final_report"]

    assert plan.question_type == "fundamental_analysis"
    assert plan.planned_agent_order == ["fundamental", "news"]
    assert result["fundamental_analysis"] is not None
    assert result["market_analysis"] is None
    assert "market" in plan.skipped_agents
    assert report is not None
    assert "시장 분석은 상세 실행 대상에서 제외되었습니다" in report.limitations


def test_news_risk_analysis_route(monkeypatch, tmp_path) -> None:
    result = _run_with_mocks(monkeypatch, tmp_path, "최근 악재 때문에 위험해?")

    plan = result["supervisor_plan"]
    report = result["final_report"]

    assert plan.question_type == "news_risk_analysis"
    assert plan.planned_agent_order == ["news", "market", "fundamental"]
    assert result["news_analysis"] is not None
    assert result["graph_context"] is not None
    assert report is not None
    assert "리스크" in f"{report.risk_factors}\n{report.scenario_analysis}\n{report.graph_context_section}"


def test_comprehensive_report_route(monkeypatch, tmp_path) -> None:
    result = _run_with_mocks(monkeypatch, tmp_path, "이 종목 종합 보고서 만들어줘")

    plan = result["supervisor_plan"]

    assert plan.question_type == "comprehensive_report"
    assert result["market_analysis"] is not None
    assert result["fundamental_analysis"] is not None
    assert result["news_analysis"] is not None
    assert result["graph_context"] is not None
    assert result["final_report"] is not None
    assert result["evaluation_result"] is not None


def test_evaluator_fail_then_rewrite_loop(monkeypatch, tmp_path) -> None:
    import src.graph.workflow as workflow

    evaluation_passes: list[bool] = []
    _patch_storage(monkeypatch, tmp_path)
    _patch_mock_agents(monkeypatch)

    def unsafe_coordinator(state: dict) -> dict:
        report = ResearchReport(
            title="AAPL unsafe report",
            ticker="AAPL",
            data_date=date(2026, 5, 12),
            executive_summary="이 종목은 무조건 매수 관점입니다.",
            market_section="RSI는 58.2입니다. [근거: market_001]",
            fundamental_section="PER은 28.1입니다. [근거: fundamental_001]",
            news_section="최근 악재와 규제 리스크가 있습니다. [근거: news_001]",
            graph_context_section=(
                "관계 기반 리스크 및 근거 요약\n"
                "리스크 관계:\n"
                "- risk:규제 -> company:AAPL: negative_risk, evidence_id=news_001"
            ),
            scenario_analysis="관망 시나리오, 분할 접근 시나리오, 리스크 회피 시나리오를 비교합니다.",
            risk_factors="가격 변동성과 규제 리스크를 확인해야 합니다. [근거: news_001]",
            limitations="공개 데이터와 제한된 근거에 기반합니다.",
            evidence_summary=(
                "1. evidence_id=market_001; source_type=market; metric_name=RSI; metric_value=58.2; "
                "description=RSI는 중립 구간입니다.\n"
                "2. evidence_id=fundamental_001; source_type=fundamental; metric_name=PER; metric_value=28.1; "
                "description=PER 기준 밸류에이션 비교가 필요합니다.\n"
                "3. evidence_id=news_001; source_type=news; metric_name=news_item; metric_value=None; "
                "description=최근 악재와 규제 리스크가 언급되었습니다."
            ),
            disclaimer=REQUIRED_DISCLAIMER,
        )
        return {"status": "success", "draft_report": report}

    original_evaluator = workflow.run_evaluator_agent

    def tracking_evaluator(state: dict) -> dict:
        result = original_evaluator(state)
        evaluation_passes.append(result["evaluation_result"].overall_pass)
        return result

    monkeypatch.setattr(workflow, "run_coordinator_agent", unsafe_coordinator)
    monkeypatch.setattr(workflow, "run_evaluator_agent", tracking_evaluator)

    result = run_research_workflow(
        "aapl",
        "장기",
        "중립형",
        user_query="이 종목 종합 보고서 만들어줘",
    )

    assert evaluation_passes[0] is False
    assert len(evaluation_passes) >= 2
    assert result["rewrite_attempts"] >= 1
    assert "무조건 매수" not in _report_text(result["final_report"])


def test_max_rewrite_failure_stops_at_two_attempts(monkeypatch, tmp_path) -> None:
    import src.graph.workflow as workflow

    rewrite_calls: list[int] = []
    _patch_storage(monkeypatch, tmp_path)
    _patch_mock_agents(monkeypatch)

    def unsafe_coordinator(state: dict) -> dict:
        report = ResearchReport(
            title="AAPL unsafe report",
            ticker="AAPL",
            data_date=date(2026, 5, 12),
            executive_summary="이 종목은 무조건 매수 관점입니다.",
            market_section="RSI는 58.2입니다. [근거: market_001]",
            fundamental_section="PER은 28.1입니다. [근거: fundamental_001]",
            news_section="최근 악재와 규제 리스크가 있습니다. [근거: news_001]",
            graph_context_section="관계 기반 리스크 및 근거 요약\n리스크 관계: risk:규제",
            scenario_analysis="관망 시나리오와 리스크 회피 시나리오를 비교합니다.",
            risk_factors="가격 변동성과 규제 리스크를 확인해야 합니다. [근거: news_001]",
            limitations="공개 데이터와 제한된 근거에 기반합니다.",
            evidence_summary=(
                "1. evidence_id=market_001; source_type=market; metric_name=RSI; metric_value=58.2; "
                "description=RSI는 중립 구간입니다.\n"
                "2. evidence_id=fundamental_001; source_type=fundamental; metric_name=PER; metric_value=28.1; "
                "description=PER 기준 밸류에이션 비교가 필요합니다.\n"
                "3. evidence_id=news_001; source_type=news; metric_name=news_item; metric_value=None; "
                "description=최근 악재와 규제 리스크가 언급되었습니다."
            ),
            disclaimer=REQUIRED_DISCLAIMER,
        )
        return {"status": "success", "draft_report": report}

    def ineffective_rewrite(state: dict) -> dict:
        rewrite_calls.append(state.get("rewrite_attempts", 0) + 1)
        return {
            "status": "success",
            "draft_report": state["draft_report"],
            "rewrite_attempts": state.get("rewrite_attempts", 0) + 1,
        }

    monkeypatch.setattr(workflow, "run_coordinator_agent", unsafe_coordinator)
    monkeypatch.setattr(workflow, "run_rewrite_agent", ineffective_rewrite)

    result = run_research_workflow(
        "aapl",
        "장기",
        "중립형",
        user_query="이 종목 종합 보고서 만들어줘",
    )

    assert rewrite_calls == [1, 2]
    assert result["rewrite_attempts"] == 2
    assert result["status"] == "failed"
    assert result["final_report"] is not None
    assert result["evaluation_result"].overall_pass is False


def test_graph_context_failure_degraded_mode(monkeypatch, tmp_path) -> None:
    import src.graph.workflow as workflow

    _patch_storage(monkeypatch, tmp_path)
    _patch_mock_agents(monkeypatch)

    def failing_graph_context_builder(*args, **kwargs):
        raise RuntimeError("graph context failed")

    monkeypatch.setattr(workflow, "build_graph_context", failing_graph_context_builder)

    result = run_research_workflow(
        "aapl",
        "장기",
        "중립형",
        user_query="이 종목 종합 보고서 만들어줘",
    )

    assert result["errors"] or result["warnings"]
    assert any("graph_context_node failed" in warning for warning in result["warnings"])
    assert result["graph_context"] is None
    assert result["final_report"] is not None
    assert result["evaluation_result"] is not None
