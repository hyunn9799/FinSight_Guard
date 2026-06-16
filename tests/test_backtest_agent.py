"""Tests for the backtest agent, routing gate, and workflow integration."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

import src.agents.backtest_agent as backtest_agent
from src.agents.backtest_agent import run_backtest_agent
from src.graph.routing import route_after_supervisor
from src.graph.state import BacktestAnalysis, SupervisorPlan
from src.graph.workflow import run_research_workflow


def _synthetic_prices(n: int = 160) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    t = np.arange(n)
    close = 100 + 12 * np.sin(t / 7.0) + t * 0.05
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 1_000),
        },
        index=idx,
    )


def test_backtest_agent_returns_evidence_and_analysis(monkeypatch) -> None:
    monkeypatch.setattr(backtest_agent, "load_price_history", lambda *a, **k: _synthetic_prices())

    result = run_backtest_agent({"ticker": "aapl", "evidence": []})

    assert result["status"] == "success"
    analysis = result["backtest_analysis"]
    assert isinstance(analysis, BacktestAnalysis)
    assert analysis.ticker == "AAPL"
    # All emitted evidence is typed as backtest and frames itself as simulation.
    assert result["evidence"]
    assert all(item.source_type == "backtest" for item in result["evidence"])
    assert any(item.metric_name == "backtest_profit_pct" for item in result["evidence"])
    # No-advice positioning is preserved.
    assert "권유하지 않습니다" in analysis.performance_summary


def test_backtest_agent_degrades_when_data_unavailable(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise ValueError("no data")

    monkeypatch.setattr(backtest_agent, "load_price_history", boom)

    result = run_backtest_agent({"ticker": "AAPL", "evidence": []})

    assert result["status"] == "degraded"
    assert result["backtest_analysis"].missing_data_notes
    assert result["errors"]


def test_route_to_backtest_when_enabled_and_not_run() -> None:
    state = {
        "supervisor_plan": SupervisorPlan(
            next_node="coordinator_node",
            needs_backtest=True,
        )
    }

    assert route_after_supervisor(state) == "backtest_node"


def test_route_skips_backtest_once_analysis_present() -> None:
    state = {
        "supervisor_plan": SupervisorPlan(
            next_node="coordinator_node",
            needs_backtest=True,
        ),
        "backtest_analysis": BacktestAnalysis(ticker="AAPL"),
    }

    assert route_after_supervisor(state) == "graph_context_node"


def test_route_default_without_backtest() -> None:
    state = {"supervisor_plan": SupervisorPlan(next_node="coordinator_node")}

    assert route_after_supervisor(state) == "graph_context_node"


def _backtest_evidence():
    from src.evidence.evidence_builder import build_backtest_evidence

    return build_backtest_evidence(
        ticker="AAPL",
        metric_name="backtest_profit_pct",
        metric_value=12.34,
        description="AAPL 과거 시뮬레이션 누적 수익률 12.34% (미래 수익 보장 아님)",
        collected_at=datetime(2026, 6, 16, tzinfo=UTC),
    )


def test_workflow_includes_backtest_section_when_enabled(monkeypatch, tmp_path) -> None:
    import src.graph.workflow as workflow
    from tests.test_workflow_e2e import _patch_mock_agents, _patch_storage

    _patch_storage(monkeypatch, tmp_path)
    _patch_mock_agents(monkeypatch)

    evidence = _backtest_evidence()

    def fake_backtest_agent(state: dict) -> dict:
        return {
            "status": "success",
            "evidence": [*state.get("evidence", []), evidence],
            "backtest_analysis": BacktestAnalysis(
                ticker="AAPL",
                summary="고정 규칙 기반 과거 시뮬레이션 참고 정보입니다.",
                period_summary="시뮬레이션 구간: 2025-05-12 ~ 2026-06-16, 사용된 거래일 270일.",
                performance_summary=(
                    "과거 시뮬레이션 누적 수익률은 12.34%입니다. 매수·매도·보유를 권유하지 않습니다."
                ),
                signal_summary="RSI 다이버전스 신호는 총 3건 확인되었습니다.",
                evidence=[evidence],
            ),
        }

    monkeypatch.setattr(workflow, "run_backtest_agent", fake_backtest_agent)

    result = run_research_workflow(
        "aapl",
        "장기",
        "중립형",
        user_query="이 종목 종합 보고서 만들어줘",
        enable_backtest=True,
    )

    assert result["backtest_analysis"] is not None
    report = result["final_report"]
    assert report is not None
    assert "전략 백테스트 참고" in report.backtest_section
    assert result["evaluation_result"].overall_pass is True
