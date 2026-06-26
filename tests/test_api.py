"""Tests for FastAPI endpoints."""

from datetime import date

import main
import pytest
from pydantic import ValidationError
from src.graph.state import EvaluationResult, ResearchReport
from src.observability.metrics import record_run, reset_metrics
from src.storage.report_store import save_report_json


def _report() -> ResearchReport:
    return ResearchReport(
        title="AAPL Research Report",
        ticker="AAPL",
        data_date=date(2026, 5, 12),
        executive_summary="요약",
        market_section="시장",
        fundamental_section="재무",
        news_section="뉴스",
        scenario_analysis="관망 시나리오\n분할 접근 시나리오\n리스크 회피 시나리오",
        risk_factors="리스크",
        limitations="한계",
        evidence_summary="근거",
        disclaimer="고지문",
    )


def _evaluation() -> EvaluationResult:
    return EvaluationResult(
        overall_pass=True,
        source_grounding_score=1.0,
        numeric_consistency_score=1.0,
        safety_score=1.0,
        risk_disclosure_score=1.0,
        freshness_score=1.0,
    )


def test_health_endpoint() -> None:
    paths = {route.path for route in main.app.routes}

    assert "/health" in paths
    assert main.health() == {"status": "ok"}


def test_analyze_endpoint_with_monkeypatched_workflow(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_workflow(ticker: str, investment_horizon: str, risk_profile: str) -> dict:
        calls["count"] += 1
        assert ticker == "AAPL"
        assert investment_horizon == "장기"
        assert risk_profile == "중립형"
        return {
            "run_id": "run123",
            "status": "success",
            "final_report": _report(),
            "evaluation_result": _evaluation(),
            "report_path": "reports/run123_AAPL_20260512T000000Z.json",
            "errors": [],
        }

    monkeypatch.setattr(main, "run_research_workflow", fake_workflow)

    payload = main.analyze(
        main.AnalyzeRequest(
            ticker="AAPL",
            investment_horizon="장기",
            risk_profile="중립형",
        )
    )

    assert calls["count"] == 1
    assert payload["run_id"] == "run123"
    assert payload["status"] == "success"
    assert payload["final_report"]["ticker"] == "AAPL"
    assert payload["evaluation_result"]["overall_pass"] is True
    assert payload["report_path"].endswith(".json")
    assert payload["errors"] == []


def test_backtest_endpoint_passes_enable_flag_and_returns_analysis(monkeypatch) -> None:
    captured = {}

    def fake_workflow(**kwargs) -> dict:
        captured.update(kwargs)
        return {
            "run_id": "bt123",
            "status": "success",
            "final_report": _report(),
            "evaluation_result": _evaluation(),
            "backtest_analysis": {"ticker": "AAPL", "summary": "과거 시뮬레이션 참고"},
            "report_path": "reports/bt123_AAPL.json",
            "request_id": "33333333-3333-3333-3333-333333333333",
            "errors": [],
        }

    monkeypatch.setattr(main, "run_research_workflow", fake_workflow)

    payload = main.backtest(
        main.BacktestRequest(
            ticker="AAPL",
            investment_horizon="장기",
            risk_profile="중립형",
        )
    )

    assert captured["enable_backtest"] is True
    assert payload["run_id"] == "bt123"
    assert payload["request_id"] == "33333333-3333-3333-3333-333333333333"
    assert payload["backtest_analysis"]["ticker"] == "AAPL"


def test_analyze_endpoint_rejects_invalid_payload_before_workflow(monkeypatch) -> None:
    def fake_workflow(ticker: str, investment_horizon: str, risk_profile: str) -> dict:
        raise AssertionError("workflow should not run for invalid request payload")

    monkeypatch.setattr(main, "run_research_workflow", fake_workflow)

    with pytest.raises(ValidationError):
        main.AnalyzeRequest(
            ticker="",
            investment_horizon="초장기",
            risk_profile="중립형",
        )


def test_metrics_endpoint() -> None:
    reset_metrics()
    record_run(True, 1.0)
    record_run(False, 0.5)

    payload = main.metrics()

    assert payload == {
        "total_runs": 2,
        "successful_runs": 1,
        "failed_runs": 1,
        "average_evaluation_score": 0.75,
    }


def test_health_and_metrics_contract_survive_persisted_request_id(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    reset_metrics()
    record_run(True, 0.8)

    def _fake_run(ticker, user_query=None, **kwargs):
        return {
            "run_id": "run-with-pg",
            "status": "success",
            "ticker": ticker,
            "request_id": "22222222-2222-2222-2222-222222222222",
            "final_report": None,
            "errors": [],
        }

    monkeypatch.setattr(main, "run_research_workflow", _fake_run, raising=False)
    client = TestClient(main.app)

    analyze_response = client.post(
        "/analyze",
        json={
            "ticker": "AAPL",
            "investment_horizon": "장기",
            "risk_profile": "중립형",
        },
    )
    assert analyze_response.status_code == 200
    assert analyze_response.json()["request_id"] == "22222222-2222-2222-2222-222222222222"

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/metrics").json() == {
        "total_runs": 1,
        "successful_runs": 1,
        "failed_runs": 0,
        "average_evaluation_score": 0.8,
    }


def test_get_report_endpoint(monkeypatch, tmp_path) -> None:
    import src.storage.report_store as report_store

    monkeypatch.setattr(report_store, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(main, "REPORT_DIR", tmp_path)
    path = save_report_json("run123", {"run_id": "run123", "final_report": _report()})

    payload = main.get_report("run123")

    assert payload["run_id"] == "run123"
    assert payload["final_report"]["ticker"] == "AAPL"
    assert path.endswith(".json")


def test_get_report_endpoint_returns_404_for_unknown_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "REPORT_DIR", tmp_path)

    with pytest.raises(main.HTTPException) as exc_info:
        main.get_report("missing-run")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Report not found."


def test_analyze_response_includes_request_id(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    def _fake_run(ticker, user_query=None, **kwargs):
        return {
            "run_id": "run-1",
            "status": "success",
            "ticker": ticker,
            "request_id": "11111111-1111-1111-1111-111111111111",
            "final_report": None,
        }

    monkeypatch.setattr(main, "run_research_workflow", _fake_run, raising=False)
    client = TestClient(main.app)
    resp = client.post("/analyze", json={"ticker": "AAPL", "investment_horizon": "장기", "risk_profile": "중립형"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run-1"
    assert body["request_id"] == "11111111-1111-1111-1111-111111111111"
