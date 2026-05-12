"""Tests for report storage behavior."""

from datetime import date
from pathlib import Path

from src.graph.state import ResearchReport
from src.observability.metrics import get_metrics, record_run, reset_metrics
from src.storage.report_store import (
    load_report_json,
    save_report_json,
    save_report_markdown,
)


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


def test_save_report_json_creates_file_and_loads(monkeypatch, tmp_path) -> None:
    import src.storage.report_store as report_store

    monkeypatch.setattr(report_store, "REPORT_DIR", tmp_path)
    result = {"run_id": "run123", "final_report": _report(), "status": "success"}

    path = save_report_json("run123", result)
    loaded = load_report_json(path)

    assert Path(path).exists()
    assert Path(path).name.startswith("run123_AAPL_")
    assert Path(path).suffix == ".json"
    assert loaded["run_id"] == "run123"
    assert loaded["final_report"]["ticker"] == "AAPL"


def test_save_report_markdown_creates_file(monkeypatch, tmp_path) -> None:
    import src.storage.report_store as report_store

    monkeypatch.setattr(report_store, "REPORT_DIR", tmp_path)

    path = save_report_markdown("run123", _report())

    assert Path(path).exists()
    assert Path(path).name.startswith("run123_AAPL_")
    assert Path(path).suffix == ".md"
    assert "AAPL Research Report" in Path(path).read_text(encoding="utf-8")


def test_metrics_update_works() -> None:
    reset_metrics()

    record_run(True, 0.8)
    record_run(False, 0.4)

    metrics = get_metrics()
    assert metrics["total_runs"] == 2
    assert metrics["successful_runs"] == 1
    assert metrics["failed_runs"] == 1
    assert metrics["average_evaluation_score"] == 0.6
