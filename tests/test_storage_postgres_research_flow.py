# tests/test_storage_postgres_research_flow.py  (new file)
import os
from datetime import UTC, date, datetime

import pytest

REQUIRES_DB = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set"
)


def _sample_report(ticker="AAPL"):
    from src.graph.state import ResearchReport
    return ResearchReport(
        title=f"{ticker} 리서치", ticker=ticker, data_date=date(2026, 6, 18),
        executive_summary="요약", market_section="시장", fundamental_section="펀더",
        news_section="뉴스", scenario_analysis="시나리오", risk_factors="리스크",
        limitations="한계", evidence_summary="근거",
        disclaimer="본 자료는 교육용이며 투자 권유가 아닙니다.",
    )


def _sample_evidence(ticker="AAPL"):
    from src.evidence.evidence_schema import EvidenceItem
    return [
        EvidenceItem(
            evidence_id="ev-1", source_type="market", source_name="yfinance",
            collected_at=datetime.now(UTC), ticker=ticker, metric_name="close",
            metric_value=190.5, description="closing price",
        )
    ]


def _sample_graph_context(ticker="AAPL"):
    from src.graph_rag.graph_schema import GraphContext, GraphEdge, GraphNode
    return GraphContext(
        ticker=ticker,
        focus="comprehensive",
        nodes=[
            GraphNode(node_id="ev-source-1", node_type="event", name="실적 발표"),
            GraphNode(node_id=f"company:{ticker}", node_type="company", name=ticker),
        ],
        edges=[
            GraphEdge(
                source_id="ev-source-1",
                target_id=f"company:{ticker}",
                relation_type="positive_driver",
                evidence_id="ev-1",
                description="실적 호조가 주가를 견인",
            )
        ],
        key_relations_summary=["실적 호조 → 주가 상승"],
        evidence_ids=["ev-1"],
    )


@REQUIRES_DB
def test_persist_successful_research_run(db_session, monkeypatch):
    import src.db.persistence as persistence
    from src.graph.state import EvaluationResult
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield db_session  # reuse the rollback-bound test session

    monkeypatch.setattr(persistence, "session_scope", _scope)

    evaluation = EvaluationResult(
        overall_pass=True, source_grounding_score=1.0, numeric_consistency_score=1.0,
        safety_score=1.0, risk_disclosure_score=1.0, freshness_score=1.0,
    )
    out = persistence.persist_research_run(
        run_id="run-1", ticker="AAPL", status="success",
        report=_sample_report(), evidence=_sample_evidence(), evaluation=evaluation,
        node_runs=[{"node_name": "market_node", "status": "success", "duration_ms": 5}],
    )

    from src.db.models import (
        AnalysisRequest, Report, ReportVersion, EvidenceItemRecord, ReportEvidenceCitation,
    )
    req = db_session.get(AnalysisRequest, out["request_id"])
    assert req.status == "success"
    report = db_session.get(Report, out["report_id"])
    assert report.status == "final"
    assert report.disclaimer_present is True
    version = db_session.get(ReportVersion, out["report_version_id"])
    assert version.version_number == 1
    assert db_session.query(EvidenceItemRecord).count() == 1
    assert db_session.query(ReportEvidenceCitation).count() == 1


@REQUIRES_DB
def test_persist_degraded_run_keeps_missing_notes(db_session, monkeypatch):
    import src.db.persistence as persistence
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield db_session

    monkeypatch.setattr(persistence, "session_scope", _scope)

    out = persistence.persist_research_run(
        run_id="run-2", ticker="AAPL", status="degraded",
        report=_sample_report(), evidence=[], evaluation=None,
        missing_data_notes=["news provider unavailable"],
    )
    from src.db.models import AnalysisRequest, Report
    req = db_session.get(AnalysisRequest, out["request_id"])
    assert req.status == "degraded"
    assert "news provider unavailable" in (req.degraded_reason or "")
    report = db_session.get(Report, out["report_id"])
    assert report.status == "draft"


@REQUIRES_DB
def test_persist_failed_evaluation_records_safety_fail(db_session, monkeypatch):
    import src.db.persistence as persistence
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield db_session

    monkeypatch.setattr(persistence, "session_scope", _scope)

    out = persistence.persist_research_run(
        run_id="run-fail",
        ticker="AAPL",
        status="failed",
        report=_sample_report(),
        evidence=[],
        evaluation={"overall_pass": False, "source_grounding_score": 0.25},
    )

    from src.db.models import Report

    report = db_session.get(Report, out["report_id"])
    assert report.safety_status == "fail"
    assert report.evaluation_score == 0.25


@REQUIRES_DB
def test_persist_research_run_accepts_dict_report(db_session, monkeypatch):
    import src.db.persistence as persistence
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield db_session

    monkeypatch.setattr(persistence, "session_scope", _scope)

    report = _sample_report().model_dump(mode="json")
    out = persistence.persist_research_run(
        run_id="run-dict",
        ticker="AAPL",
        status="success",
        report=report,
        evidence=[],
        evaluation={"overall_pass": True, "source_grounding_score": 0.9},
    )

    from src.db.models import Report, ReportVersion

    report_row = db_session.get(Report, out["report_id"])
    version = db_session.get(ReportVersion, out["report_version_id"])
    assert report_row.title == "AAPL 리서치"
    assert report_row.safety_status == "pass"
    assert version.report_json["ticker"] == "AAPL"


@REQUIRES_DB
def test_save_report_node_persists_then_exports(db_session, monkeypatch, tmp_path):
    import src.db.persistence as persistence
    import src.graph.workflow as workflow
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield db_session

    monkeypatch.setattr(persistence, "session_scope", _scope)
    monkeypatch.setattr(workflow, "save_report_json", lambda run_id, payload: str(tmp_path / "r.json"))
    monkeypatch.setattr(workflow, "save_report_markdown", lambda run_id, report: str(tmp_path / "r.md"))
    monkeypatch.setattr(workflow, "save_run", lambda run_id, meta: meta)

    from src.graph.state import EvaluationResult
    evaluation = EvaluationResult(
        overall_pass=True, source_grounding_score=1.0, numeric_consistency_score=1.0,
        safety_score=1.0, risk_disclosure_score=1.0, freshness_score=1.0,
    )
    state = {
        "run_id": "run-x", "ticker": "AAPL", "status": "success",
        "draft_report": _sample_report(), "evidence": _sample_evidence(),
        "evaluation_result": evaluation,
    }
    out = workflow.save_report_node(state)
    assert out["status"] == "success"
    assert out["request_id"]  # set from persistence
    from src.db.models import AnalysisRequest
    assert db_session.get(AnalysisRequest, out["request_id"]) is not None


@REQUIRES_DB
def test_persist_run_stores_graph_evidence_path(db_session, monkeypatch):
    import src.db.persistence as persistence
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield db_session

    monkeypatch.setattr(persistence, "session_scope", _scope)

    out = persistence.persist_research_run(
        run_id="run-gc", ticker="AAPL", status="success",
        report=_sample_report(), evidence=_sample_evidence(),
        evaluation={"overall_pass": True, "source_grounding_score": 0.9},
        graph_context=_sample_graph_context(),
    )

    from src.db.models import EvidencePath, EvidencePathStep, EvidenceItemRecord

    path = db_session.get(EvidencePath, out["evidence_path_id"])
    assert path is not None
    assert path.path_type == "graph_context"
    assert path.request_id == out["request_id"]

    steps = (
        db_session.query(EvidencePathStep)
        .filter(EvidencePathStep.evidence_path_id == path.id)
        .order_by(EvidencePathStep.step_index)
        .all()
    )
    assert len(steps) == 1
    assert steps[0].step_index == 0
    assert steps[0].node_table == "evidence_items"
    assert steps[0].relationship_type == "positive_driver"
    # the step's node_id resolves to a persisted evidence row
    assert db_session.get(EvidenceItemRecord, steps[0].node_id) is not None


@REQUIRES_DB
def test_save_report_node_degrades_on_persistence_error(monkeypatch, tmp_path):
    import src.graph.workflow as workflow

    def _boom(**kwargs):
        raise RuntimeError("db down")

    json_calls = []
    md_calls = []

    def _fake_json(run_id, payload):
        json_calls.append((run_id, payload))
        return str(tmp_path / "r.json")

    def _fake_md(run_id, report):
        md_calls.append((run_id, report))
        return str(tmp_path / "r.md")

    monkeypatch.setattr(workflow, "persist_research_run", _boom)
    monkeypatch.setattr(workflow, "save_report_json", _fake_json)
    monkeypatch.setattr(workflow, "save_report_markdown", _fake_md)
    monkeypatch.setattr(workflow, "save_run", lambda *a: {})

    state = {"run_id": "run-y", "ticker": "AAPL", "status": "success", "draft_report": _sample_report()}
    out = workflow.save_report_node(state)
    assert out["status"] == "degraded"
    assert any(getattr(e, "error_type", "") == "persistence_error" for e in out["errors"])
    # Export files must always be written even when PG persistence fails.
    assert len(json_calls) == 1, "save_report_json must be called on degrade path"
    assert len(md_calls) == 1, "save_report_markdown must be called on degrade path"
    assert "report_path" in out, "degraded result must include report_path"
    assert out["report_path"] == str(tmp_path / "r.json")
    assert out.get("final_report") is not None, "degraded result must include final_report"
