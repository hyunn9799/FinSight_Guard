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
