import os
from datetime import UTC, datetime

import pytest

REQUIRES_DB = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set"
)


@REQUIRES_DB
def test_db_session_fixture_roundtrips_a_ticker(db_session):
    from src.db.models import Ticker
    db_session.add(Ticker(symbol="AAPL", market="NASDAQ"))
    db_session.flush()
    found = db_session.query(Ticker).filter_by(symbol="AAPL").one()
    assert found.market == "NASDAQ"
    assert found.id is not None


@REQUIRES_DB
def test_base_repository_get(db_session):
    from src.db.repositories.base import BaseRepository
    from src.db.models import Ticker
    from tests.fixtures.postgres import make_ticker

    ticker = make_ticker(db_session)
    repo = BaseRepository(db_session)
    assert repo.get(Ticker, ticker.id).symbol == "AAPL"
    import uuid
    assert repo.get(Ticker, uuid.uuid4()) is None


@REQUIRES_DB
def test_upsert_ticker_is_idempotent_and_uppercases(db_session):
    from src.db.repositories.analysis_repository import AnalysisRepository
    repo = AnalysisRepository(db_session)
    a = repo.upsert_ticker("aapl", "NASDAQ")
    b = repo.upsert_ticker("AAPL", "NASDAQ")
    assert a.id == b.id
    assert a.symbol == "AAPL"


@REQUIRES_DB
def test_request_node_run_result_lifecycle(db_session):
    from src.db.repositories.analysis_repository import AnalysisRepository
    repo = AnalysisRepository(db_session)
    ticker = repo.upsert_ticker("AAPL", "NASDAQ")
    req = repo.create_request(ticker.id, "research", parameters={"q": "x"})
    assert req.status == "pending"

    node = repo.record_node_run(req.id, "run-1", "market_node", "success", duration_ms=12)
    assert node.attempt_number == 1

    result = repo.add_result(
        req.id, "market", ticker_id=ticker.id, summary="s",
        missing_data_notes=["news unavailable"], status="degraded",
    )
    assert result.missing_data_notes == ["news unavailable"]

    updated = repo.update_request_status(
        req.id, "degraded", degraded_reason="news down", completed_at=datetime.now(UTC)
    )
    assert updated.status == "degraded"
    assert updated.completed_at is not None


@REQUIRES_DB
def test_node_run_unique_per_attempt(db_session):
    import pytest as _pytest
    from sqlalchemy.exc import IntegrityError
    from src.db.repositories.analysis_repository import AnalysisRepository
    repo = AnalysisRepository(db_session)
    t = repo.upsert_ticker("MSFT", "NASDAQ")
    req = repo.create_request(t.id, "research")
    repo.record_node_run(req.id, "run-1", "market_node", "success", attempt_number=1)
    with _pytest.raises(IntegrityError):
        repo.record_node_run(req.id, "run-1", "market_node", "failed", attempt_number=1)


@REQUIRES_DB
def test_report_versions_and_current_version(db_session):
    from src.db.repositories.analysis_repository import AnalysisRepository
    from src.db.repositories.report_repository import ReportRepository

    analysis = AnalysisRepository(db_session)
    ticker = analysis.upsert_ticker("AAPL", "NASDAQ")
    req = analysis.create_request(ticker.id, "research")

    reports = ReportRepository(db_session)
    report = reports.create_report(req.id, ticker.id, title="AAPL 리서치")
    assert reports.next_version_number(report.id) == 1

    v1 = reports.add_version(report.id, 1, "draft", {"a": 1}, "# md")
    assert report.current_version_id == v1.id
    assert reports.next_version_number(report.id) == 2

    v2 = reports.add_version(report.id, 2, "final", {"a": 2}, "# md2", created_by_node="rewrite_node")
    assert report.current_version_id == v2.id

    reports.set_status(report.id, status="final", safety_status="pass",
                       evaluation_score=0.9, disclaimer_present=True)
    assert report.status == "final"
    assert report.safety_status == "pass"
    assert report.disclaimer_present is True
