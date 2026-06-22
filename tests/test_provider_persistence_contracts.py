"""US2 persistence lineage tests. Skips without TEST_DATABASE_URL (conftest)."""

import pytest

from src.db.repositories.provider_repository import ProviderRepository
from src.providers.enums import NormalizationStatus


@pytest.fixture()
def repo(db_session):
    return ProviderRepository(db_session)


@pytest.fixture()
def request_and_ticker(db_session):
    from src.db.repositories.analysis_repository import AnalysisRepository
    ar = AnalysisRepository(db_session)
    ticker = ar.upsert_ticker("ACME", market="US")
    db_session.flush()
    req = ar.create_request(ticker_id=ticker.id, request_type="research", status="pending")
    db_session.flush()
    return req.id, ticker.id


def test_raw_response_persists_and_normalized_records_trace_back(repo, request_and_ticker):
    request_id, ticker_id = request_and_ticker
    raw = repo.create_raw_response(
        request_id=request_id, ticker_id=ticker_id, provider_name="provider_a",
        provider_kind="news", status="success", payload_body={"items": []},
    )
    repo.session.flush()
    ev = repo.create_news_event(
        request_id=request_id, ticker_id=ticker_id, raw_response_id=raw.id,
        title="Acme beats earnings", normalization_status=NormalizationStatus.SUCCESS.value,
    )
    repo.session.flush()
    assert ev.raw_response_id == raw.id
    linked = repo.get_normalized_for_raw(raw.id)
    assert ev.id in {n.id for n in linked["news_events"]}


def test_derived_results_trace_to_market_and_evidence_not_raw(repo, request_and_ticker):
    request_id, ticker_id = request_and_ticker
    tar = repo.create_technical_result(
        request_id=request_id, ticker_id=ticker_id,
        source_market_data_refs=["md::1"], indicator_values={"rsi_14": 55.0},
        normalization_status=NormalizationStatus.SUCCESS.value, evidence_ids=["ev1"],
    )
    repo.session.flush()
    assert tar.source_market_data_refs == ["md::1"]
    # derived row has no raw_response_id column
    assert not hasattr(tar, "raw_response_id")


def test_partial_and_failed_normalization_recoverable(repo, request_and_ticker):
    request_id, ticker_id = request_and_ticker
    raw = repo.create_raw_response(
        request_id=request_id, ticker_id=ticker_id, provider_name="provider_x",
        provider_kind="financial", status="failed", error_message="rate limited",
    )
    repo.session.flush()
    assert raw.status == "failed"
    assert raw.error_message == "rate limited"
    # partial normalized record still persists with warnings
    fm = repo.create_financial_metric(
        request_id=request_id, ticker_id=ticker_id, raw_response_id=raw.id,
        metric_name="revenue", normalization_status=NormalizationStatus.PARTIAL_SUCCESS.value,
        warnings=[{"code": "missing_period", "message": "no period"}],
    )
    repo.session.flush()
    assert fm.normalization_status == "partial_success"
    assert fm.warnings[0]["code"] == "missing_period"


def test_persist_normalization_writes_raw_then_normalized(db_session, request_and_ticker):
    from src.db.persistence import persist_normalization
    from src.providers.normalization import normalize_news
    from tests.fixtures.provider_contracts import raw_news_provider_a

    request_id, ticker_id = request_and_ticker
    res = normalize_news(
        raw_items=raw_news_provider_a(), request_id=str(request_id),
        ticker_id=str(ticker_id), raw_response_id="placeholder",
    )
    out = persist_normalization(
        db_session, request_id=request_id, ticker_id=ticker_id,
        raw_kwargs=dict(provider_name="provider_a", provider_kind="news", status="success"),
        normalization_result=res,
    )
    db_session.flush()
    assert out["raw_response_id"] is not None
    assert len(out["news_events"]) == 1
    assert out["news_events"][0].raw_response_id == out["raw_response_id"]
