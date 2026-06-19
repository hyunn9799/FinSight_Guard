"""US2 PostgreSQL tests: projection status, keyword terms, graph, evidence paths."""

import os
import uuid
from datetime import UTC, datetime

import pytest

from tests.fixtures.postgres import make_request, make_ticker

REQUIRES_DB = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set"
)


@REQUIRES_DB
def test_projection_status_lifecycle(db_session):
    from src.db.repositories.projection_repository import ProjectionRepository

    repo = ProjectionRepository(db_session)
    record = repo.upsert_status(
        source_table="document_chunks",
        source_id=uuid.uuid4(),
        target_system="pinecone",
        projection_type="chunk_embedding",
        projection_key="vec-1",
        idempotency_key="idem-1",
    )
    assert record.status == "pending"
    assert record.attempt_count == 0

    now = datetime.now(UTC)
    repo.mark_failure(record, at=now, error_message="pinecone timeout")
    assert record.status == "failed"
    assert record.attempt_count == 1
    assert record.error_message == "pinecone timeout"

    repo.mark_success(record, at=now)
    assert record.status == "success"
    assert record.attempt_count == 2
    assert record.last_success_at is not None
    assert record.error_message is None

    repo.mark_stale(record)
    assert record.status == "stale"


@REQUIRES_DB
def test_projection_upsert_is_idempotent_on_idempotency_key(db_session):
    from src.db.repositories.projection_repository import ProjectionRepository

    repo = ProjectionRepository(db_session)
    source_id = uuid.uuid4()
    first = repo.upsert_status(
        source_table="reports",
        source_id=source_id,
        target_system="opensearch",
        projection_type="report_text",
        projection_key="r1",
        idempotency_key="dup",
    )
    second = repo.upsert_status(
        source_table="reports",
        source_id=source_id,
        target_system="opensearch",
        projection_type="report_text",
        projection_key="r1-updated",
        idempotency_key="dup",
    )
    assert first.id == second.id
    assert second.projection_key == "r1-updated"
    assert len(repo.list_for_source("reports", source_id)) == 1


@REQUIRES_DB
def test_projection_failure_warning_does_not_mutate_canonical(db_session):
    from src.db.repositories.projection_repository import ProjectionRepository
    from src.db.repositories.source_document_repository import SourceDocumentRepository

    docs = SourceDocumentRepository(db_session)
    doc = docs.add_document(
        document_type="news",
        source_name="Reuters",
        content_hash="h1",
        collected_at=datetime.now(UTC),
    )
    chunk = docs.add_chunk(doc.id, 0, "본문", "ch0")

    repo = ProjectionRepository(db_session)
    record = repo.upsert_status(
        source_table="document_chunks",
        source_id=chunk.id,
        target_system="pinecone",
        projection_type="chunk_embedding",
        projection_key="k",
        idempotency_key="i",
    )
    repo.mark_failure(record, at=datetime.now(UTC), error_message="boom")
    warning = repo.failure_warning(record)

    assert warning["status"] == "failed"
    assert warning["error_message"] == "boom"
    assert warning["target_system"] == "pinecone"
    assert warning["source_table"] == "document_chunks"

    # SER-008: canonical records remain intact after a projection failure.
    db_session.refresh(chunk)
    db_session.refresh(doc)
    assert chunk.chunk_text == "본문"
    assert doc.status == "active"


@REQUIRES_DB
def test_projection_reupsert_resets_status_to_pending(db_session):
    """Pin the re-index-queued semantic: re-calling upsert_status with the same
    idempotency_key (no explicit status) resets status to 'pending', documents
    that a fresh re-projection attempt is queued (spec.md scenario #1).
    An explicit status= override is also honoured."""
    from src.db.repositories.projection_repository import ProjectionRepository

    repo = ProjectionRepository(db_session)
    source_id = uuid.uuid4()

    # Step 1: create a record (status defaults to "pending").
    record = repo.upsert_status(
        source_table="document_chunks",
        source_id=source_id,
        target_system="pinecone",
        projection_type="chunk_embedding",
        projection_key="vec-A",
        idempotency_key="reupsert-test",
    )
    original_id = record.id
    assert record.status == "pending"

    # Step 2: mark it success so we can confirm the reset.
    repo.mark_success(record, at=datetime.now(UTC))
    assert record.status == "success"

    # Step 3: re-call upsert_status with the same idempotency_key and a new
    # projection_key, but no explicit status → status must reset to "pending".
    second = repo.upsert_status(
        source_table="document_chunks",
        source_id=source_id,
        target_system="pinecone",
        projection_type="chunk_embedding",
        projection_key="vec-B",
        idempotency_key="reupsert-test",
    )
    assert second.id == original_id, "upsert must return the same row, not a new one"
    assert second.status == "pending", (
        "re-upsert without explicit status must reset to 'pending' "
        "(re-index-queued semantic)"
    )
    assert second.projection_key == "vec-B", "projection_key must be updated on re-upsert"

    # Step 4: re-call upsert_status with explicit status="success" → honoured.
    third = repo.upsert_status(
        source_table="document_chunks",
        source_id=source_id,
        target_system="pinecone",
        projection_type="chunk_embedding",
        projection_key="vec-B",
        idempotency_key="reupsert-test",
        status="success",
    )
    assert third.id == original_id
    assert third.status == "success", "explicit status= must be honoured on re-upsert"


@REQUIRES_DB
def test_keyword_term_upsert_is_unique_per_normalized_language(db_session):
    from src.db.repositories.projection_repository import ProjectionRepository

    repo = ProjectionRepository(db_session)
    a = repo.upsert_term("Apple", "apple", language="en")
    b = repo.upsert_term("APPLE", "apple", language="en")
    assert a.id == b.id  # same (normalized_term, language) returns existing

    ko = repo.upsert_term("사과", "apple", language="ko")
    assert ko.id != a.id  # different language is a distinct term

    none1 = repo.upsert_term("apple", "apple", language=None)
    none2 = repo.upsert_term("apple", "apple", language=None)
    assert none1.id == none2.id  # NULL language collides with NULL language

    assert len(repo.list_terms()) == 3


@REQUIRES_DB
def test_source_document_recrawl_preserves_revision_lineage(db_session):
    from src.db.repositories.source_document_repository import SourceDocumentRepository

    docs = SourceDocumentRepository(db_session)
    original = docs.add_document(
        document_type="news",
        source_name="Reuters",
        content_hash="h1",
        collected_at=datetime.now(UTC),
        source_url="http://example.test/a",
    )
    corrected = docs.add_correction(
        original, content_hash="h2", collected_at=datetime.now(UTC)
    )
    assert corrected.id != original.id
    assert corrected.revision_group_id == original.revision_group_id
    assert corrected.supersedes_document_id == original.id


@REQUIRES_DB
def test_document_chunk_get_or_add_is_idempotent(db_session):
    from src.db.repositories.source_document_repository import SourceDocumentRepository

    docs = SourceDocumentRepository(db_session)
    doc = docs.add_document(
        document_type="news",
        source_name="X",
        content_hash="h",
        collected_at=datetime.now(UTC),
    )
    first = docs.get_or_add_chunk(doc.id, 0, "hello", "ch0")
    second = docs.get_or_add_chunk(doc.id, 0, "hello", "ch0")
    assert first.id == second.id
    assert len(docs.list_chunks(doc.id)) == 1


@REQUIRES_DB
def test_graph_rules_scenarios_conditions_and_join(db_session):
    from src.db.repositories.graph_repository import GraphRepository

    repo = GraphRepository(db_session)
    ticker = make_ticker(db_session)
    rule = repo.add_rule(rule_code="W3-EXT", name="Wave 3 extension", rule_type="impulse")
    scenario = repo.add_scenario(
        name="Primary count", ticker_id=ticker.id, confidence_label="moderate"
    )
    condition = repo.add_invalidation_condition(
        scenario.id,
        condition_text="Breaks below the wave 1 high",
        metric_name="price",
        threshold_value=150.0,
        direction="below",
    )
    assert condition.threshold_value == {"value": 150.0}  # scalar wrapped as JSON

    link1 = repo.link_scenario_rule(scenario.id, rule.id, role="primary")
    link2 = repo.link_scenario_rule(scenario.id, rule.id, role="primary")
    assert link1.id == link2.id  # idempotent on (scenario, rule, role)

    rules = repo.list_rules_for_scenario(scenario.id)
    assert [r.rule_code for r in rules] == ["W3-EXT"]


@REQUIRES_DB
def test_wave_rule_code_is_unique(db_session):
    from sqlalchemy.exc import IntegrityError

    from src.db.repositories.graph_repository import GraphRepository

    repo = GraphRepository(db_session)
    repo.add_rule(rule_code="DUP", name="A", rule_type="impulse")
    with pytest.raises(IntegrityError):
        repo.add_rule(rule_code="DUP", name="B", rule_type="corrective")
