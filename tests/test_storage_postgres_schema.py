import importlib
import os
import pytest
from sqlalchemy import create_engine, inspect

REQUIRES_DB = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set"
)


def test_config_exposes_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    import src.config as config
    importlib.reload(config)
    assert config.DATABASE_URL == "postgresql+psycopg://u:p@localhost:5432/db"


def test_base_metadata_and_session_factory_import():
    from src.db.models import Base
    from src.db.postgres import SessionLocal, session_scope, get_engine
    assert Base.metadata is not None
    assert SessionLocal is not None
    assert callable(session_scope)
    assert callable(get_engine)


def test_status_vocabularies_present():
    from src.db import constants
    assert "pending" in constants.REQUEST_STATUSES
    assert "draft" in constants.REPORT_STATUSES
    assert "pass" in constants.SAFETY_STATUSES
    assert "success" in constants.RESULT_STATUSES


EXPECTED_TABLES = {
    "users", "tickers", "analysis_requests", "workflow_node_runs",
    "analysis_results", "reports", "report_versions", "evidence_items",
    "report_evidence_citations", "source_documents", "document_chunks",
    "structured_log_events",
}


def test_metadata_has_exactly_us1_and_us2_tables():
    from src.db.models import Base
    assert set(Base.metadata.tables) == EXPECTED_TABLES | US2_EXPECTED_TABLES


def test_key_unique_constraints_declared():
    from src.db.models import Base
    tickers = Base.metadata.tables["tickers"]
    ticker_unique = next(
        con
        for con in tickers.constraints
        if con.__class__.__name__ == "UniqueConstraint"
        and tuple(sorted(c.name for c in con.columns)) == ("market", "symbol")
    )
    uniques = {tuple(sorted(c.name for c in con.columns))
               for con in tickers.constraints
               if con.__class__.__name__ == "UniqueConstraint"}
    assert ("market", "symbol") in uniques
    assert ticker_unique.dialect_options["postgresql"]["nulls_not_distinct"] is True

    versions = Base.metadata.tables["report_versions"]
    v_uniques = {tuple(sorted(c.name for c in con.columns))
                 for con in versions.constraints
                 if con.__class__.__name__ == "UniqueConstraint"}
    assert ("report_id", "version_number") in v_uniques


def test_evaluation_score_columns_return_float_values():
    from src.db.models import Base

    for table_name in ("workflow_node_runs", "reports", "structured_log_events"):
        column = Base.metadata.tables[table_name].c.evaluation_score
        assert column.type.asdecimal is False


@REQUIRES_DB
def test_alembic_upgrade_creates_all_tables(alembic_migrated_db):
    engine = create_engine(os.environ["TEST_DATABASE_URL"], future=True)
    table_names = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES.issubset(table_names)
    engine.dispose()


US2_EXPECTED_TABLES = {
    "index_projection_status",
    "keyword_terms",
    "wave_rules",
    "wave_scenarios",
    "wave_invalidation_conditions",
    "wave_scenario_rules",
    "evidence_paths",
    "evidence_path_steps",
}


@REQUIRES_DB
def test_alembic_upgrade_creates_us2_tables(alembic_migrated_db):
    engine = create_engine(os.environ["TEST_DATABASE_URL"], future=True)
    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    assert US2_EXPECTED_TABLES.issubset(table_names)

    # Key uniqueness constraints exist (names follow the uq_ naming convention).
    proj_uniques = {c["name"] for c in insp.get_unique_constraints("index_projection_status")}
    assert any("target_system" in name or "idempotency" in name for name in proj_uniques) or any(
        set(c["column_names"]) == {"target_system", "projection_type", "idempotency_key"}
        for c in insp.get_unique_constraints("index_projection_status")
    )
    step_uniques = [
        set(c["column_names"]) for c in insp.get_unique_constraints("evidence_path_steps")
    ]
    assert {"evidence_path_id", "step_index"} in step_uniques
    engine.dispose()
