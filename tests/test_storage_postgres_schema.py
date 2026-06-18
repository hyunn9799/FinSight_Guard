import importlib


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
