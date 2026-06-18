"""Shared pytest fixtures, including the PostgreSQL test session."""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def _test_url() -> str | None:
    return os.getenv("TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def alembic_migrated_db():
    url = _test_url()
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    command.upgrade(cfg, "head")
    return url


@pytest.fixture()
def db_session(alembic_migrated_db):
    engine = create_engine(alembic_migrated_db, future=True)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()
