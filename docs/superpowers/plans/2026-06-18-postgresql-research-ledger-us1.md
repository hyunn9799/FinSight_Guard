# PostgreSQL Research Ledger (004 — US1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PostgreSQL the canonical store for one end-to-end research workflow run (request → node runs → results → report versions → evidence → citations → source documents/chunks → log events), while keeping the existing JSON/Markdown files as export artifacts.

**Architecture:** A narrow persistence boundary under `src/db` (SQLAlchemy 2.x ORM + Alembic). Agents, LangGraph nodes, API, and UI touch PostgreSQL only through repository classes. `src/storage/report_store.py` keeps its current function signatures but writes to PostgreSQL first, then emits the existing files. PostgreSQL is required at runtime; on a persistence failure the workflow returns a deterministic degraded/error result.

**Tech Stack:** Python 3.12, SQLAlchemy 2.x, Alembic, psycopg 3, PostgreSQL 16 (docker-compose), pytest. Existing: FastAPI, Streamlit, Pydantic v2, LangGraph.

## Global Constraints

- Python 3.12; match the active `.venv`.
- Deterministic tests only — NO live external provider calls (yfinance, Tavily, Firecrawl, OpenAI, Pinecone, Neo4j, OpenSearch, Redis). Tests use a real local PostgreSQL via env DSN.
- Tests connect via `TEST_DATABASE_URL`; when unset, PostgreSQL tests are skipped explicitly (never silently pass).
- Test schema is created by running **Alembic migrations** (not `create_all`); each test runs inside a transaction that is rolled back.
- PostgreSQL is **required** at runtime. Files remain export-only. No silent files-only fallback.
- Source-of-truth direction: derived stores (Pinecone/Neo4j/OpenSearch) and Redis are out of scope for this slice.
- Safety: NO field that represents brokerage connection, order placement/execution, guaranteed return, guaranteed target, or buy/sell/hold instruction. Korean final-report disclaimer is preserved. Important numeric/factual claims resolve to `evidence_items`.
- Missing provider data is stored as `missing_data_notes` — never fabricated.
- US1 scope is 12 tables only: `users`, `tickers`, `analysis_requests`, `workflow_node_runs`, `analysis_results`, `reports`, `report_versions`, `evidence_items`, `report_evidence_citations`, `source_documents`, `document_chunks`, `structured_log_events`. No US2/US3/deferred tables.
- Run all Python via `.venv/bin/python`. Commit after every task.

---

## File Structure

**Create:**
- `src/db/__init__.py` — package marker
- `src/db/postgres.py` — engine + session factory, DSN loading
- `src/db/constants.py` — status vocabularies
- `src/db/models.py` — Base, mixins, the 12 US1 ORM models
- `src/db/repositories/__init__.py` — package marker
- `src/db/repositories/base.py` — `BaseRepository` session/transaction helper
- `src/db/repositories/analysis_repository.py` — ticker, request, node run, result, log event
- `src/db/repositories/evidence_repository.py` — evidence items, citations
- `src/db/repositories/report_repository.py` — report, report version, current version, safety
- `src/db/repositories/source_document_repository.py` — source documents (+ lineage), chunks
- `src/db/persistence.py` — `persist_research_run(...)` orchestration used by the workflow
- `src/db/migrations/env.py`, `src/db/migrations/script.py.mako`, `src/db/migrations/versions/` — Alembic
- `alembic.ini` — Alembic config (repo root)
- `tests/conftest.py` — DB session fixture (session-scoped migration + per-test rollback)
- `tests/fixtures/postgres.py` — DSN helper + sample-data builders
- `tests/test_storage_postgres_schema.py` — schema catalog/constraint assertions
- `tests/test_storage_postgres_repositories.py` — repository behavior
- `tests/test_storage_postgres_research_flow.py` — full ledger integration + degraded run
- `tests/test_storage_postgres_safety.py` — no-forbidden-fields schema review

**Modify:**
- `requirements.txt` — add `sqlalchemy>=2.0`, `alembic>=1.13`, `psycopg[binary]>=3.1`
- `.env.example` — add `DATABASE_URL`, `TEST_DATABASE_URL`
- `docker-compose.yml` — add `db` (postgres:16) service + healthcheck; API/UI `depends_on: db`
- `src/config.py` — add `DATABASE_URL`
- `src/storage/report_store.py` — PG-first write inside existing functions
- `src/storage/run_store.py` — mirror run metadata to PG
- `src/graph/workflow.py:329-378` — `save_report_node` persists the full ledger
- `main.py` — `/analyze`, `/backtest`, `/backtest/optimize`, `/reports/{run_id}` use persisted IDs, backward compatible

---

## Task 1: Dependencies, config, docker-compose Postgres

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `src/config.py`
- Test: `tests/test_storage_postgres_schema.py`

**Interfaces:**
- Produces: `src.config.DATABASE_URL: str` (env `DATABASE_URL`, default `postgresql+psycopg://finsight:finsight@localhost:5432/finsight`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage_postgres_schema.py
import importlib


def test_config_exposes_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    import src.config as config
    importlib.reload(config)
    assert config.DATABASE_URL == "postgresql+psycopg://u:p@localhost:5432/db"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_schema.py::test_config_exposes_database_url -v`
Expected: FAIL with `AttributeError: module 'src.config' has no attribute 'DATABASE_URL'`

- [ ] **Step 3: Add deps, env vars, compose service, config**

Append to `requirements.txt`:

```text
sqlalchemy>=2.0
alembic>=1.13
psycopg[binary]>=3.1
```

Install: `.venv/bin/python -m pip install "sqlalchemy>=2.0" "alembic>=1.13" "psycopg[binary]>=3.1"`

Append to `.env.example`:

```text

# PostgreSQL source of truth (004 US1)
DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight
# Test database — when unset, PostgreSQL tests are skipped
TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test
```

Add to `src/config.py` (after the existing lines):

```python
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://finsight:finsight@localhost:5432/finsight",
)
```

Add the `db` service to `docker-compose.yml` and make the apps depend on it:

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: finsight
      POSTGRES_PASSWORD: finsight
      POSTGRES_DB: finsight
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U finsight"]
      interval: 5s
      timeout: 5s
      retries: 5

  finance-agent-api:
    build: .
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./reports:/app/reports
      - ./logs:/app/logs
    depends_on:
      db:
        condition: service_healthy

  finance-agent-ui:
    build: .
    command: streamlit run app.py --server.address 0.0.0.0 --server.port 8501
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./reports:/app/reports
      - ./logs:/app/logs
    depends_on:
      - finance-agent-api

volumes:
  pgdata:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_schema.py::test_config_exposes_database_url -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .env.example docker-compose.yml src/config.py tests/test_storage_postgres_schema.py
git commit -m "feat(db): add postgres deps, config, and docker-compose service"
```

---

## Task 2: Engine/session factory, status constants, ORM Base

**Files:**
- Create: `src/db/__init__.py`, `src/db/postgres.py`, `src/db/constants.py`, `src/db/models.py`
- Test: `tests/test_storage_postgres_schema.py`

**Interfaces:**
- Produces:
  - `src.db.postgres.get_engine() -> sqlalchemy.Engine` (cached, from `config.DATABASE_URL`)
  - `src.db.postgres.SessionLocal` — `sessionmaker[Session]`
  - `src.db.postgres.session_scope() -> ContextManager[Session]` (commit on success, rollback on error)
  - `src.db.models.Base` — declarative base; `Base.metadata`
  - `src.db.models.UUIDMixin` (`id`), `src.db.models.TimestampMixin` (`created_at`, `updated_at`)
  - `src.db.constants.REQUEST_STATUSES`, `RESULT_STATUSES`, `REPORT_STATUSES`, `SAFETY_STATUSES` (frozensets)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_storage_postgres_schema.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_schema.py::test_base_metadata_and_session_factory_import -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.db'`

- [ ] **Step 3: Create the modules**

`src/db/__init__.py`:

```python
"""PostgreSQL source-of-truth persistence boundary."""
```

`src/db/constants.py`:

```python
"""Shared status vocabularies for canonical persistence."""

REQUEST_STATUSES = frozenset(
    {"pending", "running", "success", "degraded", "insufficient_data", "failed", "cancelled"}
)
RESULT_STATUSES = frozenset({"success", "degraded", "insufficient_data", "failed"})
REPORT_STATUSES = frozenset({"draft", "final", "failed_review", "archived"})
SAFETY_STATUSES = frozenset({"pass", "fail", "not_evaluated"})
REPORT_VERSION_STAGES = frozenset({"draft", "rewrite", "final", "failed"})
RESULT_TYPES = frozenset(
    {
        "market", "fundamental", "news", "graph_context", "backtest",
        "optimization", "coordinator_draft", "evaluation", "rewrite",
    }
)
REQUEST_TYPES = frozenset({"research", "backtest", "robust_optimization", "graph_context"})
```

`src/db/postgres.py`:

```python
"""SQLAlchemy engine and session management."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE_URL


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine for the configured database."""
    return create_engine(DATABASE_URL, future=True, pool_pre_ping=True)


SessionLocal = sessionmaker(bind=get_engine(), class_=Session, expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session: commit on success, rollback on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

`src/db/models.py`:

```python
"""SQLAlchemy ORM models for the PostgreSQL source of truth (US1)."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class UUIDMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_schema.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/db/__init__.py src/db/postgres.py src/db/constants.py src/db/models.py tests/test_storage_postgres_schema.py
git commit -m "feat(db): add engine/session factory, status constants, ORM base"
```

---

## Task 3: Define the 12 US1 ORM models

**Files:**
- Modify: `src/db/models.py`
- Test: `tests/test_storage_postgres_schema.py`

**Interfaces:**
- Produces ORM classes (all under `src.db.models`): `User`, `Ticker`, `AnalysisRequest`, `WorkflowNodeRun`, `AnalysisResult`, `Report`, `ReportVersion`, `EvidenceItemRecord`, `ReportEvidenceCitation`, `SourceDocument`, `DocumentChunk`, `StructuredLogEvent`. Table names are the snake_case plurals listed in Global Constraints.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_storage_postgres_schema.py
EXPECTED_TABLES = {
    "users", "tickers", "analysis_requests", "workflow_node_runs",
    "analysis_results", "reports", "report_versions", "evidence_items",
    "report_evidence_citations", "source_documents", "document_chunks",
    "structured_log_events",
}


def test_metadata_has_exactly_us1_tables():
    from src.db.models import Base
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_key_unique_constraints_declared():
    from src.db.models import Base
    tickers = Base.metadata.tables["tickers"]
    uniques = {tuple(sorted(c.name for c in con.columns))
               for con in tickers.constraints
               if con.__class__.__name__ == "UniqueConstraint"}
    assert ("market", "symbol") in uniques

    versions = Base.metadata.tables["report_versions"]
    v_uniques = {tuple(sorted(c.name for c in con.columns))
                 for con in versions.constraints
                 if con.__class__.__name__ == "UniqueConstraint"}
    assert ("report_id", "version_number") in v_uniques
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_schema.py::test_metadata_has_exactly_us1_tables -v`
Expected: FAIL (metadata empty / KeyError)

- [ ] **Step 3: Append the models to `src/db/models.py`**

Add these imports to the existing import block:

```python
from sqlalchemy import ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, Boolean
from sqlalchemy.dialects.postgresql import JSONB
```

Append the model classes:

```python
class User(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "users"
    email: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    role: Mapped[str] = mapped_column(String, default="demo", nullable=False)
    status: Mapped[str] = mapped_column(String, default="active", nullable=False)
    anonymized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Ticker(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "tickers"
    __table_args__ = (UniqueConstraint("symbol", "market"),)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    market: Mapped[str | None] = mapped_column(String, nullable=True)
    exchange: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str | None] = mapped_column(String, nullable=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    sector: Mapped[str | None] = mapped_column(String, nullable=True)
    industry: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class AnalysisRequest(UUIDMixin, Base):
    __tablename__ = "analysis_requests"
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    request_type: Mapped[str] = mapped_column(String, nullable=False)
    horizon: Mapped[str | None] = mapped_column(String, nullable=True)
    risk_profile: Mapped[str | None] = mapped_column(String, nullable=True)
    parameters: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    degraded_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    warning_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WorkflowNodeRun(UUIDMixin, Base):
    __tablename__ = "workflow_node_runs"
    __table_args__ = (UniqueConstraint("request_id", "node_name", "attempt_number"),)
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    node_name: Mapped[str] = mapped_column(String, nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluation_score: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    node_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)


class AnalysisResult(UUIDMixin, Base):
    __tablename__ = "analysis_results"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    result_type: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    missing_data_notes: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Report(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "reports"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("report_versions.id", use_alter=True, name="fk_reports_current_version"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String, default="", nullable=False)
    language: Mapped[str] = mapped_column(String, default="ko", nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    safety_status: Mapped[str] = mapped_column(String, default="not_evaluated", nullable=False)
    evaluation_score: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    disclaimer_present: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class ReportVersion(UUIDMixin, Base):
    __tablename__ = "report_versions"
    __table_args__ = (UniqueConstraint("report_id", "version_number"),)
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reports.id"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    report_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    report_markdown: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_by_node: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvidenceItemRecord(UUIDMixin, Base):
    __tablename__ = "evidence_items"
    evidence_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    request_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("analysis_requests.id"), nullable=True)
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    analysis_result_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("analysis_results.id"), nullable=True)
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("source_documents.id"), nullable=True)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_name: Mapped[str] = mapped_column(String, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metric_name: Mapped[str] = mapped_column(String, nullable=False)
    metric_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ReportEvidenceCitation(UUIDMixin, Base):
    __tablename__ = "report_evidence_citations"
    __table_args__ = (
        UniqueConstraint("report_version_id", "evidence_item_id", "section_name", "claim_text"),
    )
    report_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("report_versions.id"), nullable=False)
    evidence_item_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("evidence_items.id"), nullable=False)
    section_name: Mapped[str] = mapped_column(String, default="", nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SourceDocument(UUIDMixin, Base):
    __tablename__ = "source_documents"
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    document_type: Mapped[str] = mapped_column(String, nullable=False)
    source_name: Mapped[str] = mapped_column(String, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_content_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    revision_group_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    supersedes_document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("source_documents.id"), nullable=True)
    doc_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DocumentChunk(UUIDMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("source_document_id", "chunk_index"),)
    source_document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("source_documents.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class StructuredLogEvent(UUIDMixin, Base):
    __tablename__ = "structured_log_events"
    request_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("analysis_requests.id"), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    node_name: Mapped[str | None] = mapped_column(String, nullable=True)
    event_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluation_score: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
```

> Note: columns named `metadata` use a different Python attribute (`node_metadata`, `doc_metadata`, etc.) because `metadata` is reserved on the declarative Base. The DB column name stays `metadata` via the first positional arg to `mapped_column`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_schema.py -v`
Expected: PASS (all schema tests)

- [ ] **Step 5: Commit**

```bash
git add src/db/models.py tests/test_storage_postgres_schema.py
git commit -m "feat(db): define 12 US1 canonical ORM models"
```

---

## Task 4: Alembic configuration and initial migration

**Files:**
- Create: `alembic.ini`, `src/db/migrations/env.py`, `src/db/migrations/script.py.mako`, `src/db/migrations/versions/` (dir)
- Test: `tests/conftest.py` (DSN helper used below), `tests/test_storage_postgres_schema.py`

**Interfaces:**
- Produces: a single migration revision creating all 12 US1 tables; `alembic upgrade head` builds the schema. `src.db.migrations.env` reads `TEST_DATABASE_URL`/`DATABASE_URL`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_storage_postgres_schema.py
import os
import pytest
from sqlalchemy import create_engine, inspect

REQUIRES_DB = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set"
)


@REQUIRES_DB
def test_alembic_upgrade_creates_all_tables(alembic_migrated_db):
    engine = create_engine(os.environ["TEST_DATABASE_URL"], future=True)
    table_names = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES.issubset(table_names)
    engine.dispose()
```

The `alembic_migrated_db` fixture is added in Task 5; this test stays failing (fixture error) until then. That is expected — implement the migration now, wire the fixture next.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_schema.py::test_alembic_upgrade_creates_all_tables -v`
Expected: FAIL/ERROR with `fixture 'alembic_migrated_db' not found`

- [ ] **Step 3: Create Alembic config and autogenerate the migration**

`alembic.ini` (repo root) — minimal:

```ini
[alembic]
script_location = src/db/migrations
prepend_sys_path = .

[loggers]
keys = root

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

`src/db/migrations/script.py.mako`:

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

`src/db/migrations/env.py`:

```python
"""Alembic environment for the PostgreSQL source of truth."""

import os

from alembic import context
from sqlalchemy import create_engine

from src.db.models import Base

target_metadata = Base.metadata


def _url() -> str:
    return os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or ""


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url(), future=True)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Create the versions dir and autogenerate against a clean throwaway DB (requires Postgres running and `TEST_DATABASE_URL` set):

```bash
mkdir -p src/db/migrations/versions
createdb -h localhost -U finsight finsight_test 2>/dev/null || true
TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test \
  .venv/bin/alembic revision --autogenerate -m "postgresql source of truth us1"
```

Open the generated file in `src/db/migrations/versions/` and verify it creates exactly the 12 tables, the `(symbol, market)`, `(report_id, version_number)`, `(source_document_id, chunk_index)`, `(request_id, node_name, attempt_number)`, citation, and `evidence_id` unique constraints. Fix the model + regenerate if anything is missing.

- [ ] **Step 4: Run test to verify it passes**

Run (after Task 5's fixture exists, or manually now):

```bash
TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test .venv/bin/alembic upgrade head
TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test .venv/bin/python -c "from sqlalchemy import create_engine, inspect; import os; e=create_engine(os.environ['TEST_DATABASE_URL']); print(sorted(inspect(e).get_table_names()))"
```
Expected: prints the 12 table names plus `alembic_version`.

- [ ] **Step 5: Commit**

```bash
git add alembic.ini src/db/migrations
git commit -m "feat(db): add alembic config and initial US1 migration"
```

---

## Task 5: Test DB fixtures (migration + per-test rollback)

**Files:**
- Create: `tests/conftest.py`, `tests/fixtures/postgres.py`
- Test: `tests/test_storage_postgres_schema.py` (re-run Task 4's test, now green)

**Interfaces:**
- Produces pytest fixtures:
  - `alembic_migrated_db` (session scope): runs `alembic upgrade head` against `TEST_DATABASE_URL` once.
  - `db_session` (function scope): a `Session` bound to a connection inside a transaction that is rolled back after the test.
  - `tests.fixtures.postgres.make_ticker(session, **kw) -> Ticker` and `make_request(session, ticker, **kw) -> AnalysisRequest` sample builders.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_storage_postgres_repositories.py  (new file)
import os
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_repositories.py -v`
Expected: FAIL/ERROR `fixture 'db_session' not found`

- [ ] **Step 3: Create the fixtures**

`tests/conftest.py`:

```python
"""Shared pytest fixtures, including the PostgreSQL test session."""

import os

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
    cfg = Config("alembic.ini")
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
```

`tests/fixtures/postgres.py`:

```python
"""Deterministic sample-record builders for PostgreSQL tests."""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.db.models import AnalysisRequest, Ticker


def make_ticker(session: Session, symbol: str = "AAPL", market: str = "NASDAQ", **kw) -> Ticker:
    ticker = Ticker(symbol=symbol, market=market, **kw)
    session.add(ticker)
    session.flush()
    return ticker


def make_request(session: Session, ticker: Ticker, request_type: str = "research", **kw) -> AnalysisRequest:
    request = AnalysisRequest(
        ticker_id=ticker.id,
        request_type=request_type,
        status=kw.pop("status", "pending"),
        created_at=kw.pop("created_at", datetime.now(UTC)),
        **kw,
    )
    session.add(request)
    session.flush()
    return request
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test \
  .venv/bin/python -m pytest tests/test_storage_postgres_repositories.py tests/test_storage_postgres_schema.py -v
```
Expected: PASS (including `test_alembic_upgrade_creates_all_tables`).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/fixtures/postgres.py tests/test_storage_postgres_repositories.py
git commit -m "test(db): add migration + rollback session fixtures"
```

---

## Task 6: Base repository helper

**Files:**
- Create: `src/db/repositories/__init__.py`, `src/db/repositories/base.py`
- Test: `tests/test_storage_postgres_repositories.py`

**Interfaces:**
- Produces: `class BaseRepository:` with `__init__(self, session: Session)` storing `self.session`, and `def get(self, model, id_)` returning an instance or `None`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_storage_postgres_repositories.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_repositories.py::test_base_repository_get -v`
Expected: FAIL `ModuleNotFoundError: No module named 'src.db.repositories.base'`

- [ ] **Step 3: Create the base repository**

`src/db/repositories/__init__.py`:

```python
"""Repository boundary for PostgreSQL persistence."""
```

`src/db/repositories/base.py`:

```python
"""Base repository with shared session access."""

from typing import TypeVar

from sqlalchemy.orm import Session

from src.db.models import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository:
    """Holds a session; subclasses add table-specific methods."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, model: type[ModelT], id_) -> ModelT | None:
        return self.session.get(model, id_)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test .venv/bin/python -m pytest tests/test_storage_postgres_repositories.py::test_base_repository_get -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/repositories/__init__.py src/db/repositories/base.py tests/test_storage_postgres_repositories.py
git commit -m "feat(db): add base repository helper"
```

---

## Task 7: Analysis repository (ticker, request, node run, result, log event)

**Files:**
- Create: `src/db/repositories/analysis_repository.py`
- Test: `tests/test_storage_postgres_repositories.py`

**Interfaces:**
- Consumes: `BaseRepository`, models from Task 3.
- Produces `class AnalysisRepository(BaseRepository):`
  - `upsert_ticker(self, symbol: str, market: str | None = None, **fields) -> Ticker` — uppercases symbol; returns existing row for `(symbol, market)` or creates one.
  - `create_request(self, ticker_id, request_type, *, user_id=None, parameters=None, status="pending", **fields) -> AnalysisRequest`
  - `update_request_status(self, request_id, status, *, degraded_reason=None, warning_summary=None, error_summary=None, completed_at=None) -> AnalysisRequest`
  - `record_node_run(self, request_id, run_id, node_name, status, *, attempt_number=1, duration_ms=None, error_type=None, error_message=None, evaluation_score=None, metadata=None) -> WorkflowNodeRun`
  - `add_result(self, request_id, result_type, *, ticker_id=None, summary="", metrics=None, warnings=None, missing_data_notes=None, status="success") -> AnalysisResult`
  - `add_log_event(self, *, event_name, status, request_id=None, run_id=None, ticker_id=None, node_name=None, message=None, error_message=None, evaluation_score=None, occurred_at) -> StructuredLogEvent`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_storage_postgres_repositories.py
from datetime import UTC, datetime


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
    repo.record_node_run(req.id, "run-1", "market_node", "failed", attempt_number=1)
    with _pytest.raises(IntegrityError):
        db_session.flush()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_repositories.py -k "upsert_ticker or lifecycle or unique_per_attempt" -v`
Expected: FAIL `ModuleNotFoundError: ... analysis_repository`

- [ ] **Step 3: Implement the repository**

`src/db/repositories/analysis_repository.py`:

```python
"""Repository for tickers, requests, node runs, results, and log events."""

from datetime import datetime

from src.db.models import (
    AnalysisRequest,
    AnalysisResult,
    StructuredLogEvent,
    Ticker,
    WorkflowNodeRun,
)
from src.db.repositories.base import BaseRepository


class AnalysisRepository(BaseRepository):
    def upsert_ticker(self, symbol: str, market: str | None = None, **fields) -> Ticker:
        symbol = symbol.strip().upper()
        existing = (
            self.session.query(Ticker)
            .filter(Ticker.symbol == symbol, Ticker.market == market)
            .one_or_none()
        )
        if existing is not None:
            return existing
        ticker = Ticker(symbol=symbol, market=market, **fields)
        self.session.add(ticker)
        self.session.flush()
        return ticker

    def create_request(
        self,
        ticker_id,
        request_type: str,
        *,
        user_id=None,
        parameters: dict | None = None,
        status: str = "pending",
        **fields,
    ) -> AnalysisRequest:
        request = AnalysisRequest(
            ticker_id=ticker_id,
            user_id=user_id,
            request_type=request_type,
            parameters=parameters or {},
            status=status,
            **fields,
        )
        self.session.add(request)
        self.session.flush()
        return request

    def update_request_status(
        self,
        request_id,
        status: str,
        *,
        degraded_reason: str | None = None,
        warning_summary: str | None = None,
        error_summary: str | None = None,
        completed_at: datetime | None = None,
    ) -> AnalysisRequest:
        request = self.session.get(AnalysisRequest, request_id)
        request.status = status
        if degraded_reason is not None:
            request.degraded_reason = degraded_reason
        if warning_summary is not None:
            request.warning_summary = warning_summary
        if error_summary is not None:
            request.error_summary = error_summary
        if completed_at is not None:
            request.completed_at = completed_at
        self.session.flush()
        return request

    def record_node_run(
        self,
        request_id,
        run_id: str,
        node_name: str,
        status: str,
        *,
        attempt_number: int = 1,
        duration_ms: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        evaluation_score: float | None = None,
        metadata: dict | None = None,
    ) -> WorkflowNodeRun:
        node = WorkflowNodeRun(
            request_id=request_id,
            run_id=run_id,
            node_name=node_name,
            status=status,
            attempt_number=attempt_number,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=error_message,
            evaluation_score=evaluation_score,
            node_metadata=metadata or {},
        )
        self.session.add(node)
        self.session.flush()
        return node

    def add_result(
        self,
        request_id,
        result_type: str,
        *,
        ticker_id=None,
        summary: str = "",
        metrics: dict | None = None,
        warnings: list | None = None,
        missing_data_notes: list | None = None,
        status: str = "success",
    ) -> AnalysisResult:
        result = AnalysisResult(
            request_id=request_id,
            ticker_id=ticker_id,
            result_type=result_type,
            summary=summary,
            metrics=metrics or {},
            warnings=warnings or [],
            missing_data_notes=missing_data_notes or [],
            status=status,
        )
        self.session.add(result)
        self.session.flush()
        return result

    def add_log_event(
        self,
        *,
        event_name: str,
        status: str,
        occurred_at: datetime,
        request_id=None,
        run_id: str | None = None,
        ticker_id=None,
        node_name: str | None = None,
        message: str | None = None,
        error_message: str | None = None,
        evaluation_score: float | None = None,
    ) -> StructuredLogEvent:
        event = StructuredLogEvent(
            event_name=event_name,
            status=status,
            occurred_at=occurred_at,
            request_id=request_id,
            run_id=run_id,
            ticker_id=ticker_id,
            node_name=node_name,
            message=message,
            error_message=error_message,
            evaluation_score=evaluation_score,
        )
        self.session.add(event)
        self.session.flush()
        return event
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test .venv/bin/python -m pytest tests/test_storage_postgres_repositories.py -k "upsert_ticker or lifecycle or unique_per_attempt" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/repositories/analysis_repository.py tests/test_storage_postgres_repositories.py
git commit -m "feat(db): add analysis repository"
```

---

## Task 8: Evidence repository (evidence items + citations)

**Files:**
- Create: `src/db/repositories/evidence_repository.py`
- Test: `tests/test_storage_postgres_repositories.py`

**Interfaces:**
- Consumes: `BaseRepository`, `EvidenceItemRecord`, `ReportEvidenceCitation`, and the Pydantic `EvidenceItem` from `src.evidence.evidence_schema`.
- Produces `class EvidenceRepository(BaseRepository):`
  - `add_evidence(self, item: EvidenceItem, *, request_id=None, ticker_id=None, analysis_result_id=None, source_document_id=None) -> EvidenceItemRecord` — maps the Pydantic `EvidenceItem` fields onto the row; `metric_value` is wrapped as `{"value": ...}` JSON.
  - `list_for_request(self, request_id) -> list[EvidenceItemRecord]`
  - `add_citation(self, report_version_id, evidence_item_id, *, section_name="", claim_text="") -> ReportEvidenceCitation`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_storage_postgres_repositories.py
@REQUIRES_DB
def test_add_evidence_from_pydantic_and_cite(db_session):
    from datetime import UTC, datetime
    from src.evidence.evidence_schema import EvidenceItem
    from src.db.repositories.analysis_repository import AnalysisRepository
    from src.db.repositories.evidence_repository import EvidenceRepository
    from src.db.repositories.report_repository import ReportRepository

    analysis = AnalysisRepository(db_session)
    ticker = analysis.upsert_ticker("AAPL", "NASDAQ")
    req = analysis.create_request(ticker.id, "research")

    item = EvidenceItem(
        evidence_id="ev-1", source_type="market", source_name="yfinance",
        collected_at=datetime.now(UTC), ticker="AAPL", metric_name="close",
        metric_value=190.5, description="closing price",
    )
    ev_repo = EvidenceRepository(db_session)
    row = ev_repo.add_evidence(item, request_id=req.id, ticker_id=ticker.id)
    assert row.evidence_id == "ev-1"
    assert row.metric_value == {"value": 190.5}
    assert [r.evidence_id for r in ev_repo.list_for_request(req.id)] == ["ev-1"]

    reports = ReportRepository(db_session)
    report = reports.create_report(req.id, ticker.id, title="t")
    version = reports.add_version(report.id, 1, "draft", {"k": "v"}, "md", created_by_node="coordinator_node")
    citation = ev_repo.add_citation(version.id, row.id, section_name="market", claim_text="close is 190.5")
    assert citation.id is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_repositories.py::test_add_evidence_from_pydantic_and_cite -v`
Expected: FAIL `ModuleNotFoundError: ... evidence_repository`

- [ ] **Step 3: Implement the repository**

`src/db/repositories/evidence_repository.py`:

```python
"""Repository for evidence items and report-evidence citations."""

from src.db.models import EvidenceItemRecord, ReportEvidenceCitation
from src.db.repositories.base import BaseRepository
from src.evidence.evidence_schema import EvidenceItem


class EvidenceRepository(BaseRepository):
    def add_evidence(
        self,
        item: EvidenceItem,
        *,
        request_id=None,
        ticker_id=None,
        analysis_result_id=None,
        source_document_id=None,
    ) -> EvidenceItemRecord:
        metric_value = None if item.metric_value is None else {"value": item.metric_value}
        record = EvidenceItemRecord(
            evidence_id=item.evidence_id,
            request_id=request_id,
            ticker_id=ticker_id,
            analysis_result_id=analysis_result_id,
            source_document_id=source_document_id,
            source_type=item.source_type,
            source_name=item.source_name,
            source_url=item.source_url,
            collected_at=item.collected_at,
            metric_name=item.metric_name,
            metric_value=metric_value,
            description=item.description,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def list_for_request(self, request_id) -> list[EvidenceItemRecord]:
        return (
            self.session.query(EvidenceItemRecord)
            .filter(EvidenceItemRecord.request_id == request_id)
            .order_by(EvidenceItemRecord.created_at)
            .all()
        )

    def add_citation(
        self,
        report_version_id,
        evidence_item_id,
        *,
        section_name: str = "",
        claim_text: str = "",
    ) -> ReportEvidenceCitation:
        citation = ReportEvidenceCitation(
            report_version_id=report_version_id,
            evidence_item_id=evidence_item_id,
            section_name=section_name,
            claim_text=claim_text,
        )
        self.session.add(citation)
        self.session.flush()
        return citation
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test .venv/bin/python -m pytest tests/test_storage_postgres_repositories.py::test_add_evidence_from_pydantic_and_cite -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/repositories/evidence_repository.py tests/test_storage_postgres_repositories.py
git commit -m "feat(db): add evidence repository"
```

---

## Task 9: Report repository (report container, versions, current version, safety)

**Files:**
- Create: `src/db/repositories/report_repository.py`
- Test: `tests/test_storage_postgres_repositories.py`

**Interfaces:**
- Consumes: `BaseRepository`, `Report`, `ReportVersion`.
- Produces `class ReportRepository(BaseRepository):`
  - `create_report(self, request_id, ticker_id, *, title="", language="ko", status="draft") -> Report`
  - `add_version(self, report_id, version_number, stage, report_json, report_markdown, *, created_by_node=None) -> ReportVersion` — also sets the report's `current_version_id` to the new version.
  - `set_status(self, report_id, *, status=None, safety_status=None, evaluation_score=None, disclaimer_present=None) -> Report`
  - `next_version_number(self, report_id) -> int` — 1 + max existing, or 1.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_storage_postgres_repositories.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_repositories.py::test_report_versions_and_current_version -v`
Expected: FAIL `ModuleNotFoundError: ... report_repository`

- [ ] **Step 3: Implement the repository**

`src/db/repositories/report_repository.py`:

```python
"""Repository for reports and immutable report versions."""

from sqlalchemy import func

from src.db.models import Report, ReportVersion
from src.db.repositories.base import BaseRepository


class ReportRepository(BaseRepository):
    def create_report(
        self,
        request_id,
        ticker_id,
        *,
        title: str = "",
        language: str = "ko",
        status: str = "draft",
    ) -> Report:
        report = Report(
            request_id=request_id,
            ticker_id=ticker_id,
            title=title,
            language=language,
            status=status,
        )
        self.session.add(report)
        self.session.flush()
        return report

    def next_version_number(self, report_id) -> int:
        current_max = (
            self.session.query(func.max(ReportVersion.version_number))
            .filter(ReportVersion.report_id == report_id)
            .scalar()
        )
        return (current_max or 0) + 1

    def add_version(
        self,
        report_id,
        version_number: int,
        stage: str,
        report_json: dict,
        report_markdown: str,
        *,
        created_by_node: str | None = None,
    ) -> ReportVersion:
        version = ReportVersion(
            report_id=report_id,
            version_number=version_number,
            stage=stage,
            report_json=report_json,
            report_markdown=report_markdown,
            created_by_node=created_by_node,
        )
        self.session.add(version)
        self.session.flush()
        report = self.session.get(Report, report_id)
        report.current_version_id = version.id
        self.session.flush()
        return version

    def set_status(
        self,
        report_id,
        *,
        status: str | None = None,
        safety_status: str | None = None,
        evaluation_score: float | None = None,
        disclaimer_present: bool | None = None,
    ) -> Report:
        report = self.session.get(Report, report_id)
        if status is not None:
            report.status = status
        if safety_status is not None:
            report.safety_status = safety_status
        if evaluation_score is not None:
            report.evaluation_score = evaluation_score
        if disclaimer_present is not None:
            report.disclaimer_present = disclaimer_present
        self.session.flush()
        return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test .venv/bin/python -m pytest tests/test_storage_postgres_repositories.py::test_report_versions_and_current_version -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/repositories/report_repository.py tests/test_storage_postgres_repositories.py
git commit -m "feat(db): add report repository"
```

---

## Task 10: Source document repository (documents + lineage + chunks)

**Files:**
- Create: `src/db/repositories/source_document_repository.py`
- Test: `tests/test_storage_postgres_repositories.py`

**Interfaces:**
- Consumes: `BaseRepository`, `SourceDocument`, `DocumentChunk`.
- Produces `class SourceDocumentRepository(BaseRepository):`
  - `add_document(self, *, document_type, source_name, content_hash, collected_at, ticker_id=None, source_url=None, title=None, language=None, published_at=None, raw_content_ref=None, metadata=None, revision_group_id=None, supersedes_document_id=None, status="active") -> SourceDocument`
  - `add_correction(self, prior: SourceDocument, *, content_hash, collected_at, **fields) -> SourceDocument` — reuses `prior.revision_group_id`, sets `supersedes_document_id=prior.id`.
  - `add_chunk(self, source_document_id, chunk_index, chunk_text, chunk_hash, *, token_count=None, metadata=None) -> DocumentChunk`
  - `list_chunks(self, source_document_id) -> list[DocumentChunk]` — ordered by `chunk_index`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_storage_postgres_repositories.py
@REQUIRES_DB
def test_source_document_revision_and_chunks(db_session):
    from datetime import UTC, datetime
    from src.db.repositories.source_document_repository import SourceDocumentRepository

    repo = SourceDocumentRepository(db_session)
    doc = repo.add_document(
        document_type="news", source_name="tavily", content_hash="h1",
        collected_at=datetime.now(UTC), source_url="https://x/1",
    )
    corrected = repo.add_correction(doc, content_hash="h2", collected_at=datetime.now(UTC))
    assert corrected.revision_group_id == doc.revision_group_id
    assert corrected.supersedes_document_id == doc.id

    repo.add_chunk(doc.id, 0, "first", "c0")
    repo.add_chunk(doc.id, 1, "second", "c1")
    chunks = repo.list_chunks(doc.id)
    assert [c.chunk_index for c in chunks] == [0, 1]


@REQUIRES_DB
def test_duplicate_chunk_index_rejected(db_session):
    import pytest as _pytest
    from datetime import UTC, datetime
    from sqlalchemy.exc import IntegrityError
    from src.db.repositories.source_document_repository import SourceDocumentRepository

    repo = SourceDocumentRepository(db_session)
    doc = repo.add_document(
        document_type="news", source_name="tavily", content_hash="h1",
        collected_at=datetime.now(UTC),
    )
    repo.add_chunk(doc.id, 0, "a", "c0")
    repo.add_chunk(doc.id, 0, "b", "c0b")
    with _pytest.raises(IntegrityError):
        db_session.flush()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_repositories.py -k "revision_and_chunks or duplicate_chunk_index" -v`
Expected: FAIL `ModuleNotFoundError: ... source_document_repository`

- [ ] **Step 3: Implement the repository**

`src/db/repositories/source_document_repository.py`:

```python
"""Repository for source documents (with revision lineage) and chunks."""

from datetime import datetime

from src.db.models import DocumentChunk, SourceDocument
from src.db.repositories.base import BaseRepository


class SourceDocumentRepository(BaseRepository):
    def add_document(
        self,
        *,
        document_type: str,
        source_name: str,
        content_hash: str,
        collected_at: datetime,
        ticker_id=None,
        source_url: str | None = None,
        title: str | None = None,
        language: str | None = None,
        published_at: datetime | None = None,
        raw_content_ref: str | None = None,
        metadata: dict | None = None,
        revision_group_id=None,
        supersedes_document_id=None,
        status: str = "active",
    ) -> SourceDocument:
        kwargs = dict(
            document_type=document_type,
            source_name=source_name,
            content_hash=content_hash,
            collected_at=collected_at,
            ticker_id=ticker_id,
            source_url=source_url,
            title=title,
            language=language,
            published_at=published_at,
            raw_content_ref=raw_content_ref,
            doc_metadata=metadata or {},
            supersedes_document_id=supersedes_document_id,
            status=status,
        )
        if revision_group_id is not None:
            kwargs["revision_group_id"] = revision_group_id
        document = SourceDocument(**kwargs)
        self.session.add(document)
        self.session.flush()
        return document

    def add_correction(
        self, prior: SourceDocument, *, content_hash: str, collected_at: datetime, **fields
    ) -> SourceDocument:
        return self.add_document(
            document_type=fields.pop("document_type", prior.document_type),
            source_name=fields.pop("source_name", prior.source_name),
            content_hash=content_hash,
            collected_at=collected_at,
            ticker_id=fields.pop("ticker_id", prior.ticker_id),
            revision_group_id=prior.revision_group_id,
            supersedes_document_id=prior.id,
            **fields,
        )

    def add_chunk(
        self,
        source_document_id,
        chunk_index: int,
        chunk_text: str,
        chunk_hash: str,
        *,
        token_count: int | None = None,
        metadata: dict | None = None,
    ) -> DocumentChunk:
        chunk = DocumentChunk(
            source_document_id=source_document_id,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            chunk_hash=chunk_hash,
            token_count=token_count,
            chunk_metadata=metadata or {},
        )
        self.session.add(chunk)
        self.session.flush()
        return chunk

    def list_chunks(self, source_document_id) -> list[DocumentChunk]:
        return (
            self.session.query(DocumentChunk)
            .filter(DocumentChunk.source_document_id == source_document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test .venv/bin/python -m pytest tests/test_storage_postgres_repositories.py -k "revision_and_chunks or duplicate_chunk_index" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/repositories/source_document_repository.py tests/test_storage_postgres_repositories.py
git commit -m "feat(db): add source document repository"
```

---

## Task 11: `persist_research_run` orchestration

**Files:**
- Create: `src/db/persistence.py`
- Test: `tests/test_storage_postgres_research_flow.py`

**Interfaces:**
- Consumes: `session_scope`, all four repositories, `ResearchReport`/`EvaluationResult`/`EvidenceItem` from `src.graph.state` / `src.evidence.evidence_schema`.
- Produces:
  - `persist_research_run(*, run_id: str, ticker: str, status: str, report: ResearchReport | None, evidence: list[EvidenceItem], evaluation: EvaluationResult | None, node_runs: list[dict] | None = None, missing_data_notes: list[str] | None = None) -> dict` returning `{"request_id": UUID, "report_id": UUID | None, "report_version_id": UUID | None}`.
  - Behavior: opens one `session_scope`; upserts ticker; creates an `analysis_requests` row (request_type `research`, status mapped from `status`); records node runs; if a report exists, creates a `reports` row + version 1 (stage `final` when status==`success` else `draft`), stores evidence, and adds a citation per evidence item to the version (`section_name="evidence_summary"`). `disclaimer_present` is set from whether `report.disclaimer` is non-empty. On `degraded`/`failed`, `degraded_reason`/`error_summary` carry `missing_data_notes`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'src.db.persistence'`

- [ ] **Step 3: Implement the orchestration**

`src/db/persistence.py`:

```python
"""Persist a full research workflow run to the PostgreSQL source of truth."""

from src.db.postgres import session_scope
from src.db.repositories.analysis_repository import AnalysisRepository
from src.db.repositories.evidence_repository import EvidenceRepository
from src.db.repositories.report_repository import ReportRepository

_RESULT_STATUS_FOR = {
    "success": "success",
    "degraded": "degraded",
    "insufficient_data": "insufficient_data",
    "failed": "failed",
}


def persist_research_run(
    *,
    run_id: str,
    ticker: str,
    status: str,
    report,
    evidence,
    evaluation,
    node_runs: list[dict] | None = None,
    missing_data_notes: list[str] | None = None,
) -> dict:
    node_runs = node_runs or []
    missing_data_notes = missing_data_notes or []
    request_status = _RESULT_STATUS_FOR.get(status, "failed")

    with session_scope() as session:
        analysis = AnalysisRepository(session)
        evidence_repo = EvidenceRepository(session)
        reports = ReportRepository(session)

        ticker_row = analysis.upsert_ticker(ticker)
        notes_text = "; ".join(missing_data_notes) or None
        request = analysis.create_request(
            ticker_row.id,
            "research",
            status=request_status,
            degraded_reason=notes_text if request_status == "degraded" else None,
            error_summary=notes_text if request_status == "failed" else None,
        )

        for index, run in enumerate(node_runs, start=1):
            analysis.record_node_run(
                request.id,
                run_id,
                run["node_name"],
                run.get("status", "success"),
                attempt_number=run.get("attempt_number", index),
                duration_ms=run.get("duration_ms"),
                error_type=run.get("error_type"),
                error_message=run.get("error_message"),
            )

        result = {"request_id": request.id, "report_id": None, "report_version_id": None}
        if report is None:
            return result

        stage = "final" if status == "success" else "draft"
        report_status = "final" if status == "success" else "draft"
        report_row = reports.create_report(
            request.id, ticker_row.id, title=getattr(report, "title", ""), status=report_status
        )
        version = reports.add_version(
            report_row.id,
            1,
            stage,
            report.model_dump(mode="json"),
            "",
            created_by_node="coordinator_node",
        )
        disclaimer_present = bool(getattr(report, "disclaimer", "").strip())
        eval_score = evaluation.source_grounding_score if evaluation is not None else None
        safety = "pass" if (evaluation is not None and evaluation.overall_pass) else "not_evaluated"
        reports.set_status(
            report_row.id,
            safety_status=safety,
            evaluation_score=eval_score,
            disclaimer_present=disclaimer_present,
        )

        for item in evidence:
            row = evidence_repo.add_evidence(item, request_id=request.id, ticker_id=ticker_row.id)
            evidence_repo.add_citation(
                version.id, row.id, section_name="evidence_summary", claim_text=item.description
            )

        result["report_id"] = report_row.id
        result["report_version_id"] = version.id
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test .venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/db/persistence.py tests/test_storage_postgres_research_flow.py
git commit -m "feat(db): add persist_research_run orchestration"
```

---

## Task 12: Wire persistence into `save_report_node`

**Files:**
- Modify: `src/graph/workflow.py:329-378`
- Test: `tests/test_storage_postgres_research_flow.py`

**Interfaces:**
- Consumes: `persist_research_run`.
- Produces: `save_report_node` calls `persist_research_run(...)` before writing files; on persistence error it returns a degraded result (status `degraded`, an appended `WorkflowError`) instead of raising; the returned dict gains `request_id` and `report_id` keys (stringified) when persistence succeeds. File export behavior is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_storage_postgres_research_flow.py
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
def test_save_report_node_degrades_on_persistence_error(monkeypatch, tmp_path):
    import src.graph.workflow as workflow

    def _boom(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(workflow, "persist_research_run", _boom)
    monkeypatch.setattr(workflow, "save_report_json", lambda *a: str(tmp_path / "r.json"))
    monkeypatch.setattr(workflow, "save_report_markdown", lambda *a: str(tmp_path / "r.md"))
    monkeypatch.setattr(workflow, "save_run", lambda *a: {})

    state = {"run_id": "run-y", "ticker": "AAPL", "status": "success", "draft_report": _sample_report()}
    out = workflow.save_report_node(state)
    assert out["status"] == "degraded"
    assert any(getattr(e, "error_type", "") == "persistence_error" for e in out["errors"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py -k save_report_node -v`
Expected: FAIL (`save_report_node` does not call/define `persist_research_run`; no `request_id` in result)

- [ ] **Step 3: Modify `save_report_node`**

Add the import near the other storage imports at the top of `src/graph/workflow.py` (around line 27):

```python
from src.db.persistence import persist_research_run
```

Replace the body of `save_report_node` after the `evaluation`/`final_status`/`payload` lines (currently lines 352-378) with persistence-first logic:

```python
    evaluation = state.get("evaluation_result")
    final_status = "success" if evaluation is not None and evaluation.overall_pass else "failed"

    persisted: dict = {}
    try:
        persisted = persist_research_run(
            run_id=run_id,
            ticker=state.get("ticker", ""),
            status=final_status,
            report=report,
            evidence=state.get("evidence", []),
            evaluation=evaluation,
            missing_data_notes=state.get("missing_data_notes", []),
        )
    except Exception as exc:  # persistence is required; degrade deterministically
        error = _workflow_error(
            node="save_report_node",
            message=f"Failed to persist research run to PostgreSQL: {exc}",
            error_type="persistence_error",
            recoverable=True,
        )
        log_node_error(run_id, "save_report_node", error.message)
        record_run(False, _evaluation_score(state))
        log_node_success(run_id, "save_report_node", (perf_counter() - start) * 1000)
        return {
            "run_id": run_id,
            "status": "degraded",
            "errors": [*state.get("errors", []), error],
            "completed_at": datetime.now(UTC),
        }

    payload = _report_payload(report, {**state, "status": final_status, "run_id": run_id})
    report_path = save_report_json(run_id, payload)
    markdown_path = save_report_markdown(run_id, report)
    evaluation_score = _evaluation_score(state)
    success = final_status == "success"
    record_run(success, evaluation_score)
    save_run(
        run_id,
        {
            "ticker": state.get("ticker"),
            "status": final_status,
            "evaluation_score": evaluation_score,
            "report_path": report_path,
            "markdown_path": markdown_path,
            "request_id": str(persisted.get("request_id")) if persisted.get("request_id") else None,
        },
    )
    log_node_success(run_id, "save_report_node", (perf_counter() - start) * 1000)

    return {
        "run_id": run_id,
        "status": final_status,
        "final_report": report,
        "report_path": report_path,
        "request_id": persisted.get("request_id"),
        "report_id": persisted.get("report_id"),
        "completed_at": datetime.now(UTC),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test \
  .venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py -k save_report_node -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/workflow.py tests/test_storage_postgres_research_flow.py
git commit -m "feat(workflow): persist research run to postgres before file export"
```

---

## Task 13: Expose persisted request_id through `/analyze` (backward compatible)

**Files:**
- Modify: `main.py:170-188` (`/analyze`)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: the `request_id`/`report_id` now present in workflow result state.
- Produces: `/analyze` response keeps all existing keys and adds an optional `request_id` (string or null). No existing key changes type. `/backtest` and `/backtest/optimize` are unchanged in this slice (their persistence lands with US1 follow-up but the response contract here stays as-is).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_api.py
def test_analyze_response_includes_request_id(monkeypatch):
    from fastapi.testclient import TestClient
    import main

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
    resp = client.post("/analyze", json={"ticker": "AAPL"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run-1"
    assert body["request_id"] == "11111111-1111-1111-1111-111111111111"
```

> Before writing the test, open `main.py:170-188` and confirm the exact name of the workflow entry function used by `/analyze` (it is imported at the top of `main.py`). Use that name in `raising=False` so the test patches the real symbol. Adjust `_fake_run`'s signature to match.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api.py::test_analyze_response_includes_request_id -v`
Expected: FAIL with `KeyError`/assertion: `request_id` not in response.

- [ ] **Step 3: Add `request_id` to the `/analyze` response and model**

In `main.py`, add an optional field to `AnalyzeResponse` (the Pydantic model near the top, around line 41 where `run_id: str` is declared):

```python
    request_id: str | None = None
```

In the `analyze(...)` handler (around line 178-188), include the persisted id in the returned dict, coercing to string:

```python
        "request_id": (str(result["request_id"]) if result.get("request_id") else None),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api.py::test_analyze_response_includes_request_id -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat(api): expose persisted request_id in /analyze response"
```

---

## Task 14: Safety schema review + README + full validation

**Files:**
- Create: `tests/test_storage_postgres_safety.py`
- Modify: `README.md`
- Test: the full suites listed below

**Interfaces:**
- Produces: a test asserting no US1 table column name matches forbidden trading/brokerage tokens; README storage section.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage_postgres_safety.py  (new file)
FORBIDDEN_TOKENS = (
    "broker", "brokerage", "order", "execute", "execution", "trade_exec",
    "buy_signal", "sell_signal", "guaranteed", "guarantee_target", "api_secret",
    "password", "credential", "session_token",
)


def test_no_forbidden_columns_in_us1_schema():
    from src.db.models import Base
    offenders = []
    for table in Base.metadata.tables.values():
        for column in table.columns:
            lowered = column.name.lower()
            for token in FORBIDDEN_TOKENS:
                if token in lowered:
                    offenders.append(f"{table.name}.{column.name}")
    assert offenders == [], f"forbidden columns present: {offenders}"
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_safety.py -v`
Expected: PASS immediately (the schema was designed without these fields). If it fails, a forbidden column slipped in — remove it from the model and migration before continuing.

- [ ] **Step 3: Update README storage section**

Add a "Storage architecture (PostgreSQL source of truth)" subsection to `README.md` documenting:
- PostgreSQL is the source of truth; JSON/Markdown files are export artifacts.
- Local setup: `docker compose up -d db`, set `DATABASE_URL` / `TEST_DATABASE_URL`, run `alembic upgrade head`.
- Tests require a running PostgreSQL and `TEST_DATABASE_URL`; without it, PostgreSQL tests skip.

```markdown
## Storage architecture (PostgreSQL source of truth)

PostgreSQL is the canonical store for research workflow records (requests, node runs,
results, reports, report versions, evidence, citations, source documents, chunks, log
events). The previous local JSON/Markdown files are kept as export artifacts only.

**Local setup**

    docker compose up -d db
    export DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight
    export TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test
    .venv/bin/alembic upgrade head

**Tests** require a running PostgreSQL and `TEST_DATABASE_URL`. When `TEST_DATABASE_URL`
is unset, the PostgreSQL storage tests skip (they never silently pass).
```

- [ ] **Step 4: Run the full validation suite**

```bash
.venv/bin/python -m compileall src
TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test \
  .venv/bin/python -m pytest \
  tests/test_storage_postgres_schema.py \
  tests/test_storage_postgres_repositories.py \
  tests/test_storage_postgres_research_flow.py \
  tests/test_storage_postgres_safety.py -v
# Regression: existing suites must still pass
.venv/bin/python -m pytest tests/test_report_store.py tests/test_workflow_routing.py tests/test_api.py tests/test_safety_checker.py -v
```
Expected: all green; `compileall` reports no SyntaxError.

- [ ] **Step 5: Commit**

```bash
git add tests/test_storage_postgres_safety.py README.md
git commit -m "test(db): add schema safety review and document storage setup"
```

---

## Self-Review

**1. Spec coverage (US1 portion of `specs/004`):**
- FR-001/002 canonical ownership + one table per artifact → Tasks 3, 11 (12 tables, single request/report/version path). ✅
- FR-003 stable identifiers (UUID PKs, `evidence_id` unique) → Task 3. ✅
- FR-004 unauthenticated local/demo users, nullable email, no credentials → Task 3 (`User`), Task 14 (forbidden-column test). ✅
- FR-005 ticker identity + `(symbol, market)` unique → Task 3, Task 7. ✅
- FR-006 request inputs/status/degraded/warnings → Task 3, Task 7. ✅
- FR-007 node run details + unique attempt → Task 3, Task 7. ✅
- FR-008 separate analysis results by type → Task 3, Task 7. ✅
- FR-009 report records w/ language, safety, version → Task 3, Task 9. ✅
- FR-010 EvidenceItem-compatible records → Task 3, Task 8. ✅
- FR-011 source document + chunk provenance → Task 3, Task 10. ✅
- FR-015 uniqueness rules (evidence_id, report version, chunk index, ticker, citation) → Task 3 + constraint tests in Tasks 5/7/10. ✅
- SER-002/003 evidence citations + disclaimer flag → Tasks 8, 11. ✅
- SER-004 missing-data notes, no fabrication → Tasks 7, 11 (degraded test). ✅
- SER-005 deterministic persistence-failure behavior → Task 12 (degrade test). ✅
- SC-002 full-flow representation, SC-003 citations resolve, SC-005 no forbidden fields, SC-006 degraded run → Tasks 11, 8, 14, 11. ✅
- **Deferred to later slices (correctly out of US1):** US2 projection/graph tables (FR-012/013, SER-008), US3 user UX + anonymization (FR-014/016/019/020), `keyword_terms`, deferred tables. Noted in plan scope.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to" — every code step has full code. Two steps intentionally say "open the file and confirm the exact symbol name" (Task 13 workflow entry function, Task 4 migration review) — these are verification actions against existing code the plan cannot see verbatim, not placeholders for new code.

**3. Type consistency:** Repository method names and signatures in the Interfaces blocks match their call sites: `upsert_ticker`, `create_request`, `record_node_run`, `add_result`, `add_evidence`, `add_citation`, `create_report`, `add_version`, `set_status`, `next_version_number`, `add_document`, `add_correction`, `add_chunk`, `persist_research_run`. ORM attribute names for reserved `metadata` columns (`node_metadata`, `doc_metadata`, `chunk_metadata`, `event_metadata`) are consistent between Task 3 and the repositories. `persist_research_run` return keys (`request_id`, `report_id`, `report_version_id`) match Task 12's usage.

---

## Open items to confirm during execution (not blockers)

- **Task 13**: confirm the actual workflow entry-function name imported in `main.py` and the exact `/analyze` return-dict construction; the test patch target and the added key must match real code.
- **Task 4**: review the autogenerated migration before committing — autogenerate occasionally omits server defaults or names; reconcile against the models.
- A running PostgreSQL with `TEST_DATABASE_URL` is required to execute Tasks 4–14. `docker compose up -d db` provides it.
