# PostgreSQL Research Ledger (004 — US3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the MVP user-experience records — user settings, in-app notifications, research-only portfolios and their ticker items — and implement privacy-preserving user anonymization that strips PII and removes user-owned UX records while keeping reports/evidence/audit resolvable under the (now anonymized) owner.

**Architecture:** US3 extends the existing US1/US2 persistence boundary under `src/db`. Four new ORM models join `src/db/models.py`; four new repository classes (`SettingsRepository`, `NotificationRepository`, `PortfolioRepository`, `UserRepository`) follow the established `BaseRepository` pattern. `UserRepository.anonymize_user` is the FR-019 boundary: it clears PII on the `users` row and hard-deletes that user's UX records (settings, notifications, portfolios + items), leaving `analysis_requests` (and the reports/evidence/node-runs that hang off them) linked to the anonymized user so audit history survives. A single new Alembic migration chains after the US2 revision `d305f4d1ff77`.

**Tech Stack:** Python 3.12, SQLAlchemy 2.x, Alembic, psycopg 3, PostgreSQL 16 (docker-compose), pytest. Existing: FastAPI, Streamlit, Pydantic v2, LangGraph.

## Global Constraints

- Python 3.12; run all Python via `.venv/bin/python`.
- **Branch off the US2 branch.** US3 builds on US2's schema and migration. The branch is `feature/004-postgresql-us3`, created from `feature/004-postgresql-us2` HEAD. The new migration's `down_revision` MUST be `'d305f4d1ff77'` (the US2 head).
- Deterministic tests only — NO live external provider calls. Tests use a real local PostgreSQL via `TEST_DATABASE_URL`; when unset, PostgreSQL tests skip explicitly via `@REQUIRES_DB` (never silently pass).
- Test schema is created by running **Alembic migrations** (not `create_all`); each test runs inside a transaction that is rolled back (`db_session` fixture).
- PostgreSQL is **required** at runtime. Repositories `flush`, never `commit`.
- **SER-001 / SC-005:** NO field representing brokerage connection, order placement/execution, guaranteed return, guaranteed target, or buy/sell/hold instruction. Portfolios are research/watchlist context only; `quantity_note`/`cost_basis_note` are optional research notes, never order triggers. Notifications are informational only and must not instruct trading.
- **FR-019 (privacy):** User deletion/anonymization MUST delete or anonymize PII and user-owned UX records (settings, notifications, portfolios, portfolio items) while preserving reports, evidence, source documents, workflow node runs, and audit records under an anonymized owner.
- **FR-020:** MVP notifications are app-internal records only. NO external delivery fields (email address, webhook URL, push token, channel, delivery status, retry count). Those are deferred-expansion tables and out of scope.
- **US3 scope is exactly 4 new tables:** `user_settings`, `notifications`, `portfolios`, `portfolio_items`. No deferred-expansion tables (`notification_preferences`, `notification_deliveries`, `auth_identities`, `user_sessions`, `provider_sync_runs`, `provider_payloads`, `analysis_request_links`).
- Commit after every task.

## Status Vocabulary (from data-model.md)

- User status: `active`, `disabled`, `anonymized`, `deleted`.
- In-app notification status: `unread`, `read`, `archived`, `deleted`.

## Existing building blocks (US1/US2 — already present, do NOT recreate)

- `src/db/models.py`: `Base`, `UUIDMixin` (`id`), `TimestampMixin` (`created_at`, `updated_at`), `User` (with `email`, `display_name`, `role`, `status`, `anonymized_at`, `deleted_at`), `Ticker`, `AnalysisRequest` (has nullable `user_id` FK to `users`). All imports used below — `JSONB`, `UUID`, `String`, `Text`, `Integer`, `DateTime`, `ForeignKey`, `UniqueConstraint`, `func`, `uuid`, `datetime` — are already imported at the top of the file.
- `src/db/repositories/base.py`: `BaseRepository(session)` with `self.session` and `get(model, id_)`.
- `src/db/constants.py`: existing status frozensets.
- `tests/fixtures/postgres.py`: `make_ticker(session, symbol="AAPL", market="NASDAQ", **kw)`, `make_request(session, ticker, request_type="research", **kw)`. There is **no** `make_user` yet — Task 1 adds it.
- `tests/test_storage_postgres_safety.py`: `test_no_forbidden_columns_in_schema` already scans the FULL `Base.metadata` (US1+US2+US3 will all be covered automatically); `FORBIDDEN_TOKENS` defined there.

---

## File Structure

**Create:**
- `src/db/repositories/settings_repository.py` — `SettingsRepository`: user_settings upsert/get/list/soft-delete.
- `src/db/repositories/notification_repository.py` — `NotificationRepository`: create/list/mark_read/mark_archived/soft-delete.
- `src/db/repositories/portfolio_repository.py` — `PortfolioRepository`: portfolios + portfolio_items create/list/soft-delete.
- `src/db/repositories/user_repository.py` — `UserRepository`: create_user + `anonymize_user` (FR-019 boundary).
- `tests/test_storage_postgres_ux.py` — US3 repository tests (settings, notifications, portfolios).
- `tests/test_storage_postgres_privacy.py` — FR-019 anonymization + privacy tests (quickstart Phase 4).

**Modify:**
- `src/db/constants.py` — add `USER_STATUSES`, `NOTIFICATION_STATUSES`.
- `src/db/models.py` — add the 4 US3 ORM models.
- `src/db/migrations/versions/` — one new autogenerated revision (down_revision `d305f4d1ff77`).
- `tests/fixtures/postgres.py` — add `make_user`.
- `tests/test_storage_postgres_schema.py` — assert the 4 new tables + key constraints exist.
- `README.md` — append a US3 subsection to the PostgreSQL storage section.

---

## Task 1: US3 ORM models, constants, fixture, and migration

**Files:**
- Modify: `src/db/constants.py`
- Modify: `src/db/models.py`
- Modify: `tests/fixtures/postgres.py`
- Create: `src/db/migrations/versions/<new>_postgresql_source_of_truth_us3.py` (autogenerated)
- Modify: `tests/test_storage_postgres_schema.py`

**Interfaces:**
- Produces (models, importable from `src.db.models`): `UserSettings`, `Notification`, `Portfolio`, `PortfolioItem`.
- Produces (constants): `src.db.constants.USER_STATUSES: frozenset[str]`, `src.db.constants.NOTIFICATION_STATUSES: frozenset[str]`.
- Produces (fixture): `tests.fixtures.postgres.make_user(session, *, email=None, display_name=None, role="demo", status="active") -> User`.
- Consumes: US1 `Base`, `UUIDMixin`, `TimestampMixin`, `User`, `Ticker`.

- [ ] **Step 1: Write the failing schema test**

Add to `tests/test_storage_postgres_schema.py` (the file already defines `REQUIRES_DB`, `EXPECTED_TABLES`, `US2_EXPECTED_TABLES`, and `import os`):

```python
US3_EXPECTED_TABLES = {
    "user_settings",
    "notifications",
    "portfolios",
    "portfolio_items",
}


@REQUIRES_DB
def test_alembic_upgrade_creates_us3_tables(alembic_migrated_db):
    engine = create_engine(os.environ["TEST_DATABASE_URL"], future=True)
    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    assert US3_EXPECTED_TABLES.issubset(table_names)

    settings_uniques = [
        set(c["column_names"]) for c in insp.get_unique_constraints("user_settings")
    ]
    assert {"user_id", "setting_key", "scope"} in settings_uniques
    engine.dispose()
```

Also update the exact-set contract test to include the US3 tables. Find `test_metadata_has_exactly_us1_and_us2_tables` and change its assertion and name to:

```python
def test_metadata_has_exactly_us1_us2_us3_tables():
    from src.db.models import Base
    assert set(Base.metadata.tables) == EXPECTED_TABLES | US2_EXPECTED_TABLES | US3_EXPECTED_TABLES
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_schema.py::test_alembic_upgrade_creates_us3_tables -v`
Expected: FAIL — the 4 tables do not exist yet (assertion error on `issubset`). If `TEST_DATABASE_URL` is unset the test SKIPS; set it first so you get a real failure.

- [ ] **Step 3: Add the status constants**

Append to `src/db/constants.py`:

```python
USER_STATUSES = frozenset({"active", "disabled", "anonymized", "deleted"})
NOTIFICATION_STATUSES = frozenset({"unread", "read", "archived", "deleted"})
```

- [ ] **Step 4: Add the 4 US3 models**

Append to `src/db/models.py` (all imports used below are already imported by the US1 header):

```python
class UserSettings(UUIDMixin, Base):
    __tablename__ = "user_settings"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "setting_key", "scope", postgresql_nulls_not_distinct=True
        ),
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    setting_key: Mapped[str] = mapped_column(String, nullable=False)
    setting_value: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    scope: Mapped[str] = mapped_column(String, default="user", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Notification(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "notifications"
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    notification_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, default="", nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, default="unread", nullable=False)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Portfolio(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "portfolios"
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_currency: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active", nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PortfolioItem(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "portfolio_items"
    portfolio_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    quantity_note: Mapped[str | None] = mapped_column(String, nullable=True)
    cost_basis_note: Mapped[str | None] = mapped_column(String, nullable=True)
    item_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

Design notes:
- `user_settings` uses `postgresql_nulls_not_distinct=True` so two system-default rows (NULL `user_id`) with the same `(setting_key, scope)` collide — matching the `keyword_terms`/`tickers` convention.
- `portfolio_items.item_metadata` maps to the DB column literally named `metadata` (mirrors `DocumentChunk.chunk_metadata`), because `metadata` is reserved on the Declarative class.
- No external-delivery columns on `notifications` (FR-020). No order/brokerage columns anywhere (SER-001).
- `deleted_at` on every UX table supports FR-016 soft deletion.

- [ ] **Step 5: Add the `make_user` fixture**

Append to `tests/fixtures/postgres.py` (it already imports `Session` and the models it uses; add `User` to the model import if not present):

```python
def make_user(
    session: Session,
    *,
    email: str | None = None,
    display_name: str | None = None,
    role: str = "demo",
    status: str = "active",
) -> User:
    user = User(email=email, display_name=display_name, role=role, status=status)
    session.add(user)
    session.flush()
    return user
```

If `User` is not already imported at the top of `tests/fixtures/postgres.py`, add it to the existing `from src.db.models import ...` line.

- [ ] **Step 6: Autogenerate the migration**

Run:

```bash
.venv/bin/python -m alembic revision --autogenerate -m "postgresql source of truth us3"
```

Then open the generated file under `src/db/migrations/versions/` and verify:
- `down_revision = 'd305f4d1ff77'` (chains after the US2 head — Alembic sets this automatically because `d305f4d1ff77` is the current head).
- `upgrade()` contains `op.create_table(...)` for all 4 tables; `downgrade()` drops all 4 in reverse dependency order (`portfolio_items` before `portfolios`).
- It does **not** drop or alter any US1/US2 table. Delete any spurious diffs against existing tables — they indicate drift and must not ship.

- [ ] **Step 7: Run the schema tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_schema.py -v`
Expected: PASS (all US1/US2 schema tests still pass; the new `test_alembic_upgrade_creates_us3_tables` and the renamed exact-set test pass).

- [ ] **Step 8: Sanity-check the migration round-trips**

Run: `.venv/bin/python -m alembic downgrade -1 && .venv/bin/python -m alembic upgrade head`
Expected: both commands exit 0 with no error.

- [ ] **Step 9: Commit**

```bash
git add src/db/constants.py src/db/models.py tests/fixtures/postgres.py src/db/migrations/versions/ tests/test_storage_postgres_schema.py
git commit -m "feat(db): add US3 user settings, notifications, portfolio tables"
```

---

## Task 2: SettingsRepository — user settings (FR-014, FR-015)

**Files:**
- Create: `src/db/repositories/settings_repository.py`
- Create: `tests/test_storage_postgres_ux.py`

**Interfaces:**
- Consumes: `UserSettings` (Task 1); `BaseRepository`; `make_user` fixture.
- Produces:
  - `SettingsRepository(session)`
  - `.upsert_setting(*, setting_key: str, setting_value: dict, scope: str = "user", user_id=None) -> UserSettings` (idempotent on `(user_id, setting_key, scope)`; updates value + un-deletes on collision)
  - `.get_setting(*, setting_key: str, scope: str = "user", user_id=None) -> UserSettings | None` (excludes soft-deleted)
  - `.list_for_user(user_id) -> list[UserSettings]` (excludes soft-deleted; ordered)
  - `.soft_delete(record) -> UserSettings`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_storage_postgres_ux.py`:

```python
"""US3 PostgreSQL tests: user settings, notifications, portfolios."""

import os
from datetime import UTC, datetime

import pytest

from tests.fixtures.postgres import make_ticker, make_user

REQUIRES_DB = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set"
)


@REQUIRES_DB
def test_user_setting_upsert_is_unique_per_key_scope(db_session):
    from src.db.repositories.settings_repository import SettingsRepository

    repo = SettingsRepository(db_session)
    user = make_user(db_session)
    first = repo.upsert_setting(
        setting_key="theme", setting_value={"mode": "dark"}, user_id=user.id
    )
    second = repo.upsert_setting(
        setting_key="theme", setting_value={"mode": "light"}, user_id=user.id
    )
    assert first.id == second.id  # same (user, key, scope) returns existing
    assert second.setting_value == {"mode": "light"}  # value updated
    assert len(repo.list_for_user(user.id)) == 1


@REQUIRES_DB
def test_system_default_setting_uses_null_user(db_session):
    from src.db.repositories.settings_repository import SettingsRepository

    repo = SettingsRepository(db_session)
    a = repo.upsert_setting(setting_key="rate_limit", setting_value={"rpm": 60}, scope="system")
    b = repo.upsert_setting(setting_key="rate_limit", setting_value={"rpm": 90}, scope="system")
    assert a.id == b.id  # NULL user_id collides with NULL user_id
    assert b.setting_value == {"rpm": 90}


@REQUIRES_DB
def test_setting_soft_delete_hides_from_get_and_list(db_session):
    from src.db.repositories.settings_repository import SettingsRepository

    repo = SettingsRepository(db_session)
    user = make_user(db_session)
    record = repo.upsert_setting(
        setting_key="lang", setting_value={"code": "ko"}, user_id=user.id
    )
    repo.soft_delete(record)
    assert record.deleted_at is not None
    assert repo.get_setting(setting_key="lang", user_id=user.id) is None
    assert repo.list_for_user(user.id) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_ux.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.db.repositories.settings_repository'`.

- [ ] **Step 3: Implement `SettingsRepository`**

Create `src/db/repositories/settings_repository.py`:

```python
"""Repository for user/system settings (US3)."""

from datetime import UTC, datetime

from src.db.models import UserSettings
from src.db.repositories.base import BaseRepository


class SettingsRepository(BaseRepository):
    def upsert_setting(
        self,
        *,
        setting_key: str,
        setting_value: dict,
        scope: str = "user",
        user_id=None,
    ) -> UserSettings:
        query = self.session.query(UserSettings).filter(
            UserSettings.setting_key == setting_key,
            UserSettings.scope == scope,
        )
        if user_id is None:
            query = query.filter(UserSettings.user_id.is_(None))
        else:
            query = query.filter(UserSettings.user_id == user_id)
        existing = query.one_or_none()
        if existing is not None:
            existing.setting_value = setting_value
            existing.deleted_at = None  # re-setting un-deletes
            self.session.flush()
            return existing
        record = UserSettings(
            user_id=user_id,
            setting_key=setting_key,
            setting_value=setting_value,
            scope=scope,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get_setting(
        self, *, setting_key: str, scope: str = "user", user_id=None
    ) -> UserSettings | None:
        query = self.session.query(UserSettings).filter(
            UserSettings.setting_key == setting_key,
            UserSettings.scope == scope,
            UserSettings.deleted_at.is_(None),
        )
        if user_id is None:
            query = query.filter(UserSettings.user_id.is_(None))
        else:
            query = query.filter(UserSettings.user_id == user_id)
        return query.one_or_none()

    def list_for_user(self, user_id) -> list[UserSettings]:
        return (
            self.session.query(UserSettings)
            .filter(
                UserSettings.user_id == user_id,
                UserSettings.deleted_at.is_(None),
            )
            .order_by(UserSettings.scope, UserSettings.setting_key)
            .all()
        )

    def soft_delete(self, record: UserSettings) -> UserSettings:
        record.deleted_at = datetime.now(UTC)
        self.session.flush()
        return record
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_ux.py -v`
Expected: PASS for the three settings tests.

- [ ] **Step 5: Commit**

```bash
git add src/db/repositories/settings_repository.py tests/test_storage_postgres_ux.py
git commit -m "feat(db): add user settings repository (US3)"
```

---

## Task 3: NotificationRepository — in-app notifications (FR-014, FR-020)

**Files:**
- Create: `src/db/repositories/notification_repository.py`
- Modify: `tests/test_storage_postgres_ux.py`

**Interfaces:**
- Consumes: `Notification` (Task 1); `BaseRepository`; `make_user`, `make_ticker` fixtures.
- Produces:
  - `NotificationRepository(session)`
  - `.create(*, notification_type: str, title: str, body: str = "", payload: dict | None = None, status: str = "unread", user_id=None, ticker_id=None) -> Notification`
  - `.list_for_user(user_id, *, status: str | None = None) -> list[Notification]` (excludes soft-deleted; ordered by `created_at`)
  - `.mark_read(record, *, at: datetime) -> Notification`
  - `.mark_archived(record) -> Notification`
  - `.soft_delete(record) -> Notification`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_storage_postgres_ux.py`:

```python
@REQUIRES_DB
def test_notification_lifecycle_create_read_archive_delete(db_session):
    from src.db.repositories.notification_repository import NotificationRepository

    repo = NotificationRepository(db_session)
    user = make_user(db_session)
    ticker = make_ticker(db_session)
    note = repo.create(
        notification_type="analysis_complete",
        title="분석 완료",
        body="요청하신 분석이 완료되었습니다.",
        payload={"request_id": "abc"},
        user_id=user.id,
        ticker_id=ticker.id,
    )
    assert note.status == "unread"
    assert note.read_at is None

    now = datetime.now(UTC)
    repo.mark_read(note, at=now)
    assert note.status == "read"
    assert note.read_at == now

    repo.mark_archived(note)
    assert note.status == "archived"

    repo.soft_delete(note)
    assert note.status == "deleted"
    assert note.deleted_at is not None


@REQUIRES_DB
def test_notification_list_filters_status_and_excludes_deleted(db_session):
    from src.db.repositories.notification_repository import NotificationRepository

    repo = NotificationRepository(db_session)
    user = make_user(db_session)
    unread = repo.create(notification_type="info", title="A", user_id=user.id)
    read = repo.create(notification_type="info", title="B", user_id=user.id)
    repo.mark_read(read, at=datetime.now(UTC))
    gone = repo.create(notification_type="info", title="C", user_id=user.id)
    repo.soft_delete(gone)

    all_active = repo.list_for_user(user.id)
    assert {n.title for n in all_active} == {"A", "B"}  # deleted excluded
    unread_only = repo.list_for_user(user.id, status="unread")
    assert [n.title for n in unread_only] == ["A"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_ux.py::test_notification_lifecycle_create_read_archive_delete -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.db.repositories.notification_repository'`.

- [ ] **Step 3: Implement `NotificationRepository`**

Create `src/db/repositories/notification_repository.py`:

```python
"""Repository for in-app notifications (US3). App-internal only — no external delivery."""

from datetime import UTC, datetime

from src.db.models import Notification
from src.db.repositories.base import BaseRepository


class NotificationRepository(BaseRepository):
    def create(
        self,
        *,
        notification_type: str,
        title: str,
        body: str = "",
        payload: dict | None = None,
        status: str = "unread",
        user_id=None,
        ticker_id=None,
    ) -> Notification:
        note = Notification(
            user_id=user_id,
            ticker_id=ticker_id,
            notification_type=notification_type,
            title=title,
            body=body,
            payload=payload if payload is not None else {},
            status=status,
        )
        self.session.add(note)
        self.session.flush()
        return note

    def list_for_user(self, user_id, *, status: str | None = None) -> list[Notification]:
        query = self.session.query(Notification).filter(
            Notification.user_id == user_id,
            Notification.deleted_at.is_(None),
        )
        if status is not None:
            query = query.filter(Notification.status == status)
        return query.order_by(Notification.created_at).all()

    def mark_read(self, record: Notification, *, at: datetime) -> Notification:
        record.status = "read"
        record.read_at = at
        self.session.flush()
        return record

    def mark_archived(self, record: Notification) -> Notification:
        record.status = "archived"
        self.session.flush()
        return record

    def soft_delete(self, record: Notification) -> Notification:
        record.status = "deleted"
        record.deleted_at = datetime.now(UTC)
        self.session.flush()
        return record
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_ux.py -v`
Expected: PASS (settings + notification tests).

- [ ] **Step 5: Commit**

```bash
git add src/db/repositories/notification_repository.py tests/test_storage_postgres_ux.py
git commit -m "feat(db): add in-app notification repository (US3)"
```

---

## Task 4: PortfolioRepository — research-only portfolios and items (FR-014, SER-001)

**Files:**
- Create: `src/db/repositories/portfolio_repository.py`
- Modify: `tests/test_storage_postgres_ux.py`

**Interfaces:**
- Consumes: `Portfolio`, `PortfolioItem` (Task 1); `BaseRepository`; `make_user`, `make_ticker` fixtures.
- Produces:
  - `PortfolioRepository(session)`
  - `.create_portfolio(*, name: str, description: str | None = None, base_currency: str | None = None, status: str = "active", user_id=None) -> Portfolio`
  - `.add_item(portfolio_id, ticker_id, *, label: str | None = None, quantity_note: str | None = None, cost_basis_note: str | None = None, metadata: dict | None = None) -> PortfolioItem`
  - `.list_items(portfolio_id) -> list[PortfolioItem]` (excludes soft-deleted; ordered by `created_at`)
  - `.list_for_user(user_id) -> list[Portfolio]` (excludes soft-deleted)
  - `.soft_delete_portfolio(record) -> Portfolio`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_storage_postgres_ux.py`:

```python
@REQUIRES_DB
def test_portfolio_with_items_is_research_only(db_session):
    from src.db.repositories.portfolio_repository import PortfolioRepository

    repo = PortfolioRepository(db_session)
    user = make_user(db_session)
    ticker = make_ticker(db_session)
    pf = repo.create_portfolio(name="관심 종목", base_currency="KRW", user_id=user.id)
    item = repo.add_item(
        pf.id,
        ticker.id,
        label="장기 관찰",
        quantity_note="참고용 메모",
        metadata={"note": "watchlist"},
    )
    assert item.portfolio_id == pf.id
    assert item.ticker_id == ticker.id
    assert item.item_metadata == {"note": "watchlist"}
    assert [i.id for i in repo.list_items(pf.id)] == [item.id]
    assert [p.id for p in repo.list_for_user(user.id)] == [pf.id]


@REQUIRES_DB
def test_portfolio_soft_delete_hides_from_list(db_session):
    from src.db.repositories.portfolio_repository import PortfolioRepository

    repo = PortfolioRepository(db_session)
    user = make_user(db_session)
    pf = repo.create_portfolio(name="삭제 대상", user_id=user.id)
    repo.soft_delete_portfolio(pf)
    assert pf.deleted_at is not None
    assert repo.list_for_user(user.id) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_ux.py::test_portfolio_with_items_is_research_only -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.db.repositories.portfolio_repository'`.

- [ ] **Step 3: Implement `PortfolioRepository`**

Create `src/db/repositories/portfolio_repository.py`:

```python
"""Repository for research-only portfolios and items (US3). No order execution."""

from datetime import UTC, datetime

from src.db.models import Portfolio, PortfolioItem
from src.db.repositories.base import BaseRepository


class PortfolioRepository(BaseRepository):
    def create_portfolio(
        self,
        *,
        name: str,
        description: str | None = None,
        base_currency: str | None = None,
        status: str = "active",
        user_id=None,
    ) -> Portfolio:
        portfolio = Portfolio(
            user_id=user_id,
            name=name,
            description=description,
            base_currency=base_currency,
            status=status,
        )
        self.session.add(portfolio)
        self.session.flush()
        return portfolio

    def add_item(
        self,
        portfolio_id,
        ticker_id,
        *,
        label: str | None = None,
        quantity_note: str | None = None,
        cost_basis_note: str | None = None,
        metadata: dict | None = None,
    ) -> PortfolioItem:
        item = PortfolioItem(
            portfolio_id=portfolio_id,
            ticker_id=ticker_id,
            label=label,
            quantity_note=quantity_note,
            cost_basis_note=cost_basis_note,
            item_metadata=metadata if metadata is not None else {},
        )
        self.session.add(item)
        self.session.flush()
        return item

    def list_items(self, portfolio_id) -> list[PortfolioItem]:
        return (
            self.session.query(PortfolioItem)
            .filter(
                PortfolioItem.portfolio_id == portfolio_id,
                PortfolioItem.deleted_at.is_(None),
            )
            .order_by(PortfolioItem.created_at)
            .all()
        )

    def list_for_user(self, user_id) -> list[Portfolio]:
        return (
            self.session.query(Portfolio)
            .filter(
                Portfolio.user_id == user_id,
                Portfolio.deleted_at.is_(None),
            )
            .order_by(Portfolio.created_at)
            .all()
        )

    def soft_delete_portfolio(self, record: Portfolio) -> Portfolio:
        record.deleted_at = datetime.now(UTC)
        self.session.flush()
        return record
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_ux.py -v`
Expected: PASS (settings + notifications + portfolio tests).

- [ ] **Step 5: Commit**

```bash
git add src/db/repositories/portfolio_repository.py tests/test_storage_postgres_ux.py
git commit -m "feat(db): add research-only portfolio repository (US3)"
```

---

## Task 5: UserRepository — create + privacy-preserving anonymization (FR-019)

**Files:**
- Create: `src/db/repositories/user_repository.py`
- Create: `tests/test_storage_postgres_privacy.py`

**Interfaces:**
- Consumes: `User`, `UserSettings`, `Notification`, `Portfolio`, `PortfolioItem`, `AnalysisRequest` (US1/US3); `BaseRepository`; `make_user`, `make_ticker`, `make_request` fixtures; `SettingsRepository`, `NotificationRepository`, `PortfolioRepository` (Tasks 2–4, used in the test to set up UX records).
- Produces:
  - `UserRepository(session)`
  - `.create_user(*, email: str | None = None, display_name: str | None = None, role: str = "demo", status: str = "active") -> User`
  - `.anonymize_user(user, *, at: datetime) -> User` — clears `email`/`display_name`, sets `status="anonymized"` and `anonymized_at=at`; **hard-deletes** that user's UX records (user_settings, notifications, portfolios, and their portfolio_items). Does NOT touch `analysis_requests` (and the reports/evidence/node-runs that hang off them) — those stay linked to the now-anonymized user so audit history survives.

Rationale: the spec's chosen policy (clarification Q on retention) is "anonymize": strip PII from the `users` row but keep its `id` so every FK chain (analysis_requests → reports/evidence/node_runs) stays resolvable under an anonymized owner. UX records are convenience data with potential PII (e.g. a portfolio named after a person), so they are hard-deleted, not merely soft-deleted, to actually remove the PII.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_storage_postgres_privacy.py`:

```python
"""US3 privacy tests: local/demo users and FR-019 anonymization."""

import os
from datetime import UTC, datetime

import pytest

from tests.fixtures.postgres import make_request, make_ticker, make_user

REQUIRES_DB = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set"
)


@REQUIRES_DB
def test_local_demo_user_needs_no_email_or_credentials(db_session):
    from src.db.repositories.user_repository import UserRepository

    repo = UserRepository(db_session)
    user = repo.create_user()  # no email, no display_name
    assert user.id is not None
    assert user.email is None
    assert user.role == "demo"
    assert user.status == "active"


@REQUIRES_DB
def test_anonymize_user_removes_pii_and_ux_but_preserves_audit(db_session):
    from src.db.repositories.notification_repository import NotificationRepository
    from src.db.repositories.portfolio_repository import PortfolioRepository
    from src.db.repositories.settings_repository import SettingsRepository
    from src.db.repositories.user_repository import UserRepository

    users = UserRepository(db_session)
    user = users.create_user(email="person@example.test", display_name="홍길동")
    ticker = make_ticker(db_session)

    # Audit-bearing record: an analysis request owned by the user.
    request = make_request(db_session, ticker, user_id=user.id)
    request_id = request.id

    # User-owned UX records.
    SettingsRepository(db_session).upsert_setting(
        setting_key="theme", setting_value={"mode": "dark"}, user_id=user.id
    )
    note = NotificationRepository(db_session).create(
        notification_type="info", title="hi", user_id=user.id
    )
    pf = PortfolioRepository(db_session).create_portfolio(name="내 포트폴리오", user_id=user.id)
    PortfolioRepository(db_session).add_item(pf.id, ticker.id)

    # Anonymize.
    users.anonymize_user(user, at=datetime.now(UTC))

    # PII cleared, status flipped.
    db_session.refresh(user)
    assert user.email is None
    assert user.display_name is None
    assert user.status == "anonymized"
    assert user.anonymized_at is not None

    # UX records removed.
    assert SettingsRepository(db_session).list_for_user(user.id) == []
    assert NotificationRepository(db_session).list_for_user(user.id) == []
    assert PortfolioRepository(db_session).list_for_user(user.id) == []

    # Audit preserved: the analysis request still exists, still owned by the
    # (now anonymized) user — reports/evidence/node-runs hang off it.
    from src.db.models import AnalysisRequest, PortfolioItem

    surviving = db_session.get(AnalysisRequest, request_id)
    assert surviving is not None
    assert surviving.user_id == user.id
    # Portfolio items for the deleted portfolio are gone too.
    assert (
        db_session.query(PortfolioItem)
        .filter(PortfolioItem.portfolio_id == pf.id)
        .count()
        == 0
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_privacy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.db.repositories.user_repository'`.

- [ ] **Step 3: Implement `UserRepository`**

Create `src/db/repositories/user_repository.py`:

```python
"""Repository for users and privacy-preserving anonymization (US3, FR-019)."""

from datetime import datetime

from src.db.models import (
    Notification,
    Portfolio,
    PortfolioItem,
    User,
    UserSettings,
)
from src.db.repositories.base import BaseRepository


class UserRepository(BaseRepository):
    def create_user(
        self,
        *,
        email: str | None = None,
        display_name: str | None = None,
        role: str = "demo",
        status: str = "active",
    ) -> User:
        user = User(email=email, display_name=display_name, role=role, status=status)
        self.session.add(user)
        self.session.flush()
        return user

    def anonymize_user(self, user: User, *, at: datetime) -> User:
        """Strip PII and remove user-owned UX records, preserving audit (FR-019).

        Reports, evidence, source documents, and workflow node runs hang off
        ``analysis_requests`` (which keep their ``user_id``), so they remain
        resolvable under the now-anonymized owner. UX records (settings,
        notifications, portfolios + items) are hard-deleted because they may
        contain PII and are not audit records.
        """
        # Hard-delete portfolio items first (FK child of portfolios).
        portfolio_ids = [
            pid
            for (pid,) in self.session.query(Portfolio.id).filter(
                Portfolio.user_id == user.id
            )
        ]
        if portfolio_ids:
            self.session.query(PortfolioItem).filter(
                PortfolioItem.portfolio_id.in_(portfolio_ids)
            ).delete(synchronize_session=False)
        self.session.query(Portfolio).filter(Portfolio.user_id == user.id).delete(
            synchronize_session=False
        )
        self.session.query(Notification).filter(Notification.user_id == user.id).delete(
            synchronize_session=False
        )
        self.session.query(UserSettings).filter(UserSettings.user_id == user.id).delete(
            synchronize_session=False
        )

        # Strip PII from the user row but keep the id so audit chains resolve.
        user.email = None
        user.display_name = None
        user.status = "anonymized"
        user.anonymized_at = at
        self.session.flush()
        return user
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_privacy.py -v`
Expected: PASS (both tests; they RAN, not skipped, because `TEST_DATABASE_URL` is set).

- [ ] **Step 5: Commit**

```bash
git add src/db/repositories/user_repository.py tests/test_storage_postgres_privacy.py
git commit -m "feat(db): add user repository with FR-019 anonymization (US3)"
```

---

## Task 6: Safety assertion, full validation, and docs

**Files:**
- Modify: `tests/test_storage_postgres_safety.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: full `Base.metadata` (US1 + US2 + US3 tables).

- [ ] **Step 1: Add a US3-tables-present assertion to the safety suite**

The existing `test_no_forbidden_columns_in_schema` already scans the FULL `Base.metadata`, so the 4 US3 tables are covered for forbidden trading/order/delivery tokens with no change. Add an explicit US3-presence test so the scope is asserted. Append to `tests/test_storage_postgres_safety.py`:

```python
US3_TABLES = {
    "user_settings",
    "notifications",
    "portfolios",
    "portfolio_items",
}


def test_us3_tables_present_in_metadata():
    from src.db.models import Base

    assert US3_TABLES.issubset(set(Base.metadata.tables.keys()))
```

- [ ] **Step 2: Run the safety tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_safety.py -v`
Expected: PASS. If `test_no_forbidden_columns_in_schema` reports an offender, a forbidden column slipped into a US3 model — fix the model name (never `*_order`, `*_execution`, `guaranteed_*`, `broker*`, delivery/channel columns).

- [ ] **Step 3: Run the full US3 + regression suite**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_ux.py tests/test_storage_postgres_privacy.py tests/test_storage_postgres_schema.py tests/test_storage_postgres_safety.py tests/test_storage_postgres_projection_status.py tests/test_storage_postgres_repositories.py tests/test_storage_postgres_research_flow.py -v`
Expected: PASS (US3 tests pass; all US1/US2 tests still pass — no regressions).

- [ ] **Step 4: Run the entire test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS / no new failures. (DB tests run because `TEST_DATABASE_URL` is set; pre-existing unrelated warnings are acceptable.)

- [ ] **Step 5: Document the US3 tables**

Append a short US3 subsection to the README's PostgreSQL storage section ("## Storage architecture (PostgreSQL source of truth)"). List the 4 tables and their roles (user settings, in-app notifications, research-only portfolios + items) and note the FR-019 anonymization policy: anonymizing a user strips PII from the `users` row and hard-deletes user-owned UX records, while reports/evidence/source-documents/node-runs remain resolvable under the anonymized owner. Note FR-020: notifications are app-internal only, no external delivery fields.

- [ ] **Step 6: Commit**

```bash
git add tests/test_storage_postgres_safety.py README.md
git commit -m "test(db): US3 presence assertion + docs"
```

---

## Self-Review

**1. Spec coverage:**

| Requirement | Task |
|---|---|
| FR-014 user settings/notifications/portfolios/items, no brokerage/order fields | Task 1 (models, column choices), Tasks 2–4 (repos) |
| FR-015 unique user setting key per (user, key, scope) | Task 1 (UniqueConstraint), Task 2 (upsert idempotency) |
| FR-016 soft deletion for user-facing records | Task 1 (`deleted_at` columns), Tasks 2–4 (`soft_delete*`) |
| FR-017 US3 tables in MVP set | Task 1 |
| FR-019 anonymize PII + UX records, preserve reports/evidence/audit under anon owner | Task 5 (`anonymize_user`), Task 5 test (audit preserved) |
| FR-020 in-app only, no external delivery fields | Task 1 (no delivery columns), Task 3 |
| SER-001 / SC-005 no trading/execution fields | Task 1 (column choices), Task 6 (full-schema forbidden scan covers US3) |
| In-app notification lifecycle (unread/read/archived/deleted) | Task 1 (status vocab), Task 3 |
| User deletion state (active→anonymized) | Task 5 |
| Migrations create schema (not create_all) | Task 1 (Alembic revision chained after `d305f4d1ff77`) |

Deferred-expansion tables (`notification_preferences`, `notification_deliveries`, `auth_identities`, `user_sessions`, etc.) are intentionally **out of scope** (Global Constraints) — no task, correct for this slice.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" — every code step shows full code; every command shows expected output.

**3. Type consistency:**
- `SettingsRepository.upsert_setting` keyword args (`setting_key`, `setting_value`, `scope`, `user_id`) match the Task 2 test call sites and the `get_setting`/`list_for_user` signatures.
- `NotificationRepository.create` keyword args match the Task 3/Task 5 test call sites; `mark_read(record, *, at)` matches.
- `PortfolioRepository.create_portfolio`/`add_item` match Task 4/Task 5 call sites; `add_item` uses `metadata=` kwarg mapped to the model's `item_metadata` attribute (DB column `metadata`).
- `UserRepository.create_user`/`anonymize_user(user, *, at)` match the Task 5 test.
- `make_user(session, *, email=None, display_name=None, role="demo", status="active")` (Task 1) is used keyword-free in Tasks 2–5 (e.g. `make_user(db_session)`), consistent with its all-defaulted signature.
- `Notification.payload` / `UserSettings.setting_value` / `PortfolioItem.item_metadata` are all `JSONB` with `default=dict`; tests assert exact dict equality.
