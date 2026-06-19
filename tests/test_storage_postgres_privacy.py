"""US3 privacy tests: local/demo users and FR-019 anonymization."""

import os
from datetime import UTC, datetime

import pytest

from tests.fixtures.postgres import make_request, make_ticker

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


@REQUIRES_DB
def test_anonymize_transient_user_does_not_wipe_system_rows(db_session):
    """Guard: anonymizing an unsaved user (id=None) must NOT delete NULL-owner
    rows like system-default settings (`user_id IS NULL`)."""
    from src.db.models import User
    from src.db.repositories.settings_repository import SettingsRepository
    from src.db.repositories.user_repository import UserRepository

    # A system-default setting owned by no user (NULL user_id).
    SettingsRepository(db_session).upsert_setting(
        setting_key="rate_limit", setting_value={"rpm": 60}, scope="system"
    )

    transient = User(email="ghost@example.test")  # not added/flushed → id is None
    assert transient.id is None

    UserRepository(db_session).anonymize_user(transient, at=datetime.now(UTC))

    # System-default row survives — the guard prevented an IS NULL mass delete.
    assert (
        SettingsRepository(db_session).get_setting(setting_key="rate_limit", scope="system")
        is not None
    )
