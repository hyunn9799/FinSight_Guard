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
