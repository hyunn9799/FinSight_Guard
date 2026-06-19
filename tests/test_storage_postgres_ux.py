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
