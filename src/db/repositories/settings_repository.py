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
