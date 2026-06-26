"""Repository for in-app notifications (US3). App-internal only — no external delivery."""

from datetime import UTC, datetime
from typing import Any

from src.db.models import Notification
from src.db.repositories.base import BaseRepository
from src.safety.safety_checker import find_forbidden_phrases


def _payload_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_payload_text(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return "\n".join(_payload_text(item) for item in value)
    return str(value)


def _assert_notification_safe(*, title: str, body: str, payload: dict | None) -> None:
    text = "\n".join([title, body, _payload_text(payload)])
    matches = find_forbidden_phrases(text)
    if matches:
        raise ValueError(f"Notification contains forbidden financial advice phrase: {matches[0]}")


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
        _assert_notification_safe(title=title, body=body, payload=payload)
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
