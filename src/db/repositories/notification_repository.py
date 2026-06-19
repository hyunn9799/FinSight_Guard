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
