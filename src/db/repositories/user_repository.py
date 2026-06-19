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
