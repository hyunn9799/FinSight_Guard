"""Repository for index projection status and keyword terms (US2)."""

from datetime import datetime

from src.db.models import IndexProjectionStatus, KeywordTerm
from src.db.repositories.base import BaseRepository


class ProjectionRepository(BaseRepository):
    def upsert_status(
        self,
        *,
        source_table: str,
        source_id,
        target_system: str,
        projection_type: str,
        projection_key: str,
        idempotency_key: str,
        status: str = "pending",
    ) -> IndexProjectionStatus:
        existing = (
            self.session.query(IndexProjectionStatus)
            .filter(
                IndexProjectionStatus.target_system == target_system,
                IndexProjectionStatus.projection_type == projection_type,
                IndexProjectionStatus.idempotency_key == idempotency_key,
            )
            .one_or_none()
        )
        if existing is not None:
            existing.source_table = source_table
            existing.source_id = source_id
            existing.projection_key = projection_key
            existing.status = status
            self.session.flush()
            return existing
        record = IndexProjectionStatus(
            source_table=source_table,
            source_id=source_id,
            target_system=target_system,
            projection_type=projection_type,
            projection_key=projection_key,
            idempotency_key=idempotency_key,
            status=status,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def mark_success(self, record: IndexProjectionStatus, *, at: datetime) -> IndexProjectionStatus:
        record.status = "success"
        record.attempt_count += 1
        record.last_attempt_at = at
        record.last_success_at = at
        record.error_message = None
        self.session.flush()
        return record

    def mark_failure(
        self, record: IndexProjectionStatus, *, at: datetime, error_message: str
    ) -> IndexProjectionStatus:
        record.status = "failed"
        record.attempt_count += 1
        record.last_attempt_at = at
        record.error_message = error_message
        self.session.flush()
        return record

    def mark_stale(self, record: IndexProjectionStatus) -> IndexProjectionStatus:
        record.status = "stale"
        self.session.flush()
        return record

    def list_for_source(self, source_table: str, source_id) -> list[IndexProjectionStatus]:
        return (
            self.session.query(IndexProjectionStatus)
            .filter(
                IndexProjectionStatus.source_table == source_table,
                IndexProjectionStatus.source_id == source_id,
            )
            .all()
        )

    def list_by_status(self, status: str) -> list[IndexProjectionStatus]:
        return (
            self.session.query(IndexProjectionStatus)
            .filter(IndexProjectionStatus.status == status)
            .all()
        )

    def failure_warning(self, record: IndexProjectionStatus) -> dict:
        """Return a non-mutating projection warning payload (SER-008).

        This reads the failed projection row only; it never touches the
        canonical source record identified by ``source_table``/``source_id``.
        """
        return {
            "target_system": record.target_system,
            "projection_type": record.projection_type,
            "status": record.status,
            "error_message": record.error_message,
            "source_table": record.source_table,
            "source_id": str(record.source_id),
        }

    def upsert_term(
        self, term: str, normalized_term: str, language: str | None = None
    ) -> KeywordTerm:
        query = self.session.query(KeywordTerm).filter(
            KeywordTerm.normalized_term == normalized_term
        )
        if language is None:
            query = query.filter(KeywordTerm.language.is_(None))
        else:
            query = query.filter(KeywordTerm.language == language)
        existing = query.one_or_none()
        if existing is not None:
            return existing
        record = KeywordTerm(term=term, normalized_term=normalized_term, language=language)
        self.session.add(record)
        self.session.flush()
        return record

    def list_terms(self) -> list[KeywordTerm]:
        return self.session.query(KeywordTerm).order_by(KeywordTerm.normalized_term).all()
