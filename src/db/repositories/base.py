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
