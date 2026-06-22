"""Repository boundary for PostgreSQL persistence."""

from src.db.repositories.analysis_repository import AnalysisRepository  # noqa: F401
from src.db.repositories.provider_repository import ProviderRepository  # noqa: F401

__all__ = ["AnalysisRepository", "ProviderRepository"]
