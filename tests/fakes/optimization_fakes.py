"""In-memory fakes for robust optimization tests."""
from __future__ import annotations
from typing import Any


class InMemoryOptimizationRepository:
    """Deterministic in-memory store for OptimizationRun objects."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def save(self, run_id: str, run: Any) -> None:
        self._store[run_id] = run

    def get(self, run_id: str) -> Any | None:
        return self._store.get(run_id)

    def all(self) -> list[Any]:
        return list(self._store.values())

    def clear(self) -> None:
        self._store.clear()
