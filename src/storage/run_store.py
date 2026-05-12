"""Run metadata persistence helpers."""

from datetime import UTC, datetime
from typing import Any


_RUNS: dict[str, dict[str, Any]] = {}


def save_run(run_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Store run metadata in memory for lightweight MVP tracking."""
    record = {
        **metadata,
        "run_id": run_id,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _RUNS[run_id] = record
    return record


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return stored run metadata if present."""
    return _RUNS.get(run_id)


def list_runs() -> list[dict[str, Any]]:
    """Return all stored run metadata."""
    return list(_RUNS.values())
