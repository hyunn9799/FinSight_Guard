"""Runtime metrics helpers."""

from typing import Any


total_runs = 0
successful_runs = 0
failed_runs = 0
total_evaluation_score = 0.0


def record_run(success: bool, evaluation_score: float | None) -> None:
    """Record one workflow run in process memory."""
    global total_runs, successful_runs, failed_runs, total_evaluation_score

    total_runs += 1
    if success:
        successful_runs += 1
    else:
        failed_runs += 1
    total_evaluation_score += float(evaluation_score or 0.0)


def get_metrics() -> dict[str, Any]:
    """Return current in-memory metrics."""
    average = round(total_evaluation_score / total_runs, 4) if total_runs else 0.0
    return {
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "average_evaluation_score": average,
    }


def reset_metrics() -> None:
    """Reset metrics for deterministic tests."""
    global total_runs, successful_runs, failed_runs, total_evaluation_score

    total_runs = 0
    successful_runs = 0
    failed_runs = 0
    total_evaluation_score = 0.0
