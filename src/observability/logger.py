"""Structured logging helpers."""

from datetime import UTC, datetime
import json
import logging
import sys
from typing import Any

from src.config import LOG_DIR


_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Format log records as compact JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("run_id", "node_name", "ticker", "elapsed_ms", "error"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()

    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger("finsight_guard")
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured project logger."""
    _configure_logging()
    return logging.getLogger(f"finsight_guard.{name}")


def log_node_start(run_id: str, node_name: str, ticker: str) -> None:
    """Log node start with structured fields."""
    get_logger("workflow").info(
        "node_start",
        extra={"run_id": run_id, "node_name": node_name, "ticker": ticker},
    )


def log_node_success(run_id: str, node_name: str, elapsed_ms: float) -> None:
    """Log node success with elapsed time."""
    get_logger("workflow").info(
        "node_success",
        extra={"run_id": run_id, "node_name": node_name, "elapsed_ms": round(elapsed_ms, 2)},
    )


def log_node_error(run_id: str, node_name: str, error: Exception | str) -> None:
    """Log node error with structured fields."""
    get_logger("workflow").error(
        "node_error",
        extra={"run_id": run_id, "node_name": node_name, "error": str(error)},
    )
