"""Retry helpers for external data providers."""

from collections.abc import Callable
from functools import wraps
import logging
import time
from typing import TypeVar


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)
COMMON_RETRY_EXCEPTIONS = (ConnectionError, TimeoutError, ValueError, RuntimeError)


def retry(
    max_attempts: int = 2,
    delay_seconds: float = 0.5,
    exceptions: tuple[type[Exception], ...] = COMMON_RETRY_EXCEPTIONS,
) -> Callable[[F], F]:
    """Retry a function and re-raise the final exception if all attempts fail."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_attempt = max_attempts
            for attempt in range(1, last_attempt + 1):
                try:
                    logger.info(
                        "retry_attempt_start",
                        extra={"function": func.__name__, "attempt": attempt},
                    )
                    result = func(*args, **kwargs)
                    logger.info(
                        "retry_attempt_success",
                        extra={"function": func.__name__, "attempt": attempt},
                    )
                    return result
                except exceptions as exc:
                    logger.warning(
                        "retry_attempt_failed",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "error": str(exc),
                        },
                    )
                    if attempt == last_attempt:
                        raise
                    if delay_seconds:
                        time.sleep(delay_seconds)

        return wrapper  # type: ignore[return-value]

    return decorator
