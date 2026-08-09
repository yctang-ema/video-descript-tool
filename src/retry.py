"""Retry/backoff helpers for network calls."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception looks like a 429/rate-limit response."""
    return "429" in str(exc)


def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: Callable[[Exception], bool] | None = None,
) -> T:
    """Call ``func`` repeatedly with exponential backoff until it succeeds.

    ``retryable(exc)`` can return False to abort retrying for a specific
    exception (e.g. permanent errors). All other exceptions are retried up to
    ``max_attempts`` times. 429/rate-limit responses are retried with the same
    exponential backoff.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            if retryable is not None and not retryable(exc):
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if _is_rate_limit_error(exc):
                delay = min(delay * 2, max_delay)
            jitter = random.uniform(0, delay * 0.5)  # noqa: S311
            time.sleep(delay + jitter)
    raise last_exc  # type: ignore[misc]
