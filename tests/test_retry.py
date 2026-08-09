"""Tests for retry/backoff helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retry import retry_with_backoff


def test_retry_with_backoff_succeeds_on_first_attempt() -> None:
    assert retry_with_backoff(lambda: "ok") == "ok"


def test_retry_with_backoff_retries_then_succeeds() -> None:
    func = MagicMock(side_effect=[RuntimeError("boom"), "ok"])
    assert retry_with_backoff(func, max_attempts=3, base_delay=0.0) == "ok"
    assert func.call_count == 2


def test_retry_with_backoff_respects_non_retryable() -> None:
    func = MagicMock(side_effect=ValueError("nope"))
    with pytest.raises(ValueError, match="nope"):
        retry_with_backoff(
            func,
            max_attempts=3,
            base_delay=0.0,
            retryable=lambda exc: False,
        )
    assert func.call_count == 1


def test_retry_with_backoff_raises_after_exhaustion() -> None:
    func = MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        retry_with_backoff(func, max_attempts=3, base_delay=0.0)
    assert func.call_count == 3
