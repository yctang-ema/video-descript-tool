"""Tests for the LLM module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIError

import src.llm
from src.llm import (
    generate_description,
    load_cache,
    resolve_channel_context,
    resolve_model,
    resolve_models,
    save_cache,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


def test_resolve_model_defaults() -> None:
    import os

    env_backup = os.environ.get("LLM_MODEL")
    try:
        os.environ.pop("LLM_MODEL", None)
        assert resolve_model(None) == "gpt-5.4-mini"
        assert resolve_model("gpt-5.4") == "gpt-5.4"
    finally:
        if env_backup is not None:
            os.environ["LLM_MODEL"] = env_backup


@patch("src.llm.build_client")
def test_generate_description(mock_build_client: MagicMock) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _FakeCompletion("Suggested description text")
    mock_build_client.return_value = client

    result = generate_description("Annual Forum", "transcript content", model="gpt-5.4-mini")
    assert result == "Suggested description text"
    client.chat.completions.create.assert_called_once()
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-5.4-mini"
    assert any("Annual Forum" in msg["content"] for msg in kwargs["messages"])


def test_resolve_channel_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHANNEL_CONTEXT", raising=False)
    assert resolve_channel_context(None) == ""
    monkeypatch.setenv("CHANNEL_CONTEXT", "a cooking channel")
    assert resolve_channel_context(None) == "a cooking channel"
    assert resolve_channel_context("a gaming channel") == "a gaming channel"


@patch("src.llm.build_client")
def test_generate_description_generic_prompt(
    mock_build_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHANNEL_CONTEXT", raising=False)
    client = MagicMock()
    client.chat.completions.create.return_value = _FakeCompletion("desc")
    mock_build_client.return_value = client

    generate_description("Any Video", "transcript", model="gpt-5.4-mini")

    _, kwargs = client.chat.completions.create.call_args
    system_msg = kwargs["messages"][0]["content"]
    assert "energy" not in system_msg.lower()


@patch("src.llm.build_client")
def test_generate_description_with_channel_context(
    mock_build_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHANNEL_CONTEXT", raising=False)
    client = MagicMock()
    client.chat.completions.create.return_value = _FakeCompletion("desc")
    mock_build_client.return_value = client

    generate_description(
        "Any Video",
        "transcript",
        model="gpt-5.4-mini",
        channel_context="a cooking tutorial channel",
    )

    _, kwargs = client.chat.completions.create.call_args
    system_msg = kwargs["messages"][0]["content"]
    assert "Channel context:" in system_msg
    assert "a cooking tutorial channel" in system_msg


def test_load_and_save_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    assert load_cache(cache_path) == {}

    cache = {"abc123": {"transcript": "hello", "suggested_description": "desc"}}
    save_cache(cache, cache_path)
    loaded = load_cache(cache_path)
    assert loaded == cache


def _api_error() -> APIError:
    return APIError(
        "boom", request=httpx.Request("POST", "https://example.com"), body=None
    )


def test_resolve_models_with_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.llm._last_working_model", None)
    monkeypatch.setenv("LLM_MODEL_FALLBACKS", "claude-haiku-4-5, kimi-k2.5")
    assert resolve_models("gpt-5.4-mini") == [
        "gpt-5.4-mini",
        "claude-haiku-4-5",
        "kimi-k2.5",
    ]


def test_resolve_models_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.llm._last_working_model", "kimi-k2.5")
    monkeypatch.setenv("LLM_MODEL_FALLBACKS", "gpt-5.4-mini,kimi-k2.5")
    assert resolve_models("gpt-5.4-mini") == ["kimi-k2.5", "gpt-5.4-mini"]


def test_generate_description_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.llm._last_working_model", None)
    monkeypatch.setenv("LLM_MODEL_FALLBACKS", "claude-haiku-4-5")
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _api_error(),
        _FakeCompletion("fallback description"),
    ]

    result = generate_description("Annual Forum", "transcript", model="gpt-5.4-mini", client=client)

    assert result == "fallback description"
    assert client.chat.completions.create.call_count == 2
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "claude-haiku-4-5"
    assert src.llm._last_working_model == "claude-haiku-4-5"


def test_generate_description_all_models_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.llm._last_working_model", None)
    monkeypatch.setenv("LLM_MODEL_FALLBACKS", "claude-haiku-4-5")
    client = MagicMock()
    client.chat.completions.create.side_effect = [_api_error(), _api_error()]

    with pytest.raises(RuntimeError, match="All LLM models failed"):
        generate_description("Annual Forum", "transcript", model="gpt-5.4-mini", client=client)
