"""Tests for the generator CLI helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from generator import _generate_suggested_description, _get_transcript, _process_videos


@patch("generator.generate_description")
def test_cached_description_returned_by_default(mock_gen: MagicMock) -> None:
    cache: dict[str, Any] = {"vid1": {"suggested_description": "old text"}}
    result = _generate_suggested_description(
        "vid1", "Title", "transcript", cache, "gpt-5.4-mini", None
    )
    assert result == "old text"
    mock_gen.assert_not_called()


@patch("generator.generate_description")
def test_regenerate_bypasses_cached_description(mock_gen: MagicMock) -> None:
    mock_gen.return_value = "new text"
    cache: dict[str, Any] = {
        "vid1": {"suggested_description": "old text", "transcript": "cached t"}
    }
    result = _generate_suggested_description(
        "vid1", "Title", "cached t", cache, "gpt-5.4-mini", None, regenerate=True
    )
    assert result == "new text"
    mock_gen.assert_called_once()
    assert cache["vid1"]["suggested_description"] == "new text"
    # Cached transcripts are untouched by regeneration.
    assert cache["vid1"]["transcript"] == "cached t"


@patch("generator.fetch_transcript")
def test_get_transcript_from_cache(mock_fetch: MagicMock) -> None:
    cache: dict[str, Any] = {
        "vid1": {"transcript": "cached text", "transcript_status": "success"}
    }
    args = SimpleNamespace(audio_fallback=False, whisper_model="base")
    transcript, status = _get_transcript("vid1", cache, args, None)
    assert transcript == "cached text"
    assert status == "success"
    mock_fetch.assert_not_called()


@patch("generator.fetch_transcript")
def test_get_transcript_fetch_and_cache(mock_fetch: MagicMock) -> None:
    mock_fetch.return_value = ("fetched text", "success")
    cache: dict[str, Any] = {}
    args = SimpleNamespace(audio_fallback=False, whisper_model="base")
    transcript, status = _get_transcript("vid1", cache, args, None)
    assert transcript == "fetched text"
    assert status == "success"
    mock_fetch.assert_called_once_with("vid1")


@patch("generator.write_review_csv")
@patch("generator.generate_review_html")
@patch("generator.save_cache")
@patch("generator.fetch_transcript")
@patch("generator.generate_description")
@patch("generator.jittered_sleep")
@patch("generator.tqdm")
def test_process_videos_generates_descriptions(
    mock_tqdm: MagicMock,
    mock_sleep: MagicMock,
    mock_gen: MagicMock,
    mock_fetch: MagicMock,
    mock_save_cache: MagicMock,
    mock_html: MagicMock,
    mock_csv: MagicMock,
) -> None:
    mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
    mock_fetch.return_value = ("transcript text", "success")
    mock_gen.return_value = "Suggested description text"

    rows = [
        {
            "video_id": "vid1",
            "title": "Video One",
            "video_url": "https://www.youtube.com/watch?v=vid1",
            "published_at": "2024-01-15",
            "has_description": False,
            "description_length": 0,
            "status": "ok",
        }
    ]
    args = SimpleNamespace(
        input="output/channel_video_audit.csv",
        output_csv="output/review_report.csv",
        output_html="output/review_report.html",
        cache="output/llm_cache.json",
        model="",
        channel_context="",
        regenerate=False,
        transcripts_only=False,
        audio_fallback=False,
        whisper_model="base",
        limit=0,
        sleep=0.0,
        sleep_jitter=0.0,
        batch_size=0,
        batch_rest=0,
        keep_audio=False,
    )
    cache: dict[str, Any] = {}

    results = _process_videos(rows, args, cache)

    assert len(results) == 1
    assert results[0]["video_id"] == "vid1"
    assert results[0]["suggested_description"] == "Suggested description text"
    assert results[0]["transcript"] == "transcript text"
    assert results[0]["transcript_status"] == "success"
    mock_csv.assert_called()
    mock_html.assert_called()
