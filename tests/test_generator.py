"""Tests for the generator CLI helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from generator import _generate_suggested_description, _get_transcript, _process_videos
from src.transcripts import (
    TRANSCRIPT_STATUS_AUDIO_FAILED,
    TRANSCRIPT_STATUS_BLOCKED,
)


def _make_args(**overrides: Any) -> SimpleNamespace:
    """Build a generator args namespace with test-friendly defaults."""
    defaults: dict[str, Any] = {
        "input": "output/channel_video_audit.csv",
        "output_csv": "output/review_report.csv",
        "output_html": "output/review_report.html",
        "cache": "output/llm_cache.json",
        "model": "",
        "channel_context": "",
        "regenerate": False,
        "transcripts_only": False,
        "audio_fallback": False,
        "whisper_model": "small",
        "limit": 0,
        "sleep": 0.0,
        "sleep_jitter": 0.0,
        "batch_size": 0,
        "batch_rest": 0,
        "keep_audio": False,
        "max_consecutive_blocks": 5,
        "transcript_max_chars": 20000,
        "skip_captions": False,
        "max_consecutive_audio_failures": 5,
        "audio_failure_cooldown": 180,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "video_id": f"vid{i}",
            "title": f"Video {i}",
            "video_url": f"https://www.youtube.com/watch?v=vid{i}",
            "published_at": "2024-01-15",
            "has_description": False,
            "description_length": 0,
            "status": "ok",
        }
        for i in range(count)
    ]


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
    args = SimpleNamespace(audio_fallback=False, whisper_model="small")
    transcript, status, blocked = _get_transcript("vid1", cache, args, None)
    assert transcript == "cached text"
    assert status == "success"
    mock_fetch.assert_not_called()


@patch("generator.fetch_transcript")
def test_get_transcript_fetch_and_cache(mock_fetch: MagicMock) -> None:
    mock_fetch.return_value = ("fetched text", "success")
    cache: dict[str, Any] = {}
    args = SimpleNamespace(audio_fallback=False, whisper_model="small")
    transcript, status, blocked = _get_transcript("vid1", cache, args, None)
    assert transcript == "fetched text"
    assert status == "success"
    mock_fetch.assert_called_once_with("vid1")


@patch("generator.transcribe_audio")
@patch("generator.cleanup_temp_audio")
@patch("generator.fetch_transcript")
def test_get_transcript_blocked_falls_back_to_audio(
    mock_fetch: MagicMock,
    mock_cleanup: MagicMock,
    mock_transcribe: MagicMock,
) -> None:
    mock_fetch.return_value = (None, TRANSCRIPT_STATUS_BLOCKED)
    mock_transcribe.return_value = ("audio text", "audio_transcribed")
    cache: dict[str, Any] = {}
    args = SimpleNamespace(
        audio_fallback=True, whisper_model="small", keep_audio=False
    )
    transcript, status, blocked = _get_transcript("vid1", cache, args, MagicMock())
    assert transcript == "audio text"
    assert status == "audio_transcribed"
    mock_transcribe.assert_called_once()
    # A successful audio fallback must still report the caption block, so the
    # caller's circuit breaker can stop hitting the blocked endpoint.
    assert blocked is True


@patch("generator.transcribe_audio")
@patch("generator.cleanup_temp_audio")
@patch("generator.fetch_transcript")
def test_get_transcript_skip_captions_bypasses_api(
    mock_fetch: MagicMock,
    mock_cleanup: MagicMock,
    mock_transcribe: MagicMock,
) -> None:
    """Once the breaker trips, the caption API must not be called again."""
    mock_transcribe.return_value = ("audio text", "audio_transcribed")
    cache: dict[str, Any] = {}
    args = SimpleNamespace(
        audio_fallback=True, whisper_model="small", keep_audio=False
    )
    transcript, status, blocked = _get_transcript(
        "vid1", cache, args, MagicMock(), skip_captions=True
    )
    assert transcript == "audio text"
    assert status == "audio_transcribed"
    mock_fetch.assert_not_called()


@patch("generator.transcribe_audio")
@patch("generator.cleanup_temp_audio")
@patch("generator.fetch_transcript")
def test_get_transcript_skip_captions_arg_implies_fallback(
    mock_fetch: MagicMock,
    mock_cleanup: MagicMock,
    mock_transcribe: MagicMock,
) -> None:
    """--skip-captions alone (no --audio-fallback) still uses audio."""
    mock_transcribe.return_value = ("audio text", "audio_transcribed")
    cache: dict[str, Any] = {}
    args = SimpleNamespace(
        audio_fallback=False,
        skip_captions=True,
        whisper_model="small",
        keep_audio=False,
    )
    transcript, status, blocked = _get_transcript(
        "vid1", cache, args, MagicMock(), skip_captions=True
    )
    assert transcript == "audio text"
    assert status == "audio_transcribed"
    assert blocked is True
    mock_fetch.assert_not_called()
    mock_transcribe.assert_called_once()


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
    args = _make_args()
    cache: dict[str, Any] = {}

    results = _process_videos(rows, args, cache)

    assert len(results) == 1
    assert results[0]["video_id"] == "vid1"
    assert results[0]["suggested_description"] == "Suggested description text"
    assert results[0]["transcript"] == "transcript text"
    assert results[0]["transcript_status"] == "success"
    mock_csv.assert_called()
    mock_html.assert_called()


@patch("generator.write_review_csv")
@patch("generator.generate_review_html")
@patch("generator.save_cache")
@patch("generator.fetch_transcript")
@patch("generator.jittered_sleep")
@patch("generator.tqdm")
def test_process_videos_stops_after_consecutive_blocks(
    mock_tqdm: MagicMock,
    mock_sleep: MagicMock,
    mock_fetch: MagicMock,
    mock_save_cache: MagicMock,
    mock_html: MagicMock,
    mock_csv: MagicMock,
) -> None:
    """Without an audio fallback, a sustained IP block aborts the run early."""
    mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
    mock_tqdm.write = MagicMock()
    mock_fetch.return_value = (None, TRANSCRIPT_STATUS_BLOCKED)

    rows = _make_rows(50)
    args = _make_args(max_consecutive_blocks=3, transcripts_only=True)
    cache: dict[str, Any] = {}

    results = _process_videos(rows, args, cache)

    assert len(results) == 3
    assert mock_fetch.call_count == 3
    # Progress up to the abort is still persisted.
    mock_save_cache.assert_called()


@patch("generator.write_review_csv")
@patch("generator.generate_review_html")
@patch("generator.save_cache")
@patch("generator.transcribe_audio")
@patch("generator.cleanup_temp_audio")
@patch("generator.fetch_transcript")
@patch("generator._load_whisper_model")
@patch("generator.jittered_sleep")
@patch("generator.tqdm")
def test_process_videos_switches_to_audio_after_blocks(
    mock_tqdm: MagicMock,
    mock_sleep: MagicMock,
    mock_load_whisper: MagicMock,
    mock_fetch: MagicMock,
    mock_cleanup: MagicMock,
    mock_transcribe: MagicMock,
    mock_save_cache: MagicMock,
    mock_html: MagicMock,
    mock_csv: MagicMock,
) -> None:
    """With --audio-fallback the run continues, but stops hitting captions."""
    mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
    mock_tqdm.write = MagicMock()
    mock_fetch.return_value = (None, TRANSCRIPT_STATUS_BLOCKED)
    mock_transcribe.return_value = ("audio text", "audio_transcribed")

    rows = _make_rows(8)
    args = _make_args(
        max_consecutive_blocks=2, audio_fallback=True, transcripts_only=True
    )
    cache: dict[str, Any] = {}

    results = _process_videos(rows, args, cache)

    # All rows processed, but captions only attempted until the breaker tripped.
    assert len(results) == 8
    assert mock_fetch.call_count == 2
    assert mock_transcribe.call_count == 8
    assert results[-1]["transcript"] == "audio text"


@patch("generator.write_review_csv")
@patch("generator.generate_review_html")
@patch("generator.save_cache")
@patch("generator.transcribe_audio")
@patch("generator.cleanup_temp_audio")
@patch("generator.fetch_transcript")
@patch("generator._load_whisper_model")
@patch("generator.jittered_sleep")
@patch("generator.tqdm")
def test_process_videos_skip_captions_never_calls_api(
    mock_tqdm: MagicMock,
    mock_sleep: MagicMock,
    mock_load_whisper: MagicMock,
    mock_fetch: MagicMock,
    mock_cleanup: MagicMock,
    mock_transcribe: MagicMock,
    mock_save_cache: MagicMock,
    mock_html: MagicMock,
    mock_csv: MagicMock,
) -> None:
    """--skip-captions bypasses the caption API from the very first video."""
    mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
    mock_tqdm.write = MagicMock()
    mock_transcribe.return_value = ("audio text", "audio_transcribed")

    rows = _make_rows(6)
    args = _make_args(skip_captions=True, transcripts_only=True)
    cache: dict[str, Any] = {}

    results = _process_videos(rows, args, cache)

    assert len(results) == 6
    mock_fetch.assert_not_called()
    assert mock_transcribe.call_count == 6
    assert all(r["transcript"] == "audio text" for r in results)


@patch("generator.write_review_csv")
@patch("generator.generate_review_html")
@patch("generator.save_cache")
@patch("generator.fetch_transcript")
@patch("generator.generate_description")
@patch("generator.jittered_sleep")
@patch("generator.tqdm")
def test_process_videos_resets_block_counter_on_success(
    mock_tqdm: MagicMock,
    mock_sleep: MagicMock,
    mock_gen: MagicMock,
    mock_fetch: MagicMock,
    mock_save_cache: MagicMock,
    mock_html: MagicMock,
    mock_csv: MagicMock,
) -> None:
    """Intermittent blocks interleaved with successes must not abort the run."""
    mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
    mock_tqdm.write = MagicMock()
    mock_gen.return_value = "desc"
    mock_fetch.side_effect = [
        (None, TRANSCRIPT_STATUS_BLOCKED),
        (None, TRANSCRIPT_STATUS_BLOCKED),
        ("ok text", "success"),
        (None, TRANSCRIPT_STATUS_BLOCKED),
        (None, TRANSCRIPT_STATUS_BLOCKED),
        ("ok text", "success"),
    ]

    rows = _make_rows(6)
    args = _make_args(max_consecutive_blocks=3)
    cache: dict[str, Any] = {}

    results = _process_videos(rows, args, cache)
    assert len(results) == 6


@patch("generator.time.sleep")
@patch("generator.write_review_csv")
@patch("generator.generate_review_html")
@patch("generator.save_cache")
@patch("generator.transcribe_audio")
@patch("generator.cleanup_temp_audio")
@patch("generator._load_whisper_model")
@patch("generator.jittered_sleep")
@patch("generator.tqdm")
def test_process_videos_cools_down_on_audio_failures(
    mock_tqdm: MagicMock,
    mock_sleep_jitter: MagicMock,
    mock_load_whisper: MagicMock,
    mock_cleanup: MagicMock,
    mock_transcribe: MagicMock,
    mock_save_cache: MagicMock,
    mock_html: MagicMock,
    mock_csv: MagicMock,
    mock_time_sleep: MagicMock,
) -> None:
    """A burst of audio failures triggers a cooldown, not an abort."""
    mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
    mock_tqdm.write = MagicMock()
    mock_transcribe.return_value = (None, TRANSCRIPT_STATUS_AUDIO_FAILED)

    rows = _make_rows(7)
    args = _make_args(
        skip_captions=True,
        transcripts_only=True,
        max_consecutive_audio_failures=3,
        audio_failure_cooldown=180,
    )
    cache: dict[str, Any] = {}

    results = _process_videos(rows, args, cache)

    # All rows still attempted (failures are retried on a later run).
    assert len(results) == 7
    # Cooldown fires each time the threshold is reached: 7 consecutive failures
    # with a threshold of 3 fires at #3 and again at #6.
    assert mock_time_sleep.call_count == 2
    assert all(c[0][0] == 180 for c in mock_time_sleep.call_args_list)


@patch("generator.time.sleep")
@patch("generator.write_review_csv")
@patch("generator.generate_review_html")
@patch("generator.save_cache")
@patch("generator.transcribe_audio")
@patch("generator.cleanup_temp_audio")
@patch("generator._load_whisper_model")
@patch("generator.jittered_sleep")
@patch("generator.tqdm")
def test_process_videos_audio_failure_counter_resets(
    mock_tqdm: MagicMock,
    mock_sleep_jitter: MagicMock,
    mock_load_whisper: MagicMock,
    mock_cleanup: MagicMock,
    mock_transcribe: MagicMock,
    mock_save_cache: MagicMock,
    mock_html: MagicMock,
    mock_csv: MagicMock,
    mock_time_sleep: MagicMock,
) -> None:
    """Interleaved successes prevent the audio-failure cooldown from firing."""
    mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
    mock_tqdm.write = MagicMock()
    mock_transcribe.side_effect = [
        (None, TRANSCRIPT_STATUS_AUDIO_FAILED),
        (None, TRANSCRIPT_STATUS_AUDIO_FAILED),
        ("ok", "audio_transcribed"),
        (None, TRANSCRIPT_STATUS_AUDIO_FAILED),
        (None, TRANSCRIPT_STATUS_AUDIO_FAILED),
        ("ok", "audio_transcribed"),
    ]

    rows = _make_rows(6)
    args = _make_args(
        skip_captions=True,
        transcripts_only=True,
        max_consecutive_audio_failures=3,
        audio_failure_cooldown=180,
    )
    cache: dict[str, Any] = {}

    _process_videos(rows, args, cache)
    mock_time_sleep.assert_not_called()
