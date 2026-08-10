"""Tests for the transcripts module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.transcripts import (
    TRANSCRIPT_STATUS_AUDIO,
    TRANSCRIPT_STATUS_AUDIO_FAILED,
    TRANSCRIPT_STATUS_BLOCKED,
    TRANSCRIPT_STATUS_DISABLED,
    TRANSCRIPT_STATUS_ERROR,
    TRANSCRIPT_STATUS_NO_CAPTIONS,
    TRANSCRIPT_STATUS_OK,
    _download_audio,
    cleanup_temp_audio,
    fetch_transcript,
    transcribe_audio,
)


class _FakeLine:
    def __init__(self, text: str) -> None:
        self.text = text


@patch("src.transcripts.YouTubeTranscriptApi")
def test_fetch_transcript_success(mock_api: MagicMock) -> None:
    mock_api.return_value.fetch.return_value = [
        _FakeLine("Hello world"),
        _FakeLine("Energy discussion"),
    ]
    text, status = fetch_transcript("abc123")
    assert status == TRANSCRIPT_STATUS_OK
    assert "Hello world" in text
    assert "Energy discussion" in text


@patch("src.transcripts.YouTubeTranscriptApi")
def test_fetch_transcript_disabled(mock_api: MagicMock) -> None:
    from src.transcripts import TranscriptsDisabled

    mock_api.return_value.fetch.side_effect = TranscriptsDisabled("abc123")
    text, status = fetch_transcript("abc123")
    assert status == TRANSCRIPT_STATUS_DISABLED
    assert text is None


@patch("src.transcripts.YouTubeTranscriptApi")
def test_fetch_transcript_no_captions(mock_api: MagicMock) -> None:
    from src.transcripts import NoTranscriptFound

    mock_api.return_value.fetch.side_effect = NoTranscriptFound(
        "abc123", ["en"], MagicMock()
    )
    text, status = fetch_transcript("abc123")
    assert status == TRANSCRIPT_STATUS_NO_CAPTIONS
    assert text is None


@patch("src.transcripts.YouTubeTranscriptApi")
def test_fetch_transcript_unexpected_error(mock_api: MagicMock) -> None:
    mock_api.return_value.fetch.side_effect = TypeError("boom")
    text, status = fetch_transcript("abc123")
    assert status == TRANSCRIPT_STATUS_ERROR
    assert text is None


@patch("src.transcripts.YouTubeTranscriptApi")
def test_fetch_transcript_blocked(mock_api: MagicMock) -> None:
    from src.transcripts import RequestBlocked

    mock_api.return_value.fetch.side_effect = RequestBlocked("abc123")
    text, status = fetch_transcript("abc123")
    assert status == TRANSCRIPT_STATUS_BLOCKED
    assert text is None


@patch("src.transcripts.YouTubeTranscriptApi")
def test_fetch_transcript_blocked_is_not_retried(mock_api: MagicMock) -> None:
    """An IP block is caller-wide, so retrying would only deepen the ban."""
    from src.transcripts import RequestBlocked

    mock_api.return_value.fetch.side_effect = RequestBlocked("abc123")
    _text, status = fetch_transcript("abc123")
    assert status == TRANSCRIPT_STATUS_BLOCKED
    assert mock_api.return_value.fetch.call_count == 1


@patch("src.transcripts.subprocess.run")
def test_download_audio_success(mock_run: MagicMock, tmp_path: Path) -> None:
    # yt-dlp appends the real extension to the base path.
    (tmp_path / "abc123.m4a").write_text("audio data")
    assert _download_audio("abc123", tmp_path / "abc123") is True
    mock_run.assert_called_once()


@patch("src.transcripts.subprocess.run")
def test_download_audio_does_not_require_ffmpeg(
    mock_run: MagicMock, tmp_path: Path
) -> None:
    """Native audio is kept; mp3 conversion would need ffmpeg."""
    (tmp_path / "abc123.m4a").write_text("audio data")
    _download_audio("abc123", tmp_path / "abc123")
    cmd = mock_run.call_args[0][0]
    assert "--extract-audio" not in cmd
    assert "mp3" not in cmd


@patch("src.transcripts.subprocess.run")
def test_download_audio_failure(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.side_effect = RuntimeError("boom")
    assert _download_audio("abc123", tmp_path / "abc123") is False


@patch("src.transcripts.subprocess.run")
def test_download_audio_missing_file(mock_run: MagicMock, tmp_path: Path) -> None:
    """yt-dlp exiting 0 without producing a file is still a failure."""
    assert _download_audio("abc123", tmp_path / "abc123") is False


def test_transcribe_audio_success(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "abc123.mp3"
    audio_path.write_text("audio data")

    class _FakeSegment:
        def __init__(self, text: str) -> None:
            self.text = text

    model = MagicMock()
    model.transcribe.return_value = ([_FakeSegment("Hello"), _FakeSegment("world")], None)

    text, status = transcribe_audio("abc123", model, audio_dir=audio_dir)
    assert status == TRANSCRIPT_STATUS_AUDIO
    assert text == "Hello world"


def test_transcribe_audio_failure(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "abc123.mp3"
    audio_path.write_text("audio data")

    model = MagicMock()
    model.transcribe.side_effect = RuntimeError("boom")

    text, status = transcribe_audio("abc123", model, audio_dir=audio_dir)
    assert status == TRANSCRIPT_STATUS_AUDIO_FAILED
    assert text is None


def test_cleanup_temp_audio(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "abc123.mp3").write_text("audio")
    (audio_dir / "abc123.m4a").write_text("audio")
    (audio_dir / "abc123.webm").write_text("audio")
    cleanup_temp_audio("abc123", audio_dir=audio_dir)
    assert not any(audio_dir.iterdir())
