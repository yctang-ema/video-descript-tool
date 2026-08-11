"""Tests for the channel module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.channel import (
    _csv_safe,
    evaluate_description,
    extract_flat_channel_videos,
    fetch_video_metadata,
    load_audit_csv,
    write_audit_csv,
)


def test_evaluate_description_missing():
    has, length = evaluate_description(None)
    assert has is False
    assert length == 0


def test_evaluate_description_too_short():
    has, length = evaluate_description("Short desc")
    assert has is False
    assert length == 10


def test_evaluate_description_present():
    desc = "This is a sufficiently long video description for an energy event."
    has, length = evaluate_description(desc)
    assert has is True
    assert length > 30


def test_evaluate_description_whitespace_only():
    has, length = evaluate_description("   \n\t  ")
    assert has is False
    assert length == 0


def test_write_and_load_audit_csv(tmp_path: Path):
    rows = [
        {
            "video_id": "abc123",
            "title": "Annual Conference Keynote",
            "published_at": "2024-01-15",
            "video_url": "https://www.youtube.com/watch?v=abc123",
            "has_description": True,
            "description_length": 45,
            "status": "ok",
        },
        {
            "video_id": "def456",
            "title": "Panel Discussion",
            "published_at": "2024-02-20",
            "video_url": "https://www.youtube.com/watch?v=def456",
            "has_description": False,
            "description_length": 12,
            "status": "ok",
        },
    ]
    path = tmp_path / "audit.csv"
    write_audit_csv(rows, path)
    loaded = load_audit_csv(path)
    assert len(loaded) == 2
    assert loaded[0]["video_id"] == "abc123"
    assert loaded[0]["has_description"] is True
    assert loaded[0]["description_length"] == 45
    assert loaded[0]["status"] == "ok"
    assert loaded[1]["has_description"] is False
    assert loaded[1]["description_length"] == 12
    assert loaded[1]["status"] == "ok"


def test_audit_csv_preserves_video_ids_starting_with_dash(tmp_path: Path):
    """Regression: video_id is a machine identifier and must never be altered.

    A spreadsheet formula-injection guard (``_csv_safe``) used to prepend an
    apostrophe to any cell starting with ``-``; applied to video_id it corrupted
    real YouTube IDs such as ``-acRraKkZfU``, breaking Tool 2 downloads.
    """
    rows = [
        {
            "video_id": "-acRraKkZfU",
            "title": "Some title",
            "published_at": "2025-11-05",
            "video_url": "https://www.youtube.com/watch?v=-acRraKkZfU",
            "has_description": False,
            "description_length": 0,
            "status": "ok",
        }
    ]
    path = tmp_path / "audit.csv"
    write_audit_csv(rows, path)
    loaded = load_audit_csv(path)
    assert loaded[0]["video_id"] == "-acRraKkZfU"
    assert loaded[0]["video_url"] == "https://www.youtube.com/watch?v=-acRraKkZfU"


def test_audit_csv_still_sanitises_title_formula(tmp_path: Path):
    """The formula-injection guard must still apply to free-text titles."""
    rows = [
        {
            "video_id": "abc123",
            "title": "=HYPERLINK(\"http://evil\")",
            "published_at": "2024-01-15",
            "video_url": "https://www.youtube.com/watch?v=abc123",
            "has_description": False,
            "description_length": 0,
            "status": "ok",
        }
    ]
    path = tmp_path / "audit.csv"
    write_audit_csv(rows, path)
    loaded = load_audit_csv(path)
    assert loaded[0]["title"].startswith("'")


def test_write_and_load_audit_csv_with_failed_status(tmp_path: Path):
    rows = [
        {
            "video_id": "xyz789",
            "title": "",
            "published_at": "",
            "video_url": "https://www.youtube.com/watch?v=xyz789",
            "has_description": False,
            "description_length": 0,
            "status": "metadata_failed",
        }
    ]
    path = tmp_path / "audit.csv"
    write_audit_csv(rows, path)
    loaded = load_audit_csv(path)
    assert len(loaded) == 1
    assert loaded[0]["video_id"] == "xyz789"
    assert loaded[0]["has_description"] is False
    assert loaded[0]["status"] == "metadata_failed"


def test_csv_safe_formula_injection():
    assert _csv_safe("=SUM(A1)") == "'=SUM(A1)"
    assert _csv_safe("+123") == "'+123"
    assert _csv_safe("-123") == "'-123"
    assert _csv_safe("@user") == "'@user"
    assert _csv_safe("  =SUM(A1)") == "'  =SUM(A1)"
    assert _csv_safe("Normal title") == "Normal title"
    assert _csv_safe(123) == "123"
    assert _csv_safe("") == ""


def test_evaluate_description_custom_threshold():
    has, length = evaluate_description("Short", threshold=10)
    assert has is False
    assert length == 5

    has, length = evaluate_description("Long enough text", threshold=10)
    assert has is True
    assert length == 16


@patch("src.channel.YoutubeDL")
@patch("src.channel.retry_with_backoff", side_effect=lambda func, **kwargs: func())
def test_extract_flat_channel_videos(mock_retry, mock_ytdl):
    ydl = MagicMock()
    ydl.extract_info.return_value = {
        "entries": [
            {"id": "abc123", "title": "Video One"},
            {"id": "def456", "title": "Video Two"},
        ]
    }
    mock_ytdl.return_value.__enter__.return_value = ydl

    videos = extract_flat_channel_videos("https://www.youtube.com/@test/videos")
    assert len(videos) == 2
    assert videos[0] == {
        "id": "abc123",
        "title": "Video One",
        "url": "https://www.youtube.com/watch?v=abc123",
    }
    assert videos[1]["id"] == "def456"


@patch("src.channel.YoutubeDL")
@patch("src.channel.retry_with_backoff", side_effect=lambda func, **kwargs: func())
def test_fetch_video_metadata(mock_retry, mock_ytdl):
    ydl = MagicMock()
    ydl.extract_info.return_value = {
        "id": "abc123",
        "title": "Video One",
        "upload_date": "20240115",
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "description": "A description",
    }
    mock_ytdl.return_value.__enter__.return_value = ydl

    metadata = fetch_video_metadata("https://www.youtube.com/watch?v=abc123")
    assert metadata == {
        "id": "abc123",
        "title": "Video One",
        "published_at": "2024-01-15",
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "description": "A description",
    }


@patch("src.channel.YoutubeDL")
@patch("src.channel.retry_with_backoff", side_effect=lambda func, **kwargs: func())
def test_fetch_video_metadata_failure_returns_none(mock_retry, mock_ytdl):
    ydl = MagicMock()
    ydl.extract_info.side_effect = RuntimeError("boom")
    mock_ytdl.return_value.__enter__.return_value = ydl

    assert fetch_video_metadata("https://www.youtube.com/watch?v=abc123") is None
