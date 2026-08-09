"""Tests for the indexer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from indexer import _load_checkpoint, _process_videos, _save_checkpoint


def _args(**kwargs: Any) -> argparse.Namespace:
    defaults = {
        "limit": 0,
        "sleep": 0.0,
        "sleep_jitter": 0.0,
        "batch_size": 0,
        "batch_rest": 0,
        "resume": False,
        "verbose": False,
        "description_threshold": 30,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@patch("indexer.Path.exists", return_value=False)
def test_load_checkpoint_empty(_mock_exists: MagicMock) -> None:
    checkpoint = _load_checkpoint(Path("nonexistent.json"))
    assert checkpoint == {"completed_ids": set(), "failed_ids": set()}


def test_load_checkpoint_legacy_format(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text('{"completed_ids": ["abc123"]}')
    checkpoint = _load_checkpoint(path)
    assert checkpoint["completed_ids"] == {"abc123"}
    assert checkpoint["failed_ids"] == set()


def test_load_checkpoint_full_format(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text('{"completed_ids": ["abc123"], "failed_ids": ["def456"]}')
    checkpoint = _load_checkpoint(path)
    assert checkpoint["completed_ids"] == {"abc123"}
    assert checkpoint["failed_ids"] == {"def456"}


def test_save_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    _save_checkpoint(path, {"abc123"}, {"def456"})
    text = path.read_text(encoding="utf-8")
    assert '"completed_ids"' in text
    assert '"failed_ids"' in text
    assert "abc123" in text
    assert "def456" in text


@patch("indexer.fetch_video_metadata")
@patch("indexer.jittered_sleep")
@patch("indexer.tqdm")
@patch("indexer.signal.signal")
def test_process_videos_writes_rows(
    mock_signal: MagicMock,
    mock_tqdm: MagicMock,
    mock_sleep: MagicMock,
    mock_fetch: MagicMock,
    tmp_path: Path,
) -> None:
    mock_fetch.return_value = {
        "id": "abc123",
        "title": "Video One",
        "published_at": "2024-01-15",
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "description": "A sufficiently long description for this video.",
    }
    output_path = tmp_path / "audit.csv"
    videos = [
        {"id": "abc123", "title": "Video One", "url": "https://www.youtube.com/watch?v=abc123"},
    ]
    args = _args()

    rows = _process_videos(videos, args, output_path)

    assert len(rows) == 1
    assert rows[0]["video_id"] == "abc123"
    assert rows[0]["status"] == "ok"
    assert output_path.exists()


@patch("indexer.fetch_video_metadata")
@patch("indexer.jittered_sleep")
@patch("indexer.tqdm")
@patch("indexer.signal.signal")
def test_process_videos_resume_skips_completed(
    mock_signal: MagicMock,
    mock_tqdm: MagicMock,
    mock_sleep: MagicMock,
    mock_fetch: MagicMock,
    tmp_path: Path,
) -> None:
    mock_fetch.return_value = {
        "id": "def456",
        "title": "Video Two",
        "published_at": "2024-02-20",
        "webpage_url": "https://www.youtube.com/watch?v=def456",
        "description": "Another long description here.",
    }
    output_path = tmp_path / "audit.csv"
    checkpoint_path = tmp_path / "indexer_checkpoint.json"
    checkpoint_path.write_text('{"completed_ids": ["abc123"], "failed_ids": []}')

    # Pre-create an audit CSV with the already-completed video.
    output_path.write_text(
        "video_id,title,published_at,video_url,has_description,description_length,status\n"
        "abc123,Video One,2024-01-15,https://www.youtube.com/watch?v=abc123,True,45,ok\n"
    )

    videos = [
        {"id": "abc123", "title": "Video One", "url": "https://www.youtube.com/watch?v=abc123"},
        {"id": "def456", "title": "Video Two", "url": "https://www.youtube.com/watch?v=def456"},
    ]
    args = _args(resume=True)

    rows = _process_videos(videos, args, output_path)

    assert len(rows) == 2
    assert mock_fetch.call_count == 1
    fetched_video_id = mock_fetch.call_args[0][0].split("v=")[-1]
    assert fetched_video_id == "def456"


@patch("indexer.fetch_video_metadata")
@patch("indexer.jittered_sleep")
@patch("indexer.tqdm")
@patch("indexer.signal.signal")
def test_process_videos_records_failed_metadata(
    mock_signal: MagicMock,
    mock_tqdm: MagicMock,
    mock_sleep: MagicMock,
    mock_fetch: MagicMock,
    tmp_path: Path,
) -> None:
    mock_fetch.return_value = None
    output_path = tmp_path / "audit.csv"
    checkpoint_path = tmp_path / "indexer_checkpoint.json"
    videos = [
        {"id": "abc123", "title": "Video One", "url": "https://www.youtube.com/watch?v=abc123"}
    ]
    args = _args(verbose=True)

    with patch("indexer._CHECKPOINT_FILE", str(checkpoint_path)):
        rows = _process_videos(videos, args, output_path)

    assert len(rows) == 1
    assert rows[0]["status"] == "metadata_failed"
    assert checkpoint_path.exists()
    checkpoint = _load_checkpoint(checkpoint_path)
    assert "abc123" in checkpoint["failed_ids"]
