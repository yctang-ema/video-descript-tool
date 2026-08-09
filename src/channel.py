"""Channel metadata extraction and audit CSV helpers."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

from src.retry import retry_with_backoff

# Bound network waits so a hung connection can't block the process indefinitely.
_YDL_SOCKET_TIMEOUT = 30


def _video_url(video_id: str) -> str:
    """Return a canonical YouTube watch URL for a video ID."""
    return f"https://www.youtube.com/watch?v={video_id}"


def _csv_safe(value: Any) -> str:
    """Return a CSV-safe string that neutralises spreadsheet formula injection."""
    text = str(value)
    stripped = text.lstrip()
    if stripped and stripped[0] in {"=", "+", "-", "@"}:
        return "'" + text
    return text


def _format_published_at(upload_date: str | None) -> str:
    """Convert yt-dlp's YYYYMMDD upload_date to YYYY-MM-DD."""
    if upload_date and len(upload_date) == 8 and upload_date.isdigit():
        return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    return upload_date or ""


def evaluate_description(
    description: str | None, threshold: int = 30
) -> tuple[bool, int]:
    """Return (has_description, description_length).

    A description is considered present if it has at least ``threshold``
    non-whitespace characters.
    """
    clean = (description or "").strip()
    length = len(clean)
    return length >= threshold, length


def extract_flat_channel_videos(channel_url: str) -> list[dict[str, Any]]:
    """Return a flat list of video entries from a channel URL.

    Each entry contains at least ``id``, ``title``, and ``url``.
    """
    ydl_opts = {
        "extract_flat": True,
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": _YDL_SOCKET_TIMEOUT,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    if not info:
        return []
    entries = info.get("entries") or []
    result = []
    for entry in entries:
        entry_id = entry.get("id")
        if not entry_id:
            continue
        result.append(
            {
                "id": entry_id,
                "title": entry.get("title"),
                "url": _video_url(entry_id),
            }
        )
    return result


def _extract_info(video_url: str, ydl_opts: dict[str, Any]) -> Any:
    """Extract video metadata using a yt-dlp context manager."""
    with YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(video_url, download=False)


def fetch_video_metadata(video_url: str) -> dict[str, Any] | None:
    """Fetch full metadata for a single video URL.

    Retries transient network errors with exponential backoff. Returns a dict
    with keys ``id``, ``title``, ``published_at``, ``webpage_url``,
    ``description``, or ``None`` if extraction fails after retries.
    """
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": _YDL_SOCKET_TIMEOUT,
    }
    try:
        info = retry_with_backoff(
            lambda: _extract_info(video_url, ydl_opts),
            max_attempts=3,
            base_delay=1.0,
        )
    except Exception:
        return None
    if not info:
        return None
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "published_at": _format_published_at(info.get("upload_date")),
        "webpage_url": info.get("webpage_url"),
        "description": info.get("description") or "",
    }


def jittered_sleep(base: float, jitter: float) -> None:
    """Sleep for ``base`` seconds plus a random amount up to ``jitter``."""
    delay = base + random.uniform(0, jitter)  # noqa: S311
    time.sleep(max(delay, 0))


def write_audit_csv(entries: list[dict[str, Any]], output_path: Path | str) -> None:
    """Write the channel audit CSV."""
    path = Path(output_path)
    fieldnames = [
        "video_id",
        "title",
        "published_at",
        "video_url",
        "has_description",
        "description_length",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "video_id": _csv_safe(entry.get("video_id")),
                    "title": _csv_safe(entry.get("title")),
                    "published_at": _csv_safe(entry.get("published_at")),
                    "video_url": _csv_safe(entry.get("video_url")),
                    "has_description": _csv_safe(entry.get("has_description")),
                    "description_length": _csv_safe(entry.get("description_length")),
                    "status": _csv_safe(entry.get("status", "ok")),
                }
            )


def append_audit_csv_row(entry: dict[str, Any], output_path: Path | str) -> None:
    """Append a single row to an existing audit CSV (assumes header exists)."""
    path = Path(output_path)
    fieldnames = [
        "video_id",
        "title",
        "published_at",
        "video_url",
        "has_description",
        "description_length",
        "status",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(
            {
                "video_id": _csv_safe(entry.get("video_id")),
                "title": _csv_safe(entry.get("title")),
                "published_at": _csv_safe(entry.get("published_at")),
                "video_url": _csv_safe(entry.get("video_url")),
                "has_description": _csv_safe(entry.get("has_description")),
                "description_length": _csv_safe(entry.get("description_length")),
                "status": _csv_safe(entry.get("status", "ok")),
            }
        )


def load_audit_csv(path: Path | str) -> list[dict[str, Any]]:
    """Load the audit CSV and coerce booleans/ints."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if reader.fieldnames is None:
                continue
            has_desc = row.get("has_description", "")
            row["has_description"] = has_desc.lower() in {"true", "1", "yes"}
            try:
                row["description_length"] = int(row.get("description_length", "0") or 0)
            except ValueError:
                row["description_length"] = 0
            row["status"] = row.get("status") or "ok"
            rows.append(row)
    return rows
