"""Transcript retrieval via youtube-transcript-api and optional Whisper fallback."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from src.retry import retry_with_backoff

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


TRANSCRIPT_STATUS_OK = "success"
TRANSCRIPT_STATUS_DISABLED = "disabled"
TRANSCRIPT_STATUS_NO_CAPTIONS = "no_captions"
TRANSCRIPT_STATUS_BLOCKED = "blocked"
TRANSCRIPT_STATUS_AUDIO = "audio_transcribed"
TRANSCRIPT_STATUS_AUDIO_FAILED = "audio_failed"
TRANSCRIPT_STATUS_ERROR = "error"

#: Containers yt-dlp may produce for audio-only downloads.
AUDIO_EXTENSIONS = (".m4a", ".webm", ".mp3", ".opus", ".ogg", ".mp4", ".aac")


def _ytdlp_command() -> list[str]:
    """Return the command prefix used to invoke yt-dlp.

    Prefers the ``yt-dlp`` executable on PATH, but falls back to running the
    module with the current interpreter. The fallback matters when the tool is
    launched via an interpreter path (e.g. ``.venv/bin/python generator.py``)
    without the virtualenv's ``bin`` directory on PATH.
    """
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]
    return [sys.executable, "-m", "yt_dlp"]


def fetch_transcript(
    video_id: str,
    languages: list[str] | None = None,
) -> tuple[str | None, str]:
    """Fetch a YouTube transcript for ``video_id``.

    Returns a tuple ``(transcript_text, status)``. ``status`` is one of:
    ``success``, ``disabled``, ``no_captions``, ``blocked``, ``error``.

    ``blocked`` means YouTube refused the request because the caller's IP is
    rate-limited or banned (``RequestBlocked``/``IpBlocked``). It is a
    caller-wide condition rather than a per-video one, so it is never retried
    here: retrying only deepens the ban. Callers should stop requesting
    captions and switch to the audio fallback instead.
    """
    languages = languages or ["en"]
    try:
        transcript = retry_with_backoff(
            lambda: YouTubeTranscriptApi().fetch(video_id, languages=languages),
            max_attempts=3,
            base_delay=1.0,
            retryable=lambda exc: not isinstance(
                exc,
                (
                    TranscriptsDisabled,
                    NoTranscriptFound,
                    VideoUnavailable,
                    RequestBlocked,
                ),
            ),
        )
    except TranscriptsDisabled:
        return None, TRANSCRIPT_STATUS_DISABLED
    except RequestBlocked:
        return None, TRANSCRIPT_STATUS_BLOCKED
    except (NoTranscriptFound, VideoUnavailable):
        return None, TRANSCRIPT_STATUS_NO_CAPTIONS
    except Exception as exc:
        print(
            f"Warning: transcript fetch failed for {video_id}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None, TRANSCRIPT_STATUS_ERROR

    text = " ".join(line.text for line in transcript)
    return text.strip(), TRANSCRIPT_STATUS_OK


def _download_audio(
    video_id: str, output_base: Path, cookies: Path | None = None
) -> bool:
    """Download audio only using yt-dlp.

    ``output_base`` is a path without a file extension; yt-dlp appends the
    real one. The audio is kept in its native container (usually ``.m4a``)
    rather than being converted to mp3: conversion requires ffmpeg, while
    faster-whisper decodes the native container directly via PyAV. This keeps
    the audio fallback working on machines without ffmpeg installed.

    ``cookies`` optionally points to a Netscape-format ``cookies.txt`` file
    (exported from a browser). Supplying it lets yt-dlp pass YouTube's
    "sign in to confirm you're not a bot" check when the caller's IP is
    rate-limited.

    Returns True if an audio file was produced.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        *_ytdlp_command(),
        "-f",
        "bestaudio[ext=m4a]/bestaudio",
        "-o",
        f"{output_base}.%(ext)s",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--socket-timeout",
        "30",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--retry-sleep",
        "http:exp=5:120",
        "--sleep-requests",
        "1",
    ]
    if cookies is not None:
        cmd += ["--cookies", str(cookies)]
    cmd.append(url)
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        print(
            f"Warning: audio download timed out for {video_id}", file=sys.stderr
        )
        return False
    except Exception as exc:
        print(
            f"Warning: audio download failed for {video_id}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False
    return _find_audio_file(output_base.parent, output_base.name) is not None


def _find_audio_file(audio_dir: Path, video_id: str) -> Path | None:
    """Return the downloaded audio file path if it exists."""
    for ext in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def transcribe_audio(
    video_id: str,
    model: WhisperModel,
    audio_dir: Path | str = "temp_audio",
    cookies: Path | None = None,
) -> tuple[str | None, str]:
    """Download audio and transcribe it with a faster-whisper model.

    ``cookies`` optionally points to a ``cookies.txt`` file used for the
    download (see :func:`_download_audio`).

    Returns ``(transcript_text, status)``.
    """
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    base_path = audio_dir / video_id

    audio_path = _find_audio_file(audio_dir, video_id)
    if audio_path is None:
        if not _download_audio(video_id, base_path, cookies=cookies):
            return None, TRANSCRIPT_STATUS_AUDIO_FAILED
        audio_path = _find_audio_file(audio_dir, video_id)
    if audio_path is None:
        return None, TRANSCRIPT_STATUS_AUDIO_FAILED

    try:
        segments, _ = model.transcribe(str(audio_path), language="en")
        text = " ".join(segment.text for segment in segments)
    except Exception as exc:
        print(
            f"Warning: audio transcription failed for {video_id}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        # The audio file may be a partial/corrupt download left behind by an
        # interrupted run. Delete it so the next run re-downloads it cleanly
        # instead of reusing (and failing on) the same bad file forever.
        cleanup_temp_audio(video_id, audio_dir)
        return None, TRANSCRIPT_STATUS_AUDIO_FAILED

    return text.strip(), TRANSCRIPT_STATUS_AUDIO


def cleanup_temp_audio(video_id: str, audio_dir: Path | str = "temp_audio") -> None:
    """Remove temporary audio files for ``video_id``."""
    audio_dir = Path(audio_dir)
    for ext in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{video_id}{ext}"
        if candidate.exists():
            candidate.unlink()
