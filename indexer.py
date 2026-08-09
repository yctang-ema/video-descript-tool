"""YouTube channel video auditor (Tool 1)."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from src.channel import (
    _video_url,
    append_audit_csv_row,
    evaluate_description,
    extract_flat_channel_videos,
    fetch_video_metadata,
    jittered_sleep,
    load_audit_csv,
    write_audit_csv,
)

load_dotenv()

_CHECKPOINT_FILE = "output/indexer_checkpoint.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a YouTube channel for videos with missing or short descriptions."
    )
    parser.add_argument(
        "--channel-url",
        default="",
        help="YouTube channel URL, e.g. https://www.youtube.com/@<handle>/videos",
    )
    parser.add_argument(
        "--output",
        default="output/channel_video_audit.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Base delay between metadata requests (seconds)",
    )
    parser.add_argument(
        "--sleep-jitter",
        type=float,
        default=3.0,
        help="Max random jitter added to sleep (seconds)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Videos per batch; 0 disables batching",
    )
    parser.add_argument(
        "--batch-rest",
        type=int,
        default=300,
        help="Seconds to rest between batches",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N videos; 0 = all",
    )
    parser.add_argument(
        "--description-threshold",
        type=int,
        default=30,
        help="Minimum non-whitespace characters to count as a description",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Resume from {_CHECKPOINT_FILE}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for individual video failures",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict[str, set[str]]:
    """Load the checkpoint (completed and failed video IDs)."""
    if not path.exists():
        return {"completed_ids": set(), "failed_ids": set()}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, TypeError):
        return {"completed_ids": set(), "failed_ids": set()}
    if not isinstance(data, dict):
        return {"completed_ids": set(), "failed_ids": set()}
    return {
        "completed_ids": set(data.get("completed_ids", [])),
        "failed_ids": set(data.get("failed_ids", [])),
    }


def _save_checkpoint(path: Path, completed_ids: set[str], failed_ids: set[str]) -> None:
    """Persist completed and failed video IDs."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {"completed_ids": sorted(completed_ids), "failed_ids": sorted(failed_ids)},
            f,
            indent=2,
        )


def _get_channel_url(args: argparse.Namespace) -> str:
    """Resolve channel URL from CLI argument or CHANNEL_URL env var."""
    url = args.channel_url or os.environ.get("CHANNEL_URL", "")
    if not url:
        raise SystemExit(
            "Error: --channel-url is required (or set CHANNEL_URL environment variable)."
        )
    return url


def _process_videos(
    videos: list[dict[str, Any]],
    args: argparse.Namespace,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Fetch metadata for each video and write the audit CSV incrementally.

    Failed metadata fetches are recorded in the audit CSV with
    ``status="metadata_failed"`` and tracked in the checkpoint so they are not
    retried indefinitely on every ``--resume`` run.

    CSV writes are append-only after the first header write to avoid O(n²)
    rewrites for large channels.
    """
    checkpoint_path = Path(_CHECKPOINT_FILE)
    checkpoint = _load_checkpoint(checkpoint_path) if args.resume else {"completed_ids": set(), "failed_ids": set()}
    completed_ids: set[str] = checkpoint["completed_ids"]
    failed_ids: set[str] = checkpoint["failed_ids"]
    processed_ids = completed_ids | failed_ids

    if not args.resume and output_path.exists():
        output_path.unlink()

    rows: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        rows = load_audit_csv(output_path)

    csv_initialized = args.resume and output_path.exists()

    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ARG001
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _handle_sigint)

    def _write_audit_row(row: dict[str, Any]) -> None:
        nonlocal csv_initialized
        if csv_initialized:
            append_audit_csv_row(row, output_path)
        else:
            write_audit_csv(rows, output_path)
            csv_initialized = True

    try:
        with tqdm(
            total=len(videos), initial=len(processed_ids), desc="Auditing videos"
        ) as pbar:
            for idx, video in enumerate(videos):
                if args.limit and idx >= args.limit:
                    break

                video_id = video.get("id")
                if not video_id or video_id in processed_ids:
                    pbar.update(1)
                    continue

                if interrupted:
                    break

                metadata = fetch_video_metadata(video.get("url", ""))
                if metadata is None:
                    if args.verbose:
                        print(f"Warning: failed to fetch metadata for {video_id}")
                    row = {
                        "video_id": video_id,
                        "title": "",
                        "published_at": "",
                        "video_url": _video_url(video_id),
                        "has_description": False,
                        "description_length": 0,
                        "status": "metadata_failed",
                    }
                    rows.append(row)
                    _write_audit_row(row)
                    failed_ids.add(video_id)
                    _save_checkpoint(checkpoint_path, completed_ids, failed_ids)
                    pbar.update(1)
                    continue

                has_desc, desc_len = evaluate_description(
                    metadata.get("description"), args.description_threshold
                )
                row = {
                    "video_id": metadata.get("id"),
                    "title": metadata.get("title"),
                    "published_at": metadata.get("published_at"),
                    "video_url": metadata.get("webpage_url"),
                    "has_description": has_desc,
                    "description_length": desc_len,
                    "status": "ok",
                }
                rows.append(row)
                _write_audit_row(row)
                completed_ids.add(video_id)
                _save_checkpoint(checkpoint_path, completed_ids, failed_ids)
                pbar.update(1)

                if idx < len(videos) - 1:
                    jittered_sleep(args.sleep, args.sleep_jitter)
                if (
                    args.batch_size > 0
                    and (idx + 1) % args.batch_size == 0
                    and idx < len(videos) - 1
                ):
                    tqdm.write(f"Batch complete; resting {args.batch_rest}s...")
                    time.sleep(args.batch_rest)
    finally:
        signal.signal(signal.SIGINT, original_handler)

    return rows


def main() -> None:
    args = _parse_args()
    channel_url = _get_channel_url(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting video list from: {channel_url}")
    videos = extract_flat_channel_videos(channel_url)
    print(f"Found {len(videos)} videos")

    if not videos:
        raise SystemExit("No videos found. Check the channel URL.")

    rows = _process_videos(videos, args, output_path)
    print(f"Audit complete: {len(rows)} rows written to {output_path}")


if __name__ == "__main__":
    main()
