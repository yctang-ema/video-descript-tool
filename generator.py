"""Transcript fetcher and description generator (Tool 2)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from src.channel import jittered_sleep, load_audit_csv
from src.llm import (
    TRANSCRIPT_MAX_CHARS,
    generate_description,
    load_cache,
    resolve_model,
    save_cache,
)
from src.report import (
    build_combined_results,
    generate_review_html,
    write_review_csv,
)
from src.transcripts import (
    TRANSCRIPT_STATUS_AUDIO_FAILED,
    TRANSCRIPT_STATUS_BLOCKED,
    TRANSCRIPT_STATUS_DISABLED,
    TRANSCRIPT_STATUS_NO_CAPTIONS,
    cleanup_temp_audio,
    fetch_transcript,
    transcribe_audio,
)

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch transcripts and generate YouTube video descriptions."
    )
    parser.add_argument(
        "--input",
        default="output/channel_video_audit.csv",
        help="Audit CSV produced by indexer.py",
    )
    parser.add_argument(
        "--output-csv",
        default="output/review_report.csv",
        help="Review CSV output path",
    )
    parser.add_argument(
        "--output-html",
        default="output/review_report.html",
        help="Review HTML output path",
    )
    parser.add_argument(
        "--combined-csv",
        default="output/combined_review_report.csv",
        help="Combined CSV (all cached videos with a transcript) output path",
    )
    parser.add_argument(
        "--combined-html",
        default="output/combined_review_report.html",
        help="Combined HTML (all cached videos with a transcript) output path",
    )
    parser.add_argument(
        "--model",
        default="",
        help="LLM model name (overrides LLM_MODEL env var)",
    )
    parser.add_argument(
        "--channel-context",
        default="",
        help="Short channel/topic description injected into the LLM prompt "
        "(overrides CHANNEL_CONTEXT env var); leave empty for a generic prompt",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate descriptions even if already cached "
        "(cached transcripts are still reused)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only N missing-description videos; 0 = all",
    )
    parser.add_argument(
        "--transcripts-only",
        action="store_true",
        help="Fetch transcripts only; do not call the LLM",
    )
    parser.add_argument(
        "--audio-fallback",
        action="store_true",
        help="Use local Whisper for videos with no captions",
    )
    parser.add_argument(
        "--skip-captions",
        action="store_true",
        help="Bypass the YouTube caption API entirely (implies --audio-fallback); "
        "use when your IP is known to be blocked",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="Whisper model size (tiny/base/small/medium)",
    )
    parser.add_argument(
        "--cookies",
        default="",
        help="Path to a Netscape cookies.txt file (exported from a browser) used "
        "for audio downloads, to pass YouTube's 'sign in to confirm you're not a "
        "bot' check when your IP is rate-limited",
    )
    parser.add_argument(
        "--max-consecutive-blocks",
        type=int,
        default=5,
        help="Stop requesting captions after N consecutive IP-blocked responses; "
        "0 disables the check",
    )
    parser.add_argument(
        "--transcript-max-chars",
        type=int,
        default=TRANSCRIPT_MAX_CHARS,
        help="Max transcript characters sent to the LLM (sampled head+tail)",
    )
    parser.add_argument(
        "--max-consecutive-audio-failures",
        type=int,
        default=5,
        help="Cool down briefly after N consecutive audio download/transcription "
        "failures (sign of audio-endpoint rate limiting); 0 disables the check",
    )
    parser.add_argument(
        "--audio-failure-cooldown",
        type=int,
        default=180,
        help="Seconds to wait once the audio-failure threshold is hit",
    )
    parser.add_argument(
        "--cache",
        default="output/llm_cache.json",
        help="Cache file path for transcripts and LLM output",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Base delay between transcript requests (seconds)",
    )
    parser.add_argument(
        "--sleep-jitter",
        type=float,
        default=1.0,
        help="Max random jitter added to sleep (seconds)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Transcripts per batch; 0 disables batching",
    )
    parser.add_argument(
        "--batch-rest",
        type=int,
        default=300,
        help="Seconds to rest between transcript batches",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep temporary audio files instead of deleting them",
    )
    return parser.parse_args()


def _load_whisper_model(model_size: str) -> Any:
    """Load a faster-whisper model once for reuse."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


AUDIO_FALLBACK_STATUSES = frozenset(
    {
        TRANSCRIPT_STATUS_DISABLED,
        TRANSCRIPT_STATUS_NO_CAPTIONS,
        TRANSCRIPT_STATUS_BLOCKED,
    }
)


def _get_transcript(
    video_id: str,
    cache: dict[str, Any],
    args: argparse.Namespace,
    whisper_model: Any | None,
    skip_captions: bool = False,
) -> tuple[str | None, str, bool]:
    """Return cached or freshly fetched transcript, status, and block flag.

    When ``skip_captions`` is True the caption API is bypassed entirely (the
    caller has detected an IP block) and the audio fallback is used directly.

    The third element reports whether the *caption* request was blocked. It is
    tracked separately from ``status`` because a successful audio fallback
    overwrites the status, which would otherwise hide an ongoing IP block from
    the caller's circuit breaker.
    """
    cached = cache.get(video_id, {})
    if cached.get("transcript") and cached.get("transcript_status"):
        return cached["transcript"], cached["transcript_status"], False

    if skip_captions:
        transcript, status = None, TRANSCRIPT_STATUS_BLOCKED
    else:
        transcript, status = fetch_transcript(video_id)
    captions_blocked = status == TRANSCRIPT_STATUS_BLOCKED

    audio_fallback = args.audio_fallback or getattr(args, "skip_captions", False)
    if status in AUDIO_FALLBACK_STATUSES and audio_fallback:
        if whisper_model is None:
            whisper_model = _load_whisper_model(args.whisper_model)
        cookies = getattr(args, "cookies", None) or None
        transcript, status = transcribe_audio(video_id, whisper_model, cookies=cookies)
        if not args.keep_audio:
            cleanup_temp_audio(video_id)

    return transcript, status, captions_blocked


def _generate_suggested_description(
    video_id: str,
    title: str,
    transcript: str,
    cache: dict[str, Any],
    model: str | None,
    channel_context: str | None,
    regenerate: bool = False,
    max_chars: int = TRANSCRIPT_MAX_CHARS,
) -> str:
    """Return cached or freshly generated description.

    When ``regenerate`` is True, any cached description is ignored and a new
    one is generated (cached transcripts are unaffected).
    """
    cached = cache.get(video_id, {})
    if not regenerate and cached.get("suggested_description"):
        return cached["suggested_description"]

    suggested = generate_description(
        title,
        transcript,
        model=model,
        channel_context=channel_context,
        max_chars=max_chars,
    )
    cache.setdefault(video_id, {})
    cache[video_id]["suggested_description"] = suggested
    return suggested


def _process_videos(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    cache: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fetch transcripts and generate descriptions for missing-description videos.

    Rows with a non-OK status (e.g. metadata_failed) are skipped because their
    metadata could not be retrieved.
    """
    missing = [
        row
        for row in rows
        if row.get("status", "ok") == "ok" and not row.get("has_description", False)
    ]
    if args.limit:
        # --limit caps how many *new* (not-yet-cached) videos to process.
        # Counting only uncached rows matters: without it, a canary run such as
        # ``--limit 5`` would just replay the first five already-cached videos,
        # perform zero new downloads, and misleadingly shrink the review report
        # to those five rows.
        uncached = [
            row
            for row in missing
            if not (
                cache.get(row.get("video_id", "").strip().strip("'\""), {}).get(
                    "transcript"
                )
                and cache.get(
                    row.get("video_id", "").strip().strip("'\""), {}
                ).get("transcript_status")
            )
        ]
        fresh = set()
        for row in uncached[: args.limit]:
            fresh.add(row.get("video_id", "").strip().strip("'\""))
        missing = [
            row
            for row in missing
            if row.get("video_id", "").strip().strip("'\"") in fresh
        ]

    # --skip-captions implies audio-only mode.
    skip_captions = getattr(args, "skip_captions", False)
    audio_fallback = args.audio_fallback or skip_captions

    # Resolve the cookies path once and normalise it back onto args so
    # _get_transcript sees a validated Path (or None if the file is missing).
    cookies = Path(args.cookies) if getattr(args, "cookies", "") else None
    if cookies is not None and not cookies.exists():
        print(f"Warning: --cookies file not found: {cookies}", file=sys.stderr)
        cookies = None
    args.cookies = cookies

    whisper_model = None
    if audio_fallback:
        whisper_model = _load_whisper_model(args.whisper_model)

    results: list[dict[str, Any]] = []
    consecutive_blocks = 0
    consecutive_audio_failures = 0
    for idx, row in enumerate(tqdm(missing, desc="Processing videos")):
        # Strip stray quotes/whitespace: the audit CSV may carry a leading
        # apostrophe on IDs that start with "-" (a spreadsheet formula guard
        # applied by older indexer versions), which would otherwise corrupt
        # the download URL.
        video_id = row.get("video_id", "").strip().strip("'\"")
        title = row.get("title", "")
        video_url = row.get("video_url", "")
        published_at = row.get("published_at", "")

        cache.setdefault(video_id, {})
        transcript, status, captions_blocked = _get_transcript(
            video_id, cache, args, whisper_model, skip_captions=skip_captions
        )
        cache[video_id]["transcript"] = transcript
        cache[video_id]["transcript_status"] = status
        truncated = bool(
            transcript and len(transcript) > args.transcript_max_chars
        )
        cache[video_id]["transcript_truncated"] = truncated

        suggested = ""
        if not args.transcripts_only and transcript:
            suggested = _generate_suggested_description(
                video_id,
                title,
                transcript,
                cache,
                resolve_model(args.model or None),
                args.channel_context or None,
                regenerate=args.regenerate,
                max_chars=args.transcript_max_chars,
            )

        results.append(
            {
                "video_id": video_id,
                "title": title,
                "video_url": video_url,
                "published_at": published_at,
                "transcript_status": status,
                "transcript": transcript or "",
                "suggested_description": suggested,
                "transcript_truncated": truncated,
            }
        )
        save_cache(cache, args.cache)
        write_review_csv(results, args.output_csv)
        generate_review_html(results, args.output_html)

        if captions_blocked:
            consecutive_blocks += 1
        else:
            consecutive_blocks = 0

        if status == TRANSCRIPT_STATUS_AUDIO_FAILED:
            consecutive_audio_failures += 1
        else:
            consecutive_audio_failures = 0

        if (
            args.max_consecutive_audio_failures > 0
            and consecutive_audio_failures >= args.max_consecutive_audio_failures
        ):
            tqdm.write(
                f"{consecutive_audio_failures} audio downloads failed in a row "
                f"(audio endpoint may be rate-limiting); cooling down "
                f"{args.audio_failure_cooldown}s then continuing."
            )
            time.sleep(args.audio_failure_cooldown)
            consecutive_audio_failures = 0

        if (
            args.max_consecutive_blocks > 0
            and not skip_captions
            and consecutive_blocks >= args.max_consecutive_blocks
        ):
            if audio_fallback:
                tqdm.write(
                    f"YouTube blocked {consecutive_blocks} caption requests in a row; "
                    "skipping the caption API and using audio transcription only."
                )
                skip_captions = True
            else:
                tqdm.write(
                    f"Stopping: YouTube blocked {consecutive_blocks} caption requests "
                    "in a row (your IP is rate-limited). Progress is saved; rerun "
                    "later to resume, or use --audio-fallback to transcribe audio "
                    "instead of captions."
                )
                break

        if idx < len(missing) - 1:
            jittered_sleep(args.sleep, args.sleep_jitter)
        if (
            args.batch_size > 0
            and (idx + 1) % args.batch_size == 0
            and idx < len(missing) - 1
        ):
            tqdm.write(f"Batch complete; resting {args.batch_rest}s...")
            time.sleep(args.batch_rest)

    return results


def main() -> None:
    args = _parse_args()
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)

    rows = load_audit_csv(args.input)
    print(f"Loaded {len(rows)} audit rows; {sum(1 for r in rows if not r.get('has_description', False))} missing descriptions")

    results = _process_videos(rows, args, cache)
    save_cache(cache, cache_path)

    output_csv = Path(args.output_csv)
    output_html = Path(args.output_html)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    write_review_csv(results, output_csv)
    generate_review_html(results, output_html)
    print(f"Review reports written: {output_csv}, {output_html}")

    # Combined report across every cached video with a transcript. Unlike the
    # per-run report above (which reflects only this run), this always shows the
    # full set of work completed so far, so a small --limit run never looks like
    # it "wiped" earlier progress.
    combined = build_combined_results(rows, cache)
    combined_csv = Path(args.combined_csv)
    combined_html = Path(args.combined_html)
    write_review_csv(combined, combined_csv)
    generate_review_html(combined, combined_html)
    print(
        f"Combined reports written ({len(combined)} videos with transcripts): "
        f"{combined_csv}, {combined_html}"
    )


if __name__ == "__main__":
    main()
