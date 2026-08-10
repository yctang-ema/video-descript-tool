#!/usr/bin/env bash
#
# Fetch transcripts for every missing-description video via the local Whisper
# audio fallback, looping until the run reports "no more videos".
#
# Each generator.py invocation only processes the videos not already present in
# the cache, so re-running is safe and resumes automatically. The loop exists
# to (a) re-attempt videos whose audio download was transiently rate-limited
# (HTTP 403) and (b) stop cleanly once there is genuinely nothing left to do.
#
# Usage:
#   ./run_audio_transcripts.sh
#
# Ctrl+C is safe: progress is saved to output/llm_cache.json after every video.

set -u
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON not found; activate or create the virtualenv first." >&2
    exit 1
fi

INPUT="output/channel_video_audit.csv"
CACHE="output/llm_cache.json"
PASS=0
MAX_PASSES=200          # safety valve against an infinite loop
SLEEP_BETWEEN_PASSES=30 # brief pause between full passes over the remainder

while [ "$PASS" -lt "$MAX_PASSES" ]; do
    PASS=$((PASS + 1))

    # How many missing-description videos still lack a usable transcript?
    REMAINING=$("$PYTHON" - "$INPUT" "$CACHE" <<'PY'
import csv, json, sys
csv.field_size_limit(sys.maxsize)
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
missing = [
    r for r in rows
    if (r.get("status") or "ok") == "ok"
    and (r.get("has_description") or "").lower() not in ("true", "1", "yes")
]
try:
    cache = json.load(open(sys.argv[2], encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    cache = {}
pending = [
    r["video_id"] for r in missing
    if not (cache.get(r["video_id"], {}).get("transcript")
            and cache.get(r["video_id"], {}).get("transcript_status"))
]
print(len(pending))
PY
)

    echo "=== pass $PASS: $REMAINING videos still need a transcript ==="
    if [ "$REMAINING" -eq 0 ]; then
        echo "All done: every missing-description video has a transcript."
        break
    fi

    # One pass over the remaining videos. Audio-only (--skip-captions bypasses
    # the blocked caption endpoint); the built-in cooldown backs off when the
    # audio endpoint starts rate-limiting. We do not stop on a non-zero exit so
    # a single bad video does not abort the whole loop.
    "$PYTHON" generator.py \
        --input "$INPUT" \
        --transcripts-only \
        --skip-captions \
        --whisper-model small \
        --max-consecutive-audio-failures 5 \
        --audio-failure-cooldown 180 \
        --sleep 0.3 --sleep-jitter 0.3

    echo "=== pass $PASS complete; pausing ${SLEEP_BETWEEN_PASSES}s before recheck ==="
    sleep "$SLEEP_BETWEEN_PASSES"
done

if [ "$REMAINING" -ne 0 ]; then
    echo "Stopped after $MAX_PASSES passes with $REMAINING videos still pending." >&2
    exit 1
fi
