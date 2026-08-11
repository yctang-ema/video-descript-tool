#!/usr/bin/env bash
#
# Self-diagnostic: check whether the transcription process is healthy, slow,
# or stuck. Run this from a second terminal while the main run is in progress.
#
# Usage:
#   ./check_status.sh
#
# It only reads state (cache, processes, file sizes) — it never modifies anything.

set -u

# --- find the generator process ---
PIDS=$(pgrep -f "generator.py" || true)
if [ -z "$PIDS" ]; then
    echo "Status: NO RUNNING PROCESS"
    echo "The generator is not running. If you expected it to be, it may have"
    echo "finished, crashed, or been killed."
    exit 0
fi

# Pick the main python process (not any child helper).
MAIN_PID=""
for p in $PIDS; do
    CPU=$(ps -o %cpu= -p "$p" 2>/dev/null | tr -d ' ')
    if [ -n "${CPU}" ] && [ "${CPU}" != "0.0" ]; then
        MAIN_PID="$p"
        break
    fi
done

# If none have CPU, take the first one (could be startup or a lull).
if [ -z "$MAIN_PID" ]; then
    MAIN_PID=$(echo "$PIDS" | awk '{print $1}')
fi

ELAPSED=$(ps -o etime= -p "$MAIN_PID" 2>/dev/null | tr -d ' ')
CPU_NOW=$(ps -o %cpu= -p "$MAIN_PID" 2>/dev/null | tr -d ' ')

# --- cache snapshot ---
CACHE="${1:-output/llm_cache.json}"
if [ -f "$CACHE" ]; then
    CACHE_ENTRIES=$(python3 -c "import json;print(len(json.load(open('$CACHE'))))" 2>/dev/null || echo "?")
    AUDIO_OK=$(python3 -c "
import json, sys
from collections import Counter
try:
    d = json.load(open('$CACHE'))
except Exception:
    sys.exit(0)
print(Counter(v.get('transcript_status') for v in d.values()).get('audio_transcribed', 0))
" 2>/dev/null || echo "?")
    AUDIO_FAIL=$(python3 -c "
import json, sys
from collections import Counter
try:
    d = json.load(open('$CACHE'))
except Exception:
    sys.exit(0)
print(Counter(v.get('transcript_status') for v in d.values()).get('audio_failed', 0))
" 2>/dev/null || echo "?")
else
    CACHE_ENTRIES="?"
    AUDIO_OK="?"
    AUDIO_FAIL="?"
fi

# --- temp_audio state ---
TEMP_DIR="temp_audio"
PART_FILE=""
if [ -d "$TEMP_DIR" ]; then
    PART_FILE=$(find "$TEMP_DIR" -maxdepth 1 -name "*.part" -print -quit 2>/dev/null || true)
fi

# --- verdict logic ---
echo "=========================================="
echo "  Process PID:     $MAIN_PID"
echo "  Running for:     ${ELAPSED:-?}"
echo "  CPU right now:   ${CPU_NOW:-?}%"
echo "  Cache entries:   $CACHE_ENTRIES"
echo "  Audio OK:        $AUDIO_OK"
echo "  Audio failed:    $AUDIO_FAIL"

if [ -n "$PART_FILE" ]; then
    echo "  Active download: $(basename "$PART_FILE")"
    SIZE1=$(stat -f%z "$PART_FILE" 2>/dev/null || echo 0)
    sleep 3
    SIZE2=$(stat -f%z "$PART_FILE" 2>/dev/null || echo 0)
    if [ "$SIZE1" != "$SIZE2" ]; then
        echo "  Download growth: YES (grew ${SIZE2} vs ${SIZE1})"
    else
        echo "  Download growth: NO — download may be stalled"
    fi
else
    echo "  Active download: none"
fi

echo ""
# Verdict
if [ "${CPU_NOW%.*}" -ge 50 ] 2>/dev/null; then
    echo "VERDICT: HEALTHY (high CPU = Whisper is actively transcribing)"
elif [ -n "$PART_FILE" ] && [ "$SIZE1" != "$SIZE2" ] 2>/dev/null; then
    echo "VERDICT: HEALTHY (download in progress)"
elif [ "${CPU_NOW%.*}" -lt 5 ] 2>/dev/null && [ -n "$PART_FILE" ]; then
    echo "VERDICT: DOWNLOAD STUCK (low CPU + .part file not growing)"
elif [ "${CPU_NOW%.*}" -lt 5 ] 2>/dev/null; then
    # Could be a brief model-loading pause or genuinely stuck.
    echo "VERDICT: PAUSED / SLOW (very low CPU, no download active)"
    echo "         If this persists for >5 minutes, the process may be hung."
else
    echo "VERDICT: UNCLEAR (mixed signals — check again in 30–60s)"
fi
echo "=========================================="

# Optional: if you run with ./check_status.sh --watch it loops every 30s.
if [ "${1:-}" = "--watch" ]; then
    echo ""
    echo "Watch mode: rechecking every 30s (Ctrl+C to stop)"
    sleep 30
    exec "$0" --watch
fi