# Video Descript Tool

Keyless YouTube channel auditor that finds videos with missing descriptions, fetches transcripts, and generates a stakeholder-review CSV + HTML report.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env
# Add your LLM Provider API key (e.g. OpenCode Zen, Anthropic, OpenAI) to .env

# 1. Audit the channel
python indexer.py --channel-url "https://www.youtube.com/@<handle>/videos"

# 2. Generate suggested descriptions
python generator.py --input output/channel_video_audit.csv
```

## Regenerating descriptions

`output/llm_cache.json` stores both transcripts and the generated descriptions. If you want to regenerate descriptions with a different model or `CHANNEL_CONTEXT` — without re-fetching transcripts from YouTube — use `--regenerate`:

```bash
# Regenerate all descriptions using cached transcripts
python generator.py --input output/channel_video_audit.csv --regenerate

# Regenerate with a different model
python generator.py --input output/channel_video_audit.csv --regenerate --model gpt-5.4

# Regenerate with a different channel context (CLI overrides .env)
python generator.py --input output/channel_video_audit.csv --regenerate --channel-context "a healthcare webinar channel"
```

Outputs (all written to the `output/` folder):
- `output/channel_video_audit.csv` — every video with metadata/description status, including a `status` column (`ok` or `metadata_failed`) (Output 1)
- `output/review_report.csv` and `output/review_report.html` — missing-description videos with transcripts and suggested copy (Output 2)
- `output/llm_cache.json` and `output/indexer_checkpoint.json` — resume/cache files

## Requirements
- Python 3.10+
- OpenCode Zen API Key (for description generation only)
- Note: You can use other LLM API keys but you will need to adjust the .env to reference only specific models offered by your provider.
- No YouTube API key, Google login, or OAuth needed

## CLI Options

### `indexer.py`
| Flag | Default | Description |
|---|---|---|
| `--channel-url` | required | Channel handle URL (e.g. `https://www.youtube.com/@<handle>/videos`); may also be set via `CHANNEL_URL` env var |
| `--output` | `output/channel_video_audit.csv` | Audit CSV path |
| `--sleep` | `3` | Base delay between per-video metadata requests (seconds) |
| `--sleep-jitter` | `3` | Max jitter added to sleep (0 = none) |
| `--batch-size` | `0` | Number of videos per batch; 0 disables batching |
| `--batch-rest` | `300` | Seconds to rest between batches |
| `--limit` | `0` | Stop after N videos (0 = all) |
| `--description-threshold` | `30` | Minimum non-whitespace characters to count as a description |
| `--resume` | `False` | Resume from `output/indexer_checkpoint.json` |

### `generator.py`
| Flag | Default | Description |
|---|---|---|
| `--input` | `output/channel_video_audit.csv` | Audit CSV path |
| `--output-csv` | `output/review_report.csv` | Review CSV output |
| `--output-html` | `output/review_report.html` | Review HTML output |
| `--model` | from `LLM_MODEL` env | Model name passed to OpenCode Zen |
| `--channel-context` | from `CHANNEL_CONTEXT` env | Optional free-form channel context (audience, tone, priorities) injected into the LLM prompt; generic, domain-agnostic prompt if unset |
| `--regenerate` | `False` | Regenerate descriptions even if already cached (cached transcripts are reused) |
| `--limit` | `0` | Process N missing-description videos only (0 = all) |
| `--transcripts-only` | `False` | Fetch transcripts only; do not call LLM |
| `--audio-fallback` | `False` | Use local Whisper when captions are unavailable or your IP is blocked |
| `--skip-captions` | `False` | Bypass the caption API entirely (implies `--audio-fallback`); use when your IP is known to be blocked |
| `--whisper-model` | `small` | Whisper model size (tiny/base/small/medium) |
| `--cookies` | (none) | Path to a Netscape `cookies.txt` for audio downloads; passes YouTube's anti-bot check when your IP is rate-limited |
| `--max-consecutive-blocks` | `5` | Stop requesting captions after N consecutive IP-blocked responses (0 disables) |
| `--max-consecutive-audio-failures` | `5` | Cool down after N consecutive audio failures (0 disables) |
| `--audio-failure-cooldown` | `180` | Seconds to wait once the audio-failure threshold is hit |
| `--transcript-max-chars` | `20000` | Max transcript characters sent to the LLM (sampled head+tail) |
| `--cache` | `output/llm_cache.json` | Cache file for transcripts and LLM output |
| `--batch-size` | `0` | Transcript batch size; 0 disables batching |
| `--batch-rest` | `300` | Seconds to rest between transcript batches |

## Troubleshooting

### `IpBlocked` / `RequestBlocked` when fetching transcripts
YouTube rate-limits the caption endpoint per IP. After many requests you will see
`IpBlocked: Could not retrieve a transcript ... YouTube is blocking requests from your IP`.

This is a caller-wide condition, not a per-video one, so it is never retried: retrying
only deepens the block. Such videos are recorded with `transcript_status=blocked`, and
after `--max-consecutive-blocks` consecutive blocks the run either switches to audio-only
transcription (with `--audio-fallback`) or stops early with progress saved.

Options, in order of practicality:

1. **Transcribe the audio instead.** Audio is served from a different endpoint and
   generally keeps working while captions are blocked. If you already know the caption
   endpoint is blocked, skip it entirely (saves a wasted, timed-out caption attempt on
   every video):
   ```bash
   python generator.py --input output/channel_video_audit.csv --skip-captions
   ```
   This needs no API key or proxy. ffmpeg is *not* required — audio is kept in its
   native container and decoded directly by faster-whisper.

   The audio endpoint is only *intermittently* rate-limited (occasional HTTP 403,
   which recovers in seconds), unlike the hard per-IP caption block. The built-in
   audio-failure cooldown backs off when this happens, and failed videos are retried
   automatically on the next run. To fetch transcripts for a whole channel unattended
   — re-attempting transiently failed videos until none remain — use:
   ```bash
   ./run_audio_transcripts.sh
   ```
   It loops `generator.py --transcripts-only --skip-captions` until every
   missing-description video has a cached transcript. Ctrl+C is safe; progress is
   saved after every video. Run it again any time to pick up stragglers.
2. **Wait and resume.** Blocks are usually temporary. Progress is saved after every
   video, so simply rerunning later resumes where it stopped; cached transcripts are
   reused and never re-fetched.
3. **Use a proxy.** See the
   [Working around IP bans](https://github.com/jdepoix/youtube-transcript-api?tab=readme-ov-file#working-around-ip-bans-requestblocked-or-ipblocked-exception)
   section of the `youtube-transcript-api` README. Note that datacenter/cloud IPs are
   largely pre-blocked by YouTube; residential proxies are needed in practice.

### `Sign in to confirm you're not a bot` (audio downloads)
After sustained downloading, YouTube may escalate from transient `403`s to a full
anti-bot wall: every yt-dlp request — metadata *and* audio — returns
`Sign in to confirm you're not a bot`, even for videos that worked minutes earlier.
This is a stronger, per-IP block than the intermittent 403s.

What works and what doesn't:

- **Waiting is the only cost-free fix.** These blocks are temporary; resume later with
  `./run_audio_transcripts.sh` and it picks up the stragglers from the cache.
- **Pass a `cookies.txt` file to keep going now.** `--cookies-from-browser` is
  unreliable on macOS because Chrome's cookies are encrypted (v10) and cannot be
  decrypted without Keychain access. Instead export cookies to a file:
  1. In Chrome, install a "Get cookies.txt LOCALLY" extension.
  2. Open `https://www.youtube.com` while signed in, export `cookies.txt`.
  3. Run with the file (treat it as a secret — do not commit it):
     ```bash
     python generator.py --input output/channel_video_audit.csv \
       --transcripts-only --skip-captions --cookies /path/to/cookies.txt
     ```
  4. Or set it in `.env` for the loop script: `YT_COOKIES=/path/to/cookies.txt`.
- **Slower pacing reduces recurrence.** Increase `--sleep`/`--sleep-jitter` and the
  audio-failure cooldown to lower the request rate.

### Video IDs that start with `-` failed with `audio download failed`
Older indexer versions applied a spreadsheet formula guard to the `video_id` column,
prepending an apostrophe to IDs starting with `-` (e.g. `'-acRraKkZfU`). Tool 2 then
built a download URL for the wrong ID. This is fixed: the indexer no longer alters
`video_id`, and the generator strips stray quotes defensively. If your audit CSV was
produced by an affected version, repair it once by stripping leading `'`/`"` from the
`video_id` column and deleting the corresponding poisoned cache keys.

Interrupting a run with `Ctrl+C` is always safe — the cache and both reports are written
after each video.

### Whisper model downloads
The first `--audio-fallback` run downloads the Whisper weights from Hugging Face and
prints `You are sending unauthenticated requests to the HF Hub`. This warning is
harmless and unrelated to YouTube; the weights are cached locally afterwards. Set a
`HF_TOKEN` (or `HF_HUB_DISABLE_WARNINGS=1`) if you want to silence it.

## Development
```bash
ruff check .
mypy src
pytest -q
```

## Notes
- The channel URL is never hardcoded in the source; pass it as a CLI argument or via the `CHANNEL_URL` environment variable.
- `.env` is gitignored by default. Never commit real API keys or URLs.
- `output/llm_cache.json` caches transcripts and generated descriptions per video ID. If you change `--model`, `--channel-context`, or `CHANNEL_CONTEXT`, rerun with `--regenerate` to regenerate descriptions while reusing the cached transcripts — no need to delete the cache file or re-fetch transcripts (see the **Regenerating descriptions** section above).
- Transcripts longer than `--transcript-max-chars` (default **20,000 characters**) are sampled before being sent to the LLM: roughly the first 60% of the budget from the start and the remainder from the end, joined by an explicit omission marker. Sampling both ends keeps the introduction *and* the closing takeaways in view, instead of discarding everything after a hard cut. The full transcript is still stored in `output/llm_cache.json` and `output/review_report.csv`; only the LLM input is limited. The HTML report shows a note when truncation occurred.
- If the primary LLM model fails (e.g. a provider outage), each model in the comma-separated `LLM_MODEL_FALLBACKS` env var is tried in order; the run only aborts if all candidates fail.
- `output/channel_video_audit.csv` includes a `status` column. Rows with `status=metadata_failed` are videos whose metadata could not be fetched; they are tracked in the checkpoint so they are not retried forever on `--resume`.
