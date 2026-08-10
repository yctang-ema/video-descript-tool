# Video Descript Tool — Agent Context

## Background
This tool audits a YouTube channel for videos that are missing useful descriptions, fetches transcripts, and generates a review report (CSV + HTML) so a business user can approve descriptions before they are published.

## Architecture
- **Keyless:** No YouTube API key, Google login, or OAuth is required for Tools 1 & 2.
- **Tool 1 (`indexer.py`):** Uses `yt-dlp` to enumerate a channel and fetch each video's metadata to determine whether the description is missing or too short.
- **Tool 2 (`generator.py`):** Fetches transcripts with `youtube-transcript-api`, optionally falls back to local Whisper audio transcription, and generates descriptions via the OpenCode Zen LLM endpoint.
- **Resilience:** network calls in `src/channel.py` and `src/transcripts.py` use `src/retry.py` for exponential-backoff retries, including 429 detection; hung yt-dlp calls are bounded by a `socket_timeout`. YouTube IP blocks (`RequestBlocked`/`IpBlocked`) are reported as the `blocked` transcript status and deliberately *not* retried, since they are caller-wide; `generator.py` counts consecutive blocks and either switches to audio-only transcription (`--audio-fallback`) or stops early with progress saved.
- **Audio fallback:** downloaded audio is kept in its native container (`.m4a` etc.) and fed straight to faster-whisper, which decodes it via PyAV. Do not reintroduce mp3 conversion — that would add a hard ffmpeg dependency.
- **Outputs:** `output/channel_video_audit.csv` (Output 1) and `output/review_report.csv` / `output/review_report.html` (Output 2). Generated files are never committed.
- **Caching:** `output/llm_cache.json` is keyed by video ID and stores transcripts plus generated descriptions. `generator.py --regenerate` regenerates descriptions while reusing cached transcripts (needed after changing the model or `CHANNEL_CONTEXT`).

## Tech Stack
- Python 3.10+
- `yt-dlp` (channel & metadata extraction)
- `youtube-transcript-api` (caption retrieval)
- `faster-whisper` (optional local audio fallback)
- `openai` SDK (OpenCode Zen compatible endpoint)
- `tqdm` (progress bars)
- `python-dotenv` (env loading)

## Project Structure
```
video-descript-tool/
├── indexer.py              # CLI entry point: Tool 1
├── generator.py            # CLI entry point: Tool 2
├── run_audio_transcripts.sh  # Loops Tool 2 (--transcripts-only --skip-captions) until all audio transcripts are cached
├── src/
│   ├── __init__.py
│   ├── channel.py          # yt-dlp extraction + audit CSV helpers
│   ├── transcripts.py      # transcript fetching + Whisper fallback
│   ├── llm.py              # OpenCode Zen client + cache + prompt
│   ├── report.py           # CSV + HTML review report generation
│   └── retry.py            # retry/backoff helpers for network calls
├── tests/                  # pytest suite, all network/LLM mocked
├── output/                 # Generated CSV/HTML/cache files (gitignored)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml          # ruff + mypy config
├── .env.example
├── .gitignore
├── AGENTS.md
├── plan.md
└── README.md
```

## Coding Conventions
- Follow PEP8 and the ruff rules in `pyproject.toml`.
- Type hints on public function signatures.
- Docstrings for modules and public functions.
- Keep runtime dependencies minimal.
- Handle network errors gracefully with retries and resume support.
- Respect server load: jittered polite delays, sequential requests, no concurrency.
- **Sanitisation rule:** no channel URLs, organisation names, or API keys are hardcoded in any committed file. Pass them via CLI arguments or environment variables. The same applies to channel/topic framing: the LLM prompt in `src/llm.py` is domain-agnostic by default; tailor it at runtime via `--channel-context` or the `CHANNEL_CONTEXT` env var, never by hardcoding.

## Build & Run
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt -r requirements-dev.txt`
3. Copy `.env.example` to `.env` and add your OpenCode Zen key.
4. Tool 1: `python indexer.py --channel-url "https://www.youtube.com/@<handle>/videos"`
5. Tool 2: `python generator.py --input output/channel_video_audit.csv`

## Quality Gates
- `ruff check .`
- `mypy src`
- `pytest -q`
