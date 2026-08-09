# Plan — Video Descript Tool

## Goal
Audit all videos on a YouTube channel, identify those with missing or very short descriptions, fetch transcripts, and generate a stakeholder-ready CSV/HTML review report with suggested descriptions.

## Design Decisions (Final)
- **YouTube access:** keyless, API-endpoint-based (`yt-dlp`, `youtube-transcript-api`), no raw HTML scraping or browser automation.
- **Anti-bot posture:** local residential IP, jittered polite delays, adaptive exponential backoff on 429, checkpoint/resume.
- **Naming:** `indexer.py` (Tool 1) and `generator.py` (Tool 2).
- **LLM:** OpenCode Zen endpoint (`https://opencode.ai/zen/v1`), `OPENCODE_API_KEY` from `.env`, default model `gpt-5.4-mini`, switchable to `gpt-5.4` via `LLM_MODEL` or `--model`, with automatic failover via comma-separated `LLM_MODEL_FALLBACKS`.
- **Transcripts:** `youtube-transcript-api` first; optional `--audio-fallback` using local `faster-whisper` if no captions exist.
- **Batching:** not mandatory; optional `--batch-size` / `--batch-rest` flags default to disabled.
- **Quality:** ruff + mypy + pytest; no hardcoded secrets or org names in committed files.

## Implementation Status
- [x] Rename project folder to `video-descript-tool`
- [x] Scaffold project structure and config files
- [x] Implement `src/` modules and CLI scripts
- [x] Add tests and fixtures
- [x] Set up venv and install dependencies
- [x] Run lint, type-check, and tests; verify `.env` is ignored
- [x] Test Tool 1 with `--limit 5` (waiting for channel URL / key)
- [x] Test Tool 2 with `--transcripts-only` and `--limit 2` (waiting for OpenCode Zen key)
- [x] Full channel run (Output 1)
- [x] Full generator run (Output 2)
- [x] Verify `.env` cannot be committed (`git check-ignore -v .env`)

## Build & Run Commands
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env
# edit .env with your OPENCODE_API_KEY

# Tool 1 — audit channel
python indexer.py --channel-url "https://www.youtube.com/@<handle>/videos"

# Tool 2 — fetch transcripts and generate descriptions
python generator.py --input output/channel_video_audit.csv

# Optional: test runs
python indexer.py --channel-url "https://www.youtube.com/@<handle>/videos" --limit 5
python generator.py --input output/channel_video_audit.csv --limit 2

# Optional: transcripts only (no LLM key needed)
python generator.py --input output/channel_video_audit.csv --transcripts-only
```
