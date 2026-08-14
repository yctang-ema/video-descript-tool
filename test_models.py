"""Test the shortlisted models with a real transcript + description prompt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from openai import OpenAI

from src.llm import (
    TRANSCRIPT_MAX_CHARS,
    _system_prompt,
    build_client,
    sample_transcript,
)

# Shortlisted models: token-efficient, accurate, non-Claude, confirmed working.
MODELS = [
    "grok-4.6",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k3",
    "deepseek-v4-pro",
]


def _load_sample_transcript(cache_path: Path) -> tuple[str, str]:
    """Pull one cached transcript. Returns (video_id, transcript)."""
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            cache = json.load(f)
        for vid, data in cache.items():
            t = data.get("transcript", "")
            if t and len(t) > 200:
                return vid, t
    # Fallback dummy if cache is empty or missing.
    return "demo", (
        "Welcome to SIEW News. This is a demo transcript for testing purposes."
    )


def _test_model(client: OpenAI, model: str, title: str, transcript: str) -> None:
    """Send one real description prompt and print results."""
    user_msg = (
        f"Video title: {title}\n\nTranscript:\n{sample_transcript(transcript, TRANSCRIPT_MAX_CHARS)}"
        "\n\nWrite a YouTube video description with a hook, 3-5 bullet points, and 3-5 hashtags."
    )

    print(f"\n{'=' * 60}")
    print(f"MODEL: {model}")
    print(f"{'=' * 60}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _system_prompt("SIEW News covers Singapore energy policy, market updates, and industry events.")},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return

    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    if usage:
        print(f"Tokens -> prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, total: {usage.total_tokens}")
    else:
        print("Tokens -> usage data not provided by gateway")
    print(f"Word count: {len(content.split())}")
    print(f"Output preview:\n{content[:800]}{'...' if len(content) > 800 else ''}")


if __name__ == "__main__":
    cache_path = Path("output/llm_cache.json")
    video_id, transcript = _load_sample_transcript(cache_path)
    title = f"SIEW News Energy Update ({video_id})"

    print(f"Using transcript from video: {video_id}")
    print(f"Transcript length: {len(transcript)} chars (sampled to ~{TRANSCRIPT_MAX_CHARS} for prompt)")

    client = build_client()
    for model in MODELS:
        _test_model(client, model, title, transcript)

    print("\n" + "=" * 60)
    print("Done. Pick the model whose output style, length, and token count you prefer.")
    print("Then update your .env LLM_MODEL and LLM_MODEL_FALLBACKS accordingly.")
