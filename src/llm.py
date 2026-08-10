"""OpenCode Zen LLM client, cache, and prompt builder."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

# Maximum transcript characters sent to the LLM per video. Longer transcripts
# are truncated and a flag is surfaced in the review reports.
TRANSCRIPT_MAX_CHARS = 20000


def build_client() -> OpenAI:
    """Build an OpenAI-compatible client pointing at OpenCode Zen."""
    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENCODE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(
        base_url="https://opencode.ai/zen/v1",
        api_key=api_key,
    )


def resolve_model(model: str | None) -> str:
    """Return the model name, preferring the CLI argument, then env, then default."""
    return model or os.environ.get("LLM_MODEL", "gpt-5.4-mini")


# Model that last succeeded in this process; tried first on subsequent calls.
_last_working_model: str | None = None


def resolve_models(model: str | None) -> list[str]:
    """Return ordered candidate models for failover.

    Order: the last model that worked in this process (if any), then the
    primary model (CLI arg / ``LLM_MODEL`` / default), then each entry of the
    comma-separated ``LLM_MODEL_FALLBACKS`` env var. Duplicates are removed.
    """
    primary = resolve_model(model)
    fallbacks = [
        name.strip()
        for name in os.environ.get("LLM_MODEL_FALLBACKS", "").split(",")
        if name.strip()
    ]
    candidates = ([_last_working_model] if _last_working_model else []) + [
        primary,
        *fallbacks,
    ]
    unique: list[str] = []
    for name in candidates:
        if name not in unique:
            unique.append(name)
    return unique


def resolve_channel_context(context: str | None) -> str:
    """Return the channel context, preferring the CLI argument, then env var.

    Returns an empty string when neither is set, meaning the prompt stays
    fully domain-agnostic.
    """
    if context:
        return context
    return os.environ.get("CHANNEL_CONTEXT", "")


def _system_prompt(channel_context: str = "") -> str:
    """Return the system prompt, optionally tailored with channel context.

    The base prompt is deliberately domain-agnostic: the model adapts tone,
    terminology, and hashtags to whatever the title and transcript contain,
    rather than boxing the content into a fixed industry framing. When
    ``channel_context`` is provided, it is included as a free-form "Channel
    context" block — it may be a short label or a longer brief covering the
    channel's audience, tone, and priorities.
    """
    parts = [
        "You are a senior communications editor. "
        "You write concise, high-quality YouTube video descriptions."
    ]
    if channel_context:
        parts.append(f"Channel context:\n{channel_context.strip()}")
    parts.append(
        "Given the video title and transcript, produce a description with exactly three parts:\n\n"
        "1. A hook (1-2 sentences) that explains why the video matters to its audience.\n"
        "2. 3 to 5 bullet points summarising the key discussion points, insights, or takeaways.\n"
        "3. 3 to 5 hashtags relevant to the video's subject matter.\n\n"
        "Adapt the tone, terminology, and hashtags to the topic and style evident in the "
        "title and transcript; do not force the content into any particular industry framing. "
        "Use plain language, no jargon without explanation, and keep the total output under 200 words."
    )
    return "\n\n".join(parts)


TRUNCATION_MARKER = "\n\n[... middle of transcript omitted for length ...]\n\n"

_HEAD_SHARE = 0.6


def sample_transcript(transcript: str, max_chars: int = TRANSCRIPT_MAX_CHARS) -> str:
    """Reduce ``transcript`` to at most ``max_chars`` characters.

    Transcripts that fit are returned unchanged. Longer ones are sampled from
    both ends rather than simply cut off at the front: the opening sets up the
    topic while the closing usually carries conclusions and takeaways, both of
    which the description needs. The omitted middle is marked explicitly so the
    model knows the text is not continuous.
    """
    if max_chars <= 0 or len(transcript) <= max_chars:
        return transcript

    budget = max_chars - len(TRUNCATION_MARKER)
    if budget <= 0:
        return transcript[:max_chars]

    head_chars = int(budget * _HEAD_SHARE)
    tail_chars = budget - head_chars
    head = transcript[:head_chars]
    tail = transcript[-tail_chars:] if tail_chars > 0 else ""
    return f"{head}{TRUNCATION_MARKER}{tail}"


def _chat(
    client: OpenAI,
    model_name: str,
    title: str,
    transcript: str,
    channel_context: str = "",
    max_chars: int = TRANSCRIPT_MAX_CHARS,
) -> str:
    """Request a description from a single model and return the text."""
    user_message = (
        f"Video title: {title}\n\nTranscript:\n{sample_transcript(transcript, max_chars)}"
        "\n\nWrite a YouTube video description with a hook, "
        "3-5 bullet points, and 3-5 hashtags."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _system_prompt(channel_context)},
            {"role": "user", "content": user_message},
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        return ""
    return content.strip()


def generate_description(
    title: str,
    transcript: str,
    model: str | None = None,
    client: OpenAI | None = None,
    channel_context: str | None = None,
    max_chars: int = TRANSCRIPT_MAX_CHARS,
) -> str:
    """Generate a YouTube description from a title and transcript.

    Tries the primary model first, then each model in the comma-separated
    ``LLM_MODEL_FALLBACKS`` env var, if the previous candidate fails with an
    API or connection error. Raises ``RuntimeError`` if all candidates fail.

    ``channel_context`` (or the ``CHANNEL_CONTEXT`` env var) optionally
    tailors the prompt persona; leave unset for a domain-agnostic prompt.

    ``max_chars`` bounds how much transcript text is sent to the model; longer
    transcripts are sampled from both ends by :func:`sample_transcript`.
    """
    global _last_working_model
    client = client or build_client()
    candidates = resolve_models(model)
    context = resolve_channel_context(channel_context)

    last_error: OpenAIError | None = None
    for model_name in candidates:
        try:
            result = _chat(
                client, model_name, title, transcript, context, max_chars
            )
        except OpenAIError as exc:
            last_error = exc
            print(
                f"Warning: model {model_name} failed "
                f"({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            continue
        _last_working_model = model_name
        return result

    raise RuntimeError(
        f"All LLM models failed: {', '.join(candidates)}"
    ) from last_error


def load_cache(path: Path | str) -> dict[str, Any]:
    """Load the JSON cache, returning an empty dict if it does not exist."""
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(cache: dict[str, Any], path: Path | str) -> None:
    """Write the JSON cache to disk."""
    cache_path = Path(path)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
