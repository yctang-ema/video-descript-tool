"""Diagnostic: list Zen models and probe chat completions."""

from __future__ import annotations

import os
import sys

from openai import OpenAI, OpenAIError


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        print("Error: OPENCODE_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(base_url="https://opencode.ai/zen/v1", api_key=api_key)


def list_models(client: OpenAI) -> list[str]:
    """Return model IDs from the /models endpoint."""
    try:
        resp = client.models.list()
        return [m.id for m in resp.data]
    except Exception as exc:
        print(f"   /models endpoint failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []


def probe_model(client: OpenAI, model_name: str, max_tokens: int = 10) -> tuple[bool, str]:
    """Send a minimal chat completion. Returns (ok, response_or_error)."""
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'pong' and nothing else."},
            ],
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ""
        return True, content.strip()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    client = build_client()

    print("1. Fetching model list from /models ...")
    models = list_models(client)
    if not models:
        print("   No models returned.")
        sys.exit(0)

    print(f"   Found {len(models)} model(s):")
    for m in models:
        print(f"      - {m}")

    # Pick any non-embedding model to test the completions endpoint itself.
    chat_models = [m for m in models if "embed" not in m.lower()]
    if not chat_models:
        print("\n   No chat models found (only embeddings?).")
        sys.exit(0)

    print(f"\n2. Probing completions endpoint with first chat model ({chat_models[0]}) ...")
    ok, result = probe_model(client, chat_models[0], max_tokens=10)
    if ok:
        print(f"   SUCCESS: {result}")
    else:
        print(f"   FAILED: {result}")
        print("\n   Retrying without max_tokens parameter ...")
        ok2, result2 = probe_model(client, chat_models[0], max_tokens=0)
        if ok2:
            print(f"   SUCCESS (no max_tokens): {result2}")
        else:
            print(f"   FAILED (no max_tokens): {result2}")

    # Now probe every model name (useful if one gateway exposes models but
    # only some backends are online).
    gemini_models = [m for m in models if "gemini" in m.lower()]
    if gemini_models:
        print(f"\n3. Probing {len(gemini_models)} Gemini model(s) ...")
        for name in gemini_models:
            ok, result = probe_model(client, name)
            status = f"SUCCESS -> {result}" if ok else f"FAILED -> {result}"
            print(f"   {name}: {status}")
    else:
        print("\n3. No Gemini models found in the list.")

    print("\n4. Probing all remaining models ...")
    for name in models:
        if name in gemini_models:
            continue
        ok, result = probe_model(client, name)
        if ok:
            print(f"   {name}: SUCCESS -> {result}")
        else:
            print(f"   {name}: FAILED -> {result}")
