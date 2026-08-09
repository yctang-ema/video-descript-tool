"""CSV + HTML review report generation."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

REVIEW_COLUMNS = [
    "video_id",
    "title",
    "video_url",
    "published_at",
    "transcript_status",
    "transcript",
    "transcript_truncated",
    "suggested_description",
    "approved",
]


def _coerce_text(value: Any) -> str:
    """Coerce a value to a string for CSV/HTML output."""
    if value is None:
        return ""
    return str(value)


def _csv_safe(value: Any) -> str:
    """Return a CSV-safe string that neutralises spreadsheet formula injection."""
    text = _coerce_text(value)
    stripped = text.lstrip()
    if stripped and stripped[0] in {"=", "+", "-", "@"}:
        return "'" + text
    return text


def write_review_csv(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    """Write the review CSV with an empty ``approved`` column."""
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_id": _csv_safe(row.get("video_id")),
                    "title": _csv_safe(row.get("title")),
                    "video_url": _csv_safe(row.get("video_url")),
                    "published_at": _csv_safe(row.get("published_at")),
                    "transcript_status": _csv_safe(row.get("transcript_status")),
                    "transcript": _csv_safe(row.get("transcript")),
                    "transcript_truncated": _csv_safe(row.get("transcript_truncated")),
                    "suggested_description": _csv_safe(row.get("suggested_description")),
                    "approved": "",
                }
            )


def _build_html(rows: list[dict[str, Any]]) -> str:
    """Build a self-contained HTML report."""
    cards = []
    for idx, row in enumerate(rows, start=1):
        video_id = html.escape(_coerce_text(row.get("video_id")))
        title = html.escape(_coerce_text(row.get("title")))
        video_url = html.escape(_coerce_text(row.get("video_url")))
        published_at = html.escape(_coerce_text(row.get("published_at")))
        transcript_status = html.escape(_coerce_text(row.get("transcript_status")))
        transcript = html.escape(_coerce_text(row.get("transcript")))
        transcript_truncated = bool(row.get("transcript_truncated"))
        suggested = html.escape(_coerce_text(row.get("suggested_description")))
        iframe_src = f"https://www.youtube.com/embed/{video_id}" if video_id else ""

        card = f"""
        <div class="card" id="video-{idx}">
          <h2>{idx}. {title}</h2>
          <p><strong>URL:</strong> <a href="{video_url}" target="_blank">{video_url}</a></p>
          <p><strong>Published:</strong> {published_at}</p>
          <p><strong>Transcript status:</strong> {transcript_status}</p>
          <div class="embed">
            <iframe width="560" height="315" src="{iframe_src}"
              title="{title}" frameborder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowfullscreen></iframe>
          </div>
          <details>
            <summary>Transcript</summary>
            <pre>{transcript}</pre>
          </details>
          <div class="description">
            <h3>Suggested description</h3>
            <pre>{suggested}</pre>
          </div>
          {'<p class="truncation-note"><strong>Note:</strong> Transcript was truncated before being sent to the LLM.</p>' if transcript_truncated else ''}
        </div>
        """
        cards.append(card)

    cards_html = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YouTube Description Review Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2rem; background: #f5f5f5; color: #222; }}
    h1 {{ color: #1a73e8; }}
    .card {{ background: #fff; border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
    h2 {{ margin-top: 0; }}
    .embed {{ margin: 1rem 0; }}
    iframe {{ max-width: 100%; border-radius: 4px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f8f9fa; padding: 1rem; border-radius: 4px; }}
    details {{ margin: 1rem 0; }}
    summary {{ cursor: pointer; font-weight: bold; }}
    .description {{ margin-top: 1rem; }}
    .truncation-note {{ color: #c5221f; font-size: 0.9rem; margin-top: 0.5rem; }}
  </style>
</head>
<body>
  <h1>YouTube Description Review Report</h1>
  <p>{len(rows)} videos with missing or short descriptions.</p>
  {cards_html}
</body>
</html>
"""


def generate_review_html(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    """Write a standalone HTML report."""
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as f:
        f.write(_build_html(rows))
