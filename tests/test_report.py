"""Tests for the report module."""

from __future__ import annotations

from pathlib import Path

from src.report import (
    build_combined_results,
    generate_review_html,
    write_review_csv,
)


def test_write_review_csv(tmp_path: Path) -> None:
    rows = [
        {
            "video_id": "abc123",
            "title": "Annual Forum",
            "video_url": "https://www.youtube.com/watch?v=abc123",
            "published_at": "2024-01-15",
            "transcript_status": "success",
            "transcript": "Hello world",
            "transcript_truncated": False,
            "suggested_description": "Executive summary",
        }
    ]
    path = tmp_path / "review.csv"
    write_review_csv(rows, path)
    content = path.read_text(encoding="utf-8")
    assert "video_id" in content
    assert "abc123" in content
    assert "Executive summary" in content
    assert "approved" in content


def test_generate_review_html_escapes_content(tmp_path: Path) -> None:
    rows = [
        {
            "video_id": "abc123",
            "title": "Annual <script>alert(1)</script>",
            "video_url": "https://www.youtube.com/watch?v=abc123",
            "published_at": "2024-01-15",
            "transcript_status": "success",
            "transcript": "Transcript <b>bold</b>",
            "transcript_truncated": False,
            "suggested_description": "Desc <br>",
        }
    ]
    path = tmp_path / "review.html"
    generate_review_html(rows, path)
    content = path.read_text(encoding="utf-8")
    assert "<script>" not in content
    assert "&lt;script&gt;" in content
    assert "<b>bold</b>" not in content
    assert "&lt;b&gt;bold&lt;/b&gt;" in content
    assert 'src="https://www.youtube.com/embed/abc123"' in content
    assert "Annual" in content


def test_write_review_csv_formula_safety(tmp_path: Path) -> None:
    rows = [
        {
            "video_id": "abc123",
            "title": "=SUM(A1)",
            "video_url": "https://www.youtube.com/watch?v=abc123",
            "published_at": "2024-01-15",
            "transcript_status": "success",
            "transcript": "+cmd/run",
            "transcript_truncated": False,
            "suggested_description": "@mention",
        }
    ]
    path = tmp_path / "review.csv"
    write_review_csv(rows, path)
    content = path.read_text(encoding="utf-8")
    assert "'=SUM(A1)" in content
    assert "'+cmd/run" in content
    assert "'@mention" in content


def _audit_row(video_id, has_description=False, status="ok"):
    return {
        "video_id": video_id,
        "title": f"Title {video_id}",
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "published_at": "2024-01-15",
        "has_description": has_description,
        "description_length": 0,
        "status": status,
    }


def test_build_combined_results_includes_only_cached_transcripts() -> None:
    rows = [
        _audit_row("done1"),
        _audit_row("done2"),
        _audit_row("pending1"),   # not in cache -> excluded
        _audit_row("failed1"),    # cached but no transcript -> excluded
        _audit_row("hasdesc", has_description=True),  # already described -> excluded
    ]
    cache = {
        "done1": {"transcript": "t1", "transcript_status": "audio_transcribed",
                   "suggested_description": "desc1"},
        "done2": {"transcript": "t2", "transcript_status": "success"},
        "failed1": {"transcript": None, "transcript_status": "audio_failed"},
        "hasdesc": {"transcript": "t3", "transcript_status": "success"},
    }
    combined = build_combined_results(rows, cache)
    ids = [r["video_id"] for r in combined]
    assert ids == ["done1", "done2"]
    assert all(r["transcript"] for r in combined)
    assert combined[0]["suggested_description"] == "desc1"


def test_build_combined_results_strips_apostrophe_from_video_id() -> None:
    """Audit CSVs from older indexers carry a leading ' on dash-prefixed IDs."""
    rows = [_audit_row("'-abcDash123")]
    cache = {"-abcDash123": {"transcript": "t", "transcript_status": "success"}}
    combined = build_combined_results(rows, cache)
    assert [r["video_id"] for r in combined] == ["-abcDash123"]


def test_build_combined_results_ignores_non_ok_rows() -> None:
    rows = [_audit_row("meta_fail", status="metadata_failed")]
    cache = {"meta_fail": {"transcript": "t", "transcript_status": "success"}}
    assert build_combined_results(rows, cache) == []
