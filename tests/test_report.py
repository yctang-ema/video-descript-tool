"""Tests for the report module."""

from __future__ import annotations

from pathlib import Path

from src.report import generate_review_html, write_review_csv


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
