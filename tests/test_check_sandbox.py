"""Phase 1 tests: in-process CHECK, infra vs content errors, plausibility."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from orchestrator import _check_generated
from sandbox import check_plausible_content, tail_lines


def test_valid_python_passes_with_spaces_in_path() -> None:
    spaced = "my project dir/app with spaces.py"
    assert _check_generated(spaced, "def add(a, b):\n    return a + b\n") is None


def test_invalid_python_fails_as_content_error() -> None:
    error = _check_generated("broken.py", "def broken(:\n")
    assert error is not None
    assert "syntax error" in error


def test_valid_html_passes_anywhere() -> None:
    html = "<!doctype html><html><head><title>x</title></head><body><p>hello world</p></body></html>"
    assert _check_generated("index.html", html) is None


def test_html_without_body_fails() -> None:
    error = check_plausible_content("index.html", "<!doctype html><html><head></head></html>")
    assert error is not None and "<body>" in error


def test_empty_and_missing_fail_cleanly() -> None:
    assert check_plausible_content("a.html", "") == "file is empty"
    assert check_plausible_content("a.css", "   ") == "file is empty"


def test_leftover_fences_fail() -> None:
    error = check_plausible_content("a.html", "```html\n<html><body>x</body></html>\n```")
    assert error is not None and "fences" in error


def test_prose_only_fails() -> None:
    error = check_plausible_content("a.js", "this file explains the approach in words only")
    assert error is not None


def test_tail_lines_keeps_last_30_never_char_truncated() -> None:
    big = "\n".join(f"line {i}" for i in range(100))
    tail = tail_lines(big)
    assert tail.splitlines()[0] == "line 70"
    assert len(tail.splitlines()) == 30
    assert "line 99" in tail


def test_checker_crash_is_infra_error() -> None:
    """An exception inside the checker is infra: it must propagate (the
    pipeline catches it and aborts WITHOUT a coder retry)."""
    with patch("orchestrator.compile", side_effect=RuntimeError("checker exploded")):
        with pytest.raises(RuntimeError, match="checker exploded"):
            _check_generated("app.py", "x = 1\n")


def test_infra_error_aborts_without_retry() -> None:
    """Pipeline level: infra error -> task FAILED with 'infrastructure', the
    coder is NOT retried, and no file is staged."""
    import asyncio

    from orchestrator import run_pipeline
    from tests.test_run_pipeline import CODER, PLANNER, make_fake_generate

    plan = '[{"path": "app.py", "action": "create", "description": "Python application implementing the requested calculator feature fully and correctly"}]'
    fake = make_fake_generate({
        "JSON array of tasks": plan,
        "TARGET FILE: app.py": "```python\nX = 1\n```",
    })
    with TemporaryDirectory() as tmp:
        with patch("orchestrator.compile", side_effect=RuntimeError("boom")):
            result = asyncio.run(run_pipeline(
                "create app.py with a calculator",
                tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            ))
        assert result["staged_files"] == []
        assert len(result["failed_tasks"]) == 1
        assert "infrastructure" in result["failed_tasks"][0]["error"].lower()
        check = next(t for t in result["trace"] if t["stage"] == "CHECK")
        assert check["status"] == "failed"
