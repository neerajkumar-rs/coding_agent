"""Tests for run_pipeline: ROUTE/PLAN/CODE/CHECK/VALIDATE/SUMMARY (fake LLM)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from orchestrator import classify_request, run_pipeline

PLANNER = "deepseek-r1:1.5b"
CODER = "qwen2.5-coder:3b"

WEB_PLAN = json.dumps([
    {"path": "index.html", "action": "create",
     "description": "Main HTML5 page structure with linked stylesheet and script files"},
    {"path": "style.css", "action": "create",
     "description": "Stylesheet with layout typography colors and responsive rules"},
    {"path": "script.js", "action": "create",
     "description": "JavaScript for dynamic content loading and user interactions"},
])

INDEX_HTML = (
    "<!doctype html><html><head>"
    '<link rel="stylesheet" href="style.css">'
    '<script src="script.js"></script>'
    "</head><body><h1>Blog</h1></body></html>"
)
STYLE_CSS = "body { font-family: sans-serif; margin: 0; }"
SCRIPT_JS = "document.addEventListener('DOMContentLoaded', () => { console.log('ok'); });"
VALIDATE_OK = json.dumps({
    "satisfies_request": True, "problems": [], "summary": "A blog site scaffold."
})


def make_fake_generate(routes: dict[str, str], default: str = ""):
    """routes: marker-in-prompt -> response. Coder routes by 'TARGET FILE: x'
    (matched via regex so retry feedback above the marker still routes)."""
    async def fake(model, prompt, system="", temperature=0.2):
        import re

        target = re.search(r"TARGET FILE:\s*(\S+)", prompt)
        if target:
            file_marker = f"TARGET FILE: {target.group(1)}"
            if file_marker in routes:
                return routes[file_marker]
        for marker in sorted(routes, key=len, reverse=True):
            if marker in prompt:
                return routes[marker]
        return default

    return fake


def _run(coro):
    return asyncio.run(coro)


# 1. Question input -> zero tasks, text answer
def test_question_gets_answer_no_tasks() -> None:
    fake = make_fake_generate({"QUESTION:": "<think>thinking</think>Rust compiles to native code; Python is interpreted."})
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "what is the difference between rust and python",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "answer" in result
    assert "Rust compiles" in result["answer"]
    assert "think" not in result["answer"].lower()
    assert "staged_files" not in result  # nothing staged
    stages = [t["stage"] for t in result["trace"]]
    assert stages == ["ROUTE", "ANSWER"]


def test_classify_rules() -> None:
    assert classify_request("what is the difference between rust and python") == "QUESTION"
    assert classify_request("make a blog site in html5 css and js") == "CODE_TASK"
    assert classify_request("create app.py") == "CODE_TASK"


# 2. Web request -> 3 files, VALIDATE and SUMMARY present in trace
def test_web_request_three_files_full_trace() -> None:
    fake = make_fake_generate({
        "JSON array of tasks": WEB_PLAN,
        "TARGET FILE: index.html": f"```html\n{INDEX_HTML}\n```",
        "TARGET FILE: style.css": f"```css\n{STYLE_CSS}\n```",
        "TARGET FILE: script.js": f"```javascript\n{SCRIPT_JS}\n```",
        "satisfies_request": VALIDATE_OK,
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js with dynamic content",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        staged = set(result["staged_files"])
        assert staged == {"index.html", "style.css", "script.js"}
        stages = [t["stage"] for t in result["trace"]]
        assert "VALIDATE" in stages and "SUMMARY" in stages
        assert all(t["status"] == "ok" for t in result["trace"])
        # Staged, not deployed
        assert (Path(tmp) / ".agent_staging" / "index.html").is_file()
        assert not (Path(tmp) / "index.html").exists()
        assert "blog site scaffold" in result["summary"].lower()


# 3. Forced-failing task -> not staged, listed as FAILED
def test_failing_task_not_staged() -> None:
    fake = make_fake_generate({
        "JSON array of tasks": json.dumps([
            {"path": "app.py", "action": "create",
             "description": "Python application implementing the requested feature fully"}
        ]),
        "TARGET FILE: app.py": "```python\ndef broken(:\n```",  # never compiles
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create app.py with a calculator",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == []
        assert len(result["failed_tasks"]) == 1
        assert result["failed_tasks"][0]["path"] == "app.py"
        assert not (Path(tmp) / ".agent_staging" / "app.py").exists()
        check = next(t for t in result["trace"] if t["stage"] == "CHECK")
        assert check["status"] == "failed"


# 4. Bad html link to a missing css file -> caught in CHECK
def test_bad_html_link_caught() -> None:
    bad_html = '<!doctype html><html><head><link rel="stylesheet" href="missing.css"></head><body>x</body></html>'
    fake = make_fake_generate({
        "JSON array of tasks": json.dumps([
            {"path": "index.html", "action": "create",
             "description": "Standalone HTML page containing all markup and structure"}
        ]),
        "TARGET FILE: index.html": f"```html\n{bad_html}\n```",
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create index.html page",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == []
        assert "missing.css" in result["failed_tasks"][0]["error"]


# 5. Valid index.html passes when cwd is not the project root
def test_check_independent_of_cwd(monkeypatch) -> None:
    fake = make_fake_generate({
        "JSON array of tasks": json.dumps([
            {"path": "index.html", "action": "create",
             "description": "Standalone HTML page containing all markup and structure"}
        ]),
        "TARGET FILE: index.html": (
            "```html\n<!doctype html><html><head><style>body { margin: 0; }</style>"
            "</head><body>hi</body></html>\n```"
        ),
        "satisfies_request": VALIDATE_OK,
    })
    with TemporaryDirectory() as ws, TemporaryDirectory() as elsewhere:
        monkeypatch.chdir(elsewhere)  # cwd is NOT the workspace
        result = _run(run_pipeline(
            "create index.html page",
            ws, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == ["index.html"]
        assert (Path(ws) / ".agent_staging" / "index.html").is_file()
        monkeypatch.undo()  # release the dir handle before Windows cleanup


# Plan validation: placeholders and short descriptions rejected -> fallback
def test_plan_placeholder_falls_back_to_web_defaults() -> None:
    fake = make_fake_generate({
        "JSON array of tasks": json.dumps([
            {"path": "your_code.html", "action": "create", "description": "page"},
            {"path": "example.py", "action": "create", "description": "does things"},
        ]),
        "TARGET FILE: index.html": f"```html\n{INDEX_HTML}\n```",
        "TARGET FILE: style.css": f"```css\n{STYLE_CSS}\n```",
        "TARGET FILE: script.js": f"```javascript\n{SCRIPT_JS}\n```",
        "satisfies_request": VALIDATE_OK,
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert set(result["staged_files"]) == {"index.html", "style.css", "script.js"}
        plan = next(t for t in result["trace"] if t["stage"] == "PLAN")
        assert "fallback" in plan["note"]


# VALIDATE error -> summary says UNVERIFIED
def test_validate_error_marks_unverified() -> None:
    async def fake(model, prompt, system="", temperature=0.2):
        if "satisfies_request" in prompt:
            raise RuntimeError("ollama died mid-review")
        if "JSON array of tasks" in prompt:
            return WEB_PLAN
        if "TARGET FILE: index.html" in prompt:
            return f"```html\n{INDEX_HTML}\n```"
        if "TARGET FILE: style.css" in prompt:
            return f"```css\n{STYLE_CSS}\n```"
        if "TARGET FILE: script.js" in prompt:
            return f"```javascript\n{SCRIPT_JS}\n```"
        return ""

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert "UNVERIFIED" in result["summary"]
        validate = next(t for t in result["trace"] if t["stage"] == "VALIDATE")
        assert validate["status"] == "failed"
        # Files still staged (CHECK passed); failure is reported, not hidden
        assert len(result["staged_files"]) == 3
