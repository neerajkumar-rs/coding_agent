"""Phase 3 tests: plan validation rules, fallback, coder prompt contents."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import _valid_plan, _valid_plan_task, run_pipeline
from tests.test_run_pipeline import CODER, PLANNER, VALIDATE_OK, make_fake_generate


def _task(path: str, desc: str) -> dict:
    return {"path": path, "action": "create", "description": desc}


# --- per-task rejection rules ------------------------------------------------

def test_placeholders_rejected() -> None:
    desc = "a sufficiently long description for this task here"
    for bad in ("your_code.py", "example.py", "foo.py", "file1.py", "untitled.py"):
        assert _valid_plan_task(_task(bad, desc)) is not None, bad
    assert _valid_plan_task(_task("blog_page.py", desc)) is None


def test_extension_language_mismatch_rejected() -> None:
    assert _valid_plan_task(_task("main.rs", "Python script that greets the user by name")) is not None
    assert _valid_plan_task(_task("page.py", "HTML page structure for the blog home page")) is not None
    assert _valid_plan_task(_task("main.rs", "Rust binary that greets the user by name")) is None
    assert _valid_plan_task(_task("page.html", "HTML page structure for the blog home page")) is None


def test_short_description_rejected() -> None:
    assert _valid_plan_task(_task("ok.py", "does things")) is not None
    assert _valid_plan_task(_task("ok.py", "does several concrete things for the user request")) is None


# --- plan-level rules ----------------------------------------------------------

def test_app_py_rejected_when_not_requested() -> None:
    tasks = [_task("app.py", "Python application implementing the requested feature fully")]
    assert _valid_plan(tasks, "make a blog site in html css js") != []
    # ...but allowed when explicitly requested
    assert _valid_plan(tasks, "create app.py with a calculator") == []


def test_web_plan_must_include_html_css_js() -> None:
    only_html = [_task("index.html", "HTML page with all structure and content included")]
    reasons = _valid_plan(only_html, "make a blog site in html css js")
    assert any("missing" in r for r in reasons)
    full = [
        _task("index.html", "HTML page structure linking stylesheet and script files"),
        _task("style.css", "Stylesheet with layout typography colors and responsive rules"),
        _task("script.js", "JavaScript for dynamic content loading and interactions"),
    ]
    assert _valid_plan(full, "make a blog site in html css js") == []


# --- plan task rules (post-validation lives in code) --------------------------

def test_plan_tasks_pass_through_valid_plan_rules() -> None:
    """PlanResult tasks go through the same per-task rules; rejected tasks
    are dropped in code, never trusted from the model."""
    from understand import PlanResult

    result = PlanResult({
        "tasks": [
            {"path": "your_code.py", "action": "create",
             "description": "a sufficiently long description for this task here"},
            {"path": "blog_page.py", "action": "create",
             "description": "a sufficiently long description for this task here"},
        ],
        "assumptions": [],
    })
    result.apply_task_rules(_valid_plan_task)
    assert [t["path"] for t in result.tasks] == ["blog_page.py"]


# --- coder prompt contents -----------------------------------------------------

def test_coder_prompt_contains_request_paths_language() -> None:
    captured: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "CLASSIFY MODE:" in prompt:
            return "code"
        if "PLAN TASKS JSON:" in prompt:
            return json.dumps({
                "tasks": [_task(
                    "greet.py",
                    "Python module with a greet function returning a greeting string",
                )],
                "assumptions": [],
            })
        if model == CODER:
            captured.append(prompt)
            return "```python\ndef greet(name):\n    return f'hi {name}'\n```"
        return VALIDATE_OK

    with TemporaryDirectory() as tmp:
        asyncio.run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert captured, "coder was never called"
    prompt = captured[0]
    assert "create greet.py with a greet function" in prompt   # original request
    assert "greet.py" in prompt                                  # planned paths
    assert "greet function returning a greeting" in prompt       # task description
    assert "python" in prompt.lower()                            # language from extension


def test_index_html_prompt_requires_exact_link_names() -> None:
    captured: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "CLASSIFY MODE:" in prompt:
            return "code"
        if "PLAN TASKS JSON:" in prompt:
            return json.dumps({
                "tasks": [
                    _task("index.html", "HTML page structure linking stylesheet and script files"),
                    _task("style.css", "Stylesheet with layout typography colors and responsive rules"),
                    _task("script.js", "JavaScript for dynamic content loading and user interactions"),
                ],
                "assumptions": [],
            })
        if model == CODER and "TARGET FILE: index.html" in prompt:
            captured.append(prompt)
            return ("```html\n<!doctype html><html><head>"
                    '<link rel="stylesheet" href="style.css">'
                    '<script src="script.js"></script>'
                    "</head><body><h1>Blog</h1></body></html>\n```")
        if model == CODER and "TARGET FILE: style.css" in prompt:
            return "```css\nbody { margin: 0; padding: 0; }\n```"
        if model == CODER and "TARGET FILE: script.js" in prompt:
            return "```javascript\ndocument.addEventListener('DOMContentLoaded', () => {});\n```"
        return VALIDATE_OK

    with TemporaryDirectory() as tmp:
        result = asyncio.run(run_pipeline(
            "make a blog site in html css js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert captured, "index.html coder prompt never captured"
    assert "style.css" in captured[0] and "script.js" in captured[0]
    assert set(result["staged_files"]) == {"index.html", "style.css", "script.js"}


def test_placeholder_plan_is_needs_human_no_files() -> None:
    """The planner keeps returning placeholder tasks -> dropped in code; with
    zero valid tasks the plan step fails to needs_human (no files, no loop)."""
    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "CLASSIFY MODE:" in prompt:
            return "code"
        if "PLAN TASKS JSON:" in prompt:
            return json.dumps({
                "tasks": [_task("foo.py", "x"), _task("untitled.html", "y")],
                "assumptions": [],
            })
        return VALIDATE_OK

    with TemporaryDirectory() as tmp:
        result = asyncio.run(run_pipeline(
            "make a blog site in html css js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert result["status"] == "needs_human"
    assert result["staged_files"] == []
