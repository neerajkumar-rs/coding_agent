"""Phase 3 tests: plan validation rules, fallback, coder prompt contents."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import _fallback_plan, _valid_plan, _valid_plan_task, run_pipeline
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


def test_fallback_plan_gives_three_web_files() -> None:
    plan = _fallback_plan("make a blog site in html css js")
    assert [t["path"] for t in plan] == ["index.html", "style.css", "script.js"]
    for t in plan:
        assert _valid_plan_task(t) is None


def test_fallback_plan_non_web_is_single_python_file() -> None:
    plan = _fallback_plan("fix the parser to handle empty lines", deliverable="parser")
    assert plan is not None
    assert len(plan) == 1
    assert plan[0]["path"].endswith(".py")


def test_fallback_plan_never_uses_request_text_as_content() -> None:
    """The fallback emits a fixed minimal template: the description is exactly
    the template (only the deliverable noun slots in), never raw request text."""
    request = "fix the flux capacitor calibration routine immediately please"
    plan = _fallback_plan(request, deliverable="calibration")
    assert plan is not None
    assert plan[0]["description"] == (
        "Python calibration stub implementing the requested "
        "behavior with a TODO for the concrete details."
    )
    # and the web fallback is equally fixed
    web = _fallback_plan("make a page about my grandmother's secret soup recipe")
    assert web is not None
    assert "grandmother" not in " ".join(t["description"] for t in web)
    assert "soup" not in " ".join(t["description"] for t in web)


def test_fallback_plan_without_deliverable_returns_none() -> None:
    assert _fallback_plan("xyzzy the whole thing somehow") is None


def test_fallback_plan_rejects_placeholder_filename_from_request() -> None:
    # "create your_code.py please" is a confirmed CODE_TASK whose only
    # filename is a placeholder -> no fallback (needs_human upstream).
    assert _fallback_plan("create your_code.py please") is None


def test_fallback_plan_uses_filename_from_request() -> None:
    plan = _fallback_plan("fix the bug in utils.py")
    assert plan is not None
    assert plan[0]["path"] == "utils.py"
    assert _valid_plan_task(plan[0]) is None


# --- coder prompt contents -----------------------------------------------------

def test_coder_prompt_contains_request_paths_language() -> None:
    captured: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2):
        if model == CODER:
            captured.append(prompt)
            return "```python\ndef greet(name):\n    return f'hi {name}'\n```"
        if "JSON array of tasks" in prompt:
            return json.dumps([_task(
                "greet.py",
                "Python module with a greet function returning a greeting string",
            )])
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

    async def fake(model, prompt, system="", temperature=0.2):
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
        if "JSON array of tasks" in prompt:
            return json.dumps([
                _task("index.html", "HTML page structure linking stylesheet and script files"),
                _task("style.css", "Stylesheet with layout typography colors and responsive rules"),
                _task("script.js", "JavaScript for dynamic content loading and interactions"),
            ])
        return VALIDATE_OK

    with TemporaryDirectory() as tmp:
        result = asyncio.run(run_pipeline(
            "make a blog site in html css js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert captured, "index.html coder prompt never captured"
    assert "style.css" in captured[0] and "script.js" in captured[0]
    assert set(result["staged_files"]) == {"index.html", "style.css", "script.js"}


def test_placeholder_plan_triggers_fallback() -> None:
    calls = {"planner": 0}

    async def fake(model, prompt, system="", temperature=0.2):
        if "JSON array of tasks" in prompt:
            calls["planner"] += 1
            # Both attempts return placeholder names -> deterministic fallback.
            return json.dumps([_task("foo.py", "x"), _task("untitled.html", "y")])
        if model == CODER:
            if "TARGET FILE: index.html" in prompt:
                return ("```html\n<!doctype html><html><head>"
                        '<link rel="stylesheet" href="style.css">'
                        '<script src="script.js"></script>'
                        "</head><body><h1>Blog</h1></body></html>\n```")
            if "TARGET FILE: style.css" in prompt:
                return "```css\nbody { margin: 0; padding: 0; }\n```"
            if "TARGET FILE: script.js" in prompt:
                return "```javascript\ndocument.addEventListener('DOMContentLoaded', () => {});\n```"
        return VALIDATE_OK

    with TemporaryDirectory() as tmp:
        result = asyncio.run(run_pipeline(
            "make a blog site in html css js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert calls["planner"] == 2  # one retry, then fallback
    assert set(result["staged_files"]) == {"index.html", "style.css", "script.js"}
    plan = next(t for t in result["trace"] if t["stage"] == "PLAN")
    assert "fallback" in plan["note"]
