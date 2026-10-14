"""Phase 3 tests: plan validation rules, fallback, coder prompt contents."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import _valid_plan_task, run_pipeline
from tests.test_run_pipeline import CODER, PLANNER, VALIDATE_OK, make_fake_generate


def _task(path: str, desc: str) -> dict:
    return {"path": path, "action": "create", "description": desc}


# --- per-task rejection rules ------------------------------------------------

def test_placeholders_rejected() -> None:
    desc = "a sufficiently long description for this task here"
    for bad in ("your_code.py", "example.py", "foo.py", "file1.py", "untitled.py"):
        assert _valid_plan_task(_task(bad, desc)) is not None, bad
    assert _valid_plan_task(_task("blog_page.py", desc)) is None


def test_safety_rules_only_no_quality_gates() -> None:
    """The word-count and language-mismatch gates are gone (they rejected
    legitimate plans like 'Create a README file for the project'). Only SAFETY
    rules remain: empty path, junk artifact name, unsupported type. The coder
    + CHECK catch bad content; the plan gate never judges prose quality."""
    # previously rejected by the word-count rule: now accepted
    assert _valid_plan_task(_task("readme.md", "Create a README file for the project")) is None
    # previously rejected by the language-mismatch rule: now accepted (the
    # description mentioning another language is not a safety problem)
    assert _valid_plan_task(_task("main.rs", "Python script that greets the user by name")) is None
    assert _valid_plan_task(_task("page.py", "HTML page structure for the blog home page")) is None
    # safety rules still fire
    assert _valid_plan_task(_task("foo.py", "anything at all here")) is not None  # placeholder stem
    assert _valid_plan_task(_task("", "no path at all here")) is not None  # empty path
    assert _valid_plan_task(_task("x.exe", "binary target not supported here")) is not None  # type


# --- plan-level text-matching rules are deleted with the rules router --------

def test_plan_level_rules_are_gone() -> None:
    """_valid_plan / invented_file_rejection / WEB_HINTS are deleted: the agent
    loop decides what to write; Python enforces safety only."""
    import orchestrator

    for gone in ("_valid_plan", "invented_file_rejection", "WEB_HINTS"):
        assert not hasattr(orchestrator, gone), gone


def test_placeholder_stems_stay_as_write_safety() -> None:
    """PLACEHOLDER_STEMS stays: the write gate rejects scaffold names (hygiene,
    never a routing decision)."""
    import orchestrator

    assert hasattr(orchestrator, "PLACEHOLDER_STEMS")
    assert "your_code" in orchestrator.PLACEHOLDER_STEMS


# --- coder prompt contents -----------------------------------------------------

def test_coder_prompt_contains_request_paths_language() -> None:
    captured: list[str] = []
    loop_actions = iter([
        {"action": "write_file", "path": "greet.py", "text": "a greet function returning a greeting string"},
        {"action": "finish"},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        if model == CODER and "Write greet.py." in prompt:
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
    assert "greet.py" in prompt                                  # the target file
    assert "greet function returning a greeting" in prompt       # what to write
    assert "python" in prompt.lower()                            # language from extension


def test_index_html_prompt_reaches_the_coder_with_the_request() -> None:
    """The loop's coder call for index.html carries the original request and
    the registry's type hint; the staged html passes its checks."""
    captured: list[str] = []
    loop_actions = iter([
        {"action": "write_file", "path": "index.html", "text": "the blog page structure"},
        {"action": "write_file", "path": "style.css", "text": "the blog stylesheet"},
        {"action": "write_file", "path": "script.js", "text": "the blog interactions"},
        {"action": "finish"},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        if model == CODER and "Write index.html." in prompt:
            captured.append(prompt)
            return ("```html\n<!doctype html><html><head>"
                    '<link rel="stylesheet" href="style.css">'
                    '<script src="script.js"></script>'
                    "</head><body><h1>Blog</h1></body></html>\n```")
        if model == CODER and "Write style.css." in prompt:
            return "```css\nbody { margin: 0; padding: 0; }\n```"
        if model == CODER and "Write script.js." in prompt:
            return "```javascript\ndocument.addEventListener('DOMContentLoaded', () => {});\n```"
        return VALIDATE_OK

    with TemporaryDirectory() as tmp:
        result = asyncio.run(run_pipeline(
            "make a blog site in html css js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert captured, "index.html coder prompt never captured"
    assert "make a blog site in html css js" in captured[0]  # original request
    assert set(result["staged_files"]) == {"index.html", "style.css", "script.js"}


def test_placeholder_write_is_needs_human_no_files() -> None:
    """The loop keeps writing placeholder paths -> the safety rule rejects
    every one; the turn is needs_human (no files, no loop)."""
    loop_actions = iter([
        {"action": "write_file", "path": "foo.py", "text": "x"},
        {"action": "write_file", "path": "untitled.html", "text": "y"},
        {"action": "finish"},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        return VALIDATE_OK

    with TemporaryDirectory() as tmp:
        result = asyncio.run(run_pipeline(
            "make a blog site in html css js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert result["status"] == "needs_human"
    assert result["staged_files"] == []
