"""Phase 4 tests: deterministic summary skeleton + guarded model prose."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import run_pipeline
from tests.test_run_pipeline import CODER, PLANNER, VALIDATE_OK, make_fake_generate

ONE_FILE_PLAN = json.dumps([
    {"path": "greet.py", "action": "create",
     "description": "Python module with a greet function returning a greeting string"},
])
GREET_CODE = "```python\ndef greet(name):\n    return f'Hello, {name}!'\n```"


def _run(coro):
    return asyncio.run(coro)


def _base_routes(**extra):
    routes = {
        "JSON array of tasks": ONE_FILE_PLAN,
        "TARGET FILE: greet.py": GREET_CODE,
        "satisfies_request": VALIDATE_OK,
    }
    routes.update(extra)
    return routes


def test_summary_skeleton_contents() -> None:
    routes = _base_routes(**{"Describe only features visible": ""})  # prose returns empty
    fake = make_fake_generate(routes)
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    summary = result["summary"]
    assert "Built: greet.py" in summary
    assert "How to run: run: python greet.py" in summary
    assert "Known issues: none" in summary
    assert "Total time:" in summary


def test_summary_prose_included_when_clean() -> None:
    routes = _base_routes(**{
        "Describe only features visible": "The module defines a greet function that returns a personalized greeting string."
    })
    fake = make_fake_generate(routes)
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "greet function" in result["summary"]
    assert "Built: greet.py" in result["summary"]


def test_summary_prose_dropped_when_inventing_features() -> None:
    routes = _base_routes(**{
        "Describe only features visible": "This app has dark mode, a login page, and a full-text search."
    })
    fake = make_fake_generate(routes)
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    # Invented keywords absent from the code -> prose dropped entirely
    assert "dark mode" not in result["summary"]
    assert "login" not in result["summary"]
    assert "Built: greet.py" in result["summary"]


def test_summary_falls_back_to_skeleton_when_model_fails() -> None:
    async def fake(model, prompt, system="", temperature=0.2):
        if "Describe only features visible" in prompt:
            raise RuntimeError("model unavailable")
        if "JSON array of tasks" in prompt:
            return ONE_FILE_PLAN
        if "TARGET FILE: greet.py" in prompt:
            return GREET_CODE
        return VALIDATE_OK

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "Built: greet.py" in result["summary"]
    assert "Total time:" in result["summary"]


def test_summary_lists_failed_tasks_with_reasons() -> None:
    routes = {
        "JSON array of tasks": json.dumps([
            {"path": "ok.py", "action": "create",
             "description": "Python module that compiles and runs cleanly every time"},
            {"path": "bad.py", "action": "create",
             "description": "Python module that never compiles for this test"},
        ]),
        "TARGET FILE: ok.py": "```python\nX = 1\n```",
        "TARGET FILE: bad.py": "```python\ndef broken(:\n```",
        "satisfies_request": VALIDATE_OK,
        "Describe only features visible": "",
    }
    fake = make_fake_generate(routes)
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create ok.py and bad.py modules",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert result["staged_files"] == ["ok.py"]
    assert result["failed_tasks"][0]["path"] == "bad.py"
    assert "NOT saved (failed): bad.py" in result["summary"]
    assert "syntax error" in result["summary"]


def test_unverified_summary_when_validate_errors() -> None:
    async def fake(model, prompt, system="", temperature=0.2):
        if "satisfies_request" in prompt:
            raise RuntimeError("ollama died mid-review")
        if "JSON array of tasks" in prompt:
            return ONE_FILE_PLAN
        if "TARGET FILE: greet.py" in prompt:
            return GREET_CODE
        if "Describe only features visible" in prompt:
            return ""
        return ""

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "UNVERIFIED" in result["summary"]
    # Files were still staged (CHECK passed) — failure is reported, not hidden
    assert result["staged_files"] == ["greet.py"]
    validate = next(t for t in result["trace"] if t["stage"] == "VALIDATE")
    assert validate["status"] == "failed"
