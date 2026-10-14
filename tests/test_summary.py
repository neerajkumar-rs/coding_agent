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
    assert "**Built:**" in summary and "`greet.py`" in summary
    # How-to-run is model-written (from the files); with the prose call
    # stubbed empty, the skeleton carries facts only (never a hardcoded line).
    assert "**Known issues:** none" in summary
    assert "**Total time:**" in summary


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
    assert "**Built:**" in result["summary"] and "`greet.py`" in result["summary"]


def test_summary_prose_included_as_written_by_the_model() -> None:
    """The summary prose is the MODEL's words from the files it wrote (the old
    feature-keyword text guard is deleted with the rules layer)."""
    routes = _base_routes(**{
        "Describe only features visible": "This app has dark mode, a login page, and a full-text search."
    })
    fake = make_fake_generate(routes)
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    # the model's prose is included as written (no Python keyword filter)
    assert "dark mode" in result["summary"]
    assert "**Built:**" in result["summary"] and "`greet.py`" in result["summary"]


def test_summary_falls_back_to_skeleton_when_model_fails() -> None:
    loop_actions = iter([
        {"action": "write_file", "path": "greet.py", "text": "the greet function"},
        {"action": "finish"},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        if "Describe only features visible" in prompt:
            raise RuntimeError("model unavailable")
        if "TARGET FILE: greet.py" in prompt or "Write greet.py." in prompt:
            return GREET_CODE
        return VALIDATE_OK

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "**Built:**" in result["summary"] and "`greet.py`" in result["summary"]
    assert "**Total time:**" in result["summary"]


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
    assert "NOT saved (failed): `bad.py`" in result["summary"]
    assert "syntax error" in result["summary"]


def test_unverified_summary_when_validate_errors() -> None:
    loop_actions = iter([
        {"action": "write_file", "path": "greet.py", "text": "the greet function"},
        {"action": "finish"},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        if "satisfies_request" in prompt:
            raise RuntimeError("ollama died mid-review")
        if "TARGET FILE: greet.py" in prompt or "Write greet.py." in prompt:
            return GREET_CODE
        if "Describe only features visible" in prompt:
            return ""
        return ""

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        summary = result["summary"]
        assert "UNVERIFIED" in summary
        # The UNVERIFIED line is a markdown blockquote with reason + fix...
        assert "> **UNVERIFIED**" in summary
        assert "validation could not run: ollama died mid-review" in summary
        assert "Fix:" in summary
        # ...and "Known issues: none" is NOT printed (the validation problem is
        # listed as the known issue).
        assert "Known issues:** none" not in summary
        assert "**Known issues:** validation could not run" in summary
        # Files were still staged (CHECK passed) — failure is reported, not hidden
        assert result["staged_files"] == ["greet.py"]
        validate = next(t for t in result["trace"] if t["stage"] == "VALIDATE")
        assert validate["status"] == "failed"
        # history outcome: unverified (not ok)
        import history as _history
        entries, _ = _history.read_history(tmp)
        assert entries[-1]["outcome"] == "unverified"


def test_unverified_summary_names_the_pull_fix_for_missing_model() -> None:
    """ModelNotInstalled detail carries 'ollama pull <name>' — the summary
    surfaces it as the fix."""
    loop_actions = iter([
        {"action": "write_file", "path": "greet.py", "text": "the greet function"},
        {"action": "finish"},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        if "satisfies_request" in prompt:
            raise RuntimeError(
                "model not installed: 'qwen2.5-coder:7b'. Run: ollama pull qwen2.5-coder:7b")
        if "TARGET FILE: greet.py" in prompt or "Write greet.py." in prompt:
            return GREET_CODE
        return ""

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "> **UNVERIFIED**" in result["summary"]
    assert "validation could not run:" in result["summary"]
    assert "Fix: ollama pull qwen2.5-coder:7b" in result["summary"]


def test_validated_summary_distinguishes_checks_from_validation() -> None:
    """A syntax-only CHECK pass is never 'verified': when VALIDATE ran clean,
    the summary says checks passed AND validated; CHECK alone says so."""
    routes = _base_routes(**{"Describe only features visible": ""})
    fake = make_fake_generate(routes)
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "**Checks passed** (syntax/links); **validated against your request**" in result["summary"]


def test_partial_outcome_when_a_task_fails() -> None:
    """One task fails CHECK, another passes: history outcome is 'partial'."""
    import history as _history

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
        entries, _ = _history.read_history(tmp)
        assert entries[-1]["outcome"] == "partial"
        assert result["failed_tasks"][0]["path"] == "bad.py"
