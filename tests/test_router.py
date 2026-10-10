"""Phase 2 tests: rules-only router (QUESTION / FILE_QUESTION / CODE_TASK / UNCLEAR)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from orchestrator import (
    build_file_facts,
    classify_request,
    format_file_table,
    run_pipeline,
)
from tests.test_run_pipeline import CODER, PLANNER, make_fake_generate


def _run(coro):
    return asyncio.run(coro)


# --- classify rules -------------------------------------------------------

def test_classify_file_question() -> None:
    assert classify_request("what is in my dir") == "FILE_QUESTION"
    assert classify_request("which language are these files") == "FILE_QUESTION"
    assert classify_request("summarize my files") == "FILE_QUESTION"
    assert classify_request("list files") == "FILE_QUESTION"


def test_classify_code_task_verbs_and_files() -> None:
    assert classify_request("make a blog site in html css js") == "CODE_TASK"
    assert classify_request("create app.py") == "CODE_TASK"
    assert classify_request("fix the login bug") == "CODE_TASK"


def test_classify_question_and_unclear() -> None:
    assert classify_request("tell me the difference between rust and python") == "QUESTION"
    assert classify_request("what is a closure?") == "QUESTION"
    assert classify_request("hmm") == "UNCLEAR"
    assert classify_request("") == "UNCLEAR"


# --- FILE_QUESTION end to end ----------------------------------------------

def test_file_question_returns_real_table_no_planner_call() -> None:
    calls: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2):
        calls.append(model)
        return "some answer"

    with TemporaryDirectory() as tmp:
        Path(tmp, "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
        Path(tmp, "notes.txt").write_text("hello\nworld\n", encoding="utf-8")
        result = _run(run_pipeline(
            "what is in my dir",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "answer" in result
    assert "main.py" in result["answer"]
    assert "notes.txt" in result["answer"]
    assert "python" in result["answer"]  # language column
    # No planner call (FILE_QUESTION without explain/summarize = no model call)
    assert calls == []
    stages = [t["stage"] for t in result["trace"]]
    assert stages == ["ROUTE", "ANSWER"]


def test_file_question_explain_makes_one_coder_call() -> None:
    calls: list[tuple[str, str]] = []

    async def fake(model, prompt, system="", temperature=0.2):
        calls.append((model, prompt))
        return "<think>x</think>main.py is the entry point."

    with TemporaryDirectory() as tmp:
        Path(tmp, "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
        result = _run(run_pipeline(
            "summarize my files",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert len(calls) == 1
    model, prompt = calls[0]
    assert model == CODER
    assert "Do not invent files" in prompt
    assert "main.py is the entry point" in result["answer"]
    assert "main.py" in result["answer"]  # deterministic table included


# --- QUESTION creates zero tasks --------------------------------------------

def test_question_creates_zero_tasks() -> None:
    fake = make_fake_generate({"QUESTION:": "Rust is compiled; Python is interpreted."})
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "tell me the difference between rust and python",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "answer" in result
    assert "staged_files" not in result
    assert "failed_tasks" not in result
    assert not list((Path(tmp) / ".agent_staging").rglob("*"))


# --- UNCLEAR asks one clarifying question -----------------------------------

def test_unclear_asks_clarifying_question() -> None:
    fake = make_fake_generate({})
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline("hmm", tmp, generate=fake,
                                   planner_model=PLANNER, coder_model=CODER))
    assert "answer" in result
    assert "?" in result["answer"]


# --- facts helpers -----------------------------------------------------------

def test_build_file_facts_skips_junk_dirs() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "app.py").write_text("X = 1\n", encoding="utf-8")
        (root / ".agent_runs").mkdir()
        (root / ".agent_runs" / "state.json").write_text("{}", encoding="utf-8")
        (root / "node_modules").mkdir()
        (root / "node_modules" / "pkg.js").write_text("x", encoding="utf-8")
        facts = build_file_facts(root)
        names = [f["name"] for f in facts["files"]]
        assert names == ["app.py"]
        assert "node_modules" not in facts["tree"]
        assert ".agent_runs" not in facts["tree"]


def test_format_file_table_columns() -> None:
    facts = {"files": [{"name": "a.py", "language": "python", "lines": 10, "size": 200}]}
    table = format_file_table(facts)
    assert "FILE" in table and "LANGUAGE" in table and "LINES" in table and "SIZE" in table
    assert "a.py" in table and "python" in table and "10" in table
