"""Pipeline-level behavior tests (the agent loop decides; the rules router is
deleted). Facts helpers live in orchestrator and are covered here."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from orchestrator import (
    build_file_facts,
    format_file_table,
    run_pipeline,
)
from tests.test_run_pipeline import CODER, PLANNER, make_fake_generate


def _run(coro):
    return asyncio.run(coro)


# --- pipeline behavior: greetings, slash commands, vague asks -------------------

def test_chat_zero_files_via_one_reply_step() -> None:
    """Greetings/chit-chat: ONE loop step (the model's reply action) — no
    regex fast path, no canned text, no files."""
    calls: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        calls.append(prompt)
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps({"action": "reply", "text": "Hello! What are we building?"})
        return "x"

    with TemporaryDirectory() as tmp:
        for text in ("hello", "hello lets work together", "thanks a lot for that",
                     "hi there, lets get started", "namaste"):
            result = _run(run_pipeline(
                text, tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            ))
            assert "answer" in result
            assert "staged_files" not in result
            stages = [t["stage"] for t in result["trace"]]
            assert stages == ["LOOP"]  # one loop step, the reply
    # one model call per greeting (the loop's single reply step)
    assert len(calls) == 5


def test_command_intent_never_touches_files() -> None:
    fake = make_fake_generate({})
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline("/save", tmp, generate=fake,
                                   planner_model=PLANNER, coder_model=CODER))
    assert "answer" in result
    assert "staged_files" not in result


def test_vague_request_needs_human_or_reply_never_files() -> None:
    """A vague fragment: the loop either answers with a reply step or runs out
    of steps -> needs_human. Never files."""
    fake = make_fake_generate({
        "CLASSIFY MODE:": "reply",
        "REPLY:": "I can do that — which file should I change?",
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline("make something", tmp, generate=fake,
                                   planner_model=PLANNER, coder_model=CODER))
    assert "answer" in result or result.get("status") == "needs_human"
    assert "staged_files" not in result or result["staged_files"] == []


# --- reply path: grounded via read-only tools, no files ---------------------------

def test_file_question_grounded_via_list_then_reply() -> None:
    """'what is in my dir' — the loop lists the files (deterministic table)
    then replies from that observation."""
    actions = iter([
        {"action": "list_files"},
        {"action": "reply", "text": "The workspace contains main.py (python) and notes.txt."},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(actions, {"action": "finish"}))
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
    stages = [t["stage"] for t in result["trace"]]
    assert stages == ["LOOP"]


def test_file_question_explain_via_read_then_reply() -> None:
    actions = iter([
        {"action": "read_file", "path": "main.py"},
        {"action": "reply", "text": "main.py is the entry point."},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(actions, {"action": "finish"}))
        return "unused"

    with TemporaryDirectory() as tmp:
        Path(tmp, "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
        result = _run(run_pipeline(
            "summarize my files",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "main.py is the entry point" in result["answer"]


def test_question_creates_zero_tasks() -> None:
    fake = make_fake_generate({
        "CLASSIFY MODE:": "reply",
        "REPLY:": "Rust is compiled; Python is interpreted.",
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "tell me the difference between rust and python",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "answer" in result
    assert "staged_files" not in result
    assert "failed_tasks" not in result
    assert not list((Path(tmp) / ".agent_staging").rglob("*"))


# --- facts helpers --------------------------------------------------------------------

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


# --- the deleted rules router: grep-level guarantee -------------------------------------

def test_rules_router_is_gone() -> None:
    """router.py, its markers and the canned sentences are deleted; the agent
    loop decides (grep-level check on the source tree)."""
    root = Path(__file__).parent.parent
    assert not (root / "router.py").exists()
    src = (root / "orchestrator.py").read_text(encoding="utf-8")
    for gone in ("CLASSIFY MODE:", "VERIFY SIDE EFFECT:", "ROUTER INTENT JSON:",
                 "route_rules_only", "canned_chat_reply"):
        assert gone not in src
