"""Understand eval gate (offline): the eval set drives the fast paths and the
pipeline's mode dispatch with a scripted Understand fake.

Gates: fast-path lines score 100% with ZERO model calls; reply/question lines
never create files; the marker never collides with other pipeline markers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from understand import UNDERSTAND_MARKER, fast_path

EVAL_PATH = Path(__file__).parent / "understand_eval.jsonl"


def _load() -> list[dict]:
    lines = []
    for raw in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            lines.append(json.loads(raw))
    return lines


def test_eval_set_size() -> None:
    assert len(_load()) >= 60


def test_fast_path_lines_score_100_percent() -> None:
    """Every fast_reply/fast_command line: fast path hits it, zero LLM calls."""
    calls: list[str] = []

    async def fake(model, prompt, **kwargs):
        calls.append(prompt)
        return "x"

    hits = 0
    total = 0
    for entry in _load():
        if entry["expected_mode"] not in ("fast_reply", "fast_command"):
            continue
        total += 1
        got = fast_path(entry["text"])
        want = "CHAT" if entry["expected_mode"] == "fast_reply" else "COMMAND"
        if got == want:
            hits += 1
        else:
            print(f"MISS: {entry['text']!r} -> {got} (want {want})")
    assert total > 0
    assert hits == total  # 100% on offline fast-path fixtures
    assert calls == []


def test_marker_never_collides() -> None:
    """UNDERSTAND JSON: must not appear in other pipeline markers' strings."""
    for other in ("JSON array of tasks", "TARGET FILE:", "satisfies_request",
                  "Describe only features visible", "QUESTION:", "ROUTER INTENT JSON:"):
        assert UNDERSTAND_MARKER != other
        assert UNDERSTAND_MARKER not in other


# --- pipeline-level dispatch (scripted Understand) ----------------------------

def _run(coro):
    return asyncio.run(coro)


def _understand_payload(mode: str, reply=None, tasks=None, question=None):
    return json.dumps({
        "mode": mode, "reply": reply, "tasks": tasks or [],
        "assumptions": [], "question": question,
    })


def test_reply_lines_create_zero_files() -> None:
    """Every expected_mode=reply line: Understand replies, nothing staged."""
    from orchestrator import run_pipeline

    for entry in _load():
        if entry["expected_mode"] != "reply":
            continue

        async def fake(model, prompt, system="", temperature=0.2, **kw):
            if UNDERSTAND_MARKER in prompt:
                return _understand_payload("reply", reply="Grounded answer.")
            return ""

        with TemporaryDirectory() as tmp:
            result = _run(run_pipeline(
                entry["text"], tmp, generate=fake,
                planner_model="P", coder_model="C",
            ))
            assert "staged_files" not in result, entry["text"]
            assert not list((Path(tmp) / ".agent_staging").rglob("*")), entry["text"]


def test_pending_question_merge_never_two_questions() -> None:
    """A pending question + a vague answer proceeds with assumptions; a second
    generic question is never produced for one request."""
    from agent_core.run_state import ConversationState
    from orchestrator import run_pipeline

    with TemporaryDirectory() as tmp:
        convo = ConversationState.load(tmp)
        convo.record_turn("vague", "question", "asked",
                          pending_question="Which file should I create?")

        seen_prompts: list[str] = []

        async def fake(model, prompt, system="", temperature=0.2, **kw):
            if UNDERSTAND_MARKER in prompt:
                seen_prompts.append(prompt)
                # The pending question is IN the message context: proceed.
                return _understand_payload(
                    "code",
                    tasks=[{"path": "app.py", "action": "create",
                            "description": "Python application implementing the requested calculator feature fully and correctly",
                            "uses_last_result": False}],
                )
            return "```python\nX = 1\n```"

        result = _run(run_pipeline(
            "i want answer", tmp, generate=fake,
            planner_model="P", coder_model="C",
        ))
        assert seen_prompts, "Understand never called"
        # the pending question was merged into the message
        assert "PENDING QUESTION" in seen_prompts[0]
        # and the pipeline proceeded to code (no second generic question)
        assert result.get("staged_files") == ["app.py"]
        assert ConversationState.load(tmp).pending_question == ""


def test_graceful_reply_on_total_understand_failure() -> None:
    from orchestrator import run_pipeline

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        return "certainly not json"  # never parseable

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js", tmp, generate=fake,
            planner_model="P", coder_model="C",
        ))
        assert "couldn't understand" in result["answer"]
        assert "staged_files" not in result
