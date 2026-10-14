"""Classify-then-act eval gate (offline).

- Fast-path lines score 100% with ZERO model calls.
- The classifier output is always one of the three labels (constrained
  decoding + regex/JSON extraction, junk -> safe 'reply').
- A classifier call that ERRORS surfaces the real error (no rephrase msg).
- The reply prompt carries the Python-built FACTS block with KB sizes and
  the "never explain how to do it with shell commands" rule.
- reply never creates files; code plan has no question field; validation
  retry -> plain JSON -> needs_human; 'both' runs reply then plan.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from understand import (
    CLASSIFY_MARKER,
    PLAN_MARKER,
    PlanFailed,
    build_facts_block,
    classify_mode,
    fast_path,
    plan_tasks,
    reply_step,
)

EVAL_PATH = Path(__file__).parent / "classify_eval.jsonl"


def _load() -> list[dict]:
    return [
        json.loads(raw) for raw in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]


def test_eval_set_size() -> None:
    assert len(_load()) >= 60


def test_eval_set_uses_only_known_labels() -> None:
    for entry in _load():
        assert entry["expected"] in ("fast", "reply", "code", "both"), entry["text"]


def test_markers_never_collide() -> None:
    """CLASSIFY MODE: / PLAN TASKS JSON: must not collide with other markers."""
    for other in ("JSON array of tasks", "TARGET FILE:", "satisfies_request",
                  "Describe only features visible", "QUESTION:", "ROUTER INTENT JSON:",
                  "FACTS:", "MESSAGE:"):
        assert CLASSIFY_MARKER != other and CLASSIFY_MARKER not in other
        assert PLAN_MARKER != other and PLAN_MARKER not in other
    assert CLASSIFY_MARKER != PLAN_MARKER


# --- fast paths -----------------------------------------------------------------

def test_fast_path_lines_score_100_percent() -> None:
    hits = 0
    total = 0
    for entry in _load():
        if entry["expected"] != "fast":
            continue
        total += 1
        if fast_path(entry["text"]) is not None:
            hits += 1
        else:
            print(f"MISS: {entry['text']!r}")
    assert total > 0
    assert hits == total  # 100% on offline fast-path fixtures


# --- classifier -----------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def test_classifier_returns_only_the_three_labels() -> None:
    """Constrained decoding + extraction: only reply|code|both come out."""
    for payload, want in (
        ('{"mode": "reply"}', "reply"),
        ('{"mode": "code"}', "code"),
        ('{"mode": "both"}', "both"),
        ("reply", "reply"),
        ("<think>reasoning</think>code", "code"),
        ("banana banana banana", "reply"),  # junk -> safe default, never an error
    ):
        async def fake(model, prompt, **kw):
            assert CLASSIFY_MARKER in prompt
            return payload
        got = _run(classify_mode("x", generate=fake, file_names=[], last_turn=""))
        assert got == want, f"{payload!r} -> {got}"


def test_classifier_error_surfaces_the_real_error() -> None:
    """A classifier call that raises propagates — never a rephrase message."""
    async def fake(model, prompt, **kw):
        raise RuntimeError("model 'deepseek-r1:1.5b' not found")

    with pytest.raises(RuntimeError, match="not found"):
        _run(classify_mode("list files", generate=fake, file_names=[], last_turn=""))


def test_classifier_prompt_has_no_eval_phrases() -> None:
    """The eval phrases are held back: they must NOT appear in the prompt."""
    from understand import CLASSIFY_PROMPT
    for entry in _load():
        text = entry["text"].strip('"').lower()
        if len(text) > 15:  # only meaningful phrases, not "hi"
            assert text not in CLASSIFY_PROMPT.lower(), entry["text"]


# --- reply step -----------------------------------------------------------------

def test_reply_prompt_has_python_built_facts_and_no_shell_rule() -> None:
    """The reply prompt carries the FACTS block with KB sizes computed in
    Python and the 'never explain shell commands' instruction."""
    seen: list[str] = []

    async def fake(model, prompt, **kw):
        seen.append(prompt)
        return "The workspace has app.py."

    facts = {
        "files": [{"name": "app.py", "language": "python", "lines": 3, "bytes": 2048}],
        "flags": {"readme": False, "dockerfile": True},
    }
    block = build_facts_block(facts)
    _run(reply_step("list all the files", generate=fake, facts_block=block))
    prompt = " ".join(seen[0].split())  # normalize wrapped lines
    assert "app.py" in prompt
    assert "2.0 KB" in prompt  # computed in Python, not by the model
    assert "never explain how the user could do it with shell commands" in prompt
    assert "Dockerfile" in prompt or "dockerfile" in prompt


def test_reply_never_creates_files() -> None:
    """Pipeline level: a classify=reply turn writes nothing to staging."""
    from orchestrator import run_pipeline

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if CLASSIFY_MARKER in prompt:
            return "reply"
        if "FACTS:" in prompt:
            return "Grounded answer."
        return ""

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "list all the files", tmp, generate=fake,
            planner_model="P", coder_model="C",
        ))
        assert "staged_files" not in result
        assert not list((Path(tmp) / ".agent_staging").rglob("*"))


# --- plan step ------------------------------------------------------------------

def test_plan_has_no_question_field_and_records_assumptions() -> None:
    async def fake(model, prompt, **kw):
        assert PLAN_MARKER in prompt
        return json.dumps({
            "tasks": [{"path": "app.py", "action": "create",
                       "description": "a sufficiently long description for this task here",
                       "uses_last_result": False}],
            "assumptions": ["Chose app.py as the default entry point"],
        })

    result = _run(plan_tasks("make something", generate=fake, file_names=[]))
    assert len(result.tasks) == 1
    assert result.assumptions == ["Chose app.py as the default entry point"]
    # the schema has no question field at all
    from understand import PLAN_SCHEMA
    assert "question" not in json.dumps(PLAN_SCHEMA)


def test_plan_validation_retry_then_plain_json_then_needs_human() -> None:
    calls: list[str] = []

    async def fake_bad_then_good(model, prompt, **kw):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"tasks": [{"path": "your_code.py", "action": "create",
                                          "description": "short"}]})  # placeholder
        return json.dumps({"tasks": [{"path": "app.py", "action": "create",
                                      "description": "a sufficiently long description for this task here"}],
                           "assumptions": []})

    from orchestrator import _valid_plan_task
    result = _run(plan_tasks("make x", generate=fake_bad_then_good, file_names=[],
                             task_validator=_valid_plan_task))
    assert len(calls) == 2  # one retry with the errors appended
    assert "rejected" in calls[1]
    assert result.tasks[0]["path"] == "app.py"

    async def fake_always_bad(model, prompt, **kw):
        return "not json at all"

    with pytest.raises(PlanFailed):
        _run(plan_tasks("make x", generate=fake_always_bad, file_names=[]))


# --- 'both' runs reply then plan -------------------------------------------------

def test_both_runs_reply_then_plan() -> None:
    from orchestrator import run_pipeline

    order: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if CLASSIFY_MARKER in prompt:
            order.append("classify")
            return "both"
        if PLAN_MARKER in prompt:
            order.append("plan")
            return json.dumps({"tasks": [{"path": "app.py", "action": "create",
                                          "description": "a sufficiently long description for this task here"}],
                               "assumptions": []})
        if "FACTS:" in prompt:
            order.append("reply")
            return "app.py is the entry point."
        if "TARGET FILE: app.py" in prompt:
            order.append("code")
            return "```python\nX = 1\n```"
        if "satisfies_request" in prompt:
            return json.dumps({"satisfies_request": True, "problems": [], "summary": "ok"})
        return ""

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "explain app.py and then add a health endpoint",
            tmp, generate=fake, planner_model="P", coder_model="C",
        ))
    assert order.index("reply") < order.index("plan")  # reply printed first
    assert "classify" in order and "plan" in order and "code" in order
    assert result.get("reply") == "app.py is the entry point."
    assert result["staged_files"] == ["app.py"]
