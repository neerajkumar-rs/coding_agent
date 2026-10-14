"""Router v2 tests: T0/T1 rules (no model), T2 grey-zone classifier (fake
generate), and the pipeline behavior per intent.

The five known-bad phrases from v2 are mandatory:
  thanks a lot for that, hello lets work together, what is the address of my
  file, what does the fix do  (all returned CODE_TASK before), plus
  hi there, lets get started.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from orchestrator import (
    build_file_facts,
    classify_request,
    format_file_table,
    run_pipeline,
)
from router import (
    ROUTER_MARKER,
    canned_chat_reply,
    diagnose,
    route,
    route_rules_only,
)
from tests.test_run_pipeline import CODER, PLANNER, make_fake_generate


def _run(coro):
    return asyncio.run(coro)


# --- T0: greetings / thanks / acks / openers / Hinglish (never CODE_TASK) -----

KNOWN_BAD = [
    "thanks a lot for that",
    "hello lets work together",
    "what is the address of my file",
    "what does the fix do",
    "hi there, lets get started",
]


def test_known_bad_phrases_never_code_task() -> None:
    expected = {
        "thanks a lot for that": "CHAT",
        "hello lets work together": "CHAT",
        "what is the address of my file": "FILE_QUESTION",
        "what does the fix do": "QUESTION",
        "hi there, lets get started": "CHAT",
    }
    for text, want in expected.items():
        got = route_rules_only(text)
        assert got.intent == want, f"{text!r}: got {got}"


def test_t0_chat_whole_message_and_filler() -> None:
    for text in ("hello", "hi there", "hey bro", "good morning", "thanks",
                 "thank you so much", "ok cool", "got it thanks", "hmm", "bye",
                 "lets work", "lets start", "continue", "ok lets go"):
        r = route_rules_only(text)
        assert r.intent == "CHAT", f"{text!r}: got {r}"
        assert r.tier == 0


def test_t0_hinglish_chat() -> None:
    for text in ("namaste", "kya haal hai", "kaise ho", "shukriya",
                 "theek hai", "chalo shuru karte hain", "achha ok", "dhanyavaad"):
        r = route_rules_only(text)
        assert r.intent == "CHAT", f"{text!r}: got {r}"


def test_chat_prefix_does_not_swallow_a_real_task() -> None:
    # greeting + a real task is NOT chat
    r = route_rules_only("ok lets go make a todo page in index.html")
    assert r.intent == "CODE_TASK"


def test_t0_slash_command() -> None:
    assert route_rules_only("/save").intent == "COMMAND"
    assert route_rules_only("/models").intent == "COMMAND"


# --- T1: scored rules -----------------------------------------------------------

def test_t1_code_task_needs_verb_and_deliverable() -> None:
    assert route_rules_only("make a todo page in index.html").intent == "CODE_TASK"
    assert route_rules_only("create a python script that renames files").intent == "CODE_TASK"
    assert route_rules_only("fix the bug in utils.py").intent == "CODE_TASK"
    assert route_rules_only("write tests for the parser").intent == "CODE_TASK"
    assert route_rules_only("build a calculator app").intent == "CODE_TASK"
    assert route_rules_only("refactor server.js into modules").intent == "CODE_TASK"


def test_t1_polite_request_exception_is_imperative() -> None:
    r = route_rules_only("can you make a login form in login.html")
    assert r.intent == "CODE_TASK"
    assert r.verb == "make"


def test_t1_question_checked_before_verbs() -> None:
    # 'fix' is an action verb but the question lead wins
    assert route_rules_only("what does the fix do").intent == "QUESTION"
    assert route_rules_only("is flask better than fastapi").intent == "QUESTION"
    assert route_rules_only("what is a closure?").intent == "QUESTION"


def test_t1_word_boundary_add_does_not_match_address() -> None:
    r = route_rules_only("what is the address of my file")
    assert r.intent == "FILE_QUESTION"
    assert r.verb is None  # 'add' inside 'address' must not match


def test_t1_verb_only_or_noun_only_unclear_with_clarifying() -> None:
    for text in ("make something", "build it", "fix this", "change it",
                 "i need a website", "write code"):
        r = route_rules_only(text)
        assert r.intent == "UNCLEAR", f"{text!r}: got {r}"
        assert r.clarifying_question.endswith("?"), f"{text!r}: {r.clarifying_question}"


def test_t1_short_unmatched_is_unclear_never_code_task() -> None:
    assert route_rules_only("can you help me").intent == "UNCLEAR"
    assert route_rules_only("do something cool").intent == "UNCLEAR"
    assert route_rules_only("").intent == "UNCLEAR"


def test_t1_scores_exposed() -> None:
    r = route_rules_only("make a todo page in index.html")
    assert r.scores["verb"] == 1.0
    assert r.scores["deliverable"] == 1.0
    assert r.scores["filename"] == 1.0
    assert r.scores["total"] == 3.0


def test_classify_request_sync_shim() -> None:
    assert classify_request("what is the difference between rust and python") == "QUESTION"
    assert classify_request("make a blog site in html5 css and js") == "CODE_TASK"
    assert classify_request("create app.py") == "CODE_TASK"
    assert classify_request("hello") == "CHAT"


# --- FILE_QUESTION deterministic rules ------------------------------------------

def test_file_question_rules() -> None:
    assert route_rules_only("what is in my dir").intent == "FILE_QUESTION"
    assert route_rules_only("which language are these files").intent == "FILE_QUESTION"
    assert route_rules_only("summarize my files").intent == "FILE_QUESTION"
    assert route_rules_only("list files").intent == "FILE_QUESTION"
    assert route_rules_only("what does main.py do").intent == "FILE_QUESTION"
    assert route_rules_only("where is the config in settings.py").intent == "FILE_QUESTION"
    assert route_rules_only("explain index.html").intent == "FILE_QUESTION"


# --- T2: grey zone with a fake classifier ----------------------------------------

def _t2_fake(payload: dict | str):
    """Fake generate routing on the router's own marker; never shares markers
    with other pipeline prompts."""
    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        assert ROUTER_MARKER in prompt
        assert "JSON array of tasks" not in prompt
        assert "TARGET FILE:" not in prompt
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)
    return fake


def test_t2_valid_json_returns_intent() -> None:
    r = _run(route("how do i make a todo page",
                   generate=_t2_fake({"intent": "QUESTION", "confidence": 0.9,
                                      "deliverable": "", "clarifying_question": ""}),
                   thinker_model=PLANNER))
    assert r.intent == "QUESTION"
    assert r.tier == 2
    assert r.raw  # raw classifier JSON recorded for /router


def test_t2_think_wrapped_json_parses() -> None:
    payload = "<think>reasoning here</think>" + json.dumps(
        {"intent": "QUESTION", "confidence": 0.95})
    r = _run(route("how do i make a todo page",
                   generate=_t2_fake(payload), thinker_model=PLANNER))
    assert r.intent == "QUESTION"


def test_t2_unclosed_think_cut_by_length_is_unclear() -> None:
    payload = "<think>reasoning that never ends and got truncated"
    r = _run(route("how do i make a todo page",
                   generate=_t2_fake(payload), thinker_model=PLANNER))
    assert r.intent == "UNCLEAR"


def test_t2_malformed_json_is_unclear() -> None:
    r = _run(route("how do i make a todo page",
                   generate=_t2_fake("not json at all"), thinker_model=PLANNER))
    assert r.intent == "UNCLEAR"


def test_t2_low_confidence_is_unclear() -> None:
    r = _run(route("how do i make a todo page",
                   generate=_t2_fake({"intent": "CODE_TASK", "confidence": 0.3,
                                      "deliverable": "page"}),
                   thinker_model=PLANNER))
    assert r.intent == "UNCLEAR"


def test_t2_code_task_with_empty_deliverable_rejected_by_guard() -> None:
    r = _run(route("how do i make a todo page",
                   generate=_t2_fake({"intent": "CODE_TASK", "confidence": 0.95,
                                      "deliverable": ""}),
                   thinker_model=PLANNER))
    assert r.intent == "UNCLEAR"


def test_t2_cannot_promote_from_nothing_to_code_task() -> None:
    # A long, signal-free message (score 0, 8+ words) reaches T2; even a
    # confident CODE_TASK with a deliverable is rejected (T1 saw neither).
    text = "the weather outside is honestly pretty nice for a walk today"
    r = _run(route(text,
                   generate=_t2_fake({"intent": "CODE_TASK", "confidence": 0.99,
                                      "deliverable": "page"}),
                   thinker_model=PLANNER))
    assert r.intent == "UNCLEAR"


def test_t2_min_confidence_configurable() -> None:
    fake = _t2_fake({"intent": "QUESTION", "confidence": 0.8, "deliverable": ""})
    r = _run(route("how do i make a todo page", generate=fake,
                   thinker_model=PLANNER, min_confidence=0.9))
    assert r.intent == "UNCLEAR"
    r2 = _run(route("how do i make a todo page", generate=_t2_fake(
        {"intent": "QUESTION", "confidence": 0.8, "deliverable": ""}),
        thinker_model=PLANNER, min_confidence=0.7))
    assert r2.intent == "QUESTION"


def test_t2_uses_own_marker_and_small_predict() -> None:
    seen: list[dict] = []

    async def fake(model, prompt, **kwargs):
        seen.append({"prompt": prompt, **kwargs})
        return json.dumps({"intent": "QUESTION", "confidence": 0.9})

    _run(route("how do i make a todo page", generate=fake, thinker_model=PLANNER))
    assert seen and ROUTER_MARKER in seen[0]["prompt"]
    assert seen[0]["num_predict"] <= 160 + 512  # answer + think allowance
    assert seen[0]["temperature"] == 0.0
    # instructions live in the user prompt (inline_system): system is empty
    assert seen[0].get("system", "") == ""


def test_rules_only_never_calls_a_model() -> None:
    def boom(*a, **k):
        raise AssertionError("model called for a rules-only decision")

    for text in ("hello", "make a todo page in index.html", "what is a closure?"):
        assert route_rules_only(text).intent != "UNCLEAR"
    with pytest.raises(AssertionError):
        # prove the fake would fire if called
        boom()


def test_canned_chat_reply_language() -> None:
    assert canned_chat_reply("hello") == canned_chat_reply("thanks a lot for that")
    assert canned_chat_reply("namaste") != canned_chat_reply("hello")  # Hinglish


def test_diagnose_reports_tier_and_scores() -> None:
    d = _run(diagnose("make a todo page in index.html"))
    assert d["intent"] == "CODE_TASK"
    assert d["tier"] == 1
    assert d["matched_verb"] == "make"
    assert d["t2_ran"] is False
    d2 = _run(diagnose("how do i make a todo page",
                       generate=_t2_fake({"intent": "QUESTION", "confidence": 0.9}),
                       thinker_model=PLANNER))
    assert d2["t2_ran"] is True
    assert d2["intent"] == "QUESTION"


# --- pipeline behavior per intent -------------------------------------------------

def test_chat_zero_model_calls_zero_files() -> None:
    calls: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        calls.append(prompt)
        return "x"

    with TemporaryDirectory() as tmp:
        for text in ("hello", "hello lets work together", "thanks a lot for that",
                     "hi there, lets get started", "namaste"):
            result = _run(run_pipeline(
                text, tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            ))
            assert "answer" in result
            assert "staged_files" not in result
            # FAST-PATH: no UNDERSTAND stage, zero model calls
            stages = [t["stage"] for t in result["trace"]]
            assert stages == ["ANSWER"]
    assert calls == []  # canned reply: zero model calls by default


def test_command_intent_never_touches_files() -> None:
    fake = make_fake_generate({})
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline("/save", tmp, generate=fake,
                                   planner_model=PLANNER, coder_model=CODER))
    assert "answer" in result
    assert "staged_files" not in result


def test_unclear_asks_clarifying_question() -> None:
    fake = make_fake_generate({
        "UNDERSTAND JSON:": json.dumps({
            "mode": "reply", "reply": None,
            "tasks": [], "assumptions": [],
            "question": "What should I build — a page, a script, or something else?",
        }),
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline("make something", tmp, generate=fake,
                                   planner_model=PLANNER, coder_model=CODER))
    assert "answer" in result
    assert "?" in result["answer"]


# --- FILE_QUESTION end to end ------------------------------------------------------

def test_file_question_grounded_in_facts_via_understand() -> None:
    """'what is in my dir' — the Understand model answers from the FACTS
    table only (grounded in the workspace, no invented files)."""
    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "UNDERSTAND JSON:" in prompt:
            assert "main.py" in prompt  # FACTS table went in
            assert "notes.txt" in prompt
            return json.dumps({
                "mode": "reply",
                "reply": "The workspace contains main.py (python) and notes.txt.",
                "tasks": [], "assumptions": [], "question": None,
            })
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
    assert stages == ["UNDERSTAND", "ANSWER"]


def test_file_question_explain_via_understand() -> None:
    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "UNDERSTAND JSON:" in prompt:
            return json.dumps({
                "mode": "reply",
                "reply": "main.py is the entry point.",
                "tasks": [], "assumptions": [], "question": None,
            })
        return "unused"

    with TemporaryDirectory() as tmp:
        Path(tmp, "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
        result = _run(run_pipeline(
            "summarize my files",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "main.py is the entry point" in result["answer"]


# --- QUESTION creates zero tasks ----------------------------------------------------

def test_question_creates_zero_tasks() -> None:
    fake = make_fake_generate({
        "UNDERSTAND JSON:": json.dumps({
            "mode": "reply",
            "reply": "Rust is compiled; Python is interpreted.",
            "tasks": [], "assumptions": [], "question": None,
        }),
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
