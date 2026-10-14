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
        raise RuntimeError("model 'some-missing-model:1b' not found")

    with pytest.raises(RuntimeError, match="not found"):
        _run(classify_mode("list files", generate=fake, file_names=[], last_turn=""))


def test_classifier_prompt_has_no_eval_phrases() -> None:
    """The eval phrases are held back: they must NOT appear in the prompt."""
    from understand import CLASSIFY_PROMPT
    for entry in _load():
        text = entry["text"].strip('"').lower()
        if len(text) > 15:  # only meaningful phrases, not "hi"
            assert text not in CLASSIFY_PROMPT.lower(), entry["text"]


# --- hold-out set (never used as prompt examples) -----------------------------

HOLDOUT_PATH = Path(__file__).parent / "classify_holdout.jsonl"


def _load_holdout() -> list[dict]:
    return [
        json.loads(raw)
        for raw in HOLDOUT_PATH.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]


def test_holdout_set_size_and_shape() -> None:
    entries = _load_holdout()
    assert len(entries) >= 60
    for entry in entries:
        assert entry["expected"] in ("fast", "reply", "code", "both"), entry["text"]
        assert "files_created" in entry, entry["text"]
        if entry["expected"] == "reply":
            assert entry["files_created"] == 0, entry["text"]


def test_holdout_contains_the_four_evidence_messages() -> None:
    texts = {e["text"] for e in _load_holdout()}
    evidence = {
        "make readme.md": "code",
        "let list all the file in this dir": "reply",
        "ok use cli tool and list all file and and there types": "reply",
        "what is in index.html use cat to see": "reply",
    }
    for text, label in evidence.items():
        assert text in texts, text
        entry = next(e for e in _load_holdout() if e["text"] == text)
        assert entry["expected"] == label, text


def test_holdout_phrases_never_in_prompt() -> None:
    """Hold-out phrases are NOT prompt examples (no teaching to the test)."""
    from understand import CLASSIFY_PROMPT
    for entry in _load_holdout():
        text = entry["text"].strip('"').lower()
        if len(text) > 15:
            assert text not in CLASSIFY_PROMPT.lower(), entry["text"]


def test_holdout_fast_lines_hit_the_fast_path() -> None:
    hits = total = 0
    for entry in _load_holdout():
        if entry["expected"] != "fast":
            continue
        total += 1
        if fast_path(entry["text"]) is not None:
            hits += 1
        else:
            print(f"MISS: {entry['text']!r}")
    assert total > 0 and hits == total


def test_classify_prompt_has_no_previous_result_text() -> None:
    """No 'Last turn' line and no previous-result text in the classify prompt —
    recent 'Built: index.html' results biased the label toward `code`."""
    from understand import CLASSIFY_PROMPT
    assert "last turn" not in CLASSIFY_PROMPT.lower()
    seen: list[str] = []

    async def fake(model, prompt, **kw):
        seen.append(prompt)
        return "reply"

    _run(classify_mode(
        "list the files", generate=fake, file_names=["index.html"],
    ))
    assert "index.html" in seen[0]          # file names still go in
    assert "last turn" not in seen[0].lower()


def test_classify_prompt_under_900_chars() -> None:
    """The classify prompt stays tiny (first-call latency on a 4GB card)."""
    from understand import CLASSIFY_PROMPT
    built = (
        CLASSIFY_PROMPT
        .replace("{files}", "index.html")
        .replace("{text}", "x")
        .replace("{CLASSIFY_MARKER}", "CLASSIFY MODE:")
    )
    assert len(built) < 900, len(built)


# --- verify pass (side-effect confirmation before any file work) --------------

def test_verify_pass_schema_and_budget() -> None:
    """verify_side_effect: enum yes|no, temperature 0, num_predict <= 32
    (6 truncated the enum token on the real gateway; 32 is the fixed budget)."""
    from understand import VERIFY_SCHEMA, verify_side_effect
    seen: list[dict] = {}

    async def fake(model, prompt, **kw):
        seen.update(kw)
        return "yes"

    assert _run(verify_side_effect("make notes.md", generate=fake)) is True
    assert seen.get("num_predict", 99) <= 32
    assert seen.get("temperature") == 0.0
    assert json.dumps(VERIFY_SCHEMA).count("yes") == 1  # enum yes|no


def test_verify_pass_parses_yes_no_and_junk() -> None:
    from understand import verify_side_effect

    async def say(payload):
        async def fake(model, prompt, **kw):
            return payload
        return await verify_side_effect("m", generate=fake)

    assert _run(say("no")) is False
    assert _run(say('{"side_effect": "no"}')) is False
    assert _run(say('{"side_effect": "yes"}')) is True
    assert _run(say("<think>hmm</think>yes")) is True
    assert _run(say("banana")) is True  # junk keeps the classifier's label


# --- reply step -----------------------------------------------------------------

def test_reply_prompt_has_python_built_facts_and_no_shell_rule() -> None:
    """The reply prompt carries the FACTS block with KB sizes computed in
    Python and the 'never explain shell commands' instruction."""
    seen: list[str] = []

    async def fake(model, prompt, **kw):
        seen.append(prompt)
        return "The workspace has app.py."

    facts = {
        "files": [
            {"name": "app.py", "language": "python", "lines": 3, "bytes": 2048},
            {"name": "Dockerfile", "language": "text", "lines": 10, "size": 512},
        ],
    }
    block = build_facts_block(facts)
    _run(reply_step("list all the files", generate=fake, facts_block=block))
    prompt = " ".join(seen[0].split())  # normalize wrapped lines
    assert "app.py" in prompt
    assert "2.0 KB" in prompt  # computed in Python, not by the model
    assert "never explain how the user could do it with shell commands" in prompt
    # extensionless-file flags are computed from the file table (bug 7)
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


# --- pipeline level: verify downgrade, empty plan, invented-file rejection ----

def _fixture_workspace(tmp: str) -> None:
    """The acceptance fixture: one index.html containing the text File List."""
    Path(tmp, "index.html").write_text(
        "<!doctype html><html><body><h1>File List</h1></body></html>\n",
        encoding="utf-8",
    )


def test_verify_downgrades_code_to_reply_for_list_show_read() -> None:
    """A list/show/read ask misclassified as code is downgraded by the verify
    pass BEFORE anything is planned or written."""
    from orchestrator import run_pipeline
    from tests.test_run_pipeline import make_fake_generate

    fake = make_fake_generate({
        "CLASSIFY MODE:": "code",
        "VERIFY SIDE EFFECT:": "no",
        "REPLY:": "The workspace has index.html (html).",
    })
    with TemporaryDirectory() as tmp:
        _fixture_workspace(tmp)
        result = _run(run_pipeline(
            "let list all the file in this dir", tmp, generate=fake,
            planner_model="P", coder_model="C",
        ))
        assert "staged_files" not in result          # nothing created
        assert "index.html" in result["answer"]
        stages = [t["stage"] for t in result["trace"]]
        assert stages == ["CLASSIFY", "VERIFY", "ANSWER"], stages
        verify = next(t for t in result["trace"] if t["stage"] == "VERIFY")
        assert "downgraded" in verify["note"]
        # all three calls logged in the journal (classify + verify + answer)
        events = (Path(tmp) / ".agent_runs" / result["run_id"]
                  / "events.jsonl").read_text(encoding="utf-8")
        assert events.count('"event": "llm_call"') == 3
        assert '"event": "verify_downgrade"' in events


def test_verify_keeps_real_create_requests() -> None:
    """verify=yes keeps the code path: plan + code run, file staged."""
    from orchestrator import run_pipeline
    from tests.test_run_pipeline import make_fake_generate

    fake = make_fake_generate({
        "CLASSIFY MODE:": "code",
        "VERIFY SIDE EFFECT:": "yes",
        "PLAN TASKS JSON:": json.dumps({
            "tasks": [{"path": "readme.md", "action": "create",
                       "description": "Markdown readme describing this small workspace project clearly"}],
            "assumptions": [],
        }),
        "TARGET FILE: readme.md": "```markdown\n# Project\n\nHello.\n```",
        "satisfies_request": json.dumps(
            {"satisfies_request": True, "problems": [], "summary": "ok"}),
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make readme.md", tmp, generate=fake,
            planner_model="P", coder_model="C",
        ))
        assert result["staged_files"] == ["readme.md"]
        stages = [t["stage"] for t in result["trace"]]
        assert "VERIFY" in stages and "CODE" in stages


def test_empty_plan_with_reason_is_needs_human_never_reply() -> None:
    """tasks: [] + reason -> needs_human with the reason; NEVER a chat reply
    (a code/both turn must not end as a reply)."""
    from orchestrator import run_pipeline
    from tests.test_run_pipeline import make_fake_generate

    fake = make_fake_generate({
        "CLASSIFY MODE:": "code",
        "VERIFY SIDE EFFECT:": "yes",
        "PLAN TASKS JSON:": json.dumps(
            {"tasks": [], "reason": "no file was named or inferable"}),
        "REPLY:": "Which file should I change?",
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "change it", tmp, generate=fake,
            planner_model="P", coder_model="C",
        ))
        assert result.get("status") == "needs_human"
        assert result["staged_files"] == []
        assert "no file was named or inferable" in result["summary"]
        events = (Path(tmp) / ".agent_runs" / result["run_id"]
                  / "events.jsonl").read_text(encoding="utf-8")
        assert '"event": "plan_empty"' in events


def test_invented_index_html_plan_rejected_then_needs_human() -> None:
    """A plan whose only task is an unrelated invented index.html (request has
    no web/page wording) is rejected in code, retried once, then needs_human
    (a code/both turn never falls back to a chat reply)."""
    from orchestrator import run_pipeline

    plan_calls: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "CLASSIFY MODE:" in prompt:
            return "code"
        if "VERIFY SIDE EFFECT:" in prompt:
            return "yes"
        if "PLAN TASKS JSON:" in prompt:
            plan_calls.append(prompt)
            return json.dumps({
                "tasks": [{"path": "index.html", "action": "create",
                           "description": "A static page listing files in one hardcoded row"}],
                "assumptions": [],
            })
        if "FACTS:" in prompt:
            return "The workspace already has index.html."
        return ""

    with TemporaryDirectory() as tmp:
        _fixture_workspace(tmp)
        result = _run(run_pipeline(
            "ok use cli tool and list all file and and there types",
            tmp, generate=fake, planner_model="P", coder_model="C",
        ))
        assert len(plan_calls) == 2                      # one retry
        assert "rejected" in plan_calls[1]
        assert result["status"] == "needs_human"         # never a reply
        assert result["staged_files"] == []              # nothing created


def test_invented_file_rule_allows_named_and_web_requests() -> None:
    from orchestrator import invented_file_rejection

    task = [{"path": "index.html", "action": "create", "description": "x"}]
    # file name appears in the request -> fine
    assert invented_file_rejection(task, "create index.html holding recipes") is None
    # web/page wording -> inferred names are legitimate
    assert invented_file_rejection(task, "make me a web page") is None
    # multi-task plans are out of scope for this rule
    assert invented_file_rejection(task + [{"path": "style.css"}], "make a thing") is None
    # unrelated single invented file -> rejected
    assert invented_file_rejection(task, "list all the files here") is not None
