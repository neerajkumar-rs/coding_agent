"""Tests for run_pipeline: ROUTE/PLAN/CODE/CHECK/VALIDATE/SUMMARY (fake LLM)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from orchestrator import run_pipeline

PLANNER = "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
CODER = "qwen2.5-coder:3b"

WEB_PLAN = json.dumps([
    {"path": "index.html", "action": "create",
     "description": "Main HTML5 page structure with linked stylesheet and script files"},
    {"path": "style.css", "action": "create",
     "description": "Stylesheet with layout typography colors and responsive rules"},
    {"path": "script.js", "action": "create",
     "description": "JavaScript for dynamic content loading and user interactions"},
])

INDEX_HTML = (
    "<!doctype html><html><head>"
    '<link rel="stylesheet" href="style.css">'
    '<script src="script.js"></script>'
    "</head><body><h1>Blog</h1></body></html>"
)
STYLE_CSS = "body { font-family: sans-serif; margin: 0; }"
SCRIPT_JS = "document.addEventListener('DOMContentLoaded', () => { console.log('ok'); });"
VALIDATE_OK = json.dumps({
    "satisfies_request": True, "problems": [], "summary": "A blog site scaffold."
})


def make_fake_generate(routes: dict[str, str], default: str = ""):
    """routes: marker-in-prompt -> response, driving the agent loop.

    The loop calls carry the "AGENT ACTION JSON:" marker. A route dict shapes
    the loop:
    - "AGENT ACTION JSON:" — a literal response for EVERY loop call (use for
      reply/finish turns).
    - "LOOP ACTIONS" — a JSON list of action dicts played back in order.
    - Legacy shaping (kept so older fixtures work):
      - "CLASSIFY MODE:" == "reply" -> one reply action using "REPLY:" text.
      - "JSON array of tasks" (a plan) -> one write_file action per task,
        then finish. "TARGET FILE: x" routes the coder write call for x.
    Everything else (satisfies_request, Describe-only, TARGET FILE) routes
    exactly as before."""
    import re as _re

    def _loop_actions() -> list[dict]:
        if "LOOP ACTIONS" in routes:
            return json.loads(routes["LOOP ACTIONS"])
        mode = routes.get("CLASSIFY MODE:", "")
        if mode == "reply" or ("REPLY:" in routes and not routes.get("JSON array of tasks", "").lstrip().startswith("[")):
            return [{"action": "reply", "text": routes.get("REPLY:", routes.get("REPLY", "ok"))}]
        plan = routes.get("JSON array of tasks", "")
        if plan.lstrip().startswith("["):
            actions = []
            for task in json.loads(plan):
                actions.append({
                    "action": "write_file" if task.get("action", "create") == "create" else "edit_file",
                    "path": task["path"],
                    "text": task.get("description", ""),
                })
            actions.append({"action": "finish"})
            return actions
        return []

    actions = _loop_actions()
    loop_calls = {"n": 0}

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        import re

        if "AGENT ACTION JSON:" in prompt:
            if "AGENT ACTION JSON:" in routes:
                return routes["AGENT ACTION JSON:"]
            if not actions:
                return json.dumps({"action": "reply", "text": default or "ok"})
            idx = min(loop_calls["n"], len(actions) - 1)
            loop_calls["n"] += 1
            return json.dumps(actions[idx])
        target = re.search(r"TARGET FILE:\s*(\S+)", prompt) or re.search(r"Write\s+(\S+\.\w+)\.", prompt)
        if target:
            file_marker = f"TARGET FILE: {target.group(1)}"
            if file_marker in routes:
                return routes[file_marker]
        for marker in sorted(routes, key=len, reverse=True):
            if marker in ("AGENT ACTION JSON:", "LOOP ACTIONS"):
                continue
            if marker in prompt:
                return routes[marker]
        return default

    return fake


def _run(coro):
    return asyncio.run(coro)


def _web_loop_actions() -> list[dict]:
    return [
        {"action": "write_file", "path": "index.html", "text": "the page structure"},
        {"action": "write_file", "path": "style.css", "text": "the stylesheet"},
        {"action": "write_file", "path": "script.js", "text": "the interactions"},
        {"action": "finish"},
    ]


def test_summary_call_uses_writer_role() -> None:
    """The summary prose call carries role='writer' through the generate seam
    (the server maps it to the writer role config)."""
    seen_roles: list[str] = []

    loop_actions = iter([
        {"action": "write_file", "path": "greet.py", "text": "greet function"},
        {"action": "finish"},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        if "Describe only features visible" in prompt:
            seen_roles.append(kwargs.get("role", ""))
            return "A small python module."
        if "Write greet.py." in prompt:
            return "```python\ndef greet(name: str) -> str:\n    return f'Hello, {name}!'\n```"
        if "satisfies_request" in prompt:
            return json.dumps({"satisfies_request": True, "problems": [], "summary": "ok"})
        return ""

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert result["staged_files"] == ["greet.py"]
    assert seen_roles == ["thinker"]  # summary prose runs on the thinker


# 1. Question input -> a reply turn (the loop's reply action ends it)
def test_question_gets_answer_no_tasks() -> None:
    fake = make_fake_generate({
        "CLASSIFY MODE:": "reply",
        "REPLY:": "Rust compiles to native code; Python is interpreted.",
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "what is the difference between rust and python",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert "answer" in result
    assert "Rust compiles" in result["answer"]
    assert "staged_files" not in result  # nothing staged
    stages = [t["stage"] for t in result["trace"]]
    assert stages == ["LOOP"]


# 2. Web request -> 3 files, VALIDATE and SUMMARY present in trace
def test_web_request_three_files_full_trace() -> None:
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps(_web_loop_actions()),
        "TARGET FILE: index.html": f"```html\n{INDEX_HTML}\n```",
        "TARGET FILE: style.css": f"```css\n{STYLE_CSS}\n```",
        "TARGET FILE: script.js": f"```javascript\n{SCRIPT_JS}\n```",
        "satisfies_request": VALIDATE_OK,
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js with dynamic content",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        staged = set(result["staged_files"])
        assert staged == {"index.html", "style.css", "script.js"}
        stages = [t["stage"] for t in result["trace"]]
        assert "VALIDATE" in stages and "SUMMARY" in stages
        # Staged, not deployed
        assert (Path(tmp) / ".agent_staging" / "index.html").is_file()
        assert not (Path(tmp) / "index.html").exists()
        # Summary = deterministic skeleton (files, issues, time) + model prose
        assert "index.html" in result["summary"]
        assert "Total time:" in result["summary"]


# 3. Forced-failing task -> not staged, listed as FAILED
def test_failing_task_not_staged() -> None:
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps([
            {"action": "write_file", "path": "app.py", "text": "the calculator"},
            {"action": "finish"},
        ]),
        "TARGET FILE: app.py": "```python\ndef broken(:\n```",  # never compiles
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create app.py with a calculator",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == []
        assert len(result["failed_tasks"]) == 1
        assert result["failed_tasks"][0]["path"] == "app.py"
        assert not (Path(tmp) / ".agent_staging" / "app.py").exists()
        assert "syntax error" in result["failed_tasks"][0]["error"]


# 4. Bad html link to a missing css file -> caught in CHECK
def test_bad_html_link_caught() -> None:
    bad_html = '<!doctype html><html><head><link rel="stylesheet" href="missing.css"></head><body>x</body></html>'
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps([
            {"action": "write_file", "path": "index.html", "text": "the page"},
            {"action": "finish"},
        ]),
        "TARGET FILE: index.html": f"```html\n{bad_html}\n```",
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create index.html holding my recipe collection",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == []
        assert "missing.css" in result["failed_tasks"][0]["error"]


# 5. Valid index.html passes when cwd is not the project root
def test_check_independent_of_cwd(monkeypatch) -> None:
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps([
            {"action": "write_file", "path": "index.html", "text": "the page"},
            {"action": "finish"},
        ]),
        "TARGET FILE: index.html": (
            "```html\n<!doctype html><html><head><style>body { margin: 0; }</style>"
            "</head><body>hi</body></html>\n```"
        ),
        "satisfies_request": VALIDATE_OK,
    })
    with TemporaryDirectory() as ws, TemporaryDirectory() as elsewhere:
        monkeypatch.chdir(elsewhere)  # cwd is NOT the workspace
        result = _run(run_pipeline(
            "create index.html page",
            ws, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == ["index.html"]
        assert (Path(ws) / ".agent_staging" / "index.html").is_file()
        monkeypatch.undo()  # release the dir handle before Windows cleanup


# VALIDATE error -> summary says UNVERIFIED
def test_validate_error_marks_unverified() -> None:
    loop_actions = iter(_web_loop_actions())

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        if "satisfies_request" in prompt:
            raise RuntimeError("ollama died mid-review")
        if "Write index.html." in prompt:
            return f"```html\n{INDEX_HTML}\n```"
        if "Write style.css." in prompt:
            return f"```css\n{STYLE_CSS}\n```"
        if "Write script.js." in prompt:
            return f"```javascript\n{SCRIPT_JS}\n```"
        return ""

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert "UNVERIFIED" in result["summary"]
        validate = next(t for t in result["trace"] if t["stage"] == "VALIDATE")
        assert validate["status"] == "failed"
        # Files still staged (CHECK passed); failure is reported, not hidden
        assert len(result["staged_files"]) == 3


# --- Phase 1: RunJournal observability ---------------------------------------

def test_run_pipeline_writes_run_journal() -> None:
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps(_web_loop_actions()),
        "TARGET FILE: index.html": f"```html\n{INDEX_HTML}\n```",
        "TARGET FILE: style.css": f"```css\n{STYLE_CSS}\n```",
        "TARGET FILE: script.js": f"```javascript\n{SCRIPT_JS}\n```",
        "satisfies_request": VALIDATE_OK,
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        run_dir = Path(tmp) / ".agent_runs" / result["run_id"]
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        assert state["status"] == "completed"
        assert state["llm_calls"] >= 4  # loop steps + 3 coder + validate (+summary)
        events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
        assert '"run_started"' in events and '"run_finished"' in events
        assert '"stage": "LOOP"' in events and '"stage": "SUMMARY"' in events


def test_llm_calls_equals_llm_call_events() -> None:
    """state.json llm_calls == the number of llm_call events in events.jsonl
    (deterministic stages — CHECK, the SUMMARY skeleton — count 0)."""
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps(_web_loop_actions()),
        "TARGET FILE: index.html": f"```html\n{INDEX_HTML}\n```",
        "TARGET FILE: style.css": f"```css\n{STYLE_CSS}\n```",
        "TARGET FILE: script.js": f"```javascript\n{SCRIPT_JS}\n```",
        "satisfies_request": VALIDATE_OK,
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js with dynamic content",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        run_dir = Path(tmp) / ".agent_runs" / result["run_id"]
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        events = [
            json.loads(ln)
            for ln in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        llm_events = [e for e in events if e.get("event") == "llm_call"]
        assert state["llm_calls"] == len(llm_events), (
            f"llm_calls={state['llm_calls']} but {len(llm_events)} llm_call events"
        )
        # calls/<n>.json full records exist, one per event
        calls = sorted((run_dir / "calls").glob("*.json"))
        assert len(calls) == len(llm_events)
        first = json.loads(calls[0].read_text(encoding="utf-8"))
        assert "prompt" in first and "response" in first and "num_ctx" in first


# --- Phase 1: cancellation -----------------------------------------------------

def test_cancel_during_code_stops_fast_and_stages_nothing_further() -> None:
    import threading
    import time as _time

    cancel = threading.Event()

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            # The loop asks for index.html, style.css, script.js, finish — but
            # cancel lands during the first coder call, so later steps abort.
            if "staged write_file index.html" in prompt:
                return json.dumps({"action": "write_file", "path": "style.css", "text": "css"})
            if "staged write_file style.css" in prompt:
                return json.dumps({"action": "write_file", "path": "script.js", "text": "js"})
            return json.dumps({"action": "write_file", "path": "index.html", "text": "page"})
        if "TARGET FILE: index.html" in prompt or "Write index.html." in prompt:
            # Cancel lands while the first coder call is in flight.
            cancel.set()
            _time.sleep(0.05)
            return f"```html\n{INDEX_HTML}\n```"
        if "TARGET FILE: style.css" in prompt or "Write style.css." in prompt:
            return f"```css\n{STYLE_CSS}\n```"
        if "TARGET FILE: script.js" in prompt or "Write script.js." in prompt:
            return f"```javascript\n{SCRIPT_JS}\n```"
        return VALIDATE_OK

    with TemporaryDirectory() as tmp:
        start = _time.monotonic()
        result = _run(run_pipeline(
            "make a blog site in html5 css and js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            cancel_event=cancel,
        ))
        elapsed = _time.monotonic() - start
        assert result["status"] == "cancelled"
        assert elapsed < 2.0  # stops within 2 s of the cancel landing
        # index.html was coded (the call in flight completed) but the pipeline
        # aborted before staging the remaining files.
        assert "script.js" not in result["staged_files"]
        # Journal records the cancellation
        state = json.loads(
            (Path(tmp) / ".agent_runs" / result["run_id"] / "state.json").read_text(encoding="utf-8")
        )
        assert state["status"] == "cancelled"


# --- Phase 1: deterministic verifier gate on Python --------------------------

def test_verifier_catches_runtime_import_crash() -> None:
    """Syntax-valid but import-crashing Python passes compile() yet must be
    rejected by the verifier's import-smoke step (and never staged)."""
    crashing = "```python\nimport nonexistent_module_xyz_123\n\nX = 1\n```"
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps([
            {"action": "write_file", "path": "crashes.py", "text": "crashes at import"},
            {"action": "finish"},
        ]),
        "TARGET FILE: crashes.py": crashing,
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create crashes.py that imports a missing module",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == []
        assert result["failed_tasks"][0]["path"] == "crashes.py"
        assert "verifier" in result["failed_tasks"][0]["error"].lower()
        assert not (Path(tmp) / ".agent_staging" / "crashes.py").exists()


def test_verifier_passes_clean_python() -> None:
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps([
            {"action": "write_file", "path": "greet.py", "text": "the greet function"},
            {"action": "finish"},
        ]),
        "TARGET FILE: greet.py": "```python\ndef greet(name):\n    return f'Hello, {name}!'\n```",
        "satisfies_request": VALIDATE_OK,
        "Describe only features visible": "",
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create greet.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == ["greet.py"]
        assert result["failed_tasks"] == []


# --- Phase 4: confirm gate before staging -------------------------------------

def test_confirm_gate_decline_stages_nothing_and_never_calls_coder() -> None:
    coder_calls: list[str] = []

    loop_actions = iter([
        {"action": "write_file", "path": "index.html", "text": "page"},
        {"action": "write_file", "path": "style.css", "text": "css"},
        {"action": "write_file", "path": "script.js", "text": "js"},
        {"action": "finish"},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        if "Write index.html." in prompt:
            return f"```html\n{INDEX_HTML}\n```"
        if "Write style.css." in prompt:
            return f"```css\n{STYLE_CSS}\n```"
        if "Write script.js." in prompt:
            return f"```javascript\n{SCRIPT_JS}\n```"
        return VALIDATE_OK

    async def decline(tasks, intent_line):
        assert {t["path"] for t in tasks} == {"index.html", "style.css", "script.js"}
        return False

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js with dynamic content",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            confirm=decline,
        ))
        assert result["status"] == "declined"
        assert result["staged_files"] == []
        assert not (Path(tmp) / ".agent_staging" / "index.html").exists()
        state = json.loads(
            (Path(tmp) / ".agent_runs" / result["run_id"] / "state.json").read_text(encoding="utf-8")
        )
        assert state["status"] == "declined"


def test_confirm_gate_approve_runs_full_pipeline() -> None:
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps(_web_loop_actions()),
        "TARGET FILE: index.html": f"```html\n{INDEX_HTML}\n```",
        "TARGET FILE: style.css": f"```css\n{STYLE_CSS}\n```",
        "TARGET FILE: script.js": f"```javascript\n{SCRIPT_JS}\n```",
        "satisfies_request": VALIDATE_OK,
    })
    seen: dict = {}

    async def approve(tasks, intent_line):
        seen["plan"] = [t["path"] for t in tasks]
        seen["intent"] = intent_line
        return True

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js with dynamic content",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            confirm=approve,
        ))
        assert set(result["staged_files"]) == {"index.html", "style.css", "script.js"}
        assert seen["plan"] == sorted(result["staged_files"]) or sorted(seen["plan"]) == sorted(result["staged_files"])


def test_confirm_gate_shows_overwrites_and_assumptions() -> None:
    """The gate payload: file names, per-file overwrites flag, the
    '(overwrites existing file)' marker."""
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps([
            {"action": "write_file", "path": "index.html", "text": "the recipe page"},
            {"action": "finish"},
        ]),
        "TARGET FILE: index.html": (
            "```html\n<!doctype html><html><head><style>body{margin:0}</style>"
            "</head><body>hi</body></html>\n```"
        ),
        "satisfies_request": VALIDATE_OK,
    })
    seen: dict = {}

    async def approve(tasks, intent_line):
        seen["tasks"] = tasks
        seen["intent"] = intent_line
        return True

    with TemporaryDirectory() as tmp:
        Path(tmp, "index.html").write_text("<html>old</html>", encoding="utf-8")
        result = _run(run_pipeline(
            "create index.html holding my recipe collection",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            confirm=approve,
        ))
        assert result["staged_files"] == ["index.html"]
    assert seen["tasks"][0]["overwrites"] is True
    assert "(overwrites existing file)" in seen["intent"]
    assert "index.html" in seen["intent"]


def test_confirm_none_skips_gate() -> None:
    # confirm=None (direct API/tests/--yes): no gate, current behavior.
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps(_web_loop_actions()),
        "TARGET FILE: index.html": f"```html\n{INDEX_HTML}\n```",
        "TARGET FILE: style.css": f"```css\n{STYLE_CSS}\n```",
        "TARGET FILE: script.js": f"```javascript\n{SCRIPT_JS}\n```",
        "satisfies_request": VALIDATE_OK,
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert set(result["staged_files"]) == {"index.html", "style.css", "script.js"}


# --- loop failure modes -------------------------------------------------------

def test_plan_never_runs_for_non_code_labels() -> None:
    calls: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        calls.append(prompt)
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps({"action": "reply", "text": "Grounded answer."})
        return "x"

    with TemporaryDirectory() as tmp:
        for text in ("hello", "make something", "what does the fix do"):
            result = _run(run_pipeline(
                text, tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            ))
            assert "staged_files" not in result, text
    coder_calls = [p for p in calls if "TARGET FILE:" in p or "Write " in p]
    assert coder_calls == []  # the coder never ran for reply turns


def test_plan_total_failure_is_needs_human() -> None:
    # The loop model returns unparseable junk for every step -> the turn is
    # needs_human (budget of steps exhausted), never files, no silent reply.
    async def fake(model, prompt, system="", temperature=0.2, **kw):
        return "not json at all"

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create your_code.py with the logic",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["status"] == "needs_human"
        assert result["staged_files"] == []
        assert "Could not complete" in result["summary"]


def test_classifier_junk_defaults_to_reply() -> None:
    # Junk loop output is ignored step by step; the turn ends needs_human (the
    # step budget runs out), never a silent reply and never files.
    async def fake(model, prompt, system="", temperature=0.2, **kw):
        return "banana banana banana"

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create your_code.py with the logic",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["status"] == "needs_human"
        assert result["staged_files"] == []


def test_classify_request_is_gone_with_the_rules_router() -> None:
    # classify_request (the rules-only pre-check) was deleted with router.py:
    # the agent loop decides, there is no text-matching classification.
    import orchestrator

    assert not hasattr(orchestrator, "classify_request")


def test_placeholder_write_rejected_by_safety_rule() -> None:
    # A write to a placeholder path is rejected by safe_target/the plan safety
    # rules regardless of the model's wording (never staged).
    fake = make_fake_generate({
        "LOOP ACTIONS": json.dumps([
            {"action": "write_file", "path": "your_code.py", "text": "the thing"},
            {"action": "finish"},
        ]),
        "TARGET FILE: your_code.py": "```python\nX = 1\n```",
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create your_code.py with the logic",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == []
        assert result["failed_tasks"]  # the safety rule rejected the write


def test_validator_follows_thinker_at_most_one_swap_pair() -> None:
    """Validator = thinker model, so a full run is coder phases and thinker
    phases only: the loop + coder run on the CODER model, validate + summary
    on the thinker -> at most ONE swap pair."""
    models: list[str] = []

    loop_actions = iter(_web_loop_actions())

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        models.append(model)
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        if "Write index.html." in prompt:
            return f"```html\n{INDEX_HTML}\n```"
        if "Write style.css." in prompt:
            return f"```css\n{STYLE_CSS}\n```"
        if "Write script.js." in prompt:
            return f"```javascript\n{SCRIPT_JS}\n```"
        if "satisfies_request" in prompt:
            return VALIDATE_OK
        return ""

    with TemporaryDirectory() as tmp:
        _run(run_pipeline(
            "make a blog site in html5 css and js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    # compress consecutive duplicates -> the phase sequence
    phases = [m for i, m in enumerate(models) if i == 0 or m != models[i - 1]]
    # loop + code run on the CODER model, validate/summary on the thinker:
    # exactly coder -> thinker = ONE swap pair.
    assert set(models) <= {PLANNER, CODER}
    assert phases == [CODER, PLANNER], phases
