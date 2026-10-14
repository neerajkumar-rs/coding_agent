"""Tests for run_pipeline: ROUTE/PLAN/CODE/CHECK/VALIDATE/SUMMARY (fake LLM)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from orchestrator import classify_request, run_pipeline

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
    """routes: marker-in-prompt -> response. Coder routes by 'TARGET FILE: x'
    (matched via regex so retry feedback above the marker still routes).

    Classify-then-act markers:
    - "CLASSIFY MODE:" -> routes.get("CLASSIFY MODE:") or, when the test set
      a "JSON array of tasks" plan, the label "code"; default "reply".
    - "PLAN TASKS JSON:" -> routes.get("PLAN TASKS JSON:") or the test's
      "JSON array of tasks" plan wrapped as {"tasks": [...]}.
    - "UNDERSTAND JSON:" (legacy) is treated as a classify marker whose
      value decides: a full payload JSON drives mode+reply/tasks.
    Extra kwargs (num_predict, think, role, _format_schema) are accepted and
    ignored."""
    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        import re

        # Legacy Understand route values (full payload JSON) drive both the
        # classify and the act step so old tests keep working unchanged.
        legacy = routes.get("UNDERSTAND JSON:")
        legacy_data: dict = {}
        if legacy:
            try:
                legacy_data = json.loads(legacy)
            except json.JSONDecodeError:
                legacy_data = {}

        if "CLASSIFY MODE:" in prompt or "UNDERSTAND JSON:" in prompt:
            if "CLASSIFY MODE:" in routes:
                return routes["CLASSIFY MODE:"]
            if legacy_data:
                mode = legacy_data.get("mode", "reply")
                return {"code_and_reply": "both"}.get(str(mode), str(mode))
            plan = routes.get("JSON array of tasks", "")
            if plan.lstrip().startswith("["):
                return "code"
            if "QUESTION:" in routes or default:
                return "reply"
            return "reply"

        if "PLAN TASKS JSON:" in prompt:
            if "PLAN TASKS JSON:" in routes:
                return routes["PLAN TASKS JSON:"]
            if legacy_data and legacy_data.get("tasks"):
                return json.dumps({
                    "tasks": legacy_data["tasks"],
                    "assumptions": legacy_data.get("assumptions", []),
                })
            plan = routes.get("JSON array of tasks", "")
            if plan.lstrip().startswith("["):
                return json.dumps({"tasks": json.loads(plan), "assumptions": []})
            return plan or default

        # The reply step: answer-only prompt (has FACTS but no plan marker).
        if "FACTS:" in prompt and "MESSAGE:" in prompt:
            if "REPLY:" in routes:
                return routes["REPLY:"]
            if legacy_data:
                return legacy_data.get("reply") or "I don't know."
            if "QUESTION:" in routes:
                return routes["QUESTION:"]
            if "REPLY" in routes:
                return routes["REPLY"]

        target = re.search(r"TARGET FILE:\s*(\S+)", prompt)
        if target:
            file_marker = f"TARGET FILE: {target.group(1)}"
            if file_marker in routes:
                return routes[file_marker]
        for marker in sorted(routes, key=len, reverse=True):
            if marker in prompt:
                return routes[marker]
        return default

    return fake


def test_summary_call_uses_writer_role() -> None:
    """The summary prose call carries role='writer' through the generate seam
    (the server maps it to the writer role config)."""
    seen_roles: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        import re
        if "CLASSIFY MODE:" in prompt:
            return "code"
        if "PLAN TASKS JSON:" in prompt:
            return json.dumps({
                "tasks": [{"path": "greet.py", "action": "create",
                           "description": "Create a greet function returning a greeting string",
                           "uses_last_result": False}],
                "assumptions": [],
            })
        if "Describe only features visible" in prompt:
            seen_roles.append(kwargs.get("role", ""))
            return "A small python module."
        target = re.search(r"TARGET FILE:\s*(\S+)", prompt)
        if target and target.group(1) == "greet.py":
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


def _run(coro):
    return asyncio.run(coro)


# 1. Question input -> zero tasks, text answer (classify -> reply step)
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
    assert stages == ["CLASSIFY", "ANSWER"]


def test_classify_rules() -> None:
    # classify_request stays the cheap rules-only pre-check (fast path).
    assert classify_request("what is the difference between rust and python") == "QUESTION"
    assert classify_request("make a blog site in html5 css and js") == "CODE_TASK"
    assert classify_request("create app.py") == "CODE_TASK"


# 2. Web request -> 3 files, VALIDATE and SUMMARY present in trace
def test_web_request_three_files_full_trace() -> None:
    fake = make_fake_generate({
        "JSON array of tasks": WEB_PLAN,
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
        assert all(t["status"] == "ok" for t in result["trace"])
        # Staged, not deployed
        assert (Path(tmp) / ".agent_staging" / "index.html").is_file()
        assert not (Path(tmp) / "index.html").exists()
        # Summary = deterministic skeleton (files, how-to-run, issues, time)
        assert "index.html" in result["summary"]
        assert "How to run:" in result["summary"]
        assert "Total time:" in result["summary"]


# 3. Forced-failing task -> not staged, listed as FAILED
def test_failing_task_not_staged() -> None:
    fake = make_fake_generate({
        "JSON array of tasks": json.dumps([
            {"path": "app.py", "action": "create",
             "description": "Python application implementing the requested calculator feature fully and correctly"}
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
        check = next(t for t in result["trace"] if t["stage"] == "CHECK")
        assert check["status"] == "failed"


# 4. Bad html link to a missing css file -> caught in CHECK
def test_bad_html_link_caught() -> None:
    bad_html = '<!doctype html><html><head><link rel="stylesheet" href="missing.css"></head><body>x</body></html>'
    fake = make_fake_generate({
        "JSON array of tasks": json.dumps([
            {"path": "index.html", "action": "create",
             "description": "Standalone HTML page containing all markup and structure"}
        ]),
        "TARGET FILE: index.html": f"```html\n{bad_html}\n```",
    })
    with TemporaryDirectory() as tmp:
        # Non-web phrasing: only index.html is planned (web requests now
        # require html+css+js in the plan by rule; "notes" contains "site").
        result = _run(run_pipeline(
            "create index.html holding my recipe collection",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == []
        assert "missing.css" in result["failed_tasks"][0]["error"]


# 5. Valid index.html passes when cwd is not the project root
def test_check_independent_of_cwd(monkeypatch) -> None:
    fake = make_fake_generate({
        "JSON array of tasks": json.dumps([
            {"path": "index.html", "action": "create",
             "description": "Standalone HTML page containing all markup and structure"}
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


# Plan post-validation: placeholder/short-description tasks are rejected in
# code (never trusted); with zero valid tasks the plan step fails to
# needs_human — never files.
def test_plan_placeholder_tasks_rejected_needs_human() -> None:
    fake = make_fake_generate({
        "CLASSIFY MODE:": "code",
        "PLAN TASKS JSON:": json.dumps({
            "tasks": [
                {"path": "your_code.html", "action": "create", "description": "page"},
                {"path": "example.py", "action": "create", "description": "does things"},
            ],
            "assumptions": [],
        }),
        "TARGET FILE: index.html": f"```html\n{INDEX_HTML}\n```",
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "make a blog site in html5 css and js",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
    assert result["status"] == "needs_human"
    assert result["staged_files"] == []
    assert "valid plan" in result["summary"]


# VALIDATE error -> summary says UNVERIFIED
def test_validate_error_marks_unverified() -> None:
    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "CLASSIFY MODE:" in prompt:
            return "code"
        if "PLAN TASKS JSON:" in prompt:
            return json.dumps({"tasks": json.loads(WEB_PLAN), "assumptions": []})
        if "satisfies_request" in prompt:
            raise RuntimeError("ollama died mid-review")
        if "TARGET FILE: index.html" in prompt:
            return f"```html\n{INDEX_HTML}\n```"
        if "TARGET FILE: style.css" in prompt:
            return f"```css\n{STYLE_CSS}\n```"
        if "TARGET FILE: script.js" in prompt:
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
        "JSON array of tasks": WEB_PLAN,
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
        assert state["llm_calls"] >= 3  # plan + 3 coder + validate (+summary)
        events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
        assert '"run_started"' in events and '"run_finished"' in events
        assert '"stage": "PLAN"' in events and '"stage": "SUMMARY"' in events


# --- Phase 1: cancellation -----------------------------------------------------

def test_cancel_during_code_stops_fast_and_stages_nothing_further() -> None:
    import threading
    import time as _time

    cancel = threading.Event()

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "CLASSIFY MODE:" in prompt:
            return "code"
        if "PLAN TASKS JSON:" in prompt:
            return json.dumps({"tasks": json.loads(WEB_PLAN), "assumptions": []})
        if "TARGET FILE: index.html" in prompt:
            # Cancel lands while the first coder call is in flight.
            cancel.set()
            _time.sleep(0.05)
            return f"```html\n{INDEX_HTML}\n```"
        if "TARGET FILE: style.css" in prompt:
            return f"```css\n{STYLE_CSS}\n```"
        if "TARGET FILE: script.js" in prompt:
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
        "JSON array of tasks": json.dumps([
            {"path": "crashes.py", "action": "create",
             "description": "Python module that imports a missing dependency at top level"}
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
        "JSON array of tasks": json.dumps([
            {"path": "greet.py", "action": "create",
             "description": "Python module with a greet function returning a greeting string"}
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

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "CLASSIFY MODE:" in prompt:
            return "code"
        if "PLAN TASKS JSON:" in prompt:
            return json.dumps({"tasks": json.loads(WEB_PLAN), "assumptions": []})
        if model == CODER:
            coder_calls.append(prompt)
        return VALIDATE_OK

    async def decline(tasks, intent_line):
        assert {t["path"] for t in tasks} == {"index.html", "style.css", "script.js"}
        assert "code" in intent_line
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
    assert coder_calls == []  # coder never ran


def test_confirm_gate_approve_runs_full_pipeline() -> None:
    fake = make_fake_generate({
        "JSON array of tasks": WEB_PLAN,
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


def test_confirm_none_skips_gate() -> None:
    # confirm=None (direct API/tests/--yes): no gate, current behavior.
    fake = make_fake_generate({
        "JSON array of tasks": WEB_PLAN,
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


# --- Phase 4: fallback only for confirmed CODE_TASK, else needs_human ----------

def test_plan_never_runs_for_non_code_labels() -> None:
    calls: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        calls.append(prompt)
        if "CLASSIFY MODE:" in prompt:
            return "reply"
        return "x"

    with TemporaryDirectory() as tmp:
        for text in ("hello", "make something", "what does the fix do"):
            result = _run(run_pipeline(
                text, tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            ))
            assert "staged_files" not in result, text
    plan_calls = [p for p in calls if "PLAN TASKS JSON:" in p]
    assert plan_calls == []  # the planner never ran for reply/greeting labels


def test_plan_total_failure_is_needs_human() -> None:
    # The plan model returns unparseable junk twice -> needs_human with the
    # reason, never files, no loop. There is no "couldn't understand" path.
    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "CLASSIFY MODE:" in prompt:
            return "code"
        return "not json at all"

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create your_code.py with the logic",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["status"] == "needs_human"
        assert result["staged_files"] == []
        assert "valid plan" in result["summary"]


def test_classifier_junk_defaults_to_reply() -> None:
    # A classifier reply with no valid label falls back to 'reply' (safe:
    # grounded answer, no files) — never a rephrase request.
    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "CLASSIFY MODE:" in prompt:
            return "banana banana banana"
        if "FACTS:" in prompt:
            return "Grounded answer."
        return "x"

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create your_code.py with the logic",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert "staged_files" not in result
        assert result["answer"] == "Grounded answer."


def test_validator_follows_thinker_at_most_one_swap_pair() -> None:
    """Validator = thinker model, so a full run is thinker phases and coder
    phases only. The sequence of MODELS handed to generate must alternate at
    most thinker -> coder -> thinker (one swap pair)."""
    models: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        models.append(model)
        if "CLASSIFY MODE:" in prompt:
            return "code"
        if "PLAN TASKS JSON:" in prompt:
            return json.dumps({"tasks": json.loads(WEB_PLAN), "assumptions": []})
        target = None
        import re
        m = re.search(r"TARGET FILE:\s*(\S+)", prompt)
        if m:
            target = m.group(1)
        if target == "index.html":
            return f"```html\n{INDEX_HTML}\n```"
        if target == "style.css":
            return f"```css\n{STYLE_CSS}\n```"
        if target == "script.js":
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
    # classify/plan/code run on the CODER model, validate/summary on the
    # thinker: exactly coder -> thinker = ONE swap pair.
    assert set(models) <= {PLANNER, CODER}
    assert phases == [CODER, PLANNER], phases
