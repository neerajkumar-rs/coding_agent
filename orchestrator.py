"""The single server-side pipeline used by POST /agent/run.

ROUTE -> PLAN -> CODE -> CHECK -> VALIDATE -> SUMMARY, with a per-stage
trace and no hidden stages. agent_core is a library this pipeline calls:
RunJournal (logs/events), safe_target/StagingArea (staging/safety),
ContextBudget (85% prompt budget), schemas (structured planner/validator),
and verifier (deterministic Python CHECK).
"""

from __future__ import annotations

import asyncio
import json
import os
import py_compile
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent_core.coder import ALLOWED_TARGET_SUFFIXES, parse_coder_output, apply_file_edit
from agent_core.context import ContextBudget
from agent_core.run_state import (
    RUNS_DIR_NAME,
    ConversationState,
    RunJournal,
    diagnostic_hash,
)
from agent_core.schemas import PipelinePlan, ValidateResult
from agent_core.staging import safe_target
from agent_core.verifier import verify_files
from ollama_client import (
    TEMPERATURE_CODE,
    TEMPERATURE_PLAN,
    OllamaCancelledError,
    strip_think_blocks,
)
from router import route as route_request
from sandbox import check_plausible_content, tail_lines
from workspace_staging import WorkspaceStagingManager

from agent_core.engine import (
    DEFAULT_MAX_LLM_CALLS,
    DEFAULT_MAX_WALL_SECONDS,
    MAX_ATTEMPTS_PER_TASK,
    BudgetExhausted,
    RunBudgets,
    RunCancelled,
)
from router import (
    RouteResult,
    canned_chat_reply,
)
from router import route as router_route
from router import route_rules_only as _route_rules_only

# Re-exported so older imports (tests, e2e, tui) keep resolving.
__all__ = [
    "BudgetExhausted",
    "RunBudgets",
    "RunCancelled",
    "RUNS_DIR_NAME",
    "MAX_ATTEMPTS_PER_TASK",
    "DEFAULT_MAX_WALL_SECONDS",
    "DEFAULT_MAX_LLM_CALLS",
    "classify_request",
    "run_pipeline",
]


# ===========================================================================
# THE request pipeline: ROUTE -> PLAN -> CODE -> CHECK -> VALIDATE -> SUMMARY
# ===========================================================================

LANGUAGE_BY_SUFFIX = {
    ".py": "python", ".html": "html", ".css": "css",
    ".js": "javascript", ".json": "json", ".md": "markdown",
    ".rs": "rust",
}

PLACEHOLDER_STEMS = {"your_code", "example", "foo", "file1", "untitled", "file", "code", "test"}
WEB_HINTS = ("html", "css", "website", "web page", "webpage", "site", "blog", "landing", "page")

MAX_TASK_RETRIES = 2  # retries AFTER the first attempt
VALIDATE_FILE_HEAD_LINES = 60
SUMMARY_FILE_HEAD_LINES = 40
# Feature keywords the summary model may NOT invent: dropped when absent from
# every generated file's actual content.
SUMMARY_FEATURE_KEYWORDS = ("search", "login", "database", "dark mode", "comments")


def classify_request(user_request: str) -> str:
    """Sync rules-only classification (router v2 tiers T0+T1, no model call).

    Kept as the cheap pre-check / backwards-compatible import. The pipeline
    itself uses :func:`router.route` (which adds the T2 grey-zone classifier).
    Never returns CODE_TASK for a generic statement.
    """
    return _route_rules_only(user_request).intent


# ---------------------------------------------------------------------------
# FILE_QUESTION: deterministic workspace facts (depth-2 tree + per-file stats)
# ---------------------------------------------------------------------------

FILE_Q_SKIP_DIRS = {".git", ".agent_staging", ".agent_runs",
                    "node_modules", "__pycache__", "env", ".venv", "venv"}
FILE_Q_MAX_ENTRIES = 200
FILE_Q_MAX_DETAIL = 20
FILE_Q_HEAD_LINES = 30
TEXT_SUFFIXES = set(LANGUAGE_BY_SUFFIX) | {".txt", ".toml", ".yml", ".yaml", ".cfg", ".ini"}


def _file_language(path: Path) -> str:
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


def build_file_facts(workspace: Path) -> dict[str, Any]:
    """Deterministic scan: tree (depth 2) + per-text-file stats and heads."""
    workspace = Path(workspace).resolve()
    tree_lines: list[str] = []
    details: list[dict[str, Any]] = []
    count = 0
    for root, dirs, files in os.walk(workspace):
        dirs[:] = sorted(d for d in dirs if d not in FILE_Q_SKIP_DIRS and not d.startswith("."))
        rel_root = Path(root).relative_to(workspace)
        depth = len(rel_root.parts)
        if depth > 2:
            continue
        indent = "  " * depth
        tree_lines.append(f"{indent}{rel_root.name if rel_root.name else '.'}/")
        for name in sorted(files):
            if count >= FILE_Q_MAX_ENTRIES:
                break
            if name.startswith(".") or name.endswith(".pyc"):
                continue
            count += 1
            path = Path(root) / name
            rel = path.relative_to(workspace).as_posix()
            tree_lines.append(f"{indent}  {name}")
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue  # skip binaries
            if len(details) >= FILE_Q_MAX_DETAIL:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            details.append({
                "name": rel,
                "language": _file_language(path),
                "lines": text.count("\n") + 1,
                "size": path.stat().st_size,
                "head": "\n".join(text.splitlines()[:FILE_Q_HEAD_LINES]),
            })
    return {"tree": "\n".join(tree_lines), "files": details}


def format_file_table(facts: dict[str, Any]) -> str:
    """Deterministic table: name, language, lines, size."""
    rows = [f"{'FILE':<40}{'LANGUAGE':<12}{'LINES':>7}{'SIZE':>9}"]
    for f in facts["files"]:
        size = f["size"]
        size_str = f"{size} B" if size < 1024 else f"{size / 1024:.1f} KB"
        rows.append(f"{f['name']:<40}{f['language']:<12}{f['lines']:>7}{size_str:>9}")
    if len(rows) == 1:
        rows.append("(no text files found)")
    return "\n".join(rows)


def _language_of(path: str) -> str:
    return LANGUAGE_BY_SUFFIX.get(Path(path).suffix.lower(), "text")


def _valid_plan_task(task: dict[str, Any]) -> str | None:
    """Return a rejection reason, or None when the plan task is acceptable."""
    from scanner import is_junk_file

    path = str(task.get("path", "")).strip().replace("\\", "/")
    desc = str(task.get("description", "")).strip()
    if not path:
        return "empty path"
    # Never plan around leftover run artifacts (tasks.md, summary.txt,
    # taskN_code.txt, images, etc.) — same junk filter the scanner uses.
    if is_junk_file(Path(path).name):
        return f"junk artifact filename: {path}"
    stem = Path(path).stem.lower()
    if stem in PLACEHOLDER_STEMS:
        return f"placeholder filename: {path}"
    suffix = Path(path).suffix.lower()
    if suffix not in ALLOWED_TARGET_SUFFIXES | {".json", ".md", ".rs"}:
        return f"unsupported file type: {path}"
    if len(desc.split()) < 8:
        return f"description under 8 words: {desc!r}"
    # Extension/language mismatch: description names a DIFFERENT language.
    lang_words = {
        ".html": ("html",), ".css": ("css",),
        ".js": ("javascript", "js"), ".py": ("python",),
        ".rs": ("rust", "rs"),
    }
    mentioned = {lang for words in lang_words.values() for lang in words if f" {lang} " in f" {desc.lower()} "}
    own = set(lang_words.get(suffix, ()))
    if mentioned and not (mentioned & own):
        return f"extension/language mismatch: {path} vs {sorted(mentioned)}"
    return None


def _valid_plan(tasks: list[dict[str, str]], user_request: str) -> list[str]:
    """Plan-LEVEL rejection reasons (per-task rules live in _valid_plan_task)."""
    reasons: list[str] = []
    requested_app_py = bool(re.search(r"\bapp\.py\b", user_request.lower()))
    for task in tasks:
        if Path(task["path"]).name.lower() == "app.py" and not requested_app_py:
            reasons.append("app.py planned but never requested — name files descriptively")
    lower = user_request.lower()
    # Completeness fires when the request EXPLICITLY spans html+css+js (or a
    # multi-file web plan was produced): all three must be present then.
    # A single-file page request ("create index.html") stays exempt — cross-
    # file link checks cannot run before sibling files exist anyway.
    mentions_all_three = all(k in lower for k in ("html", "css")) and (
        "js" in lower or "javascript" in lower
    )
    is_multi_file_web = len(tasks) > 1 and any(
        h in lower for h in WEB_HINTS
    )
    if mentions_all_three or is_multi_file_web:
        suffixes = {Path(t["path"]).suffix.lower() for t in tasks}
        missing = {".html", ".css", ".js"} - suffixes
        if missing and ".html" not in missing:
            reasons.append(
                f"web app plan must include html, css and js files (missing: {sorted(missing)})"
            )
    return reasons


# _fallback_plan was retired with the rules router (v4): the Understand step
# produces plans; a total Understand failure is a graceful reply, never a
# deterministic scaffold.



def _parse_plan_json(text: str) -> list[dict[str, Any]]:
    """Extract a JSON task array from planner output (think already stripped)."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [t for t in data if isinstance(t, dict)]


def _extract_code(raw: str) -> str:
    """Longest fenced block, or the stripped raw text when no fences exist."""
    matches = re.findall(r"```(?:\w+)?\n(.*?)```", raw, re.DOTALL)
    if matches:
        return max(matches, key=len).rstrip() + "\n"
    return raw.strip() + "\n"


def _check_generated(rel_path: str, content: str) -> str | None:
    """Per-file deterministic CHECK, fully in-process. Returns error or None.

    Content problems return a string; an exception escaping this function is
    an INFRA error (checker crashed) and aborts WITHOUT a coder retry.
    """
    suffix = Path(rel_path).suffix.lower()
    if suffix == ".py":
        # compile() in-memory: no temp files, no __pycache__, no os.devnull
        # (py_compile with os.devnull fails on Windows: "nul is a non-regular
        # file"), no dependence on cwd or paths with spaces.
        try:
            compile(content, rel_path, "exec")
        except SyntaxError as exc:
            return f"syntax error: {exc.msg} (line {exc.lineno})"
        return None
    return check_plausible_content(rel_path, content)


def _check_html_links(
    rel_path: str, content: str, planned_paths: set[str], workspace: Path
) -> str | None:
    """Every local href/src must resolve to a planned or existing file."""
    if Path(rel_path).suffix.lower() != ".html":
        return None
    bad_links: list[str] = []
    for ref in re.findall(r'(?:href|src)="([^"]+)"', content):
        if ref.startswith(("#", "mailto:", "data:")) or "://" in ref:
            continue
        ref_path = ref.split("#", 1)[0].split("?", 1)[0].strip()
        if not ref_path:
            continue
        resolved = (Path(rel_path).parent / ref_path).as_posix().lstrip("./")
        if resolved not in planned_paths and not (workspace / resolved).is_file():
            bad_links.append(ref)
    if bad_links:
        return f"html links do not resolve to existing files: {bad_links}"
    return None


async def _summary_prose(
    user_request: str,
    generated: dict[str, str],
    generate: Callable[..., Awaitable[str]],
    summary_model: str,
) -> str:
    """One thinker-side call: 2-3 sentences describing ONLY features visible in
    the first 40 lines of each file. Dropped entirely when it names a feature
    keyword (search/login/database/dark mode/comments) absent from all files,
    or when the call fails (the skeleton alone is returned then)."""
    if not generated:
        return ""
    sections = []
    for name, content in generated.items():
        head = "\n".join(content.splitlines()[:SUMMARY_FILE_HEAD_LINES])
        sections.append(f"FILE: {name}\n{head}")
    prompt = (
        "Describe only features visible in the code shown, in 2-3 sentences. "
        "Do not invent features.\n\n"
        + "\n\n".join(sections)
        + "\n\nREQUEST: " + user_request
    )
    try:
        raw = await generate(
            summary_model, prompt,
            system="You describe only what the code visibly does.",
            temperature=TEMPERATURE_PLAN,
            role="thinker",  # summary prose runs on the thinker (writer parked)
        )
    except Exception:
        return ""
    _, prose = strip_think_blocks(raw)
    prose = prose.strip()
    if not prose:
        return ""
    # Feature-keyword guard: drop prose that invents features.
    all_content = "\n".join(generated.values()).lower()
    for keyword in SUMMARY_FEATURE_KEYWORDS:
        if keyword in prose.lower() and keyword not in all_content:
            return ""
    return prose


def _summary_skeleton(
    user_request: str,
    generated: dict[str, str],
    failed_tasks: list[dict[str, str]],
    validate_problems: list[str],
    validate_failed: bool,
    total_seconds: float,
) -> str:
    """Deterministic part of the summary — built in code, never by a model."""
    file_list = sorted(generated)
    lines: list[str] = []
    if validate_failed:
        lines.append("UNVERIFIED — validation could not run.")
    lines.append(
        "Built: " + (", ".join(file_list) if file_list else "(no files passed CHECK)")
    )
    if failed_tasks:
        lines.append("Failed:")
        for t in failed_tasks:
            lines.append(f"  NOT saved (failed): {t['path']} - {t['error']}")
    how: list[str] = []
    if any(f.endswith((".html", ".htm")) for f in file_list):
        page = next(f for f in file_list if f.endswith((".html", ".htm")))
        how.append(f"open {page} in a browser")
    if any(f.endswith(".py") for f in file_list):
        script = next(f for f in file_list if f.endswith(".py"))
        how.append(f"run: python {script}")
    lines.append("How to run: " + ("; ".join(how) if how else "see file contents"))
    issues = list(validate_problems)
    lines.append("Known issues: " + ("; ".join(issues) if issues else "none"))
    lines.append(f"Total time: {total_seconds:.1f}s")
    return "\n".join(lines)


async def _build_summary(
    *,
    user_request: str,
    generated: dict[str, str],
    failed_tasks: list[dict[str, str]],
    validate_problems: list[str],
    validate_failed: bool,
    total_seconds: float,
    generate: Callable[..., Awaitable[str]],
    summary_model: str,
) -> str:
    skeleton = _summary_skeleton(
        user_request, generated, failed_tasks,
        validate_problems, validate_failed, total_seconds,
    )
    prose = await _summary_prose(user_request, generated, generate, summary_model)
    return (prose + "\n\n" + skeleton) if prose else skeleton


async def run_pipeline(
    user_request: str,
    workspace: str | Path,
    *,
    generate: Callable[..., Awaitable[str]],
    planner_model: str,
    coder_model: str,
    unload: Callable[[str], Awaitable[None]] | None = None,
    cancel_event: Any | None = None,
    journal: RunJournal | None = None,
    num_ctx: int = 8192,
    structured_generate: Callable[..., Awaitable[Any]] | None = None,
    budgets: RunBudgets | None = None,
    confirm: Callable[[list[dict[str, Any]], str], Awaitable[bool]] | None = None,
    history: list[str] | None = None,
    router_min_confidence: float = 0.7,
    chat_use_llm: bool = False,
    think_supported: bool | None = None,
    think_budget: int = 512,
    log_llm_calls: bool = True,
) -> dict[str, Any]:
    """ROUTE -> PLAN -> CODE -> CHECK -> VALIDATE -> SUMMARY.

    ``generate(model, prompt, system=..., temperature=...) -> str`` is the
    only LLM access (think-stripped by the caller's layer or stripped here).
    Returns {summary, staged_files, failed_tasks, trace} for code tasks, or
    {answer, trace} for questions.

    Observability: a RunJournal under .agent_runs/<run_id>/ records state.json
    and events.jsonl (one event per stage). ``cancel_event`` (a threading or
    asyncio event with ``is_set()``) aborts the run between stages and before
    every LLM call, returning status ``cancelled``.
    """
    workspace = Path(workspace).resolve()
    staging = WorkspaceStagingManager(workspace)
    if journal is None:
        journal = RunJournal(workspace)
    trace: list[dict[str, Any]] = []
    pipeline_start = time.monotonic()
    # 85% of the context window is the input budget; the plan/code prompts are
    # trimmed against it so a large request can never blow the window.
    budget = ContextBudget(num_ctx, reserve_output_tokens=2048)
    if budgets is None:
        budgets = RunBudgets()
    llm_calls = 0

    def _cancelled() -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    def _check_cancel() -> None:
        if _cancelled():
            raise RunCancelled("run cancelled by user")

    def _check_wall_time() -> None:
        if time.monotonic() - pipeline_start > budgets.max_wall_seconds:
            raise BudgetExhausted(
                f"max wall time exceeded ({budgets.max_wall_seconds:.0f}s)"
            )

    def record(stage: str, status: str, seconds: float, note: str = "",
               model: str = "") -> None:
        entry = {"stage": stage, "status": status,
                 "seconds": round(seconds, 1), "note": note}
        if model:
            entry["model"] = model
        trace.append(entry)
        journal.emit_event({"event": "stage", **entry})

    def _finish(status: str, extra: dict[str, Any] | None = None) -> None:
        state: dict[str, Any] = {
            "run_id": journal.run_id,
            "status": status,
            "request": user_request,
            "intent": trace[0].get("note") if trace else "",
            "roles": {"thinker": planner_model, "coder": coder_model},
            "num_ctx": num_ctx,
            "llm_calls": llm_calls,
            "elapsed_seconds": round(time.monotonic() - pipeline_start, 3),
        }
        if extra:
            state.update(extra)
        journal.save_state(state)
        journal.emit_event(
            {"event": "run_finished", "run_id": journal.run_id, "status": status}
        )

    journal.save_state({"run_id": journal.run_id, "status": "running",
                        "request": user_request})
    journal.emit_event({"event": "run_started", "run_id": journal.run_id,
                        "request": user_request[:500]})

    async def _llm(model: str, prompt: str, system: str = "", temperature: float = 0.2, **kwargs: Any) -> str:
        """Single LLM choke point: budget the prompt, count + cap the call,
        honour cancellation immediately before hitting the model, and log the
        raw call (truncated to 20KB) into the run journal under its role."""
        nonlocal llm_calls
        _check_cancel()
        _check_wall_time()
        if llm_calls >= budgets.max_llm_calls:
            raise BudgetExhausted(
                f"max total LLM calls exceeded ({budgets.max_llm_calls})"
            )
        llm_calls += 1
        role = kwargs.get("role") or ("coder" if model == coder_model else "thinker")
        # Role-scoped calls (understand) may pass model=None: resolve the model
        # name for logging + the generate seam from the role.
        if model is None:
            model = {"understand": coder_model, "coder": coder_model,
                     "writer": planner_model}.get(role, planner_model)
        trimmed = budget.trim(prompt)
        call_t0 = time.monotonic()
        response = ""
        error: str | None = None
        try:
            response = await generate(model, trimmed, system=system, temperature=temperature, **kwargs)
            return response
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            # On the server path the gateway's on_call hook logs every call
            # with full Ollama meta (log_llm_calls=False there); the pipeline
            # logs only when it owns the generate seam (tests, headless use).
            if log_llm_calls:
                journal.log_llm_call(
                    role, model=model, prompt=trimmed, response=response,
                    temperature=temperature, num_ctx=num_ctx,
                    seconds=round(time.monotonic() - call_t0, 3), error=error,
                )

    _check_cancel()

    try:
        # ------------------------------------------------------------------ ROUTE
        # Classify-then-act: fast paths (no model, no files) first, then ONE
        # tiny CLASSIFY call, then ONE act call per label (reply facts / plan).
        from understand import (
            PlanFailed,
            build_facts_block,
            classify_mode,
            fast_path,
            plan_tasks,
            reply_step,
        )

        t0 = time.monotonic()
        fp = fast_path(user_request)
        if fp == "COMMAND":
            record("FAST-PATH", "ok", 0.0, "command handled by the client")
            _finish("completed")
            return {
                "answer": "Slash commands are handled by the client UI — "
                          "type /help for the list.",
                "trace": trace,
                "run_id": journal.run_id,
            }
        if fp == "CHAT":
            reply = canned_chat_reply(user_request)
            record("FAST-PATH", "ok", 0.0, "canned", model="")
            _finish("completed")
            return {"answer": reply, "trace": trace, "run_id": journal.run_id}

        # Conversation state: the models resolve references from it.
        convo = ConversationState.load(workspace)
        # Merge a pending question into the message; after one question we
        # proceed with assumptions — never two questions in a row.
        message = user_request
        if convo.pending_question:
            message = (
                f"PENDING QUESTION FROM THE AGENT: {convo.pending_question}\n"
                f"USER ANSWER: {user_request}"
            )
        facts = build_file_facts(workspace)
        file_names = [str(f.get("name", "")) for f in facts.get("files", [])][:40]
        last_turn_line = ""
        if convo.turns:
            last = convo.turns[-1]
            last_turn_line = f"{last.get('user', '')[:80]} -> {last.get('mode', '')}"

        # ---------------------------------------------------------- CLASSIFY
        mode = await classify_mode(
            message,
            generate=_llm,
            file_names=file_names,
            last_turn=last_turn_line,
        )
        record("CLASSIFY", "ok", time.monotonic() - t0, mode, model=coder_model)

        reply_text: str | None = None

        if mode in ("reply", "both"):
            # ---------------------------------------------------------- ANSWER
            t0 = time.monotonic()
            facts_block = build_facts_block(
                facts,
                turns=convo.turns,
                last_result=convo.last_result,
                last_trace=convo.last_trace,
                pending_question=convo.pending_question,
            )
            reply_text = await reply_step(
                message, generate=_llm, facts_block=facts_block,
            )
            record("ANSWER", "ok", time.monotonic() - t0, model=coder_model)
            if mode == "reply":
                convo.record_turn(
                    user_request, "reply", reply_text[:120],
                    trace=trace,
                    last_result={"type": "reply", "text": reply_text},
                )
                _finish("completed")
                return {"answer": reply_text, "trace": trace, "run_id": journal.run_id}
            # both: print the reply first, then continue to the plan step.
            journal.emit_event({"event": "reply", "text": reply_text[:500]})

        # -------------------------------------------------------------- PLAN
        t0 = time.monotonic()
        try:
            plan = await plan_tasks(
                message,
                generate=_llm,
                file_names=file_names,
                task_validator=_valid_plan_task,
            )
        except PlanFailed as exc:
            record("PLAN", "failed", time.monotonic() - t0, str(exc)[:200])
            _finish("needs_human", {"staged_files": [], "failed_tasks": []})
            return {
                "status": "needs_human",
                "summary": f"Could not produce a valid plan: {exc}",
                "staged_files": [],
                "failed_tasks": [],
                "trace": trace,
                "run_id": journal.run_id,
            }
        tasks = plan.tasks
        assumptions = plan.assumptions
        record("PLAN", "ok", time.monotonic() - t0,
               f"{len(tasks)} tasks"
               + (f" (assumptions: {'; '.join(assumptions[:3])})" if assumptions else ""),
               model=coder_model)

        planned_paths = {t["path"] for t in tasks}

        # ------------------------------------------------------- CONFIRM GATE
        # Before anything is staged: show planned files + one-line intent and
        # require approval (Enter/y at the client). ``confirm=None`` (direct
        # API calls, tests, --yes) skips the gate.
        if confirm is not None:
            _check_cancel()
            intent_line = (
                f"{mode}: plan is "
                + ", ".join(f"{t['action']} {t['path']}" for t in tasks)
            )
            journal.emit_event({"event": "confirm_requested",
                                "plan": [t["path"] for t in tasks],
                                "intent": intent_line})
            approved = await confirm(tasks, intent_line)
            if not approved:
                record("PLAN", "ok", 0.0, "declined by user — nothing staged")
                _finish("declined", {"staged_files": [], "failed_tasks": []})
                return {
                    "status": "declined",
                    "summary": "Plan declined by user — nothing was staged.",
                    "staged_files": [],
                    "failed_tasks": [],
                    "trace": trace,
                    "run_id": journal.run_id,
                }
            journal.emit_event({"event": "confirm_approved"})

        # ------------------------------------------------------------------ CODE
        t_code = time.monotonic()
        generated: dict[str, str] = {}
        failed_tasks: list[dict[str, str]] = []
        check_seconds = 0.0
        infra_error: str | None = None

        for task in tasks:
            path, desc = task["path"], task["description"]
            language = _language_of(path)
            extra = ""
            if Path(path).name.lower() == "index.html":
                links = [
                    p for p in ("style.css", "script.js") if p in planned_paths
                ]
                if links:
                    extra = (
                        "\nThe HTML MUST reference "
                        + " and ".join(links)
                        + " with <link>/<script> tags using those exact filenames."
                    )
            # Understand flags: this task consumes the previous turn's result.
            last_result_block = ""
            if task.get("uses_last_result") and convo.last_result.get("text"):
                last_result_block = (
                    "\nPREVIOUS RESULT (use this as input data):\n"
                    + str(convo.last_result.get("text", ""))[:1_500]
                )
            assumptions_block = ""
            if assumptions:
                assumptions_block = "\nASSUMPTIONS: " + "; ".join(assumptions[:3])
            code_prompt = (
                f"Implement exactly one file.\n"
                f"ORIGINAL USER REQUEST: {user_request}\n"
                f"PLANNED FILES: {', '.join(sorted(planned_paths))}\n"
                f"TARGET FILE: {path}\n"
                f"TASK: {desc}\n"
                f"LANGUAGE: {language}\n"
                f"Return ONLY the complete file contents inside one "
                f"```{language} fenced block. No commentary.{extra}"
                f"{assumptions_block}{last_result_block}"
            )
            content: str | None = None
            error: str | None = None
            previous_diag_hash: str | None = None
            no_progress = False
            for attempt in range(1 + MAX_TASK_RETRIES):
                prompt = code_prompt
                if error:
                    prompt += (
                        f"\n\nPREVIOUS ATTEMPT FAILED THIS CHECK:\n{error}\n"
                        "Fix exactly this and return the full corrected file."
                    )
                raw = await _llm(
                    coder_model, prompt,
                    system=f"You output only complete {language} files in one fenced block.",
                    temperature=TEMPERATURE_CODE,
                )
                _, clean = strip_think_blocks(raw)
                content = _extract_code(clean)

                c0 = time.monotonic()
                try:
                    error = _check_generated(path, content)
                except Exception as exc:  # infra error: checker itself crashed
                    infra_error = f"CHECK infrastructure error on {path}: {exc}"
                    check_seconds += time.monotonic() - c0
                    break
                check_seconds += time.monotonic() - c0
                if error is None:
                    break
                # No-progress rule: the same diagnostic twice in a row means the
                # coder is stuck — stop early instead of burning the last retry.
                diag_hash = diagnostic_hash(error)
                if previous_diag_hash is not None and diag_hash == previous_diag_hash:
                    no_progress = True
                    break
                previous_diag_hash = diag_hash

            if infra_error:
                failed_tasks.append({"path": path, "error": infra_error})
                break  # abort: never retry coder on infra errors
            if error is not None or content is None:
                reason = error or "no content generated"
                if no_progress:
                    reason += " (needs human: identical failure twice)"
                failed_tasks.append({"path": path, "error": reason})
                continue
            safe_target(workspace, path)  # raises on escape/protected
            staging.stage_file(path, content)
            generated[path] = content

        # Cross-file CHECK: html links resolve against the FULL generated set
        # (sibling files exist only after every task has been coded).
        if not infra_error:
            for name in [n for n in generated if n.endswith(".html")]:
                link_error = _check_html_links(name, generated[name], planned_paths, workspace)
                if link_error:
                    staging.get_staged_file_path(name).unlink(missing_ok=True)
                    generated.pop(name)
                    failed_tasks.append({"path": name, "error": link_error})

        # Deterministic verifier gate for Python: py_compile -> ruff -> import
        # smoke -> pytest (each step skipped, never failed, when its tool is
        # unavailable). Runs on the STAGED files; a failure unstages the file
        # and marks its task failed. Runs in a thread: subprocesses would
        # otherwise block the event loop.
        python_staged = {
            name: staging.get_staged_file_path(name)
            for name in generated
            if name.endswith(".py")
        }
        verify_result: Any | None = None
        if python_staged and not infra_error:
            v0 = time.monotonic()
            try:
                verify_result = await asyncio.to_thread(
                    verify_files, python_staged, staging.staging_dir, None, workspace
                )
            except Exception as exc:  # verifier crashed -> infra error, no retry
                infra_error = f"verifier infrastructure error: {exc}"
                check_seconds += time.monotonic() - v0
            else:
                check_seconds += time.monotonic() - v0
                if not verify_result.passed:
                    diagnostics = tail_lines(verify_result.diagnostics(), 30)
                    for name in python_staged:
                        staging.get_staged_file_path(name).unlink(missing_ok=True)
                        generated.pop(name, None)
                        failed_tasks.append({"path": name, "error": f"verifier: {diagnostics}"})
            journal.emit_event(
                {"event": "verifier", "passed": not any(
                    t["error"].startswith("verifier:") for t in failed_tasks
                )}
            )

        record("CODE", "ok" if generated else ("failed" if failed_tasks else "ok"),
               time.monotonic() - t_code,
               f"{len(generated)} coded, {len(failed_tasks)} failed")
        record("CHECK", "failed" if (failed_tasks or infra_error) else "ok",
               check_seconds,
               infra_error or f"{len(generated)} passed, {len(failed_tasks)} failed")

        # sandbox channel: every deterministic CHECK outcome, one block per
        # run (per-file checks + the verifier step chain + infra errors).
        sandbox_lines: list[str] = []
        for name in sorted(generated):
            sandbox_lines.append(f"[PASS] {name}")
        for t in failed_tasks:
            sandbox_lines.append(f"[FAIL] {t['path']}: {t['error']}")
        if verify_result is not None:
            for step in verify_result.steps:
                status = "SKIP" if step.skipped else ("PASS" if step.passed else "FAIL")
                sandbox_lines.append(f"[{status}] verifier.{step.step}")
                if step.output and not step.passed:
                    sandbox_lines.append(step.output[:2_000])
        if infra_error:
            sandbox_lines.append(f"[INFRA] {infra_error}")
        journal.log_role("sandbox", "\n".join(sandbox_lines) or "(no checks ran)")

        # --------------------------------------------------------------- VALIDATE
        validate_note = ""
        validate_problems: list[str] = []
        validator_summary = ""
        t0 = time.monotonic()
        validate_failed = False
        if not generated:
            record("VALIDATE", "skipped", 0.0, "nothing passed CHECK")
        else:
            try:
                if unload is not None:
                    await unload(coder_model)
                head_sections = []
                for name, content in generated.items():
                    head = "\n".join(content.splitlines()[:VALIDATE_FILE_HEAD_LINES])
                    head_sections.append(f"FILE: {name}\n{head}")
                validate_prompt = (
                    "Review whether the generated files satisfy the request. "
                    "Reply with ONLY JSON: "
                    '{"satisfies_request": true|false, "problems": ["..."], '
                    '"summary": "one sentence"}' + "\n\n"
                    f"REQUEST: {user_request}\n"
                    f"FILES: {', '.join(sorted(generated))}\n\n"
                    + "\n\n".join(head_sections)
                )
                data: dict[str, Any] = {}
                if structured_generate is not None:
                    try:
                        _check_cancel()
                        llm_calls += 1
                        vobj = await structured_generate(
                            planner_model, budget.trim(validate_prompt), ValidateResult
                        )
                        data = vobj.model_dump()
                    except Exception:
                        data = {}  # fall through to the prompt path
                if not data:
                    raw = await _llm(
                        planner_model, validate_prompt,
                        system="You are a strict reviewer outputting only JSON.",
                        temperature=TEMPERATURE_PLAN,
                        role="validator",
                    )
                    _, clean = strip_think_blocks(raw)
                    match = re.search(r"\{.*\}", clean, re.DOTALL)
                    data = json.loads(match.group(0)) if match else {}
                validate_problems = [str(p) for p in data.get("problems", [])]
                validator_summary = str(data.get("summary", ""))

                if data.get("satisfies_request") is False:
                    # ONE fix pass through the coder, then re-run CHECK.
                    fix_prompt = (
                        "Fix the reported problems. Return every corrected file as "
                        "```FILE: path blocks with full contents.\n"
                        f"PROBLEMS: {json.dumps(validate_problems)}\n"
                        f"REQUEST: {user_request}\n\n"
                        + "\n\n".join(
                            f"FILE: {n}\n```\n{c}\n```" for n, c in generated.items()
                        )
                    )
                    raw_fix = await _llm(
                        coder_model, fix_prompt,
                        system="You output only corrected FILE blocks.",
                        temperature=TEMPERATURE_CODE,
                    )
                    _, clean_fix = strip_think_blocks(raw_fix)
                    try:
                        for file_edit in parse_coder_output(clean_fix):
                            base = generated.get(file_edit.path)
                            fixed = apply_file_edit(base, file_edit)
                            fix_error = _check_generated(file_edit.path, fixed)
                            if fix_error is None:
                                staging.stage_file(file_edit.path, fixed)
                                generated[file_edit.path] = fixed
                            else:
                                validate_problems.append(
                                    f"{file_edit.path}: fix failed CHECK ({fix_error})"
                                )
                    except ValueError:
                        validate_problems.append("fix pass returned unusable output")
                    validate_note = "fix pass applied" if not validate_problems else "problems remain"
                else:
                    validate_note = "satisfies request"
                record("VALIDATE", "ok", time.monotonic() - t0, validate_note)
            except Exception as exc:
                validate_failed = True
                record("VALIDATE", "failed", time.monotonic() - t0, str(exc)[:300])

        # ---------------------------------------------------------------- SUMMARY
        # Runs on the THINKER (validator's model): after VALIDATE the thinker
        # is already resident, so a full run is at most ONE swap pair
        # (thinker -> coder -> thinker).
        t0 = time.monotonic()
        summary = await _build_summary(
            user_request=user_request,
            generated=generated,
            failed_tasks=failed_tasks,
            validate_problems=validate_problems,
            validate_failed=validate_failed,
            total_seconds=time.monotonic() - pipeline_start,
            generate=generate,
            summary_model=planner_model,
        )
        file_list = sorted(generated)
        record("SUMMARY", "ok", time.monotonic() - t0,
               "UNVERIFIED" if validate_failed else "")

        _finish("completed", {"staged_files": file_list,
                              "failed_tasks": failed_tasks})
        # Conversation state: every turn recorded (reply-first for
        # code_and_reply was emitted as an event at PLAN time).
        convo.record_turn(
            user_request, mode, summary[:120],
            trace=trace,
            last_result={
                "type": "code",
                "text": summary,
                "files": file_list,
            },
        )
        out: dict[str, Any] = {
            "summary": summary,
            "staged_files": file_list,
            "failed_tasks": failed_tasks,
            "trace": trace,
            "run_id": journal.run_id,
        }
        if mode == "both" and reply_text:
            out["reply"] = reply_text
        return out
    except (RunCancelled, OllamaCancelledError):
        _finish("cancelled")
        return {
            "status": "cancelled",
            "summary": "Run cancelled by user.",
            "staged_files": sorted(generated) if "generated" in locals() else [],
            "failed_tasks": failed_tasks if "failed_tasks" in locals() else [],
            "trace": trace,
            "run_id": journal.run_id,
        }
    except BudgetExhausted as exc:
        _finish("budget_exhausted")
        return {
            "status": "budget_exhausted",
            "summary": f"Run stopped: {exc}",
            "staged_files": sorted(generated) if "generated" in locals() else [],
            "failed_tasks": failed_tasks if "failed_tasks" in locals() else [],
            "trace": trace,
            "run_id": journal.run_id,
        }
