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

from agent_core.coder import ALLOWED_TARGET_SUFFIXES
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

# Re-exported so older imports (tests, e2e, tui) keep resolving.
__all__ = [
    "BudgetExhausted",
    "RunBudgets",
    "RunCancelled",
    "RUNS_DIR_NAME",
    "MAX_ATTEMPTS_PER_TASK",
    "DEFAULT_MAX_WALL_SECONDS",
    "DEFAULT_MAX_LLM_CALLS",
    "run_pipeline",
]


# ===========================================================================
# THE request pipeline: ROUTE -> PLAN -> CODE -> CHECK -> VALIDATE -> SUMMARY
# ===========================================================================

# The one file-type registry (filetypes.py) is the ONLY suffix table; these
# are read-through views so existing attribute access keeps working.
from filetypes import (
    ALLOWED_SUFFIXES,
    PROSE_SUFFIXES,
    TEXT_SUFFIXES,
    check_file,
    extract_content,
    language_of,
)

LANGUAGE_BY_SUFFIX = {  # read-through view over the registry (never edit here)
    ext: e["language"] for ext, e in
    __import__("filetypes").FILE_TYPES.items()
}

# PLACEHOLDER_STEMS stays as WRITE safety (never a content decision): the
# loop's write gate rejects scaffold names like your_code.py/untitled.* —
# a workspace-hygiene rule, not a text-matching routing decision.
PLACEHOLDER_STEMS = {"your_code", "example", "foo", "file1", "untitled", "file", "code", "test"}

MAX_TASK_RETRIES = 2  # retries AFTER the first attempt
VALIDATE_FILE_HEAD_LINES = 60
SUMMARY_FILE_HEAD_LINES = 40


# classify_request (the rules-only pre-check) is deleted with the rules router:
# the agent loop decides, there is no text-matching classification anywhere.


# ---------------------------------------------------------------------------
# FILE_QUESTION: deterministic workspace facts (depth-2 tree + per-file stats)
# ---------------------------------------------------------------------------

FILE_Q_SKIP_DIRS = {".git", ".agent_staging", ".agent_runs",
                    "node_modules", "__pycache__", "env", ".venv", "venv"}
FILE_Q_MAX_ENTRIES = 200
FILE_Q_MAX_DETAIL = 20
FILE_Q_HEAD_LINES = 30
# TEXT_SUFFIXES comes from the registry (imported above).


def _file_language(path: Path) -> str:
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


def build_file_facts(workspace: Path) -> dict[str, Any]:
    """Deterministic scan: tree (depth 2) + per-text-file stats and heads.

    Returns {"tree", "files", "all_files"}: ``files`` carries text-file
    details; ``all_files`` lists EVERY relative path (binaries included) so
    the reply layer can tell "named but binary" apart from "absent"."""
    workspace = Path(workspace).resolve()
    tree_lines: list[str] = []
    details: list[dict[str, Any]] = []
    all_files: list[str] = []
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
            all_files.append(rel)
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
    return {"tree": "\n".join(tree_lines), "files": details, "all_files": all_files}


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
    return language_of(path)


def _valid_plan_task(task: dict[str, Any]) -> str | None:
    """Return a rejection reason, or None when the plan task is acceptable.

    SAFETY rules only (empty path, junk artifact name, unsupported type). The
    old word-count and language-mismatch rules rejected legitimate plans (a
    7-word "Create a README file for the project") and are gone — the model's
    description quality is not a safety concern; the coder + CHECK catch bad
    content."""
    from scanner import is_junk_file

    path = str(task.get("path", "")).strip().replace("\\", "/")
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
    if suffix not in ALLOWED_SUFFIXES:
        return f"unsupported file type: {path}"
    return None


# The plan-level text-matching rules (_valid_plan, invented_file_rejection,
# WEB_HINTS) are deleted with the rules router: the agent loop decides what to
# write; Python enforces safety (safe_target + the write gate), never content
# decisions.



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


# Extraction + the per-file CHECK come from the filetypes registry (one
# suffix table, one check per type). Kept as thin wrappers so existing call
# sites and tests keep working.


def _extract_code(raw: str, rel_path: str = "") -> str:
    """Per-type extraction: prose unwraps one outer fence (inner fences stay);
    code takes the longest fenced block. Registry: filetypes.extract_content."""
    return extract_content(raw, rel_path)


def _check_generated(rel_path: str, content: str) -> str | None:
    """Per-file deterministic CHECK via the registry. Returns error or None.
    An exception escaping this function is an INFRA error (checker crashed)
    and aborts WITHOUT a coder retry."""
    return check_file(rel_path, content)


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
        "Describe only features visible in the code shown, in 2-3 sentences, "
        "then one line on how to run/open what you wrote (from the files, "
        "never invented). Do not invent features. GitHub-flavored markdown: "
        "short, file names in backticks, no headings.\n\n"
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
    return prose


def _summary_skeleton(
    user_request: str,
    generated: dict[str, str],
    failed_tasks: list[dict[str, str]],
    validate_problems: list[str],
    validate_failed: bool,
    total_seconds: float,
    validate_error: str = "",
    validate_ran: bool = False,
) -> str:
    """Deterministic part of the summary — built in code, never by a model.

    GitHub-flavored markdown: bullet lists, file names in backticks, bold
    labels; the UNVERIFIED line is a blockquote. Honest verification wording:
    a syntax-only CHECK pass is never reported as "verified"; when VALIDATE
    fails or is skipped the summary says WHY and how to fix it, and the
    validation problem is listed as the known issue (never "none")."""
    file_list = sorted(generated)
    lines: list[str] = []
    issues = list(validate_problems)
    if validate_failed:
        reason = (validate_error or "unknown error").replace("\n", " ")[:200]
        fix_match = re.search(r"ollama pull \S+", reason)
        fix = fix_match.group(0) if fix_match else (
            "check /models and the validator role config"
        )
        lines.append(
            f"> **UNVERIFIED** — validation could not run: {reason}. Fix: {fix}"
        )
        issues.append(f"validation could not run: {reason}")
    elif validate_ran and generated:
        # Distinguish deterministic checks from the model review explicitly.
        lines.append(
            "**Checks passed** (syntax/links); **validated against your request**: "
            + ("no problems found" if not validate_problems else "see issues below")
        )
    elif generated:
        lines.append("**Checks passed** (syntax/links) — not validated against your request.")
    if file_list:
        lines.append("**Built:**")
        lines.extend(f"- `{f}`" for f in file_list)
    else:
        lines.append("**Built:** — (no files passed CHECK)")
    if failed_tasks:
        lines.append("**Failed:**")
        for t in failed_tasks:
            lines.append(f"- NOT saved (failed): `{t['path']}` — {t['error']}")
    lines.append("**Known issues:** " + ("; ".join(issues) if issues else "none"))
    lines.append(f"**Total time:** {total_seconds:.1f}s")
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
    validate_error: str = "",
    validate_ran: bool = False,
) -> str:
    skeleton = _summary_skeleton(
        user_request, generated, failed_tasks,
        validate_problems, validate_failed, total_seconds,
        validate_error=validate_error, validate_ran=validate_ran,
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
    understand_model: str | None = None,
    validator_missing: str | None = None,
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
    convo_box: dict[str, Any] = {"convo": None}  # set once ConversationState loads

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

    def _reported_llm_calls() -> int:
        """The honest count: real model calls = llm_call events. The local
        ``llm_calls`` counter stays the BUDGET counter (it pre-increments
        before a call and counts one per structured-generate invocation); the
        journal counts exactly one event per actual call."""
        return journal.llm_call_count if journal.llm_call_count else llm_calls

    def _finish(status: str, extra: dict[str, Any] | None = None) -> None:
        state: dict[str, Any] = {
            "run_id": journal.run_id,
            "status": status,
            "request": user_request,
            "intent": trace[0].get("note") if trace else "",
            "roles": {"thinker": planner_model, "coder": coder_model},
            "num_ctx": num_ctx,
            "llm_calls": _reported_llm_calls(),
            "elapsed_seconds": round(time.monotonic() - pipeline_start, 3),
        }
        if extra:
            state.update(extra)
        journal.save_state(state)
        journal.emit_event(
            {"event": "run_finished", "run_id": journal.run_id, "status": status}
        )

    def _record_history(
        path: str,
        outcome: str,
        *,
        label: str = "",
        answer: str = "",
        files_planned: list[str] | None = None,
        files_staged: list[str] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """One history.jsonl entry per turn. Never raises into the flow."""
        try:
            import history as _history

            # Pre-turn snapshot (captured right after load): the state the
            # model actually saw for THIS turn — fixtures depend on it.
            convo_snap: dict[str, Any] = convo_box.get("pre_turn_snapshot") or {}
            _history.append_turn(workspace, {
                "text": user_request,
                "path": path,
                "label": label or None,
                "run_id": journal.run_id,
                "models": {"thinker": planner_model, "coder": coder_model},
                "trace": [
                    {k: t.get(k) for k in ("stage", "status", "seconds", "model") if t.get(k) is not None}
                    for t in trace
                ],
                "llm_calls": _reported_llm_calls(),
                "seconds": round(time.monotonic() - pipeline_start, 3),
                "outcome": outcome,
                "error": error,
                "files_planned": files_planned or [],
                "files_staged": files_staged or [],
                "answer_preview": answer[: _history.ANSWER_PREVIEW],
                "state_snapshot": convo_snap,
            })
        except Exception:
            pass  # history is observability — never breaks a turn

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
        # name for logging + the generate seam from the role. The understand
        # role honors roles.understand.model (the A/B-tested classifier model),
        # defaulting to the coder.
        if model is None:
            model = {"understand": understand_model or coder_model,
                     "coder": coder_model,
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

    try:
        # ------------------------------------------------------------------ ROUTE
        _check_cancel()  # a pre-set cancel still records history via the except
        # The bounded agent loop (agent_loop.py) replaces the old
        # classify/verify/plan dispatch: the model decides (one structured call
        # per step), Python executes tools and enforces safety.
        from agent_loop import agent_turn

        t0 = time.monotonic()
        fp = user_request.strip()
        if fp.startswith("/") and re.match(r"^/\w", fp):
            # Slash commands are dispatched by the client UI; one that reaches
            # the server gets a polite no-op, never files.
            record("FAST-PATH", "ok", 0.0, "command handled by the client")
            _finish("completed")
            _record_history("fast-path", "ok",
                            answer="Slash commands are handled by the client UI.")
            return {
                "answer": "Slash commands are handled by the client UI — "
                          "type /help for the list.",
                "trace": trace,
                "run_id": journal.run_id,
            }

        # Conversation state: the loop sees the last 2 turns as one-line
        # descriptions (what the agent DID, never raw reply text).
        convo = ConversationState.load(workspace)
        convo_box["convo"] = convo
        import history as _history_mod

        convo_box["pre_turn_snapshot"] = _history_mod.state_snapshot(convo)
        message = user_request
        if convo.pending_question:
            message = (
                f"PENDING QUESTION FROM THE AGENT: {convo.pending_question}\n"
                f"USER ANSWER: {user_request}"
            )
        route_model = understand_model or coder_model
        history_lines = [
            f"{t.get('user', '')[:80]} -> {t.get('mode', '')} ({t.get('result', '')[:80]})"
            for t in convo.turns[-2:]
        ]

        # ------------------------------------------------------------- LOOP
        # The model drives: read-only tools first (evidence), writes collected
        # and gated, reply/finish end the turn. Max 6 model steps per turn.
        async def _loop_llm(model, prompt, system="", temperature=0.1, **kwargs):
            return await _llm(model, prompt, system=system, temperature=temperature, **kwargs)

        loop = await agent_turn(
            message,
            workspace,
            generate=_loop_llm,
            llm=_loop_llm,
            history_lines=history_lines,
        )
        record("LOOP", "ok" if loop.stopped_by in ("reply", "finish") else "failed",
               time.monotonic() - t0,
               f"{loop.steps} steps, stopped by {loop.stopped_by}",
               model=route_model)
        for obs in loop.observations:
            journal.emit_event({"event": "loop_step", "text": obs[:300]})

        # A reply with NO pending writes ends the turn as an answer.
        if loop.stopped_by == "reply" and not loop.pending_writes:
            reply_text = loop.answer or "I could not answer that from the workspace."
            convo.record_turn(
                user_request, "reply", f"answered: {reply_text[:80]}",
                trace=trace,
                last_result={"type": "reply",
                             "text": f"answered: {reply_text[:80]}"},
            )
            _finish("completed")
            _record_history("reply", "ok", answer=reply_text)
            return {"answer": reply_text, "trace": trace, "run_id": journal.run_id}

        # The loop ended with no usable outcome: needs_human, never a silent
        # chat fallback. Check failures the loop hit ARE reported as failed
        # tasks (a write that failed CHECK is never invisible).
        if loop.stopped_by in ("budget", "no_progress", "error") and not loop.pending_writes:
            reason = {
                "budget": f"ran out of steps ({loop.steps}) without an answer",
                "no_progress": "the model repeated the same action without progress",
                "error": "the loop hit an error",
            }.get(loop.stopped_by, loop.stopped_by)
            failed_from_checks = [
                {"path": rel, "error": err}
                for rel, err in loop.checks.items() if err
            ]
            record("LOOP", "failed", 0.0, reason)
            _finish("needs_human", {"staged_files": [],
                                    "failed_tasks": failed_from_checks})
            _record_history("code", "needs_human",
                            error={"stage": "LOOP", "detail": reason})
            convo.record_turn(
                user_request, "code", f"needs_human: {reason[:80]}",
                trace=trace,
                last_result={"type": "code", "text": f"needs_human: {reason}"},
            )
            return {
                "status": "needs_human",
                "summary": f"Could not complete that: {reason}",
                "staged_files": [],
                "failed_tasks": failed_from_checks,
                "trace": trace,
                "run_id": journal.run_id,
            }

        # ------------------------------------------------------- CONFIRM GATE
        # The loop collected writes; show them (paths, overwrites, first lines)
        # and require approval before anything is staged. Nothing touched the
        # workspace before this point.
        pending = sorted(loop.pending_writes)
        # A finish/reply with NOTHING to write but check failures on record:
        # needs_human (never a silent "completed" with zero files).
        if not pending and loop.checks and any(loop.checks.values()):
            failed_from_checks = [
                {"path": rel, "error": err}
                for rel, err in loop.checks.items() if err
            ]
            record("LOOP", "failed", 0.0,
                   f"all writes failed CHECK: {len(failed_from_checks)}")
            _finish("needs_human", {"staged_files": [],
                                    "failed_tasks": failed_from_checks})
            _record_history("code", "needs_human",
                            error={"stage": "CHECK",
                                   "detail": "; ".join(t["error"] for t in failed_from_checks)[:500]})
            return {
                "status": "needs_human",
                "summary": "Could not complete that: every write failed its check.",
                "staged_files": [],
                "failed_tasks": failed_from_checks,
                "trace": trace,
                "run_id": journal.run_id,
            }
        if confirm is not None and pending:
            _check_cancel()
            if validator_missing:
                journal.emit_event({
                    "event": "validator_unavailable", "model": validator_missing})
                console_note = (
                    f"\nWARNING: validator model '{validator_missing}' is not "
                    f"installed — validation will fail. Fix: ollama pull "
                    f"{validator_missing}"
                )
            else:
                console_note = ""
            gate_tasks = []
            for rel in pending:
                overwrites = (workspace / rel).is_file()
                content = loop.writes.get(rel) or loop.edits.get(rel) or ""
                gate_tasks.append({
                    "path": rel,
                    "action": "edit" if rel in loop.edits else "create",
                    "description": f"{'; '.join(content.splitlines()[:2])[:120]}",
                    "overwrites": overwrites,
                })
            intent_line = ", ".join(
                f"{t['action']} {t['path']}"
                + (" (overwrites existing file)" if t["overwrites"] else "")
                for t in gate_tasks
            ) + console_note
            journal.emit_event({"event": "confirm_requested",
                                "plan": [t["path"] for t in gate_tasks],
                                "intent": intent_line})
            approved = await confirm(gate_tasks, intent_line)
            if not approved:
                record("LOOP", "ok", 0.0, "declined by user — nothing staged")
                _finish("declined", {"staged_files": [], "failed_tasks": []})
                _record_history("code", "cancelled",
                                answer="Plan declined by user — nothing was staged.",
                                files_planned=pending)
                return {
                    "status": "declined",
                    "summary": "Plan declined by user — nothing was staged.",
                    "staged_files": [],
                    "failed_tasks": [],
                    "trace": trace,
                    "run_id": journal.run_id,
                }
            journal.emit_event({"event": "confirm_approved"})

        # --------------------------------------------------------------- STAGE
        # Writes the loop collected are staged now (post-gate). CHECK already
        # ran per write inside the loop; the verifier gates .py below.
        t_code = time.monotonic()
        generated: dict[str, str] = {}
        failed_tasks: list[dict[str, str]] = []
        check_seconds = 0.0
        infra_error: str | None = None

        for rel in pending:
            content = loop.writes.get(rel) or loop.edits.get(rel) or ""
            err = loop.checks.get(rel)
            if err:
                failed_tasks.append({"path": rel, "error": err})
                continue
            try:
                safe_target(workspace, rel)
            except Exception as exc:
                failed_tasks.append({"path": rel, "error": str(exc)})
                continue
            staging.stage_file(rel, content)
            generated[rel] = content
        # Writes that failed CHECK inside the loop never reached pending —
        # surface them as failed tasks (never invisible).
        for rel, err in loop.checks.items():
            if err and rel not in {t["path"] for t in failed_tasks} and rel not in generated:
                failed_tasks.append({"path": rel, "error": err})

        record("CODE", "ok" if generated else ("failed" if failed_tasks else "ok"),
               time.monotonic() - t_code,
               f"{len(generated)} staged, {len(failed_tasks)} failed",
               model=coder_model)
        record("CHECK", "failed" if (failed_tasks or infra_error) else "ok",
               check_seconds,
               infra_error or f"{len(generated)} passed, {len(failed_tasks)} failed",
               model="-")  # deterministic stage: no model

        # Cross-file CHECK: html links resolve against the FULL generated set
        # (sibling files exist only after every task has been coded).
        if not infra_error:
            generated_paths = set(generated)
            for name in [n for n in generated if n.endswith(".html")]:
                link_error = _check_html_links(name, generated[name], generated_paths, workspace)
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
        validate_error = ""
        validate_ran = False
        if not generated:
            record("VALIDATE", "skipped", 0.0, "nothing passed CHECK", model="-")
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
                        # Budget tick only — the REPORTED llm_calls count comes
                        # from journal llm_call events (one per actual call).
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
                validate_ran = True

                if data.get("satisfies_request") is False:
                    # ONE more loop pass with the validator's problems as
                    # context: the model fixes through the loop's tools (never
                    # a separate fix-call path). New writes stage after their
                    # own checks; the confirm gate already covered this turn.
                    missing_note = "; ".join(validate_problems)[:400]
                    follow_up = await agent_turn(
                        f"{message}\n\nVALIDATION PROBLEMS TO FIX: {missing_note}",
                        workspace,
                        generate=_loop_llm,
                        llm=_loop_llm,
                        history_lines=history_lines,
                        max_steps=3,
                    )
                    for obs in follow_up.observations:
                        journal.emit_event({"event": "loop_step",
                                            "text": ("fix: " + obs)[:300]})
                    for rel, content in {**follow_up.writes, **follow_up.edits}.items():
                        err = follow_up.checks.get(rel)
                        if err is not None:
                            validate_problems.append(f"{rel}: fix failed CHECK ({err})")
                            continue
                        try:
                            safe_target(workspace, rel)
                        except Exception as exc:
                            validate_problems.append(f"{rel}: fix rejected ({exc})")
                            continue
                        staging.stage_file(rel, content)
                        generated[rel] = content
                    validate_note = "fix pass applied" if not validate_problems else "problems remain"
                else:
                    validate_note = "satisfies request"
                record("VALIDATE", "ok", time.monotonic() - t0, validate_note,
                       model=planner_model)
            except Exception as exc:
                validate_failed = True
                validate_error = str(exc)[:300]
                record("VALIDATE", "failed", time.monotonic() - t0, validate_error,
                       model=planner_model)

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
            validate_error=validate_error,
            validate_ran=validate_ran,
        )
        file_list = sorted(generated)
        record("SUMMARY", "ok", time.monotonic() - t0,
               "UNVERIFIED" if validate_failed else "", model=planner_model)

        _finish("completed", {"staged_files": file_list,
                              "failed_tasks": failed_tasks})
        # Conversation state: every turn recorded, described (never raw text).
        convo.record_turn(
            user_request, "code",
            "wrote " + ", ".join(file_list) if file_list else "no files passed CHECK",
            trace=trace,
            last_result={
                "type": "code",
                "text": (
                    "wrote " + ", ".join(file_list) if file_list
                    else "needs_human: no files passed CHECK"
                ),
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
        if loop.answer:
            out["reply"] = loop.answer
        # Honest outcome vocabulary: ok | unverified (code ran but validation
        # failed/was skipped) | partial (some tasks failed) — see also
        # needs_human / failed / cancelled / error on the other paths.
        outcome = (
            "unverified" if validate_failed
            else ("partial" if failed_tasks else "ok")
        )
        _record_history(
            "code", outcome, answer=summary, files_planned=pending,
            files_staged=file_list,
        )
        return out
    except (RunCancelled, OllamaCancelledError):
        _finish("cancelled")
        _record_history(
            "code", "cancelled",
            files_staged=sorted(generated) if "generated" in locals() else [],
        )
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
        _record_history(
            "code", "failed",
            error={"stage": "budget", "detail": str(exc)[:500]},
            files_staged=sorted(generated) if "generated" in locals() else [],
        )
        return {
            "status": "budget_exhausted",
            "summary": f"Run stopped: {exc}",
            "staged_files": sorted(generated) if "generated" in locals() else [],
            "failed_tasks": failed_tasks if "failed_tasks" in locals() else [],
            "trace": trace,
            "run_id": journal.run_id,
        }
    except Exception as exc:
        # Any other failure: the turn still lands in history with the error,
        # and the traceback goes into events.jsonl. Re-raised so main.py's
        # structured-error envelope carries stage + detail + run_id.
        import traceback as _tb

        tb_text = _tb.format_exc()
        stage = trace[-1]["stage"] if trace else "unknown"
        try:
            journal.emit_event({"event": "exception", "stage": stage,
                                "error": str(exc)[:500], "traceback": tb_text})
        except Exception:
            pass
        _finish("failed")
        _record_history(
            "error", "error",
            error={"stage": stage, "detail": str(exc)[:500]},
        )
        raise
