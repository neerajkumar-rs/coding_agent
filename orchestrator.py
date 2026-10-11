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
from agent_core.run_state import RUNS_DIR_NAME, RunJournal, diagnostic_hash
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

QUESTION_HINTS = (
    "what ", "why ", "how ", "explain", "difference", " vs ", " versus ",
    "who ", "when ", "which ", "is it ", "are there", "can you explain",
)
CODE_HINTS = (
    "create", "make", "build", "write", "implement", "add", "fix",
    "refactor", "generate", "update", "modify", "site", "page", "app",
    "script", "function", "endpoint", "html", "css", "blog",
)
PLACEHOLDER_STEMS = {"your_code", "example", "foo", "file1", "untitled", "file", "code", "test"}
WEB_HINTS = ("html", "css", "website", "web page", "webpage", "site", "blog", "landing", "page")

MAX_TASK_RETRIES = 2  # retries AFTER the first attempt
VALIDATE_FILE_HEAD_LINES = 60
SUMMARY_FILE_HEAD_LINES = 40
# Feature keywords the summary model may NOT invent: dropped when absent from
# every generated file's actual content.
SUMMARY_FEATURE_KEYWORDS = ("search", "login", "database", "dark mode", "comments")


FILE_QUESTION_HINTS = (
    "what is in my", "what's in my", "whats in my", "list files", "list the files",
    "which files", "what files", "show files", "show me the files",
    "which language", "what language", "summarize my files", "summarise my files",
    "describe my files", "my workspace", "my project", "in this dir", "in my dir",
    "in this folder", "in this directory",
)
CODE_VERBS = (
    "create", "write", "make", "build", "add", "fix", "implement",
    "generate", "update", "modify", "refactor",
)


def classify_request(user_request: str) -> str:
    """Rules-only ROUTE: QUESTION / FILE_QUESTION / CODE_TASK / UNCLEAR.
    No model call is involved. Files are only ever created for CODE_TASK."""
    r = user_request.strip().lower()
    if not r:
        return "UNCLEAR"
    has_ext = bool(re.search(r"\b\w+\.(py|html?|css|js|json|md|txt)\b", r))
    has_verb = any(v in r for v in CODE_VERBS)
    if has_verb or has_ext:
        return "CODE_TASK"
    if any(h in r for h in FILE_QUESTION_HINTS):
        return "FILE_QUESTION"
    if any(h in r for h in QUESTION_HINTS) or r.endswith("?"):
        return "QUESTION"
    # A bare statement with no verb, no file, and no question mark tells us
    # nothing — ask instead of guessing.
    if len(r.split()) < 4:
        return "UNCLEAR"
    return "CODE_TASK"


# ---------------------------------------------------------------------------
# FILE_QUESTION: deterministic workspace facts (depth-2 tree + per-file stats)
# ---------------------------------------------------------------------------

FILE_Q_SKIP_DIRS = {".git", ".agent_staging", ".agent_runs", ".agent_logs",
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


def _fallback_plan(user_request: str) -> list[dict[str, str]]:
    """Deterministic plan when the planner fails validation twice."""
    lower = user_request.lower()
    if any(h in lower for h in WEB_HINTS):
        return [
            {"path": "index.html", "action": "create",
             "description": "Main HTML5 page structure that links the stylesheet and script files and presents the content sections."},
            {"path": "style.css", "action": "create",
             "description": "Stylesheet with layout, typography, colors and responsive rules for the page structure."},
            {"path": "script.js", "action": "create",
             "description": "JavaScript providing the dynamic content and interactions requested for the page."},
        ]
    words = user_request.split()
    desc = "Python implementation for the request: " + " ".join(words[:40])
    if len(desc.split()) < 8:
        desc += " with clear structure and entry point"
    return [{"path": "app.py", "action": "create", "description": desc}]


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
    coder_model: str,
) -> str:
    """One coder-model call: 2-3 sentences describing ONLY features visible in
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
            coder_model, prompt,
            system="You describe only what the code visibly does.",
            temperature=TEMPERATURE_PLAN,
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
    coder_model: str,
) -> str:
    skeleton = _summary_skeleton(
        user_request, generated, failed_tasks,
        validate_problems, validate_failed, total_seconds,
    )
    prose = await _summary_prose(user_request, generated, generate, coder_model)
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

    def record(stage: str, status: str, seconds: float, note: str = "") -> None:
        trace.append({"stage": stage, "status": status,
                      "seconds": round(seconds, 1), "note": note})
        journal.emit_event(
            {"event": "stage", "stage": stage, "status": status,
             "seconds": round(seconds, 1), "note": note}
        )

    def _finish(status: str, extra: dict[str, Any] | None = None) -> None:
        state: dict[str, Any] = {
            "run_id": journal.run_id,
            "status": status,
            "request": user_request,
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

    async def _llm(model: str, prompt: str, system: str = "", temperature: float = 0.2) -> str:
        """Single LLM choke point: budget the prompt, count + cap the call,
        and honour cancellation immediately before hitting the model."""
        nonlocal llm_calls
        _check_cancel()
        _check_wall_time()
        if llm_calls >= budgets.max_llm_calls:
            raise BudgetExhausted(
                f"max total LLM calls exceeded ({budgets.max_llm_calls})"
            )
        llm_calls += 1
        return await generate(model, budget.trim(prompt), system=system, temperature=temperature)

    _check_cancel()

    try:
        # ------------------------------------------------------------------ ROUTE
        t0 = time.monotonic()
        route = classify_request(user_request)
        record("ROUTE", "ok", time.monotonic() - t0, route)

        if route == "QUESTION":
            t0 = time.monotonic()
            # Depth-1 file tree as context; the workspace name itself never
            # enters the prompt.
            facts = build_file_facts(workspace)
            depth1 = "\n".join(
                line for line in facts["tree"].splitlines() if not line.startswith("  ")
            )
            raw = await _llm(
                coder_model,
                "Answer concisely and accurately. The user's directory contains:\n"
                + depth1 + "\n\nQUESTION: " + user_request,
                system="You are a precise technical assistant.",
                temperature=TEMPERATURE_PLAN,
            )
            _, answer = strip_think_blocks(raw)
            record("ANSWER", "ok", time.monotonic() - t0)
            _finish("completed")
            return {"answer": answer.strip(), "trace": trace, "run_id": journal.run_id}

        if route == "FILE_QUESTION":
            t0 = time.monotonic()
            facts = build_file_facts(workspace)
            table = format_file_table(facts)
            wants_prose = any(
                w in user_request.lower() for w in ("explain", "summarize", "summarise", "describe")
            )
            if wants_prose and facts["files"]:
                # ONE model call, facts only — never invent files, never name the
                # workspace. The deterministic table is included in the answer.
                facts_block = "\n".join(
                    f"- {f['name']} ({f['language']}, {f['lines']} lines):\n{f['head']}"
                    for f in facts["files"]
                )
                raw = await _llm(
                    coder_model,
                    "Describe only the files listed. Do not invent files.\n\n"
                    + facts_block + "\n\nREQUEST: " + user_request,
                    system="You describe only the files you are shown.",
                    temperature=TEMPERATURE_PLAN,
                )
                _, prose = strip_think_blocks(raw)
                answer = table + "\n\n" + prose.strip()
            else:
                answer = table
            record("ANSWER", "ok", time.monotonic() - t0,
                   f"{len(facts['files'])} files scanned")
            _finish("completed")
            return {"answer": answer, "trace": trace, "run_id": journal.run_id}

        if route == "UNCLEAR":
            record("ANSWER", "ok", 0.0, "needs clarification")
            _finish("completed")
            return {
                "answer": "Could you tell me a bit more — should I create or change "
                          "code (which file/feature), or answer a question?",
                "trace": trace,
                "run_id": journal.run_id,
            }

        # ------------------------------------------------------------------ PLAN
        t0 = time.monotonic()
        plan_prompt = (
            "You are a software planner. Return ONLY a JSON array of tasks, each "
            '{"path": "<relative file path>", "action": "create"|"edit", '
            '"description": "<what to build, at least 8 words>"}. '
            "Invent concrete, descriptive filenames from the request itself. "
            "Every description must be at least 8 words and match the file's "
            "language. A web page request must produce an HTML file, a CSS file "
            "and a JavaScript file. No code, JSON only.\n\nREQUEST: " + user_request
        )
        tasks: list[dict[str, Any]] = []
        last_rejects: list[str] = []

        def _normalize(candidate: dict[str, Any]) -> dict[str, Any] | None:
            """Accept any plan-task shape at the edge; normalize to the single
            unified format {path, action, description}."""
            path = candidate.get("path") or candidate.get("file") or ""
            files = candidate.get("target_files")
            if not path and isinstance(files, list) and files:
                path = str(files[0])
            desc = candidate.get("description") or candidate.get("spec") or candidate.get("name") or ""
            action = str(candidate.get("action", "create")).lower()
            if action not in ("create", "edit"):
                action = "create"
            path = str(path).strip().replace("\\", "/")
            if not path:
                return None
            return {"path": path, "action": action, "description": str(desc).strip()}

        # Schema-constrained attempt first (when the seam supports it): Ollama
        # `format` + pydantic validation, THEN the rules. Falls back to the
        # JSON-regex prompt path below on any structured failure.
        if structured_generate is not None:
            try:
                _check_cancel()
                llm_calls += 1
                plan_obj = await structured_generate(
                    planner_model,
                    budget.trim(plan_prompt),
                    PipelinePlan,
                )
                for task in plan_obj.tasks:
                    normalized = _normalize(task.model_dump())
                    if normalized is None:
                        continue
                    reason = _valid_plan_task(normalized)
                    if reason:
                        last_rejects.append(reason)
                    else:
                        tasks.append(normalized)
            except Exception as exc:
                last_rejects.append(f"structured plan failed: {exc}")
                tasks = []

        for plan_attempt in range(2 if not tasks else 0):
            prompt = plan_prompt
            if last_rejects:
                prompt += "\n\nPrevious plan was rejected: " + "; ".join(last_rejects)
            raw = await _llm(
                planner_model, prompt,
                system="You output only strict JSON task arrays.",
                temperature=TEMPERATURE_PLAN,
            )
            _, clean = strip_think_blocks(raw)
            candidates = _parse_plan_json(clean)
            tasks, last_rejects = [], []
            for candidate in candidates:
                normalized = _normalize(candidate)
                if normalized is None:
                    last_rejects.append("task has no file path")
                    continue
                reason = _valid_plan_task(normalized)
                if reason:
                    last_rejects.append(reason)
                else:
                    tasks.append(normalized)
            if tasks:
                # Plan-level rules (whole-set): app.py only when requested, web
                # requests must cover html+css+js.
                plan_reasons = _valid_plan(tasks, user_request)
                if plan_reasons:
                    last_rejects.extend(plan_reasons)
                    tasks = []
            if tasks:
                break
            if plan_attempt == 0 and not last_rejects:
                last_rejects = ["planner returned no parseable JSON task array"]
        if not tasks:
            tasks = _fallback_plan(user_request)
            record("PLAN", "ok", time.monotonic() - t0,
                   f"deterministic fallback ({len(tasks)} tasks)")
        else:
            record("PLAN", "ok", time.monotonic() - t0, f"{len(tasks)} tasks")

        planned_paths = {t["path"] for t in tasks}

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
            code_prompt = (
                f"Implement exactly one file.\n"
                f"ORIGINAL USER REQUEST: {user_request}\n"
                f"PLANNED FILES: {', '.join(sorted(planned_paths))}\n"
                f"TARGET FILE: {path}\n"
                f"TASK: {desc}\n"
                f"LANGUAGE: {language}\n"
                f"Return ONLY the complete file contents inside one "
                f"```{language} fenced block. No commentary.{extra}"
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
        if python_staged and not infra_error:
            v0 = time.monotonic()
            try:
                verify = await asyncio.to_thread(
                    verify_files, python_staged, staging.staging_dir, None, workspace
                )
            except Exception as exc:  # verifier crashed -> infra error, no retry
                infra_error = f"verifier infrastructure error: {exc}"
                check_seconds += time.monotonic() - v0
            else:
                check_seconds += time.monotonic() - v0
                if not verify.passed:
                    diagnostics = tail_lines(verify.diagnostics(), 30)
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
        t0 = time.monotonic()
        summary = await _build_summary(
            user_request=user_request,
            generated=generated,
            failed_tasks=failed_tasks,
            validate_problems=validate_problems,
            validate_failed=validate_failed,
            total_seconds=time.monotonic() - pipeline_start,
            generate=generate,
            coder_model=coder_model,
        )
        file_list = sorted(generated)
        record("SUMMARY", "ok", time.monotonic() - t0,
               "UNVERIFIED" if validate_failed else "")

        _finish("completed", {"staged_files": file_list,
                              "failed_tasks": failed_tasks})
        return {
            "summary": summary,
            "staged_files": file_list,
            "failed_tasks": failed_tasks,
            "trace": trace,
            "run_id": journal.run_id,
        }
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
