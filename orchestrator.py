"""Retry-bounded agentic coding pipeline (Thinker -> Coder -> Validator).

Phase 1 hardening:
- No Writer/Fetcher theatre: the pipeline never creates tasks.md, summary.txt
  or taskN_code.txt inside the workspace. Run metadata lives in
  ``.agent_runs/<run_id>/{state.json,events.jsonl,tasks.md}`` (git-ignored).
- No silent scaffolding: if the coder returns nothing usable the attempt FAILS
  with a clear reason (``_fallback_files``/``_scaffold`` are gone).
- Run budgets: max wall time, max total LLM calls, max attempts per task (2),
  and a no-progress rule (two consecutive attempts with the same diagnostic
  hash -> the task is stopped and marked ``needs_human``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from ollama_client import (
    TEMPERATURE_CODE,
    TEMPERATURE_PLAN,
    OllamaEngine,
    OllamaTruncatedError,
)

# Native local Ollama models supporting strict VRAM limits (RTX 3050 4GB ceiling)
THINKER_MODEL = "deepseek-r1:1.5b"
DEBUGGER_MODEL = THINKER_MODEL
CODER_MODEL = "qwen2.5-coder:3b"
MAX_RETRIES = 3
MAX_ATTEMPTS_PER_TASK = 2
MAX_SPEC_WORDS = 300
THINKER_INPUT_CHARS = 16_000
CODER_SPEC_CHARS = 8_000
MAX_LOG_CHARS = 8_000
MAX_GENERATED_CODE_CHARS = 100_000
ALLOWED_TARGET_SUFFIXES = {".py", ".html", ".css", ".js"}
WORKSPACE_FILE_SUFFIXES = ALLOWED_TARGET_SUFFIXES | {".json", ".md", ".txt"}
MAX_WORKSPACE_FILES = 60
MAX_WORKSPACE_CONTEXT_CHARS = 12_000
MAX_FILE_SNIPPET_CHARS = 3_000
MAX_TASKS = 12
RUNS_DIR_NAME = ".agent_runs"
VALIDATION_TIMEOUT_SECONDS = int(os.getenv("VIBE_VALIDATION_TIMEOUT_SECONDS", "300"))

DEFAULT_MAX_WALL_SECONDS = float(os.getenv("VIBE_MAX_WALL_SECONDS", "900"))
DEFAULT_MAX_LLM_CALLS = int(os.getenv("VIBE_MAX_LLM_CALLS", "40"))


class BudgetExhausted(RuntimeError):
    """Raised when a run budget (wall time / LLM calls) is exceeded."""


@dataclass
class RunBudgets:
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    max_attempts_per_task: int = MAX_ATTEMPTS_PER_TASK


@dataclass
class RunResult:
    status: str
    run_id: str = ""
    target_file: str = ""
    files: list[str] = field(default_factory=list)
    specification: str = ""
    code: str = ""
    attempts: int = 0
    stdout: str = ""
    stderr: str = ""
    llm_validation: str = ""
    errors: list[str] = field(default_factory=list)
    needs_human: list[str] = field(default_factory=list)
    llm_calls: int = 0
    elapsed_seconds: float = 0.0


class _RunJournal:
    """Persists run metadata under .agent_runs/<run_id>/ (never in workspace root)."""

    def __init__(self, workspace: Path, run_id: str | None = None) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"{stamp}-{uuid4().hex[:6]}"
        self.run_dir = workspace / RUNS_DIR_NAME / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.tasks_path = self.run_dir / "tasks.md"

    def write_task_plan(self, tasks: list[dict[str, str]]) -> None:
        lines = ["# Agent run task plan", "", "Tasks are processed sequentially.", ""]
        for task in tasks:
            lines.extend(
                [
                    f"## TASK {task['number']}: {task['name']} [{task['status']}]",
                    task["spec"][:CODER_SPEC_CHARS],
                    "",
                ]
            )
        self.tasks_path.write_text("\n".join(lines), encoding="utf-8")

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def emit_event(self, event: dict[str, Any]) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


class VibeOrchestrator:
    def __init__(
        self,
        engine: OllamaEngine | None = None,
        workspace: Path | str = ".",
        max_retries: int = MAX_ATTEMPTS_PER_TASK,
        budgets: RunBudgets | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.engine = engine or OllamaEngine(workspace_root=self.workspace)
        if budgets is None:
            budgets = RunBudgets(max_attempts_per_task=max_retries)
        budgets.max_attempts_per_task = max(1, min(budgets.max_attempts_per_task, MAX_RETRIES))
        self.budgets = budgets
        self._llm_calls = 0
        self._start = 0.0

    # ------------------------------------------------------------------
    # Budget-guarded LLM access
    # ------------------------------------------------------------------

    def _chat(self, model: str, prompt: str, *, max_tokens: int, temperature: float) -> str:
        if self._llm_calls >= self.budgets.max_llm_calls:
            raise BudgetExhausted(
                f"max total LLM calls exceeded ({self.budgets.max_llm_calls})"
            )
        self._llm_calls += 1
        return self.engine.chat(model, prompt, max_tokens=max_tokens, temperature=temperature)

    def _check_wall_time(self) -> None:
        elapsed = time.monotonic() - self._start
        if elapsed > self.budgets.max_wall_seconds:
            raise BudgetExhausted(
                f"max wall time exceeded ({self.budgets.max_wall_seconds:.0f}s)"
            )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        user_request: str,
        target_file: str | None = None,
        test_command: list[str] | None = None,
        progress: Callable[[str], None] | None = None,
        workspace_dir: str | Path | None = None,
    ) -> RunResult:
        def report(message: str) -> None:
            if progress:
                progress(message)

        if not user_request.strip():
            raise ValueError("request must not be empty")
        report("preparing workspace and safety checks")
        self._set_workspace(workspace_dir)
        self._llm_calls = 0
        self._start = time.monotonic()
        journal = _RunJournal(self.workspace)
        result = RunResult("running", run_id=journal.run_id)
        journal.emit_event({"event": "run_started", "run_id": journal.run_id, "request": user_request[:500]})
        journal.save_state({"run_id": journal.run_id, "status": "running", "request": user_request})

        try:
            report("tool • workspace.list_files: scanning the selected workspace")
            workspace_context = self._workspace_context(user_request, report)
            report("stage 1/3 • thinker: breaking request into sequential tasks")
            plan = self._think(user_request[:THINKER_INPUT_CHARS], workspace_context)
            tasks = self._parse_tasks(plan)
            journal.write_task_plan(tasks)
            journal.emit_event({"event": "plan_ready", "tasks": len(tasks)})
            result.specification = plan

            for task_number, task in enumerate(tasks, 1):
                self._check_wall_time()
                report(f"task {task_number}/{len(tasks)} • {task['name']}")
                journal.emit_event(
                    {"event": "task_started", "task": task_number, "name": task["name"]}
                )
                outcome = self._run_task(
                    task_number, task, test_command, journal, report, result
                )
                journal.write_task_plan(tasks)
                journal.save_state(
                    {
                        "run_id": journal.run_id,
                        "status": result.status if outcome != "complete" else "running",
                        "tasks": tasks,
                        "llm_calls": self._llm_calls,
                    }
                )
                if outcome == "complete":
                    report(f"task {task_number} complete • moving to next task")
                    continue
                if outcome == "needs_human":
                    result.status = "needs_human"
                    journal.emit_event(
                        {"event": "task_needs_human", "task": task_number, "name": task["name"]}
                    )
                    return self._finalize(result, journal)
                result.status = "failed"
                return self._finalize(result, journal)

            result.status = "completed"
            report("all tasks complete • validated code is ready")
        except BudgetExhausted as exc:
            result.status = "budget_exhausted"
            result.errors.append(str(exc))
            journal.emit_event({"event": "budget_exhausted", "reason": str(exc)})
            report(f"run stopped: {exc}")
        return self._finalize(result, journal)

    def _finalize(self, result: RunResult, journal: _RunJournal) -> RunResult:
        result.llm_calls = self._llm_calls
        result.elapsed_seconds = round(time.monotonic() - self._start, 3)
        journal.save_state(
            {
                "run_id": journal.run_id,
                "status": result.status,
                "attempts": result.attempts,
                "llm_calls": result.llm_calls,
                "elapsed_seconds": result.elapsed_seconds,
                "files": result.files,
                "needs_human": result.needs_human,
                "errors": result.errors,
            }
        )
        journal.emit_event(
            {
                "event": "run_finished",
                "run_id": journal.run_id,
                "status": result.status,
                "llm_calls": result.llm_calls,
                "seconds": result.elapsed_seconds,
            }
        )
        return result

    # ------------------------------------------------------------------
    # One task: coder -> validate (deterministic) -> LLM judge, bounded
    # ------------------------------------------------------------------

    def _run_task(
        self,
        task_number: int,
        task: dict[str, str],
        test_command: list[str] | None,
        journal: _RunJournal,
        report: Callable[[str], None],
        result: RunResult,
    ) -> str:
        task_spec = task["spec"]
        previous_diag_hash: str | None = None

        for attempt in range(1, self.budgets.max_attempts_per_task + 1):
            self._check_wall_time()
            result.attempts += 1
            report(f"task {task_number} • coder • attempt {attempt}")
            journal.emit_event(
                {"event": "attempt_started", "task": task_number, "attempt": attempt}
            )

            # --- Coder (truncation is retryable with a smaller budget, never a FAIL;
            #     unusable output FAILS the attempt with a clear reason — no scaffolds)
            files: dict[str, str] | None = None
            coder_error: str | None = None
            spec_chars = CODER_SPEC_CHARS
            for _trunc_retry in range(2):
                try:
                    files = self._code(task_spec[:spec_chars])
                    coder_error = None
                    break
                except OllamaTruncatedError as exc:
                    report(f"task {task_number} • {exc}; retrying with reduced spec")
                    journal.emit_event(
                        {
                            "event": "response_truncated",
                            "task": task_number,
                            "attempt": attempt,
                            "action": "retry_with_reduced_spec",
                        }
                    )
                    spec_chars = max(spec_chars // 2, 1_000)
                except ValueError as exc:
                    coder_error = str(exc)
                    break

            if files is None:
                diagnostics = coder_error or (
                    "coder response truncated even after reducing the spec budget"
                )
                local_passed, llm_passed = False, False
                stdout, stderr = "", diagnostics
            else:
                total_size = sum(len(content) for content in files.values())
                if total_size > MAX_GENERATED_CODE_CHARS:
                    raise ValueError("generated task output exceeds the configured safety limit")

                for relative_name, content in files.items():
                    target = self._safe_target(relative_name)
                    report(f"tool • filesystem.write_file: {target}")
                    self._write_file(target, content)
                result.files = list(dict.fromkeys(result.files + list(files)))
                result.code = "\n\n".join(files.values())

                empty_files = [name for name, content in files.items() if not content.strip()]
                if empty_files:
                    local_passed, llm_passed = False, False
                    stdout, stderr = "", "FILE EMPTY: " + ", ".join(empty_files)
                    llm_feedback = ""
                else:
                    local_passed, stdout, stderr = self._validate(files, test_command)
                    if local_passed:
                        report(f"task {task_number} • validator: LLM review")
                        llm_passed, llm_feedback = self._llm_validate(task_spec, files)
                        result.llm_validation = llm_feedback
                    else:
                        llm_passed, llm_feedback = False, ""
                result.stdout, result.stderr = stdout, stderr
                diagnostics = (
                    stdout + "\n" + stderr + ("\nLLM VALIDATION:\n" + llm_feedback if llm_feedback else "")
                ).strip()[-MAX_LOG_CHARS:]

                if local_passed and llm_passed:
                    task["status"] = "complete"
                    journal.emit_event(
                        {"event": "task_complete", "task": task_number, "attempt": attempt}
                    )
                    return "complete"

            # --- Attempt failed: no-progress detection + debugger refinement
            result.errors.append(f"Task {task_number} attempt {attempt}: {diagnostics[:500]}")
            task["status"] = "failed"
            journal.emit_event(
                {
                    "event": "attempt_failed",
                    "task": task_number,
                    "attempt": attempt,
                    "diagnostics": diagnostics[:1_000],
                }
            )
            diag_hash = hashlib.sha256(
                " ".join(diagnostics.split()).encode("utf-8")
            ).hexdigest()
            if previous_diag_hash is not None and diag_hash == previous_diag_hash:
                report(f"task {task_number} • no progress between attempts; needs human")
                journal.emit_event(
                    {"event": "no_progress", "task": task_number, "diagnostic_hash": diag_hash}
                )
                task["status"] = "needs_human"
                result.needs_human.append(task["name"])
                return "needs_human"
            previous_diag_hash = diag_hash

            if attempt < self.budgets.max_attempts_per_task:
                report(f"task {task_number} failed • thinker is refining this task")
                task_spec = self._debug(task_spec, diagnostics)
                task["spec"] = task_spec

        return "failed"

    # ------------------------------------------------------------------
    # Roles (thinker / coder / debugger / validator)
    # ------------------------------------------------------------------

    def _think(self, request_text: str, workspace_context: str) -> str:
        prompt = (
            "You are the thinker and task planner. Break the messy request into "
            "small ordered atomic tasks. Return ONLY Markdown using this exact shape:\n"
            "## TASK 1: short task name\n"
            "- files: relative/file.py\n"
            "- logic: signatures, parameters, and rules\n"
            "Create no more than 12 tasks and keep every task under 120 words. "
            "Do not generate code. Each task must be independently codable and testable.\n\n"
            f"WORKSPACE:\n{self.workspace}\n\nEXISTING FILE CONTEXT:\n"
            f"{workspace_context}\n\nREQUEST:\n{request_text}"
        )
        spec = self._chat(
            THINKER_MODEL, prompt, max_tokens=900, temperature=TEMPERATURE_PLAN
        )
        return spec[:CODER_SPEC_CHARS]

    def _parse_tasks(self, plan: str) -> list[dict[str, str]]:
        matches = re.findall(
            r"##\s*TASK\s*(\d+)\s*:\s*([^\r\n]+)\r?\n(.*?)(?=##\s*TASK|\Z)",
            plan,
            re.IGNORECASE | re.DOTALL,
        )
        tasks = [
            {"number": number, "name": name.strip(), "spec": body.strip(), "status": "pending"}
            for number, name, body in matches[:MAX_TASKS]
            if body.strip()
        ]
        if tasks:
            return tasks
        return [{"number": "1", "name": "primary implementation", "spec": plan, "status": "pending"}]

    def _code(self, specification: str) -> dict[str, str]:
        prompt = (
            "You are a pure code generator. Use ONLY this specification. Return "
            "one JSON object with this shape: {\"filename\": \"relative/path.ext\", "
            "\"description\": \"brief purpose\", \"code\": \"complete file contents\"}. "
            "If multiple files are required, return {\"files\": [objects with those "
            "three fields]}. Return JSON only, use relative paths, and do not invent "
            "requirements.\n\nSPECIFICATION:\n"
            + specification
        )
        response = self._chat(
            CODER_MODEL, prompt, max_tokens=1800, temperature=TEMPERATURE_CODE
        )
        try:
            parsed = json.loads(response.strip().strip("`"))
            objects = (
                parsed.get("files", [])
                if isinstance(parsed, dict) and "files" in parsed
                else [parsed]
            )
            if objects and all(isinstance(item, dict) for item in objects):
                json_files: dict[str, str] = {}
                for item in objects:
                    name = str(item.get("filename", "")).replace("\\", "/").strip()
                    content = item.get("code")
                    if not name or not isinstance(content, str):
                        raise ValueError("Coder JSON objects require filename and code")
                    if name in json_files or name.startswith("/") or ".." in name.split("/"):
                        raise ValueError(f"unsafe or duplicate generated file path: {name}")
                    if Path(name).suffix.lower() not in ALLOWED_TARGET_SUFFIXES:
                        raise ValueError(f"unsupported generated file type: {name}")
                    json_files[name] = content if content.endswith("\n") else content + "\n"
                if json_files:
                    return json_files
        except json.JSONDecodeError:
            pass
        matches = re.findall(
            r"```FILE:\s*([^\r\n`]+?)(?:```|\r?\n)(.*?)(?:```|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if not matches:
            raise ValueError(
                "coder returned no usable files (no JSON object or ```FILE: blocks found); "
                f"response excerpt: {response[:200]!r}"
            )
        files: dict[str, str] = {}
        for name, content in matches:
            # Some models accidentally attach the closing fence to the
            # filename (`FILE:pages/index.html```). Never treat fence markup
            # as part of the path.
            safe_name = name.split("```", 1)[0].strip().strip("`").replace("\\", "/")
            safe_name = re.sub(r"^FILE:\s*", "", safe_name, flags=re.IGNORECASE)
            safe_name = safe_name.rstrip("`").strip()
            if safe_name in files or safe_name.startswith("/") or ".." in safe_name.split("/"):
                raise ValueError(f"unsafe or duplicate generated file path: {name}")
            if Path(safe_name).suffix.lower() not in ALLOWED_TARGET_SUFFIXES:
                raise ValueError(f"unsupported generated file type: {name}")
            if not content.strip():
                raise ValueError(f"coder produced an empty file: {safe_name}")
            files[safe_name] = content.strip() + "\n"
        return files

    def _debug(self, specification: str, diagnostics: str) -> str:
        prompt = (
            "You are a debugging context parser. Update the Markdown specification "
            "under 300 words using ONLY the existing specification and test logs. "
            "Keep signatures and logic rules; state the minimal correction. No code.\n\n"
            f"SPECIFICATION:\n{specification}\n\nTEST LOG:\n{diagnostics}"
        )
        return " ".join(
            self._chat(DEBUGGER_MODEL, prompt, max_tokens=900, temperature=TEMPERATURE_PLAN).split()[
                :MAX_SPEC_WORDS
            ]
        )

    def _llm_validate(
        self, specification: str, files: dict[str, str]
    ) -> tuple[bool, str]:
        bundled_files = "\n\n".join(
            f"FILE: {name}\n{content[:12000]}" for name, content in files.items()
        )[:24000]
        prompt = (
            "You are the validation agent. Compare the generated files against "
            "the specification. Check required behavior, syntax-level problems, "
            "file references, and obvious runtime defects. Do not rewrite code. "
            "Respond exactly with PASS or FAIL on the first line, followed by a "
            "short reason. Choose FAIL for any required behavior that is missing.\n\n"
            f"SPECIFICATION:\n{specification}\n\nGENERATED FILES:\n{bundled_files}"
        )
        feedback = self._chat(
            DEBUGGER_MODEL, prompt, max_tokens=700, temperature=TEMPERATURE_PLAN
        )
        first_line = feedback.splitlines()[0].strip().upper() if feedback else ""
        return first_line == "PASS", feedback[-MAX_LOG_CHARS:]

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def _workspace_context(
        self, request_text: str, report: Callable[[str], None]
    ) -> str:
        """Find relevant bounded snippets for the thinker, never the coder."""
        files = self._list_workspace_files()
        if not files:
            return "(workspace is empty)"
        request_lower = request_text.lower()
        ranked = sorted(
            files,
            key=lambda path: (
                0 if path.name.lower() in request_lower else 1,
                0 if path.suffix.lower() in {".py", ".html", ".js", ".css"} else 1,
                str(path),
            ),
        )
        sections: list[str] = [f"FILES ({len(files)} found):"]
        remaining = MAX_WORKSPACE_CONTEXT_CHARS
        for path in ranked[:MAX_WORKSPACE_FILES]:
            relative = path.relative_to(self.workspace)
            sections.append(f"- {relative}")
            if remaining <= 0:
                break
            try:
                report(f"tool • filesystem.read_snippet: {path}")
                snippet = self._read_file_snippet(path)
            except OSError:
                continue
            if snippet:
                section = f"\n[{relative}]\n{snippet}"
                sections.append(section[:remaining])
                remaining -= len(section)
        return "\n".join(sections)[:MAX_WORKSPACE_CONTEXT_CHARS]

    def _list_workspace_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.workspace.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in WORKSPACE_FILE_SUFFIXES
                and not any(part.startswith(".") for part in path.relative_to(self.workspace).parts)
            ):
                files.append(path)
        return sorted(files)

    def _read_file_snippet(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_SNIPPET_CHARS]

    def _write_file(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def _safe_target(self, target_file: str) -> Path:
        candidate = (self.workspace / target_file).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise ValueError("target_file must remain inside the workspace")
        if candidate.suffix.lower() not in ALLOWED_TARGET_SUFFIXES:
            raise ValueError("target_file must end in .py, .html, .css, or .js")
        return candidate

    def _set_workspace(self, workspace_dir: str | Path | None) -> None:
        if workspace_dir is None:
            return
        raw_path = str(workspace_dir).strip().strip('"')
        candidate = Path(raw_path).expanduser().resolve()
        if not candidate.exists():
            raise ValueError(f"workspace directory does not exist: {candidate}")
        if not candidate.is_dir():
            raise ValueError(f"workspace path is not a directory: {candidate}")
        self.workspace = candidate

    def _validate(
        self, files: dict[str, str], test_command: list[str] | None
    ) -> tuple[bool, str, str]:
        if test_command is None:
            python_files = [self._safe_target(name) for name in files if name.endswith(".py")]
            if python_files:
                command = [sys.executable, "-m", "py_compile", *map(str, python_files)]
            elif not all(content.strip() for content in files.values()):
                return False, "", "generated file is empty"
            else:
                return True, "", ""
        else:
            command = test_command
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("test_command must be a non-empty list of strings")
        try:
            completed = subprocess.run(
                command, cwd=self.workspace, capture_output=True, text=True,
                timeout=VALIDATION_TIMEOUT_SECONDS, shell=False, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = str(exc.stdout or "")[-MAX_LOG_CHARS:]
            stderr = str(exc.stderr or "")[-MAX_LOG_CHARS:]
            return False, stdout, (
                f"validation timed out after {VALIDATION_TIMEOUT_SECONDS} seconds\n{stderr}"
            )
        return completed.returncode == 0, completed.stdout[-MAX_LOG_CHARS:], completed.stderr[-MAX_LOG_CHARS:]
