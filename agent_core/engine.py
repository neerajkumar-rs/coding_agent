"""Engine: the single agentic pipeline. Everything goes through Engine.run().

Pipeline: context scan -> plan -> per task [coder -> stage -> verify -> audit]
with budgets, truncation handling, and the no-progress rule. Files are staged
during the run and deployed to the workspace on success (auto_deploy).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_core import auditor, coder, context, planner
from agent_core.llm_gateway import LLMGateway
from agent_core.run_state import (
    STATUS_BUDGET_EXHAUSTED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NEEDS_HUMAN,
    RunJournal,
    diagnostic_hash,
)
from agent_core.staging import StagingArea
from agent_core.verifier import verify_files
from agent_config import AgentConfig, load_agent_config
from ollama_client import (
    TEMPERATURE_PLAN,
    OllamaEngine,
    OllamaTruncatedError,
)

MAX_ATTEMPTS_PER_TASK = 2
MAX_RETRIES = 3
CODER_SPEC_CHARS = 8_000
MAX_SPEC_WORDS = 300
MAX_LOG_CHARS = 8_000
MAX_GENERATED_CODE_CHARS = 100_000
THINKER_INPUT_CHARS = 16_000

DEFAULT_MAX_WALL_SECONDS = float(os.getenv("VIBE_MAX_WALL_SECONDS", "900"))
DEFAULT_MAX_LLM_CALLS = int(os.getenv("VIBE_MAX_LLM_CALLS", "40"))


class BudgetExhausted(RuntimeError):
    """Raised when a run budget (wall time / LLM calls) is exceeded."""


class RunCancelled(RuntimeError):
    """Raised when cancellation is requested mid-run."""


@dataclass
class RunBudgets:
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    max_attempts_per_task: int = MAX_ATTEMPTS_PER_TASK


@dataclass
class RunRequest:
    prompt: str
    target_file: str | None = None
    test_command: list[str] | None = None
    progress: Callable[[str], None] | None = None
    workspace_dir: str | Path | None = None
    auto_deploy: bool = True
    budgets: RunBudgets | None = None


@dataclass
class RunReport:
    status: str
    run_id: str = ""
    files: list[str] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    specification: str = ""
    code: str = ""
    attempts: int = 0
    stdout: str = ""
    stderr: str = ""
    audits: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    needs_human: list[str] = field(default_factory=list)
    llm_calls: int = 0
    elapsed_seconds: float = 0.0
    deployed_files: list[str] = field(default_factory=list)


class Engine:
    """One workspace, one pipeline. All LLM access is budget-guarded."""

    def __init__(
        self,
        workspace: Path | str = ".",
        llm: Any | None = None,
        config: AgentConfig | None = None,
        budgets: RunBudgets | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.config = config or load_agent_config(self.workspace)
        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLMGateway(
                OllamaEngine(config=self.config, workspace_root=self.workspace)
            )
        self.budgets = budgets or RunBudgets()
        self._llm_calls = 0
        self._start = 0.0
        self._cancel_event = threading.Event()

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        self._cancel_event.set()

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise RunCancelled("run cancelled by user")

    def _check_wall_time(self) -> None:
        if time.monotonic() - self._start > self.budgets.max_wall_seconds:
            raise BudgetExhausted(
                f"max wall time exceeded ({self.budgets.max_wall_seconds:.0f}s)"
            )

    def _llm_text(self, model: str, prompt: str, **kwargs: Any) -> str:
        if self._llm_calls >= self.budgets.max_llm_calls:
            raise BudgetExhausted(
                f"max total LLM calls exceeded ({self.budgets.max_llm_calls})"
            )
        self._llm_calls += 1
        return self.llm.text(model, prompt, **kwargs)

    def _llm_structured(self, model: str, prompt: str, schema: Any, **kwargs: Any) -> Any:
        if self._llm_calls >= self.budgets.max_llm_calls:
            raise BudgetExhausted(
                f"max total LLM calls exceeded ({self.budgets.max_llm_calls})"
            )
        self._llm_calls += 1
        return self.llm.structured(model, prompt, schema, **kwargs)

    # ------------------------------------------------------------------
    # The pipeline
    # ------------------------------------------------------------------

    def run(self, request: RunRequest) -> RunReport:
        def report(message: str) -> None:
            if request.progress:
                request.progress(message)

        if not request.prompt.strip():
            raise ValueError("request must not be empty")
        if request.workspace_dir is not None:
            self._set_workspace(request.workspace_dir)

        budgets = request.budgets or self.budgets
        budgets.max_attempts_per_task = max(1, min(budgets.max_attempts_per_task, MAX_RETRIES))
        self.budgets = budgets
        self._llm_calls = 0
        self._start = time.monotonic()
        self._cancel_event.clear()

        journal = RunJournal(self.workspace)
        staging = StagingArea(self.workspace)
        result = RunReport(status="running", run_id=journal.run_id)
        journal.emit_event(
            {"event": "run_started", "run_id": journal.run_id, "request": request.prompt[:500]}
        )
        journal.save_state(
            {"run_id": journal.run_id, "status": "running", "request": request.prompt}
        )

        try:
            report("tool • workspace.list_files: scanning the selected workspace")
            ws_context = context.build_workspace_context(
                self.workspace, request.prompt, report
            )
            report("stage 1/3 • planner: breaking request into sequential tasks")
            tasks, raw_plan = self._plan(request.prompt[:THINKER_INPUT_CHARS], ws_context)
            journal.write_task_plan(tasks)
            journal.emit_event({"event": "plan_ready", "tasks": len(tasks)})
            result.specification = raw_plan
            result.tasks = tasks

            for task_number, task in enumerate(tasks, 1):
                self._check_wall_time()
                self._check_cancelled()
                report(f"task {task_number}/{len(tasks)} • {task['name']}")
                journal.emit_event(
                    {"event": "task_started", "task": task_number, "name": task["name"]}
                )
                outcome = self._run_task(
                    task_number, task, request.test_command, journal, staging, report, result
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
                if outcome == STATUS_NEEDS_HUMAN:
                    result.status = STATUS_NEEDS_HUMAN
                    journal.emit_event(
                        {"event": "task_needs_human", "task": task_number, "name": task["name"]}
                    )
                    return self._finalize(result, journal)
                result.status = STATUS_FAILED
                return self._finalize(result, journal)

            result.status = STATUS_COMPLETED
            if request.auto_deploy and result.files:
                deployed = staging.deploy()
                result.deployed_files = deployed
                journal.emit_event({"event": "deployed", "files": deployed})
            report("all tasks complete • validated code is ready")
        except RunCancelled as exc:
            result.status = STATUS_CANCELLED
            result.errors.append(str(exc))
            journal.emit_event({"event": "run_cancelled"})
        except BudgetExhausted as exc:
            result.status = STATUS_BUDGET_EXHAUSTED
            result.errors.append(str(exc))
            journal.emit_event({"event": "budget_exhausted", "reason": str(exc)})
            report(f"run stopped: {exc}")
        return self._finalize(result, journal)

    def _finalize(self, result: RunReport, journal: RunJournal) -> RunReport:
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
                "deployed_files": result.deployed_files,
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
    # One task: coder -> stage -> verify -> audit, bounded
    # ------------------------------------------------------------------

    def _run_task(
        self,
        task_number: int,
        task: dict[str, Any],
        test_command: list[str] | None,
        journal: RunJournal,
        staging: StagingArea,
        report: Callable[[str], None],
        result: RunReport,
    ) -> str:
        task_spec = task["spec"]
        previous_diag_hash: str | None = None

        for attempt in range(1, self.budgets.max_attempts_per_task + 1):
            self._check_wall_time()
            self._check_cancelled()
            result.attempts += 1
            report(f"task {task_number} • coder • attempt {attempt}")
            journal.emit_event(
                {"event": "attempt_started", "task": task_number, "attempt": attempt}
            )

            # --- Coder: truncation is retryable with a smaller spec budget;
            #     unusable output FAILS the attempt with a clear reason.
            files: dict[str, str] | None = None
            coder_error: str | None = None
            spec_chars = CODER_SPEC_CHARS
            for _trunc_retry in range(2):
                try:
                    files = coder.generate_files(self, self.config.coder_model, task_spec[:spec_chars])
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
                verify_passed = False
                verify_stdout, verify_stderr = "", diagnostics
            else:
                total_size = sum(len(content) for content in files.values())
                if total_size > MAX_GENERATED_CODE_CHARS:
                    raise ValueError("generated task output exceeds the configured safety limit")

                staged = staging.stage_files(files)
                for name in files:
                    report(f"tool • filesystem.stage_file: {name}")
                result.files = list(dict.fromkeys(result.files + list(files)))
                result.code = "\n\n".join(files.values())

                verify = verify_files(staged, staging.manager.staging_dir, test_command)
                verify_passed = verify.passed
                verify_stdout, verify_stderr = verify.stdout, verify.stderr
                journal.emit_event(
                    {
                        "event": "verify_result",
                        "task": task_number,
                        "attempt": attempt,
                        "passed": verify.passed,
                        "steps": [
                            {"step": s.step, "passed": s.passed, "output": s.output[:500]}
                            for s in verify.steps
                        ],
                    }
                )

                audit_note = ""
                if verify_passed:
                    report(f"task {task_number} • auditor: LLM review of diff")
                    diff_text = staging.diff_for(files)
                    audit = auditor.audit_diff(
                        self, self.config.planner_model, task_spec, diff_text
                    )
                    audit_dict = {
                        "verdict": audit.verdict,
                        "issues": [i.model_dump() for i in audit.issues],
                    }
                    result.audits.append(audit_dict)
                    journal.emit_event(
                        {"event": "audit", "task": task_number, **audit_dict}
                    )
                    if audit.issues:
                        audit_note = "\nAUDIT CONCERNS:\n" + "\n".join(
                            f"- [{i.severity}] {i.message}" for i in audit.issues
                        )

                result.stdout, result.stderr = verify_stdout, verify_stderr
                diagnostics = (
                    verify_stdout + "\n" + verify_stderr + audit_note
                ).strip()[-MAX_LOG_CHARS:]

                if verify_passed:
                    # The auditor NEVER blocks a deterministic pass.
                    task["status"] = "complete"
                    journal.emit_event(
                        {"event": "task_complete", "task": task_number, "attempt": attempt}
                    )
                    return "complete"

            # --- Attempt failed: no-progress detection + spec refinement
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
            diag_hash = diagnostic_hash(diagnostics)
            if previous_diag_hash is not None and diag_hash == previous_diag_hash:
                report(f"task {task_number} • no progress between attempts; needs human")
                journal.emit_event(
                    {"event": "no_progress", "task": task_number, "diagnostic_hash": diag_hash}
                )
                task["status"] = "needs_human"
                result.needs_human.append(task["name"])
                return STATUS_NEEDS_HUMAN
            previous_diag_hash = diag_hash

            if attempt < self.budgets.max_attempts_per_task:
                report(f"task {task_number} failed • planner is refining this task")
                task_spec = self._refine_spec(task_spec, diagnostics)
                task["spec"] = task_spec

        return "failed"

    # ------------------------------------------------------------------
    # LLM-facing helpers (Engine implements the llm protocol for coder/
    # planner/auditor by routing through the budget guards)
    # ------------------------------------------------------------------

    @property
    def supports_structured(self) -> bool:
        return bool(getattr(self.llm, "supports_structured", False))

    def text(self, model: str, prompt: str, **kwargs: Any) -> str:
        return self._llm_text(model, prompt, **kwargs)

    def structured(self, model: str, prompt: str, schema: Any, **kwargs: Any) -> Any:
        return self._llm_structured(model, prompt, schema, **kwargs)

    def _plan(
        self, request_text: str, workspace_context: str
    ) -> tuple[list[dict[str, Any]], str]:
        return planner.plan_tasks(
            self, self.config.planner_model, request_text, workspace_context,
            str(self.workspace),
        )

    def _refine_spec(self, specification: str, diagnostics: str) -> str:
        prompt = (
            "You are a debugging context parser. Update the Markdown specification "
            "under 300 words using ONLY the existing specification and test logs. "
            "Keep signatures and logic rules; state the minimal correction. No code.\n\n"
            f"SPECIFICATION:\n{specification}\n\nTEST LOG:\n{diagnostics}"
        )
        return " ".join(
            self._llm_text(
                self.config.planner_model, prompt,
                max_tokens=900, temperature=TEMPERATURE_PLAN,
            ).split()[:MAX_SPEC_WORDS]
        )

    def _set_workspace(self, workspace_dir: str | Path) -> None:
        raw_path = str(workspace_dir).strip().strip('"')
        candidate = Path(raw_path).expanduser().resolve()
        if not candidate.exists():
            raise ValueError(f"workspace directory does not exist: {candidate}")
        if not candidate.is_dir():
            raise ValueError(f"workspace path is not a directory: {candidate}")
        self.workspace = candidate


# ----------------------------------------------------------------------
# Per-workspace session registry: one Engine per resolved workspace path.
# ----------------------------------------------------------------------

_sessions: dict[str, Engine] = {}
_sessions_lock = threading.Lock()


def get_engine(workspace: Path | str) -> Engine:
    """Return the session Engine for a workspace, creating it on first use."""
    key = str(Path(workspace).resolve())
    with _sessions_lock:
        engine = _sessions.get(key)
        if engine is None:
            engine = Engine(workspace=key)
            _sessions[key] = engine
        return engine


def run(request: RunRequest, workspace: Path | str = ".") -> RunReport:
    """Module-level entry point: ``engine.run(request, workspace)``."""
    return get_engine(workspace).run(request)
