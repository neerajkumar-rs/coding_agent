"""Compatibility shim. The real pipeline lives in ``agent_core.engine``.

``VibeOrchestrator`` keeps its Phase 1 constructor and ``run()`` signature so
existing callers and tests keep working, but all work is delegated to
``agent_core.engine.Engine`` — there is exactly one pipeline now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_core.engine import (
    DEFAULT_MAX_LLM_CALLS,
    DEFAULT_MAX_WALL_SECONDS,
    MAX_ATTEMPTS_PER_TASK,
    BudgetExhausted,
    Engine,
    RunBudgets,
    RunRequest,
)
from agent_core.llm_gateway import LegacyLLMAdapter
from agent_core.run_state import RUNS_DIR_NAME

# Re-exported so older imports (tests, e2e, tui) keep resolving.
__all__ = [
    "BudgetExhausted",
    "RunBudgets",
    "RunResult",
    "VibeOrchestrator",
    "RUNS_DIR_NAME",
    "MAX_ATTEMPTS_PER_TASK",
    "DEFAULT_MAX_WALL_SECONDS",
    "DEFAULT_MAX_LLM_CALLS",
]


@dataclass
class RunResult:
    """Legacy result shape returned by VibeOrchestrator.run()."""

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


class VibeOrchestrator:
    """Thin adapter: legacy interface in, agent_core Engine out."""

    def __init__(
        self,
        engine: Any | None = None,
        workspace: Path | str = ".",
        max_retries: int = MAX_ATTEMPTS_PER_TASK,
        budgets: RunBudgets | None = None,
    ) -> None:
        if budgets is None:
            budgets = RunBudgets(max_attempts_per_task=max_retries)
        budgets.max_attempts_per_task = max(1, min(budgets.max_attempts_per_task, 3))
        self.budgets = budgets
        # A provided "engine" is a legacy chat engine (or test stub): adapt it
        # so it satisfies the Engine's llm protocol without consuming extra
        # scripted responses on structured calls.
        llm = LegacyLLMAdapter(engine) if engine is not None else None
        self._engine = Engine(workspace=workspace, llm=llm, budgets=budgets)
        self.workspace = self._engine.workspace

    def run(
        self,
        user_request: str,
        target_file: str | None = None,
        test_command: list[str] | None = None,
        progress: Callable[[str], None] | None = None,
        workspace_dir: str | Path | None = None,
    ) -> RunResult:
        report = self._engine.run(
            RunRequest(
                prompt=user_request,
                target_file=target_file,
                test_command=test_command,
                progress=progress,
                workspace_dir=workspace_dir,
                auto_deploy=True,
            )
        )
        self.workspace = self._engine.workspace
        return RunResult(
            status=report.status,
            run_id=report.run_id,
            target_file=target_file or "",
            files=report.files,
            specification=report.specification,
            code=report.code,
            attempts=report.attempts,
            stdout=report.stdout,
            stderr=report.stderr,
            llm_validation="; ".join(
                issue["message"]
                for audit in report.audits
                for issue in audit.get("issues", [])
            ),
            errors=report.errors,
            needs_human=report.needs_human,
            llm_calls=report.llm_calls,
            elapsed_seconds=report.elapsed_seconds,
        )
