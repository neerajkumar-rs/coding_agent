"""Run budgets and the per-workspace cancellation registry.

The agentic pipeline itself is retired: ``orchestrator.run_pipeline`` is the
ONLY pipeline. This module keeps (a) the shared budget dataclass and
exceptions the pipeline uses, and (b) a slim per-workspace holder of the
threading.Event that ``/agent/cancel`` sets and the pipeline polls between
stages and before every LLM call.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

MAX_ATTEMPTS_PER_TASK = 2
MAX_RETRIES = 3

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


class WorkspaceSession:
    """Slim per-workspace session holder: just the cancel event.

    The retired Engine used to own the whole pipeline; the only cross-request
    state the live code actually needs is one cancellation event per
    workspace (set by ``/agent/cancel``, cleared at the start of each run).
    """

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).resolve()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()


# Back-compat alias: call sites (main.py, tests) still say "engine".
Engine = WorkspaceSession


_sessions: dict[str, WorkspaceSession] = {}
_sessions_lock = threading.Lock()


def get_engine(workspace: Path | str) -> WorkspaceSession:
    """Return the session holder for a workspace, creating it on first use."""
    key = str(Path(workspace).resolve())
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = WorkspaceSession(key)
            _sessions[key] = session
        return session
