"""Run lifecycle state and metadata persistence.

Everything a run produces that is NOT a requested target file lives under
``.agent_runs/<run_id>/`` (git-ignored): state.json, events.jsonl, tasks.md.
Nothing here ever writes to the workspace root.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

RUNS_DIR_NAME = ".agent_runs"

# Run terminal statuses
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_NEEDS_HUMAN = "needs_human"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
STATUS_CANCELLED = "cancelled"
TERMINAL_STATUSES = {
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NEEDS_HUMAN,
    STATUS_BUDGET_EXHAUSTED,
    STATUS_CANCELLED,
}

# Task statuses
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_COMPLETE = "complete"
TASK_FAILED = "failed"
TASK_NEEDS_HUMAN = "needs_human"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def diagnostic_hash(text: str) -> str:
    """Whitespace-insensitive hash used by the no-progress detector."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


@dataclass
class RunJournal:
    """Persists run metadata under .agent_runs/<run_id>/ (never workspace root)."""

    workspace: Path
    run_id: str = ""
    run_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()
        if not self.run_id:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.run_id = f"{stamp}-{uuid4().hex[:6]}"
        self.run_dir = self.workspace / RUNS_DIR_NAME / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.tasks_path = self.run_dir / "tasks.md"
        self.backup_dir = self.run_dir / "backup"

    def write_task_plan(self, tasks: list[dict[str, Any]], spec_chars: int = 8_000) -> None:
        lines = ["# Agent run task plan", "", "Tasks are processed sequentially.", ""]
        for task in tasks:
            lines.extend(
                [
                    f"## TASK {task.get('number', '?')}: {task.get('name', '')} "
                    f"[{task.get('status', TASK_PENDING)}]",
                    str(task.get("spec", ""))[:spec_chars],
                    "",
                ]
            )
        self.tasks_path.write_text("\n".join(lines), encoding="utf-8")

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now_iso()
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def emit_event(self, event: dict[str, Any]) -> dict[str, Any]:
        record = {"ts": utc_now_iso(), **event}
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def read_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events
