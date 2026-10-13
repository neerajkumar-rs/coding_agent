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


# ---------------------------------------------------------------------------
# Log channel registry — the plug-and-play core of run logging.
#
# Every log a run can produce is ONE entry here: name -> filename under
# .agent_runs/<run_id>/. Writers (RunJournal.log_*) and readers (both TUIs'
# /logs) go through this registry, so adding a channel (a fourth role, a
# bench log, a calibration log) is a one-line entry — no reader/writer
# changes anywhere else.
# ---------------------------------------------------------------------------
LOG_CHANNELS: dict[str, str] = {
    "thinker": "thinker.log",
    "coder": "coder.log",
    "validator": "validator.log",
    "writer": "writer.log",
    "sandbox": "sandbox.log",
    "events": "events.jsonl",
}

# Channels that hold role logs (everything except "events"); used when /logs
# is called with no channel filter.
ROLE_CHANNELS: tuple[str, ...] = tuple(c for c in LOG_CHANNELS if c != "events")


def channel_path(run_dir: Path, channel: str) -> Path:
    """The file a channel writes to, under the given run dir."""
    filename = LOG_CHANNELS.get(channel, f"{channel}.log")
    return Path(run_dir) / filename


def latest_run_dir(workspace: Path | str) -> Path | None:
    """The most recent .agent_runs/<run_id>/ for a workspace, or None."""
    runs_dir = Path(workspace) / RUNS_DIR_NAME
    if not runs_dir.is_dir():
        return None
    run_dirs = sorted((d for d in runs_dir.iterdir() if d.is_dir()), reverse=True)
    return run_dirs[0] if run_dirs else None


def tail_run_logs(
    workspace: Path | str,
    channel: str = "",
    lines: int = 60,
    run_id: str = "",
) -> dict[str, Any]:
    """Read a run's logs for a workspace. Pure data — both TUIs render this
    their own way.

    ``channel`` picks one registered channel ("" = all role channels).
    ``lines`` is a LINE count (tail semantics). ``run_id`` selects a specific
    run (exact name or unique prefix; "" = latest). Never raises: an
    unreadable file becomes {"error": ...}. Returns {"run_id", "searched",
    "sections"}, where "searched" is the exact path checked when nothing
    matched.
    """
    runs_dir = Path(workspace) / RUNS_DIR_NAME
    chosen: Path | None = None
    if run_id:
        if runs_dir.is_dir():
            exact = runs_dir / run_id
            if exact.is_dir():
                chosen = exact
            else:
                matches = [d for d in runs_dir.iterdir()
                           if d.is_dir() and d.name.startswith(run_id)]
                chosen = matches[0] if len(matches) == 1 else None
        if chosen is None:
            return {"run_id": None,
                    "searched": str(runs_dir / run_id),
                    "sections": [],
                    "error": f"no run matching {run_id!r}"}
    else:
        chosen = latest_run_dir(workspace)
        if chosen is None:
            return {"run_id": None, "searched": str(runs_dir), "sections": []}

    channels = [channel] if channel in LOG_CHANNELS else list(ROLE_CHANNELS)
    sections: list[dict[str, Any]] = []
    for name in channels:
        path = channel_path(chosen, name)
        if not path.exists():
            continue
        try:
            tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
            sections.append({"channel": name, "path": str(path), "lines": tail})
        except OSError as exc:
            sections.append({"channel": name, "path": str(path), "error": str(exc)})
    return {"run_id": chosen.name, "searched": str(chosen), "sections": sections}


def format_run_logs(result: dict[str, Any]) -> str:
    """Render tail_run_logs output as plain text (shared by both TUIs)."""
    if result.get("run_id") is None:
        if result.get("error"):
            return f"{result['error']} — searched {result.get('searched')}"
        return f"No runs yet — searched {result.get('searched')}"
    parts = [f"--- latest run: {result['run_id']} ---"]
    sections = result.get("sections", [])
    if not sections:
        parts.append(f"(no logs in {result.get('searched')})")
    for section in sections:
        header = f"--- {LOG_CHANNELS.get(section['channel'], section['channel'])} ---"
        if "error" in section:
            parts.append(f"{header}\n(unreadable: {section['error']})")
        else:
            parts.append(header + "\n" + "\n".join(section["lines"]))
    return "\n".join(parts)


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

    # ------------------------------------------------------------------
    # Per-role logs + raw LLM call capture (the ONE logging path for a run)
    # ------------------------------------------------------------------
    RAW_LOG_CHARS = 20_000  # raw prompt/response truncated to this per call

    def role_log_path(self, role: str) -> Path:
        """The file a role/channel writes to (registry-driven)."""
        return channel_path(self.run_dir, role)

    def log_role(self, role: str, content: str) -> None:
        """Append a timestamped block to the role's log file."""
        stamp = utc_now_iso()
        header = f"\n{'='*70}\n[{stamp}]\n"
        path = self.role_log_path(role)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(header + content + "\n")
        except OSError:
            pass

    def log_llm_call(
        self,
        role: str,
        *,
        model: str,
        prompt: str,
        response: str = "",
        temperature: float | None = None,
        num_ctx: int | None = None,
        seconds: float | None = None,
        error: str | None = None,
        think_chars: int | None = None,
        done_reason: str | None = None,
        eval_count: int | None = None,
        eval_duration_ns: int | None = None,
        size_vram: int | None = None,
        size_total: int | None = None,
    ) -> None:
        """Record one raw LLM call (prompt + response, each truncated to
        RAW_LOG_CHARS) in the role's log, and emit an llm_call event.

        All metadata fields are optional so old call sites keep working;
        new fields are simply added to the header line when present. This is
        the plug-and-play contract: extend the signature, and the block gets
        richer without any reader change."""
        def _cut(text: str) -> str:
            text = text or ""
            return text if len(text) <= self.RAW_LOG_CHARS else (
                text[: self.RAW_LOG_CHARS] + "\n...[truncated to 20KB]"
            )

        tok_per_s: float | None = None
        if eval_count is not None and eval_duration_ns:
            tok_per_s = round(eval_count / (eval_duration_ns / 1e9), 1)
        meta = (
            f"num_ctx: {num_ctx}  temperature: {temperature}  seconds: {seconds}"
            f"  think_chars: {think_chars}  done_reason: {done_reason}"
            f"  eval_count: {eval_count}  tok/s: {tok_per_s}"
            f"  size_vram: {size_vram}  size_total: {size_total}"
        )
        lines = [
            f"MODEL: {model}",
            meta,
            "--- PROMPT ---",
            _cut(prompt),
            "--- RESPONSE ---",
            _cut(response),
        ]
        if error:
            lines.append(f"--- ERROR ---\n{error}")
        self.log_role(role, "\n".join(lines))
        self.emit_event({
            "event": "llm_call", "role": role, "model": model,
            "seconds": seconds, "num_ctx": num_ctx,
            "prompt_chars": len(prompt), "response_chars": len(response),
            "think_chars": think_chars, "done_reason": done_reason,
            "eval_count": eval_count, "tok_per_s": tok_per_s,
            "size_vram": size_vram, "size_total": size_total,
            "error": error,
        })
