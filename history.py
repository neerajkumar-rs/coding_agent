"""Per-turn history store: one append-only JSONL per workspace.

`<workspace>/.agent_runs/history.jsonl` — one JSON object per user turn
(fast paths, replies, code runs, failures, errors; state-changing slash
commands like /save; pure display commands excluded). Enough detail to debug
a bad answer in minutes; every entry links to its run dir for the heavy
per-call data (prompts, raw responses, meta) the journal keeps.

Rotation: past 20 MB or 1000 entries the file is renamed history-<n>.jsonl
and a new one starts; readers span rotated files newest first. Corrupt or
truncated lines are skipped (counted). Writes are atomic under a file lock
and never raise into the user flow.

No gateway/Ollama calls anywhere in this module.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_core.run_state import RUNS_DIR_NAME

logger = logging.getLogger("history")

HISTORY_NAME = "history.jsonl"
MAX_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_ENTRIES = 1000
TEXT_CAP = 4 * 1024          # user text capped at 4 KB
ANSWER_PREVIEW = 500
STATE_RESULT_CAP = 1_500

# One lock per workspace path (Textual workers + controller may both write).
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def history_path(workspace: str | Path) -> Path:
    return Path(workspace).resolve() / RUNS_DIR_NAME / HISTORY_NAME


def _rotated_files(workspace: str | Path) -> list[Path]:
    """All history files, newest first: history.jsonl, history-1.jsonl, ..."""
    base = history_path(workspace)
    files: list[Path] = []
    if base.exists():
        files.append(base)
    for other in sorted(base.parent.glob("history-*.jsonl"),
                         key=lambda p: p.name, reverse=True):
        files.append(other)
    return files


def _next_id(workspace: str | Path) -> int:
    """Monotonic id = max existing id + 1 (scans rotated files, tolerates
    corrupt lines)."""
    highest = 0
    for path in _rotated_files(workspace):
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    entry = json.loads(raw)
                    highest = max(highest, int(entry.get("id", 0)))
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            continue
    return highest + 1


def _maybe_rotate(workspace: str | Path) -> None:
    path = history_path(workspace)
    try:
        size = path.stat().st_size
        lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
    except OSError:
        return
    if size <= MAX_BYTES and lines < MAX_ENTRIES:
        return
    n = 1
    while (path.parent / f"history-{n}.jsonl").exists():
        n += 1
    try:
        path.rename(path.parent / f"history-{n}.jsonl")
    except OSError:
        logger.warning("history rotation failed", exc_info=True)


def append_turn(workspace: str | Path, entry: dict[str, Any]) -> int | None:
    """Append one turn. Returns the assigned id, or None on write failure
    (never raises into the user flow — a warning is logged instead)."""
    try:
        path = history_path(workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = _lock_for(path)
        with lock:
            _maybe_rotate(workspace)
            entry = dict(entry)
            entry.setdefault("id", _next_id(workspace))
            entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
            if len(str(entry.get("text", ""))) > TEXT_CAP:
                entry["text"] = str(entry.get("text", ""))[:TEXT_CAP]
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return int(entry["id"])
    except Exception:
        logger.warning("history append failed", exc_info=True)
        return None


def read_history(
    workspace: str | Path,
    *,
    limit: int | None = None,
    before: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Read turns newest-last (chronological). Returns (entries, skipped).

    Spans rotated files. ``before`` pages backwards (entries with id < before).
    Corrupt/truncated lines are skipped and counted in ``skipped``.
    """
    entries: list[dict[str, Any]] = []
    skipped = 0
    # Oldest file first so ids come out chronological.
    for path in reversed(_rotated_files(workspace)):
        try:
            raw_lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in raw_lines:
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
                if not isinstance(entry, dict):
                    raise ValueError
                entries.append(entry)
            except (json.JSONDecodeError, ValueError):
                skipped += 1
    entries.sort(key=lambda e: int(e.get("id", 0)))
    if before is not None:
        entries = [e for e in entries if int(e.get("id", 0)) < before]
    if limit is not None:
        entries = entries[-limit:]
    return entries, skipped


def get_turn(workspace: str | Path, turn_id: int) -> dict[str, Any] | None:
    entries, _skipped = read_history(workspace)
    for entry in entries:
        if int(entry.get("id", -1)) == turn_id:
            return entry
    return None


def search_history(workspace: str | Path, needle: str) -> list[dict[str, Any]]:
    """Substring search over text and answer_preview (chronological)."""
    entries, _skipped = read_history(workspace)
    lowered = needle.lower()
    return [
        e for e in entries
        if lowered in str(e.get("text", "")).lower()
        or lowered in str(e.get("answer_preview", "")).lower()
    ]


def state_snapshot(convo: Any) -> dict[str, Any]:
    """The compact ConversationState fed to the model this turn (for the
    record). Pure data — never raises."""
    try:
        return {
            "turns": list(getattr(convo, "turns", []) or [])[-3:],
            "last_result": {
                **dict(getattr(convo, "last_result", {}) or {}),
                "text": str((getattr(convo, "last_result", {}) or {}).get("text", ""))[:STATE_RESULT_CAP],
            },
            "pending_question": str(getattr(convo, "pending_question", "") or ""),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Shared rendering for /history (both TUIs call these)
# ---------------------------------------------------------------------------

def format_history_table(workspace: str | Path, n: int = 15) -> str:
    """`/history [N]`: id, time, path/label, outcome, seconds, text preview."""
    entries, skipped = read_history(workspace, limit=n)
    if not entries:
        return "No turns yet."
    lines = [f"{'id':>4}  {'time':<19}  {'path/label':<18}  {'outcome':<12}  {'sec':>6}  text"]
    for e in entries:
        ts = str(e.get("ts", ""))[:19].replace("T", " ")
        label = e.get("label") or e.get("path", "?")
        text = str(e.get("text", "")).replace("\n", " ")[:60]
        lines.append(
            f"{int(e.get('id', 0)):>4}  {ts:<19}  {str(label):<18}  "
            f"{str(e.get('outcome', '?')):<12}  {float(e.get('seconds', 0.0)):>6.1f}  {text}"
        )
    if skipped:
        lines.append(f"(skipped {skipped} corrupt line(s))")
    return "\n".join(lines)


def format_history_detail(workspace: str | Path, turn_id: int) -> str:
    """`/history <id>`: full text, state snapshot, trace, plan/files, answer,
    error + run folder path."""
    entry = get_turn(workspace, turn_id)
    if entry is None:
        return f"No turn with id {turn_id}."
    lines = [f"--- turn {turn_id} ---"]
    lines.append(f"time: {entry.get('ts', '?')}   path: {entry.get('path', '?')}   "
                 f"outcome: {entry.get('outcome', '?')}   seconds: {entry.get('seconds', '?')}")
    if entry.get("label"):
        lines.append(f"label: {entry['label']}")
    if entry.get("models"):
        lines.append("models: " + "; ".join(
            f"{r}={m}" for r, m in entry["models"].items()))
    lines.append(f"run_id: {entry.get('run_id', '?')}")
    run_dir = Path(workspace).resolve() / RUNS_DIR_NAME / str(entry.get("run_id", ""))
    if run_dir.is_dir():
        lines.append(f"run folder: {run_dir}")
    lines.append("\ntext:\n" + str(entry.get("text", "")))
    snap = entry.get("state_snapshot") or {}
    if snap:
        lines.append("\nstate snapshot:\n" + json.dumps(snap, indent=2)[:2_000])
    trace = entry.get("trace") or []
    if trace:
        lines.append("\ntrace:")
        for t in trace:
            lines.append(f"  {t.get('stage', '?')} {t.get('status', '?')} "
                         f"{t.get('seconds', '?')}s {t.get('model', '') or ''}")
    if entry.get("files_planned"):
        lines.append("planned: " + ", ".join(entry["files_planned"]))
    if entry.get("files_staged"):
        lines.append("staged: " + ", ".join(entry["files_staged"]))
    if entry.get("answer_preview"):
        lines.append("\nanswer:\n" + str(entry["answer_preview"]))
    if entry.get("error"):
        err = entry["error"]
        lines.append(f"\nerror — stage {err.get('stage', '?')}: {err.get('detail', '')}")
    return "\n".join(lines)


def format_history_search(workspace: str | Path, needle: str) -> str:
    """`/history search <text>`: matching turns (id, time, preview)."""
    hits = search_history(workspace, needle)
    if not hits:
        return f"No turns matching {needle!r}."
    lines = [f"{len(hits)} turn(s) matching {needle!r}:"]
    for e in hits[-20:]:
        ts = str(e.get("ts", ""))[:19].replace("T", " ")
        text = str(e.get("text", "")).replace("\n", " ")[:60]
        lines.append(f"  #{int(e.get('id', 0))}  {ts}  {text}")
    return "\n".join(lines)
