"""Textual front-end — OpenCode-style two-column layout (layout only).

All pipeline logic, endpoints and internal names live in tui.py /
orchestrator.py and are imported unchanged. If Textual is not importable,
tui.cli() falls back to the Rich TUI — this module is never loaded then.

Layout: left = scrollable chat log + input bar fixed at the bottom (accent on
the input's left edge). Right = fixed 30-column sidebar (SYSTEM, LAST RUN,
workspace + wordmark pinned to the bottom). Bottom status bar: key hints on
the left, active model on the right.
"""

from __future__ import annotations

import asyncio
import json as _json
import time
import urllib.request as _req
from pathlib import Path

from rich.rule import Rule
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, Static

import tui
from tui import (
    EXPECTED_STAGES,
    INIT_URL,
    OLLAMA_BASE,
    STATUS_URL,
    _vital_gpu,
    _vital_loaded_model,
    _vital_services,
    pluralize,
)

# ---------------------------------------------------------------------------
# Style constants (the one constants block)
# ---------------------------------------------------------------------------
ACCENT = "bright_cyan"
DIM = "dim"
RED = "red"
GREEN = "bright_green"
COLOR_FOUR = "#4fc3f7"
COLOR_PIE = "#ce93d8"
WORDMARK = f"[{DIM}]<_[/][bold {COLOR_FOUR}]FOUR[/][{DIM}]_[/][bold {COLOR_PIE}]PIE[/]"

QUESTION_LIKE_ROUTES = {"QUESTION", "FILE_QUESTION", "UNCLEAR"}


# ---------------------------------------------------------------------------
# Vitals probes — each in its own try/except, "n/a" on failure
# ---------------------------------------------------------------------------

def _probe(func) -> str:
    try:
        return func()
    except Exception:
        return "n/a"


def collect_system_lines() -> list[str]:
    return [_probe(_vital_gpu), _probe(_vital_loaded_model), _probe(_vital_services)]


def _route_of(trace: list[dict]) -> str:
    for entry in trace:
        if entry.get("stage") == "ROUTE":
            return str(entry.get("note", ""))
    return ""


def trace_stage_rows(trace: list[dict]) -> list[tuple[str, str, str]]:
    """(stage, value, color) rows. Route-aware:
    - QUESTION/FILE_QUESTION/UNCLEAR: only the stages present, never red SKIPPED.
    - CODE_TASK: every expected stage; missing = red SKIPPED, pipeline-skipped
      = dim n/a.
    """
    route = _route_of(trace)
    rows: list[tuple[str, str, str]] = []
    if route in QUESTION_LIKE_ROUTES:
        for entry in trace:
            status = entry.get("status", "ok")
            seconds = f"{entry.get('seconds', 0.0):.1f}s"
            value = f"ok {seconds}" if status == "ok" else status
            color = GREEN if status == "ok" else RED
            rows.append((entry.get("stage", "?"), value, color))
        return rows
    seen = {t.get("stage"): t for t in trace}
    for stage in EXPECTED_STAGES:
        entry = seen.get(stage)
        if entry is None:
            rows.append((stage, "SKIPPED", RED))
            continue
        status = entry.get("status", "ok")
        if status == "ok":
            rows.append((stage, f"ok {entry.get('seconds', 0.0):.1f}s", GREEN))
        elif status == "skipped":
            rows.append((stage, "n/a", DIM))
        else:
            rows.append((stage, status, RED))
    return rows


def trace_line(trace: list[dict]) -> str:
    """Single dim chat line, e.g. 'ROUTE ok 0.0s | PLAN ok 12.3s | ...'."""
    return " | ".join(f"[{color}]{stage} {value}[/{color}]"
                      for stage, value, color in trace_stage_rows(trace))


def _post_json_sync(url: str, payload: dict) -> dict:
    data = _json.dumps(payload).encode()
    request = _req.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _req.urlopen(request, timeout=600) as r:
        return _json.loads(r.read())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

class Sidebar(Static):
    """SYSTEM, LAST RUN, then pinned to the bottom: workspace + wordmark."""

    def compose(self) -> ComposeResult:
        yield Static("", id="sb-system")
        yield Static("", id="sb-lastrun")
        yield Static("", id="sb-workspace")
        wordmark = Text.from_markup(WORDMARK)
        wordmark.no_wrap = True
        yield Static(wordmark, id="sb-wordmark")  # LAST widget, bottom-left

    def render_system(self, lines: list[str], age_seconds: int) -> None:
        body = "\n".join(lines) + f"\n[{DIM}]updated {age_seconds}s ago[/]"
        self.query_one("#sb-system", Static).update(f"[bold]SYSTEM[/]\n{body}")

    def render_last_run(self, trace: list[dict]) -> None:
        lines = ["[bold]LAST RUN[/]"]
        if not trace:
            lines.append(f"[{DIM}]n/a[/]")
        else:
            for stage, value, color in trace_stage_rows(trace):
                lines.append(f"[{color}]{stage}[/{color}] [{color}]{value}[/{color}]")
        self.query_one("#sb-lastrun", Static).update("\n".join(lines))

    def render_workspace(self, workspace: str) -> None:
        short = workspace if len(workspace) <= 26 else "..." + workspace[-23:]
        self.query_one("#sb-workspace", Static).update(f"[{DIM}]{short}[/]")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class OllamaAgentApp(App):
    CSS = """
    Screen { background: $surface; }
    #chat-col { width: 1fr; }
    #chat {
        height: 1fr;
        border: none;
        padding: 0 1;
        scrollbar-size-vertical: 2;
        scrollbar-color: $primary;
        scrollbar-color-hover: $primary-lighten-2;
        scrollbar-background: $surface-darken-1;
    }
    #input-bar {
        height: 3;
        border: none;
        border-left: thick $primary;
        background: $boost;
        padding: 0 1;
    }
    #sidebar { width: 30; background: $panel; padding: 1 2; }
    #sb-system, #sb-lastrun { padding-bottom: 1; }
    #sb-workspace { dock: bottom; }
    #sb-wordmark { dock: bottom; text-align: left; padding: 0 1; }
    #status-bar { height: 1; background: $boost; dock: bottom; }
    #status-left { width: 1fr; padding: 0 1; color: $text-muted; }
    #status-right { width: auto; padding: 0 1; color: $primary; }
    """

    BINDINGS = [Binding("ctrl+c", "quit", "quit")]

    def __init__(self, workspace: str) -> None:
        super().__init__()
        self.workspace = workspace
        self._save_pending = False
        self._workspace_pending = False
        self._vitals_lines: list[str] = ["n/a", "n/a", "n/a"]
        self._vitals_at = 0.0
        self._polling = False

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="chat-col"):
                yield RichLog(id="chat", markup=True, wrap=True, auto_scroll=False)
                yield Input(placeholder="message or /command", id="input-bar")
            yield Sidebar(id="sidebar")
        with Horizontal(id="status-bar"):
            yield Static("Enter send · /help · Ctrl+C quit", id="status-left")
            yield Static("", id="status-right")

    def on_mount(self) -> None:
        sidebar = self.query_one(Sidebar)
        sidebar.render_workspace(self.workspace)
        sidebar.render_last_run(list(tui._last_trace))
        try:
            from agent_config import load_agent_config

            model = load_agent_config(self.workspace).coder_model
        except Exception:
            model = "qwen2.5-coder:3b"
        self.query_one("#status-right", Static).update(f"coder: {model}")
        self.query_one("#input-bar", Input).focus()
        self._write(f"[{DIM}]Workspace: {self.workspace} — type /help for commands.[/]")
        # Vitals polling every second, in a thread worker; UI never blocks.
        self.set_interval(1.0, self._tick)

    # ------------------------------------------------------------------ chat

    def _write(self, markup) -> None:
        """Write to the log; auto-scroll follows ONLY when already at bottom."""
        chat = self.query_one("#chat", RichLog)
        at_bottom = chat.scroll_y >= chat.max_scroll_y - 1
        chat.write(markup)
        if at_bottom:
            chat.scroll_end(animate=False)

    def _tick(self) -> None:
        age = int(time.monotonic() - self._vitals_at) if self._vitals_at else 0
        sidebar = self.query_one(Sidebar)
        sidebar.render_system(self._vitals_lines, age)
        sidebar.render_last_run(list(tui._last_trace))  # live as stages land
        if not self._polling:
            self._poll_vitals()

    @work(thread=True)
    def _poll_vitals(self) -> None:
        self._polling = True
        try:
            lines = collect_system_lines()
            self.call_from_thread(self._store_vitals, lines)
        finally:
            self._polling = False

    def _store_vitals(self, lines: list[str]) -> None:
        self._vitals_lines = lines
        self._vitals_at = time.monotonic()

    # ------------------------------------------------------------------ input

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        # Inline prompts capture the next input (no modal, no "you" echo).
        if self._save_pending:
            self._save_pending = False
            await self._answer_save(text.lower())
            return
        if self._workspace_pending:
            self._workspace_pending = False
            self._answer_workspace(text)
            return

        # Turn separation: dim rule, then the message ONCE with accent prefix.
        self._write(Rule(style=DIM))
        self._write(f"[bold {ACCENT}]you[/] {text}")

        command = text.lower()
        if command in {"/quit", "/exit", "q"}:
            self.exit()
        elif command == "/help":
            self._write(f"[{DIM}]/run · /benchmark · /save · /status · /workspace · "
                        f"/cancel · /models · /logs · /clear · /quit[/]\n")
        elif command == "/clear":
            self.query_one("#chat", RichLog).clear()
        elif command in {"/save", "/deploy"}:
            self._prepare_save()
        elif command.startswith("/benchmark"):
            arg_string = text.strip()[len("/benchmark"):].strip()
            asyncio.create_task(tui.run_benchmark_slash(arg_string, self.workspace))
        elif command == "/status":
            self._status_work()
        elif command == "/workspace":
            self._workspace_pending = True
            self._write(f"[bold {ACCENT}]New workspace path:[/]")
        elif command == "/cancel":
            self._cancel_work()
        elif command == "/models":
            self._models_work()
        elif command == "/logs":
            self._write(_tail_logs_text())
        elif command == "/run":
            self._write(f"[{DIM}]Type the request directly — /run is the default pipeline.[/]")
        else:
            self._pipeline_work(text)

    # ------------------------------------------------------------- workers

    @work(thread=True)
    def _pipeline_work(self, request_text: str) -> None:
        try:
            _post_json_sync(INIT_URL, {"root_path": self.workspace})
            result = _post_json_sync(
                tui.RUN_URL, {"prompt": request_text, "workspace_dir": self.workspace}
            )
        except Exception as exc:
            self.call_from_thread(self._pipeline_error, str(exc))
            return
        self.call_from_thread(self._pipeline_done, result)

    @work(thread=True)
    def _status_work(self) -> None:
        try:
            health = _post_get_sync(tui.HEALTH_URL)
            self.call_from_thread(
                self._write,
                f"[{DIM}]{health.get('status', '?')} — {health.get('active_workspace', '')}[/]",
            )
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]{exc}[/]")

    @work(thread=True)
    def _cancel_work(self) -> None:
        try:
            res = _post_json_sync(tui.CANCEL_URL, {"workspace_dir": self.workspace})
            self.call_from_thread(self._write, f"[{DIM}]{res}[/]")
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]{exc}[/]")

    @work(thread=True)
    def _models_work(self) -> None:
        try:
            data = _post_get_sync(OLLAMA_BASE + "/api/tags")
            names = [m.get("name", "") for m in data.get("models", [])]
            self.call_from_thread(self._write, f"[{DIM}]" + ("\n".join(names) or "(none)") + "[/]")
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]{exc}[/]")

    @work(thread=True)
    def _prepare_save_work(self) -> None:
        try:
            diff_status = _post_json_sync(STATUS_URL, {"workspace_dir": self.workspace})
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]{exc}[/]")
            return
        self.call_from_thread(self._show_save_prompt, diff_status)

    def _prepare_save(self) -> None:
        self._prepare_save_work()

    # ------------------------------------------------------------- pipeline UI

    def _pipeline_error(self, message: str) -> None:
        self._write(f"[{RED}]Pipeline error: {message}[/]\n")

    def _pipeline_done(self, result: dict) -> None:
        trace = result.get("trace", [])
        tui._last_trace = trace
        self._write(f"[{DIM}]" + trace_line(trace) + "[/]")

        if "answer" in result:
            self._write(result["answer"])
            self._write("")
            return

        self._write(result.get("summary", "(no summary)"))
        for failed in result.get("failed_tasks", []):
            self._write(f"[{RED}]NOT saved (failed): {failed['path']} — {failed['error']}[/]")
        self._write("")
        if result.get("staged_files"):
            self._prepare_save()
        else:
            self._write(f"[{DIM}]No staged changes.[/]")

    # ----------------------------------------------------------------- save

    def _show_save_prompt(self, diff_status: dict) -> None:
        if not diff_status.get("has_changes"):
            self._write(f"[{DIM}]No staged changes.[/]")
            return
        dest_root = str(Path(self.workspace).resolve())
        lines = [f"[yellow]Saving to {dest_root}:[/]"]
        for diff_obj in diff_status.get("diffs", []):
            tag = "(new)" if diff_obj["action"] == "create" else "(overwrites existing file)"
            lines.append(f"  {diff_obj['file']}  [{DIM}]{tag}[/]")
        # --yes auto-approves: only CHECK-passed files are staged, so saving
        # the staged set is exactly "approve what passed CHECK".
        if tui._cli_yes:
            self._write("\n".join(lines))
            self._write(f"[{DIM}]--yes: auto-saving staged (CHECK-passed) files.[/]")
            self._deploy_work()
            return
        # The literal "[y/N]" must be escaped so Rich does not eat it as markup.
        lines.append(f"[bold {ACCENT}]Save? \\[y/N][/]")
        self._write("\n".join(lines))
        self._save_pending = True

    async def _answer_save(self, answer: str) -> None:
        if answer != "y":
            self._write(f"[{DIM}]Not saved. Files remain in .agent_staging/. "
                        f"Type /save to save them later.[/]\n")
            return
        self._deploy_work()

    @work(thread=True)
    def _deploy_work(self) -> None:
        dest_root = str(Path(self.workspace).resolve())
        try:
            res = _post_json_sync(tui.DEPLOY_URL, {"workspace_dir": self.workspace})
            saved = res.get("deployed_files", [])
            self.call_from_thread(
                self._write,
                f"[bold {ACCENT}]Saved {pluralize(len(saved), 'file')} to {dest_root}[/]\n",
            )
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]{exc}[/]\n")

    def _answer_workspace(self, raw: str) -> None:
        candidate = Path(raw.strip().strip('"')).expanduser()
        if not candidate.is_dir():
            self._write(f"[{RED}]Not a valid directory — workspace unchanged.[/]")
            return
        self.workspace = str(candidate.resolve())
        tui._remember_workspace(self.workspace)
        self.query_one(Sidebar).render_workspace(self.workspace)
        self._write(f"[{DIM}]Workspace: {self.workspace}[/]")


def _post_get_sync(url: str) -> dict:
    with _req.urlopen(url, timeout=5) as r:
        return _json.loads(r.read())


def _tail_logs_text() -> str:
    parts: list[str] = []
    log_dir = Path(tui.PROJECT_ROOT) / ".agent_logs"
    for log_file in ("planner.log", "coder.log", "sandbox.log"):
        path = log_dir / log_file
        if path.exists():
            tail = path.read_text(encoding="utf-8", errors="replace")[-800:]
            parts.append(f"[{DIM}]--- {log_file} ---\n{tail}[/]")
    return "\n".join(parts) if parts else f"[{DIM}]No logs yet.[/]"


def run_textual(workspace: str) -> None:
    """Entry point used by tui.cli() when Textual is available."""
    OllamaAgentApp(workspace).run()
