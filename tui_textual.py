"""Textual front-end for the Ollama Agent — OpenCode-style two-column layout.

Layout only. All logic (pipeline calls, slash commands, vitals probes) is
imported from tui.py unchanged; endpoints and internal names are untouched.

Left:  scrollable chat log (RichLog) + input bar fixed at the bottom.
Right: 30-column sidebar — logo, SYSTEM vitals, LAST RUN stage times,
       workspace pinned at the bottom.
Bottom: status bar with key hints (left) and active model (right).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Input, RichLog, Static

import tui
from tui import (
    EXPECTED_STAGES,
    HEALTH_URL,
    INIT_URL,
    OLLAMA_BASE,
    RUN_URL,
    STATUS_URL,
    _check_ollama_running,
    _http_get,
    _vital_gpu,
    _vital_loaded_model,
    _vital_services,
    pick_workspace,
    pluralize,
    post_json,
)

ACCENT = "bright_cyan"


def _probe(func, default: str = "n/a") -> str:
    """Run a vitals probe; never raise."""
    try:
        return func()
    except Exception:
        return default


def _collect_system_lines() -> list[str]:
    return [_probe(_vital_gpu), _probe(_vital_loaded_model), _probe(_vital_services)]


def _last_run_lines() -> list[tuple[str, str]]:
    """(stage, "Xs") pairs; stages absent from the trace are dim/red."""
    seen = {t.get("stage"): t for t in tui._last_trace}
    lines: list[tuple[str, str]] = []
    for stage in EXPECTED_STAGES:
        entry = seen.get(stage)
        if entry is None:
            lines.append((stage, "skipped"))
        else:
            status = entry.get("status", "ok")
            seconds = entry.get("seconds", 0.0)
            lines.append((stage, f"{seconds:.1f}s" if status == "ok" else status))
    return lines


class Sidebar(Static):
    """Right-hand 30-column sidebar: logo, SYSTEM, LAST RUN, workspace."""

    system_text = reactive("")
    workspace = reactive("")

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {ACCENT}]OLLAMA AGENT[/]", id="sb-logo")
        yield Static("", id="sb-system")
        yield Static("", id="sb-lastrun")
        yield Static("", id="sb-workspace")

    def refresh_vitals(self) -> None:
        self.system_text = "\n".join(_collect_system_lines())
        system = [f"[bold]SYSTEM[/]", self.system_text]
        self.query_one("#sb-system", Static).update("\n".join(system))

        run_lines = ["[bold]LAST RUN[/]"]
        for stage, value in _last_run_lines():
            if value == "skipped":
                run_lines.append(f"[red]{stage} SKIPPED[/]")
            elif value in ("failed",):
                run_lines.append(f"[red]{stage} {value}[/]")
            elif value == "n/a":
                run_lines.append(f"[dim]{stage} n/a[/]")
            else:
                run_lines.append(f"[dim]{stage}[/] {value}")
        if not tui._last_trace:
            run_lines.append("[dim]n/a[/]")
        self.query_one("#sb-lastrun", Static).update("\n".join(run_lines))

        short = self.workspace
        if len(short) > 26:
            short = "..." + short[-23:]
        self.query_one("#sb-workspace", Static).update(f"[dim]{short}[/]")


class OllamaAgentApp(App):
    """OpenCode-style TUI for the Ollama agent."""

    CSS = """
    Screen { background: $surface; }
    #chat-col { width: 1fr; }
    #chat { height: 1fr; border: none; padding: 0 1; }
    #input-bar {
        height: 3;
        border-left: thick $primary;
        background: $boost;
        padding: 0 1;
    }
    #sidebar {
        width: 30;
        background: $panel;
        padding: 1 2;
    }
    #sb-logo { text-align: center; padding-bottom: 1; }
    #sb-system, #sb-lastrun { padding-bottom: 1; }
    #sb-workspace { dock: bottom; }
    Footer { background: $boost; }
    """

    def __init__(self, workspace: str) -> None:
        super().__init__()
        self.workspace = workspace
        self._pending_save = False

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="chat-col"):
                yield RichLog(id="chat", markup=True, wrap=True)
                yield Input(placeholder="message or /command", id="input-bar")
            yield Sidebar(id="sidebar")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Ollama Agent"
        self.sub_title = "Enter send · /help · Ctrl+C quit"
        sidebar = self.query_one(Sidebar)
        sidebar.workspace = self.workspace
        sidebar.refresh_vitals()
        # Background vitals refresh every 2s; never blocks input.
        self.set_interval(2.0, sidebar.refresh_vitals)
        self.query_one("#input-bar", Input).focus()
        self._write(f"[dim]Workspace: {self.workspace} — type /help for commands.[/]")

    # ------------------------------------------------------------------ I/O

    def _write(self, markup: str) -> None:
        self.query_one("#chat", RichLog).write(markup)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        # User message appears exactly once.
        self._write(f"[bold {ACCENT}]you[/] {text}")
        command = text.lower()
        if command in {"/quit", "/exit", "q"}:
            self.exit()
            return
        if command == "/help":
            self._write(
                "[dim]/run · /benchmark · /save · /status · /workspace · "
                "/cancel · /models · /logs · /clear · /quit[/]"
            )
            return
        if command == "/clear":
            self.query_one("#chat", RichLog).clear()
            return
        # Everything else (slash commands + requests) runs in a worker thread
        # so the UI never freezes.
        asyncio.create_task(self._dispatch(text))

    async def _dispatch(self, text: str) -> None:
        # A pending inline question ("Save? [y/N]") captures the next input.
        pending = getattr(self, "_pending_answer", None)
        if self._pending_save and pending is not None and not pending.done():
            self._pending_save = False
            self._pending_answer = None
            pending.set_result(text)
            return

        command = text.lower()
        if command == "/save" or command == "/deploy":
            await self._save_flow()
        elif command.startswith("/benchmark"):
            arg_string = text.strip()[len("/benchmark"):].strip()
            await tui.run_benchmark_slash(arg_string, self.workspace)
        elif command == "/status":
            try:
                health = await asyncio.to_thread(_http_get, HEALTH_URL)
                self._write(f"[dim]{health.get('status', '?')} — {health.get('active_workspace', '')}[/]")
            except Exception as exc:
                self._write(f"[red]{exc}[/]")
        elif command == "/workspace":
            new = await asyncio.to_thread(pick_workspace)
            self.workspace = new
            self.query_one(Sidebar).workspace = new
        elif command == "/cancel":
            try:
                res = await post_json(tui.CANCEL_URL, {"workspace_dir": self.workspace})
                self._write(f"[dim]{res}[/]")
            except Exception as exc:
                self._write(f"[red]{exc}[/]")
        elif command == "/models":
            try:
                data = await asyncio.to_thread(_http_get, OLLAMA_BASE + "/api/tags")
                names = [m.get("name", "") for m in data.get("models", [])]
                self._write("[dim]" + ("\n".join(names) or "(none)") + "[/]")
            except Exception as exc:
                self._write(f"[red]{exc}[/]")
        elif command == "/logs":
            for line in _tail_logs():
                self._write(line)
        elif command == "/run":
            self._write("[dim]Type the request directly — /run is the default pipeline.[/]")
        else:
            await self._pipeline_flow(text)

    async def _pipeline_flow(self, request_text: str) -> None:
        """One /agent/run call; print trace, summary, then save prompt inline."""
        try:
            await post_json(INIT_URL, {"root_path": self.workspace})
            result = await asyncio.to_thread(
                _post_json_sync, RUN_URL,
                {"prompt": request_text, "workspace_dir": self.workspace},
            )
        except Exception as exc:
            self._write(f"[red]Pipeline error: {exc}[/]")
            return

        trace = result.get("trace", [])
        tui._last_trace = trace
        self._write("[dim]" + _trace_line(trace) + "[/]")
        self.query_one(Sidebar).refresh_vitals()

        if "answer" in result:
            self._write(result["answer"])
            return

        self._write(result.get("summary", "(no summary)"))
        for failed in result.get("failed_tasks", []):
            self._write(f"[red]NOT saved (failed): {failed['path']} — {failed['error']}[/]")

        await self._save_flow()

    async def _save_flow(self) -> None:
        """Inline save confirmation in the chat log (not a modal), default N."""
        try:
            diff_status = await post_json(STATUS_URL, {"workspace_dir": self.workspace})
        except Exception as exc:
            self._write(f"[red]{exc}[/]")
            return
        if not diff_status.get("has_changes"):
            self._write("[dim]No staged changes.[/]")
            return
        dest_root = str(Path(self.workspace).resolve())
        lines = [f"[yellow]Saving to {dest_root}:[/]"]
        for diff_obj in diff_status.get("diffs", []):
            tag = "(new)" if diff_obj["action"] == "create" else "(overwrites existing file)"
            lines.append(f"  {diff_obj['file']}  [dim]{tag}[/]")
        self._write("\n".join(lines))
        # Prompt line in the chat log, default N.
        answer = await self._ask("Save? [y/N]")
        if answer != "y":
            self._write("[dim]Not saved. Files remain in .agent_staging/. "
                        "Type /save to save them later.[/]")
            return
        res = await post_json(tui.DEPLOY_URL, {"workspace_dir": self.workspace})
        saved = res.get("deployed_files", [])
        self._write(f"[bold {ACCENT}]Saved {pluralize(len(saved), 'file')} to {dest_root}[/]")

    async def _ask(self, question: str) -> str:
        """Inline one-line prompt in the chat log; returns lowercase answer."""
        self._write(f"[bold {ACCENT}]{question}[/]")
        self._pending_save = True
        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_answer = future
        return (await future).strip().lower()


def _post_json_sync(url: str, payload: dict) -> dict:
    """Blocking POST for use inside a worker thread."""
    import json as _json
    import urllib.request as _req

    data = _json.dumps(payload).encode()
    request = _req.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _req.urlopen(request, timeout=600) as r:
        return _json.loads(r.read())


def _run_sync_print(app: "OllamaAgentApp") -> None:
    """Worker guard (benchmark output goes to its own reports + console)."""
    return None


def _trace_line(trace: list[dict]) -> str:
    parts = []
    seen = {t.get("stage") for t in trace}
    for entry in trace:
        stage = entry.get("stage", "?")
        status = entry.get("status", "ok")
        seconds = entry.get("seconds", 0.0)
        parts.append(f"{stage} {status} {seconds:.1f}s")
    for stage in EXPECTED_STAGES:
        if stage not in seen:
            parts.append(f"{stage} SKIPPED")
    return " | ".join(parts)


def _tail_logs() -> list[str]:
    lines: list[str] = []
    log_dir = Path(tui.PROJECT_ROOT) / ".agent_logs"
    for log_file in ("planner.log", "coder.log", "sandbox.log"):
        path = log_dir / log_file
        if path.exists():
            tail = path.read_text(encoding="utf-8", errors="replace")[-800:]
            lines.append(f"[dim]--- {log_file} ---\n{tail}[/]")
    return lines or ["[dim]No logs yet.[/]"]


def run_textual(workspace: str) -> None:
    """Entry point used by tui.cli() when Textual is available."""
    app = OllamaAgentApp(workspace)
    app.run()
