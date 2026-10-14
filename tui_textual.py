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

QUESTION_LIKE_ROUTES = {"reply", "fast-path"}


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
    """The classifier label for this trace: CLASSIFY note, or 'fast-path'
    when the turn never reached the classifier."""
    for entry in trace:
        if entry.get("stage") == "CLASSIFY":
            return str(entry.get("note", "")).split(" ", 1)[0]
        if entry.get("stage") == "FAST-PATH":
            return "fast-path"
    return ""


def trace_stage_rows(trace: list[dict]) -> list[tuple[str, str, str]]:
    """(stage, value, color) rows. Only stages that ACTUALLY RAN appear —
    never a synthetic SKIPPED row. Red only for failed stages; an
    explicitly pipeline-skipped stage renders as dim n/a."""
    rows: list[tuple[str, str, str]] = []
    for entry in trace:
        status = entry.get("status", "ok")
        seconds = f"{entry.get('seconds', 0.0):.1f}s"
        if status == "ok":
            rows.append((entry.get("stage", "?"), f"ok {seconds}", GREEN))
        elif status == "skipped":
            rows.append((entry.get("stage", "?"), "n/a", DIM))
        else:
            rows.append((entry.get("stage", "?"), status, RED))
    return rows


def trace_line(trace: list[dict]) -> str:
    """Single dim chat line, e.g. 'CLASSIFY ok 0.4s (reply) | ANSWER ok 2.1s'.
    Stages that ran a model show it in parentheses; FAST-PATH stages show no
    model (0 model calls)."""
    models = {t.get("stage"): t.get("model", "") for t in trace}
    parts: list[str] = []
    for stage, value, color in trace_stage_rows(trace):
        model = models.get(stage, "")
        parts.append(f"[{color}]{stage} {value}[/{color}]"
                     + (f" [{DIM}]({model})[/]" if model else ""))
    return " | ".join(parts)


def _post_json_sync(url: str, payload: dict) -> dict:
    data = _json.dumps(payload).encode()
    request = _req.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _req.urlopen(request, timeout=600) as r:
        return _json.loads(r.read())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

class Sidebar(Static):
    """SYSTEM, LAST RUN at the top; a spacer pushes workspace + wordmark to
    the very bottom of the sidebar (wordmark last, bottom-left)."""

    def compose(self) -> ComposeResult:
        yield Static("", id="sb-system")
        yield Static("", id="sb-lastrun")
        yield Static("", id="sb-spacer")  # expands, pinning the rest to bottom
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
    #sb-spacer { height: 1fr; }
    #sb-workspace { padding: 0 1; }
    #sb-wordmark { text-align: left; padding: 0 1; }
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
        self._confirm_pending = False
        self._setup_pending = False
        self._vitals_lines: list[str] = ["n/a", "n/a", "n/a"]
        self._vitals_at = 0.0
        self._polling = False
        self._confirm_polling = False
        self._status_bar_at = 0.0
        # Shell-style input recall (per workspace, persisted from history):
        # Up walks backwards through prior inputs, Down forwards, Esc clears.
        self._recall: list[str] = []
        self._recall_idx: int = -1

    def _load_recall(self) -> None:
        """Previous inputs, oldest-first, from history (persisted per
        workspace — survives an app restart)."""
        try:
            import history as _history

            entries, _ = _history.read_history(self.workspace, limit=100)
            self._recall = [str(e.get("text", "")) for e in entries
                            if str(e.get("text", "")).strip()]
        except Exception:
            self._recall = []
        self._recall_idx = len(self._recall)

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
        self.query_one("#status-right", Static).update("loading…")
        self.query_one("#input-bar", Input).focus()
        self._write(f"[{DIM}]Workspace: {self.workspace} — type /help for commands.[/]")
        self._load_recall()
        # Vitals polling every second, in a thread worker; UI never blocks.
        self.set_interval(1.0, self._tick)
        self._refresh_status_bar()

    def on_key(self, event) -> None:
        """Up/Down in the input bar recalls previous inputs; Esc clears."""
        try:
            input_widget = self.query_one("#input-bar", Input)
        except Exception:
            return
        if not input_widget.has_focus:
            return
        if event.key == "up" and self._recall:
            self._recall_idx = max(0, self._recall_idx - 1)
            input_widget.value = self._recall[self._recall_idx]
            event.stop()
        elif event.key == "down" and self._recall:
            self._recall_idx = min(len(self._recall), self._recall_idx + 1)
            input_widget.value = (
                self._recall[self._recall_idx]
                if self._recall_idx < len(self._recall) else ""
            )
            event.stop()
        elif event.key == "escape":
            input_widget.value = ""
            self._recall_idx = len(self._recall)
            event.stop()

    # ------------------------------------------------------------------ chat

    def _write(self, markup) -> None:
        """Write to the log; auto-scroll follows ONLY when already at bottom."""
        chat = self.query_one("#chat", RichLog)
        at_bottom = chat.scroll_y >= chat.max_scroll_y - 1
        chat.write(markup)
        if at_bottom:
            chat.scroll_end(animate=False)

    @work(thread=True)
    def _poll_vitals(self) -> None:
        self._polling = True
        try:
            lines = collect_system_lines()
            self.call_from_thread(self._store_vitals, lines)
        finally:
            self._polling = False

    @work(thread=True)
    def _refresh_status_bar(self) -> None:
        """Status bar: all three roles + ctx + resident model + GPU%."""
        try:
            models = _post_get_sync(tui.MODELS_URL)
            gpu = _vital_gpu()
        except Exception:
            return
        roles = models.get("roles", {})
        resident = next(
            (c.get("model") for c in roles.values() if c.get("resident")), "—"
        )
        # Full configured names, per the exact-name rule: a wrong name must be
        # visible at a glance. U/T/C/V only; the writer (W) is parked and
        # appears only when configured AND used (never today).
        parts = [
            f"U:{roles.get('understand', {}).get('model') or '—'}@{roles.get('understand', {}).get('num_ctx') or '—'}",
            f"T:{roles.get('thinker', {}).get('model') or '—'}@{roles.get('thinker', {}).get('num_ctx') or '—'}",
            f"C:{roles.get('coder', {}).get('model') or '—'}@{roles.get('coder', {}).get('num_ctx') or '—'}",
            f"V:{roles.get('validator', {}).get('model') or '—'}@{roles.get('validator', {}).get('num_ctx') or '—'}",
            f"res:{resident}",
        ]
        gpu_pct = ""
        if "%" in gpu:
            gpu_pct = gpu.split("-")[-1].strip()
        if gpu_pct:
            parts.append(f"gpu:{gpu_pct}")
        text = " ".join(parts)
        self.call_from_thread(self.query_one("#status-right", Static).update, text)

    def _store_vitals(self, lines: list[str]) -> None:
        self._vitals_lines = lines
        self._vitals_at = time.monotonic()

    # ------------------------------------------------------------------ input

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if text:
            # recall list: current session's inputs join the persisted ones
            self._recall.append(text)
            self._recall_idx = len(self._recall)

        # Inline prompts capture the next input (no modal, no "you" echo).
        # Plan-confirm gate: Enter or y approves; anything else declines.
        if self._confirm_pending:
            self._confirm_pending = False
            self._answer_confirm(text.lower() in ("", "y", "yes"))
            return
        if self._setup_pending:
            # Enter (empty) or anything but y = N — never auto-runs.
            self._setup_pending = False
            if text.lower() == "y":
                self._setup_work()
            else:
                self._write(f"[{DIM}]Calibration skipped — nothing changed.[/]")
            return
        if not text:
            return
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
                        f"/cancel · /router <text> · /models · /logs [role|events] [N] · /clear · /quit[/]\n")
        elif command == "/clear":
            self.query_one("#chat", RichLog).clear()
        elif command in {"/save", "/deploy"}:
            self._prepare_save()
        elif command.startswith("/benchmark"):
            arg_string = text.strip()[len("/benchmark"):].strip()
            self._benchmark_work(arg_string)
        elif command == "/status":
            self._status_work()
        elif command == "/workspace":
            self._workspace_pending = True
            self._write(f"[bold {ACCENT}]New workspace path:[/]")
        elif command == "/cancel":
            self._cancel_work()
        elif command == "/models":
            self._models_work()
        elif command.startswith("/logs"):
            from agent_core.run_state import LOG_CHANNELS as _LOG_CH
            args = text.strip()[len("/logs"):].split()
            channel, run_id = "", ""
            count = 60
            positional = [a for a in args if not a.isdigit()]
            digits = [a for a in args if a.isdigit()]
            if digits:
                count = int(digits[0])
            if positional:
                if positional[0] in _LOG_CH:
                    channel = positional[0]
                else:
                    run_id = positional[0]
                if len(positional) > 1:
                    if positional[1] in _LOG_CH:
                        channel = positional[1]
                    else:
                        run_id = run_id or positional[1]
            self._write(_tail_logs_text(self.workspace, channel, count, run_id=run_id))
        elif command.startswith("/history"):
            args = text.strip()[len("/history"):].strip()
            self._write(_history_text(self.workspace, args))
        elif command.startswith("/retry"):
            args = text.strip()[len("/retry"):].strip()
            if not args.isdigit():
                self._write(f"[{DIM}]Usage: /retry <id> — re-sends that turn's text as a NEW turn (old state is not restored).[/]")
            else:
                import history as _history
                entry = _history.get_turn(self.workspace, int(args))
                if entry is None:
                    self._write(f"[{DIM}]No turn with id {args}.[/]")
                else:
                    self._write(f"[{DIM}]Re-sending turn {args} as a new turn (old state is not restored).[/]")
                    self._pipeline_work(str(entry.get("text", "")))
        elif command.startswith("/export"):
            args = text.strip()[len("/export"):].strip()
            self._export_work(args)
        elif command.startswith("/router"):
            self._router_work(text.strip()[len("/router"):].strip())
        elif command == "/roles":
            self._roles_work()
        elif command.startswith("/model"):
            self._model_work(text.strip()[len("/model"):].strip())
        elif command.startswith("/ctx"):
            self._ctx_work(text.strip()[len("/ctx"):].strip())
        elif command == "/setup":
            self._setup_pending = True
            self._write(f"[yellow]Machine calibration probes every role model at several "
                        f"context sizes against the GPU guard (loads/unloads models, "
                        f"a few minutes). Never auto-runs.[/]")
            self._write(f"[bold {ACCENT}]Run calibration? \\[y/N] (default: N)[/]")
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
                tui.RUN_URL,
                {"prompt": request_text, "workspace_dir": self.workspace,
                 # Plan-confirm gate (Phase 4): unless --yes, the run pauses
                 # after PLAN until we answer via /agent/confirm (polled below).
                 "confirm": not tui._cli_yes},
            )
        except Exception as exc:
            self.call_from_thread(self._pipeline_error, str(exc))
            return
        self.call_from_thread(self._pipeline_done, result)

    @work(thread=True)
    def _router_work(self, arg_text: str) -> None:
        if not arg_text:
            self.call_from_thread(self._write, f"[{DIM}]Usage: /router <text>[/]")
            return
        try:
            diag = _post_json_sync(tui.ROUTER_URL,
                                   {"text": arg_text, "workspace_dir": self.workspace})
            self.call_from_thread(self._write, f"[{DIM}]" + tui.format_router_diagnosis(diag) + "[/]")
            return
        except Exception as exc:
            err = exc
        try:
            from router import diagnose as local_diagnose

            diag = asyncio.run(local_diagnose(arg_text))
            note = f"skipped: {type(err).__name__} (server/Ollama unreachable)"
            self.call_from_thread(
                self._write,
                f"[{DIM}]" + tui.format_router_diagnosis(diag, t2_note=note) + "[/]",
            )
        except Exception as exc2:
            self.call_from_thread(self._write, f"[{RED}]/router failed: {exc2}[/]")

    # ------------------------------------------------------- plan confirm gate

    def _tick(self) -> None:
        age = int(time.monotonic() - self._vitals_at) if self._vitals_at else 0
        sidebar = self.query_one(Sidebar)
        sidebar.render_system(self._vitals_lines, age)
        sidebar.render_last_run(list(tui._last_trace))  # live as stages land
        if not self._polling:
            self._poll_vitals()
        if not self._confirm_polling:
            self._poll_confirm()
        # status bar (roles/ctx/resident/GPU) refreshes every ~5s
        if time.monotonic() - self._status_bar_at > 5.0:
            self._status_bar_at = time.monotonic()
            self._refresh_status_bar()

    @work(thread=True)
    def _poll_confirm(self) -> None:
        """Poll the server's pending plan-confirm gate (one per second via
        _tick). Never blocks the UI; no Rich prompts inside the app."""
        from urllib.parse import quote

        self._confirm_polling = True
        try:
            data = _post_get_sync(
                tui.PENDING_CONFIRM_URL + "?workspace_dir=" + quote(self.workspace)
            )
            if data.get("pending") and not self._confirm_pending:
                self.call_from_thread(self._show_confirm, data)
        except Exception:
            pass
        finally:
            self._confirm_polling = False

    def _show_confirm(self, pending: dict) -> None:
        lines = [f"[yellow]Plan ({pending.get('intent', '')}):[/]"]
        for t in pending.get("plan", []):
            lines.append(f"  {t.get('action', 'create')}  {t.get('path', '')}")
        lines.append(f"[bold {ACCENT}]Stage and build? \\[Y/n] (Enter = yes)[/]")
        self._write("\n".join(lines))
        self._confirm_pending = True

    def _answer_confirm(self, approved: bool) -> None:
        self._write(f"[{DIM}]{'Approved — running the plan.' if approved else 'Declined — nothing will be staged.'}[/]")
        self._answer_confirm_work(approved)

    @work(thread=True)
    def _answer_confirm_work(self, approved: bool) -> None:
        try:
            _post_json_sync(tui.CONFIRM_URL,
                            {"workspace_dir": self.workspace, "approved": approved})
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]{exc}[/]")

    @work(thread=True)
    def _benchmark_work(self, arg_string: str) -> None:
        """/benchmark [args] — GPU-locked against the pipeline; output goes to
        the RichLog widget (never Rich console.print inside the Textual app)."""
        import bench
        import shlex

        args = shlex.split(arg_string) if arg_string.strip() else []
        if args and args[0] == "last":
            bench_dir = Path(self.workspace) / ".agent_runs" / "bench"
            reports = sorted(bench_dir.glob("bench-*.md"), reverse=True) if bench_dir.is_dir() else []
            if not reports:
                self.call_from_thread(self._write, f"[{DIM}]No benchmark reports yet.[/]")
            else:
                text = reports[0].read_text(encoding="utf-8", errors="replace")[:4000]
                self.call_from_thread(self._write, f"[{DIM}]{text}[/]")
            return
        if args and args[0] == "apply":
            self.call_from_thread(
                self._write,
                f"[{DIM}]Use the Rich TUI (no Textual) for /benchmark apply, or edit .agent directly.[/]",
            )
            return

        lock = tui._get_gpu_lock()
        if lock.locked():
            self.call_from_thread(
                self._write,
                f"[{DIM}]GPU busy with another operation — try again when it finishes.[/]",
            )
            return
        self.call_from_thread(self._write, f"[{DIM}]Benchmark running… (models unload when done)[/]")
        bench_argv = ["--workspace", self.workspace, *args]

        async def _run_bench():
            async with lock:
                return await asyncio.to_thread(bench.run, bench_argv)

        try:
            report, exit_code = asyncio.run(_run_bench())
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]Benchmark failed: {exc}[/]")
            return
        if exit_code == 2:
            self.call_from_thread(self._write, f"[{RED}]Ollama unreachable — start it first.[/]")
            return
        # render the table into the widget
        lines = ["[bold]Benchmark results[/]"]
        for r in report.get("results", []):
            ok = f"[{GREEN}]ok[/]" if r.get("ok") else f"[{RED}]FAIL[/]"
            lines.append(
                f"  {r.get('role', '')}/{r.get('model', '')}: {ok} "
                f"{r.get('seconds', 0):.2f}s score={r.get('score', 0):.1f} "
                f"[{DIM}]{r.get('error', '')}[/]"
            )
        best = report.get("best_per_role", {})
        if best:
            lines.append("[bold]Best per role:[/] " + ", ".join(f"{k}={v}" for k, v in best.items()))
        self.call_from_thread(self._write, "\n".join(lines))

    @work(thread=True)
    def _export_work(self, args: str) -> None:
        """`/export [target] [--full] [--zip] [--fixture] [--no-redact]` —
        writes under .agent_runs/exports/, prints the path, copies the report
        to the clipboard when possible (never fails on clipboard errors)."""
        import shlex

        import export as _export

        tokens = shlex.split(args) if args.strip() else []
        flags = {t for t in tokens if t.startswith("--")}
        positional = [t for t in tokens if not t.startswith("--")]
        target = positional[0] if positional else "last"
        try:
            result = _export.do_export(
                self.workspace, target,
                full="--full" in flags, zip_out="--zip" in flags,
                fixture="--fixture" in flags, no_redact="--no-redact" in flags,
            )
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]Export failed: {exc}[/]")
            return
        if result.get("error"):
            self.call_from_thread(self._write, f"[{DIM}]{result['error']}[/]")
            return
        lines: list[str] = []
        if result.get("warning"):
            lines.append(f"[yellow]{result['warning']}[/]")
        lines.append(f"[bold {ACCENT}]Export written:[/] {result['path']} "
                     f"({result.get('size', 0)} bytes, {result.get('turns', 0)} turn(s))")
        if result.get("zip_path"):
            lines.append(f"[{DIM}]zip: {result['zip_path']} ({result.get('zip_size', 0)} bytes)[/]")
        if result.get("note"):
            lines.append(f"[{DIM}]{result['note']}[/]")
        # Clipboard: best-effort, never fails the command.
        report = result.get("report")
        if report:
            try:
                self.copy_to_clipboard(report)
                lines.append(f"[{DIM}](report copied to the clipboard)[/]")
            except Exception:
                lines.append(f"[{DIM}](clipboard unavailable — the path above has the report)[/]")
        self.call_from_thread(self._write, "\n".join(lines))

    @work(thread=True)
    def _setup_work(self) -> None:
        """/setup — machine calibration through a local gateway, GPU-locked
        against in-TUI bench/runs (UX mutex; the VRAM guard is the real
        safety). Writes calibration.json atomically per role."""
        import calibrate
        from ollama_client import AgentGateway, OllamaSupervisorClient

        lock = tui._get_gpu_lock()
        if lock.locked():
            self.call_from_thread(
                self._write, f"[{DIM}]GPU busy — try again when the current operation finishes.[/]",
            )
            return
        self.call_from_thread(self._write, f"[{DIM}]Calibrating… (models load/unload per probe)[/]")

        async def _go():
            async with lock:
                gateway = AgentGateway(OllamaSupervisorClient(base_url=tui.OLLAMA_BASE))
                return await calibrate.run_calibration(gateway)

        try:
            data = asyncio.run(_go())
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]Calibration failed: {exc}[/]")
            return
        lines = ["[bold]Calibration complete[/]"]
        for role, entry in (data.get("roles") or {}).items():
            note = " (cancelled)" if entry.get("cancelled") else ""
            lines.append(f"  {role}: ceiling={entry.get('ceiling')}{note}")
        lines.append(f"[{DIM}]Written to {calibrate.CALIBRATION_PATH} — applies immediately.[/]")
        self.call_from_thread(self._write, "\n".join(lines))

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
            data = _post_get_sync(tui.MODELS_URL)
            lines: list[str] = []
            for role, card in data.get("roles", {}).items():
                line = (f"{role}: {card.get('model')}  "
                        f"ctx={card.get('num_ctx')}  "
                        f"digest={card.get('digest') or '—'}  "
                        f"size={card.get('size_gb') or '—'}GB  quant={card.get('quant') or '—'}  "
                        f"{'resident' if card.get('resident') else 'not resident'}")
                lines.append(line)
                if card.get("warning"):
                    lines.append(f"  [{RED}]{card['warning']}[/]")
            self.call_from_thread(self._write, f"[{DIM}]" + ("\n".join(lines) or "(none)") + "[/]")
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]{exc}[/]")

    @work(thread=True)
    def _roles_work(self) -> None:
        """/roles — the current per-role config table."""
        try:
            data = _post_get_sync(tui.CONFIG_URL)
            roles = data.get("roles", {})
            lines = [f"[bold]ROLES[/]  [{DIM}](workspace: {data.get('workspace')})[/]"]
            for role in ("thinker", "coder", "validator", "writer"):
                cfg = roles.get(role, {})
                lines.append(
                    f"  {role}: model={cfg.get('model')}  ctx={cfg.get('num_ctx')}  "
                    f"predict={cfg.get('num_predict')}  temp={cfg.get('temperature')}  "
                    f"keep_alive={cfg.get('keep_alive')}"
                )
            router = data.get("router", {})
            lines.append(f"  router: min_confidence={router.get('min_confidence')}  "
                         f"chat_use_llm={router.get('chat_use_llm')}")
            self.call_from_thread(self._write, "\n".join(lines))
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]{exc}[/]")

    @work(thread=True)
    def _model_work(self, args: str) -> None:
        """/model <thinker|coder|validator> <name> — validate + persist."""
        parts = args.split()
        if len(parts) != 2:
            self.call_from_thread(self._write, f"[{DIM}]Usage: /model <thinker|coder|validator|writer> <name>[/]")
            return
        role, name = parts
        try:
            res = _post_json_sync(tui.ROLE_URL, {
                "role": role, "model": name, "workspace_dir": self.workspace,
            })
            cfg = res.get("config", {})
            self.call_from_thread(
                self._write,
                f"[bold {ACCENT}]{res.get('role', role)} → {cfg.get('model')}[/] "
                f"[{DIM}](ctx={cfg.get('num_ctx')}; applies on the next call)[/]",
            )
        except Exception as exc:
            self.call_from_thread(self._write, f"[{RED}]{exc}[/]")

    @work(thread=True)
    def _ctx_work(self, args: str) -> None:
        """/ctx <role> <n> — clamp to the role ceiling, persist."""
        parts = args.split()
        if len(parts) != 2 or not parts[1].isdigit():
            self.call_from_thread(self._write, f"[{DIM}]Usage: /ctx <thinker|coder|validator|writer> <n>[/]")
            return
        role, n = parts[0], int(parts[1])
        try:
            res = _post_json_sync(tui.ROLE_URL, {
                "role": role, "num_ctx": n, "workspace_dir": self.workspace,
            })
            cfg = res.get("config", {})
            self.call_from_thread(
                self._write,
                f"[bold {ACCENT}]{res.get('role', role)} ctx → {cfg.get('num_ctx')}[/] "
                f"[{DIM}](clamped to the role ceiling)[/]",
            )
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

        # Structured error envelope (never a bare 500): stage + detail + run_id.
        if result.get("error"):
            self._write(
                f"[{RED}]Run failed at {result.get('stage', '?')}: "
                f"{result.get('error')}: {result.get('detail', '')}[/]\n"
                f"[{DIM}]run_id: {result.get('run_id', '?')}[/]"
                + (f"\n[yellow]{result['hint']}[/]" if result.get("hint") else "")
            )
            return

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


def _history_text(workspace: str, args: str) -> str:
    """`/history` shared dispatch: [N] | <id> | search <text>."""
    import history as _history

    args = args.strip()
    if not args:
        return f"[{DIM}]" + _history.format_history_table(workspace) + "[/]"
    if args.startswith("search "):
        return f"[{DIM}]" + _history.format_history_search(workspace, args[7:].strip()) + "[/]"
    if args.isdigit():
        # /history <id> shows one turn; /history N (small) could mean a count —
        # disambiguate: a bare number that matches an existing id wins as id.
        n = int(args)
        if _history.get_turn(workspace, n) is not None:
            return f"[{DIM}]" + _history.format_history_detail(workspace, n) + "[/]"
        return f"[{DIM}]" + _history.format_history_table(workspace, n) + "[/]"
    return f"[{DIM}]Usage: /history [N] | /history <id> | /history search <text>[/]"


def _short_model(name) -> str:
    """Compact model label for the status bar: strip org prefix and tag."""
    if not name:
        return "—"
    base = str(name).split("/")[-1]
    return base.split(":")[0]


def _tail_logs_text(workspace: str, channel: str = "", count: int = 60,
                    run_id: str = "") -> str:
    """Read the ACTIVE workspace's run logs from .agent_runs/<run_id>/.

    Thin markup wrapper over the registry-driven reader in
    agent_core.run_state: ``channel`` is any registered channel
    (thinker/coder/validator/sandbox/events), ``count`` is a LINE count,
    ``run_id`` selects a specific run (exact or unique prefix; default latest).
    When nothing matches, the exact path searched is printed.
    """
    from agent_core.run_state import tail_run_logs

    result = tail_run_logs(workspace, channel, count, run_id=run_id)
    if result.get("run_id") is None:
        if result.get("error"):
            return f"[{DIM}]{result['error']} — searched {result.get('searched')}[/]"
        return f"[{DIM}]No runs yet — searched {result.get('searched')}[/]"
    parts: list[str] = [f"[{DIM}]--- latest run: {result['run_id']} ---[/]"]
    sections = result.get("sections", [])
    if not sections:
        parts.append(f"[{DIM}](no logs in {result.get('searched')})[/]")
    from agent_core.run_state import LOG_CHANNELS
    for section in sections:
        filename = LOG_CHANNELS.get(section["channel"], section["channel"])
        if "error" in section:
            parts.append(f"[{DIM}]--- {filename} ---\n(unreadable: {section['error']})[/]")
        else:
            parts.append(f"[{DIM}]--- {filename} ---\n" + "\n".join(section["lines"]) + "[/]")
    return "\n".join(parts)


def run_textual(workspace: str) -> None:
    """Entry point used by tui.cli() when Textual is available."""
    OllamaAgentApp(workspace).run()
