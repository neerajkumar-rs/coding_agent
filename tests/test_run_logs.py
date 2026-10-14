"""/logs readers: registry-driven, latest run, line-count tails, both TUIs.

Covers the plug-and-play contract: every channel in LOG_CHANNELS is readable
through the same reader, and the two TUI formatters are thin wrappers over it.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent_core.run_state import (
    LOG_CHANNELS,
    ROLE_CHANNELS,
    RunJournal,
    format_run_logs,
    latest_run_dir,
    tail_run_logs,
)
import tui
import tui_textual


def _seed_run_with_logs(workspace: str | Path, run_id: str = "") -> RunJournal:
    journal = RunJournal(workspace, run_id=run_id)
    journal.save_state({"run_id": journal.run_id, "status": "completed"})
    journal.log_llm_call(
        "thinker", model="m", prompt="plan this", response="the plan",
        num_ctx=8192, temperature=0.2, seconds=1.5, think_chars=42,
        done_reason="stop", eval_count=100, eval_duration_ns=2_000_000_000,
        size_vram=1_100_000_000, size_total=1_100_000_000,
    )
    journal.log_llm_call("coder", model="m2", prompt="code this", response="the code")
    journal.log_role("validator", "verdict block")
    journal.log_role("writer", "summary prose block")
    journal.log_role("sandbox", "check output block")
    journal.emit_event({"event": "run_started", "run_id": journal.run_id})
    journal.emit_event({"event": "run_finished", "status": "completed"})
    return journal


# ---------------------------------------------------------------------------
# The registry is the single source of truth for channels.
# ---------------------------------------------------------------------------

def test_registry_covers_required_channels() -> None:
    for channel in ("thinker", "coder", "validator", "sandbox", "events"):
        assert channel in LOG_CHANNELS
    assert "events" not in ROLE_CHANNELS  # events is not a role log


def test_every_registered_channel_is_readable() -> None:
    """Plug-and-play proof: iterate the registry itself, never a hardcoded
    list — any channel added later is automatically covered by this test."""
    with TemporaryDirectory() as tmp:
        journal = _seed_run_with_logs(tmp)
        for channel in LOG_CHANNELS:
            result = tail_run_logs(tmp, channel)
            assert result["run_id"] == journal.run_id
            channels_seen = {s["channel"] for s in result["sections"]}
            assert channel in channels_seen, f"channel {channel} not readable"


def test_no_filter_reads_all_role_channels() -> None:
    with TemporaryDirectory() as tmp:
        _seed_run_with_logs(tmp)
        result = tail_run_logs(tmp)
        channels_seen = {s["channel"] for s in result["sections"]}
        for channel in ROLE_CHANNELS:
            assert channel in channels_seen
        assert "events" not in channels_seen  # default view = role logs only


def test_events_channel_selected_explicitly() -> None:
    with TemporaryDirectory() as tmp:
        _seed_run_with_logs(tmp)
        result = tail_run_logs(tmp, "events")
        assert [s["channel"] for s in result["sections"]] == ["events"]
        assert any("run_started" in line for line in result["sections"][0]["lines"])


# ---------------------------------------------------------------------------
# run_id selection (exact + unique prefix), with and without a channel.
# ---------------------------------------------------------------------------

def test_run_id_exact_and_prefix_selection(tmp_path: Path) -> None:
    older = _seed_run_with_logs(tmp_path, run_id="20261012T000000Z-aaa111")
    newer = _seed_run_with_logs(tmp_path, run_id="20261013T000000Z-bbb222")
    # exact id picks that run even though it is older
    result = tail_run_logs(tmp_path, "thinker", 60, run_id=older.run_id)
    assert result["run_id"] == older.run_id
    # unique prefix picks it too
    result = tail_run_logs(tmp_path, run_id="20261012")
    assert result["run_id"] == older.run_id
    # empty run_id = latest
    assert tail_run_logs(tmp_path)["run_id"] == newer.run_id


def test_run_id_no_match_reports_searched_path(tmp_path: Path) -> None:
    _seed_run_with_logs(tmp_path, run_id="20261013T000000Z-bbb222")
    result = tail_run_logs(tmp_path, run_id="nonexistent-run")
    assert result["run_id"] is None
    assert "nonexistent-run" in result["error"]
    assert "nonexistent-run" in result["searched"]
    # ambiguous prefix matches nothing uniquely
    _seed_run_with_logs(tmp_path, run_id="20261013T235959Z-ccc333")
    result = tail_run_logs(tmp_path, run_id="20261013")
    assert result["run_id"] is None


def test_run_id_via_rich_tui_formatter(tmp_path: Path) -> None:
    older = _seed_run_with_logs(tmp_path, run_id="20261012T000000Z-aaa111")
    _seed_run_with_logs(tmp_path, run_id="20261013T000000Z-bbb222")
    text = tui.format_run_logs_text(str(tmp_path), run_id=older.run_id)
    assert older.run_id in text
    text = tui.format_run_logs_text(str(tmp_path), run_id="nope")
    assert "nope" in text and "searched" in text


# ---------------------------------------------------------------------------
# Latest-run selection and line-count tails.
# ---------------------------------------------------------------------------

def test_latest_run_dir_picks_newest(tmp_path: Path) -> None:
    _seed_run_with_logs(tmp_path, run_id="20261012T000000Z-old000")
    newer = _seed_run_with_logs(tmp_path, run_id="20261013T000000Z-new999")
    assert latest_run_dir(tmp_path) == newer.run_dir
    assert latest_run_dir(tmp_path / "nonexistent") is None


def test_latest_run_dir_excludes_bench_and_stateless_dirs(tmp_path: Path) -> None:
    """A benchmark run writes .agent_runs/bench/ with reports but no state.json;
    /logs must never pick it as 'latest run'."""
    real = _seed_run_with_logs(tmp_path, run_id="20261012T000000Z-aaa111")
    # bench dir + a stateless junk dir, both NEWER by name
    bench_dir = tmp_path / ".agent_runs" / "bench"
    bench_dir.mkdir(parents=True)
    (bench_dir / "bench-20261013T000000Z.md").write_text("# report", encoding="utf-8")
    (tmp_path / ".agent_runs" / "20261013T999999Z-nostate").mkdir()
    assert latest_run_dir(tmp_path) == real.run_dir
    # run_id selection also refuses bench
    assert tail_run_logs(tmp_path, run_id="bench")["run_id"] is None


def test_line_count_tail_not_char_count() -> None:
    with TemporaryDirectory() as tmp:
        journal = RunJournal(tmp)
        journal.save_state({"run_id": journal.run_id, "status": "completed"})
        journal.log_role("sandbox", "\n".join(f"line {i}" for i in range(100)))
        result = tail_run_logs(tmp, "sandbox", lines=5)
        lines = result["sections"][0]["lines"]
        assert len(lines) == 5
        assert lines[-1] == "line 99"  # the TAIL


def test_rich_metadata_in_llm_call_block() -> None:
    with TemporaryDirectory() as tmp:
        _seed_run_with_logs(tmp)
        result = tail_run_logs(tmp, "thinker")
        block = "\n".join(result["sections"][0]["lines"])
        assert "think_chars: 42" in block
        assert "done_reason: stop" in block
        assert "tok/s: 50.0" in block  # 100 evals / 2 s
        assert "size_vram: 1100000000" in block


# ---------------------------------------------------------------------------
# No-runs case: the exact searched path is reported.
# ---------------------------------------------------------------------------

def test_no_runs_reports_exact_path_searched(tmp_path: Path) -> None:
    result = tail_run_logs(tmp_path, "thinker", 40)
    assert result["run_id"] is None
    assert result["searched"] == str(tmp_path / ".agent_runs")
    text = format_run_logs(result)
    assert str(tmp_path / ".agent_runs") in text
    assert "No runs yet" in text


def test_empty_runs_dir_reports_searched_path(tmp_path: Path) -> None:
    (tmp_path / ".agent_runs").mkdir()
    text = format_run_logs(tail_run_logs(tmp_path))
    assert "No runs yet" in text and ".agent_runs" in text


# ---------------------------------------------------------------------------
# Both TUIs are thin wrappers over the same reader.
# ---------------------------------------------------------------------------

def test_rich_tui_formatter_matches_registry(tmp_path: Path) -> None:
    _seed_run_with_logs(tmp_path)
    text = tui.format_run_logs_text(str(tmp_path), "coder", 60)
    assert "coder.log" in text
    assert "code this" in text
    assert "plan this" not in text  # thinker content filtered out


def test_textual_wrapper_has_markup_and_same_data(tmp_path: Path) -> None:
    _seed_run_with_logs(tmp_path)
    text = tui_textual._tail_logs_text(str(tmp_path), "events", 10)
    assert "[dim]" in text  # markup wrapper
    assert "run_started" in text


def test_textual_no_runs_prints_path(tmp_path: Path) -> None:
    text = tui_textual._tail_logs_text(str(tmp_path))
    assert "No runs yet" in text
    assert str(tmp_path / ".agent_runs") in text


def test_workspaces_outside_repo(tmp_path: Path) -> None:
    """The reader never touches the repo or cwd — only the given workspace."""
    outside = tmp_path / "some" / "deep" / "workspace"
    outside.mkdir(parents=True)
    _seed_run_with_logs(outside)
    result = tail_run_logs(outside, "validator")
    assert result["run_id"] is not None
    assert result["sections"][0]["lines"], "validator log readable outside the repo"


# ---------------------------------------------------------------------------
# The retired workspace-level log dir is gone from code, docs and config.
# ---------------------------------------------------------------------------

def test_no_retired_log_dir_references_in_repo() -> None:
    import re

    root = Path(tui.PROJECT_ROOT)
    needle = re.compile(re.escape(".agent" + "_logs"))  # built, never spelled
    offenders: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".json"} and path.name != ".gitignore":
            continue
        if path.name == Path(__file__).name:
            continue  # this test holds the pattern (built, not spelled)
        if any(part in {"env", ".git", "node_modules", "__pycache__",
                        ".agent_runs", ".agent_staging", ".pytest_cache"}
               for part in path.parts):
            continue
        try:
            if needle.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(str(path))
        except OSError:
            continue
    assert offenders == [], f"retired log-dir references remain: {offenders}"


# ---------------------------------------------------------------------------
# Textual hygiene: no Rich console output or blocking prompts inside the app.
# ---------------------------------------------------------------------------

def test_textual_app_has_no_rich_console_or_blocking_prompts() -> None:
    """Inside tui_textual.py, output goes to the RichLog widget only: no
    console.print, Prompt.ask/Confirm.ask, input(), rich.print, breakpoint().
    (Docstrings/comments are stripped before scanning.)"""
    import ast

    source = (Path(tui.PROJECT_ROOT) / "tui_textual.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    banned_calls = {"print", "input", "breakpoint"}  # rich.print / bare input / breakpoint
    banned_attrs = {("Prompt", "ask"), ("Confirm", "ask")}  # Prompt.ask / Confirm.ask
    violations: list[str] = []

    class _Names(ast.NodeVisitor):
        def __init__(self) -> None:
            self.console_names: set[str] = set()

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Name) and func.id in banned_calls:
                violations.append(f"line {node.lineno}: {func.id}()")
            elif isinstance(func, ast.Attribute):
                if (isinstance(func.value, ast.Name)
                        and (func.value.id, func.attr) in banned_attrs):
                    violations.append(f"line {node.lineno}: {func.value.id}.{func.attr}()")
                if func.attr == "print" and isinstance(func.value, ast.Name) \
                        and func.value.id in self.console_names:
                    violations.append(f"line {node.lineno}: {func.value.id}.print()")
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            # Track names bound to a Rich Console() so `x.print(...)` is caught.
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) \
                    and node.value.func.id == "Console":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.console_names.add(target.id)
            self.generic_visit(node)

    _Names().visit(tree)
    assert violations == [], f"Textual hygiene violations: {violations}"
