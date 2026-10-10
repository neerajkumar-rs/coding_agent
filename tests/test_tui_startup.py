"""Tests for tui.py startup hardening: port parsing, terminate_pid, picker, CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tui


# ---------------------------------------------------------------------------
# netstat parser: LISTENING only, exact port, never PID 0/4
# ---------------------------------------------------------------------------

NETSTAT_FIXTURE = """
Active Connections
  Proto  Local Address          Foreign Address        State           PID
  TCP    127.0.0.1:8000         0.0.0.0:0              LISTENING       4321
  TCP    127.0.0.1:8000         127.0.0.1:55000        TIME_WAIT       0
  TCP    127.0.0.1:8000         127.0.0.1:55001        ESTABLISHED     999
  TCP    0.0.0.0:80001          0.0.0.0:0              LISTENING       7777
  TCP    [::]:8000              [::]:0                 LISTENING       4321
  TCP    0.0.0.0:8000           0.0.0.0:0              LISTENING       4
"""


def test_netstat_parser_only_listening_exact_port() -> None:
    pids = tui._parse_netstat_listeners(NETSTAT_FIXTURE, 8000)
    assert pids == [4321]            # LISTENING only; dedupes IPv4+IPv6 rows
    assert 0 not in pids             # TIME_WAIT PID 0 excluded
    assert 4 not in pids             # System PID excluded
    assert 999 not in pids           # ESTABLISHED excluded
    assert 7777 not in pids          # :80001 decoy excluded


def test_netstat_parser_excludes_own_pid() -> None:
    own = os.getpid()
    fixture = f"  TCP    127.0.0.1:8000  0.0.0.0:0  LISTENING  {own}\n"
    assert tui._parse_netstat_listeners(fixture, 8000) == []


# ---------------------------------------------------------------------------
# terminate_pid: never raises, refuses 0/4/self
# ---------------------------------------------------------------------------

def test_terminate_pid_refuses_zero_and_four() -> None:
    assert tui.terminate_pid(0) is False
    assert tui.terminate_pid(4) is False
    assert tui.terminate_pid(os.getpid()) is False
    assert tui.terminate_pid(-1) is False


def test_terminate_pid_swallows_file_not_found() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("no powershell")):
        with patch.dict(sys.modules, {"psutil": None}):
            assert tui.terminate_pid(12345) is False  # no exception


def test_terminate_pid_swallows_permission_error() -> None:
    with patch("subprocess.run", side_effect=PermissionError("denied")):
        with patch.dict(sys.modules, {"psutil": None}):
            assert tui.terminate_pid(12345) is False


def test_terminate_pid_taskkill_success() -> None:
    ok = MagicMock(returncode=0)
    with patch("subprocess.run", return_value=ok) as run_mock:
        with patch.dict(sys.modules, {"psutil": None}):
            if sys.platform == "win32":
                assert tui.terminate_pid(12345) is True
                args = run_mock.call_args[0][0]
                assert "taskkill" in args[0].lower()
                assert "/PID" in args and "12345" in args


# ---------------------------------------------------------------------------
# Busy-port fallback
# ---------------------------------------------------------------------------

def test_next_free_port_finds_fallback() -> None:
    with patch.object(tui, "_pids_on_port", side_effect=lambda p: [111] if p == 8000 else []):
        assert tui._next_free_port(8000) == 8001


def test_next_free_port_none_when_all_busy() -> None:
    with patch.object(tui, "_pids_on_port", return_value=[111]):
        assert tui._next_free_port(8000) is None


def test_set_fastapi_port_repoints_urls() -> None:
    original = tui.FASTAPI_PORT
    try:
        tui._set_fastapi_port(8005)
        assert tui.FASTAPI_PORT == 8005
        assert tui.HEALTH_URL.endswith(":8005/health")
        assert tui.RUN_URL.endswith(":8005/agent/run")
    finally:
        tui._set_fastapi_port(original)


def test_kill_port_reuses_our_controller() -> None:
    with patch.object(tui, "_pids_on_port", return_value=[4321]), \
         patch.object(tui, "_port_owner_is_our_controller", return_value=True), \
         patch.object(tui, "terminate_pid") as kill_mock:
        assert tui._kill_port(8000, "FastAPI") is True
        kill_mock.assert_not_called()


def test_kill_port_never_kills_ollama() -> None:
    with patch.object(tui, "_pids_on_port", return_value=[1234]), \
         patch.object(tui, "_port_owner_is_our_controller", return_value=False), \
         patch.object(tui, "_pid_image_name", return_value="ollama.exe"), \
         patch.object(tui, "terminate_pid") as kill_mock:
        assert tui._kill_port(11434, "FastAPI") is False
        kill_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Workspace picker
# ---------------------------------------------------------------------------

def test_picker_default_last_used(tmp_path, monkeypatch) -> None:
    last = tmp_path / "last_ws"
    last.mkdir()
    monkeypatch.setattr(tui, "TUI_STATE_PATH", tmp_path / ".agent_tui_state.json")
    tui._save_tui_state({"last_workspace": str(last), "recent": [str(last)]})
    monkeypatch.setattr(tui.Prompt, "ask", lambda *a, **k: "1")
    assert tui.pick_workspace() == str(last.resolve())


def test_picker_current_dir_default_when_no_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "TUI_STATE_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(tui.Prompt, "ask", lambda *a, **k: "2")
    assert tui.pick_workspace() == str(Path.cwd())


def test_picker_custom_path(tmp_path, monkeypatch) -> None:
    custom = tmp_path / "custom"
    custom.mkdir()
    monkeypatch.setattr(tui, "TUI_STATE_PATH", tmp_path / "state.json")
    answers = iter(["3", str(custom)])
    monkeypatch.setattr(tui.Prompt, "ask", lambda *a, **k: next(answers))
    assert tui.pick_workspace() == str(custom.resolve())


def test_picker_custom_invalid_then_valid(tmp_path, monkeypatch) -> None:
    valid = tmp_path / "ok"
    valid.mkdir()
    monkeypatch.setattr(tui, "TUI_STATE_PATH", tmp_path / "state.json")
    answers = iter(["3", str(valid)])  # "3" then a valid path directly
    monkeypatch.setattr(tui.Prompt, "ask", lambda *a, **k: next(answers))
    assert tui.pick_workspace() == str(valid.resolve())


def test_picker_corrupt_state_file(tmp_path, monkeypatch) -> None:
    bad = tmp_path / ".agent_tui_state.json"
    bad.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(tui, "TUI_STATE_PATH", bad)
    monkeypatch.setattr(tui.Prompt, "ask", lambda *a, **k: "2")
    assert tui.pick_workspace() == str(Path.cwd())


def test_picker_missing_state_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "TUI_STATE_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(tui.Prompt, "ask", lambda *a, **k: "2")
    assert tui.pick_workspace() == str(Path.cwd())


def test_state_roundtrip_atomic(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".agent_tui_state.json"
    monkeypatch.setattr(tui, "TUI_STATE_PATH", path)
    tui._remember_workspace(str(tmp_path))
    state = tui._load_tui_state()
    assert state["last_workspace"] == str(tmp_path)
    assert state["recent"][0] == str(tmp_path)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def test_cli_workspace_flag() -> None:
    args = tui._parse_cli(["--workspace", "E:/proj"])
    assert args.workspace == "E:/proj"
    assert args.benchmark is False


def test_cli_port_flag() -> None:
    args = tui._parse_cli(["--port", "8003"])
    assert args.port == 8003


def test_cli_benchmark_forwards_unknown_args() -> None:
    args = tui._parse_cli(["--benchmark", "--quick", "--coexist", "--repeat", "3"])
    assert args.benchmark is True
    assert args.bench_args == ["--quick", "--coexist", "--repeat", "3"]


def test_cli_yes_flag() -> None:
    args = tui._parse_cli(["--yes"])
    assert args.yes is True
