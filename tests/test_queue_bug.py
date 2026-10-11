"""Regression tests kept after retiring the /agent/plan + /agent/step queue:
stale-server detection (build_id), scanner junk filtering, html plausibility,
and plan-stage rejection of non-source/junk targets (now in run_pipeline's
_valid_plan_task)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main
from main import _build_id
from orchestrator import _valid_plan_task
from sandbox import check_plausible_content
from scanner import WorkspaceScanner, is_junk_file


@pytest.fixture
def ws():
    with TemporaryDirectory() as tmp:
        main.state.rebind_workspace(tmp)
        yield tmp


def test_health_exposes_build_id(ws) -> None:
    client = TestClient(main.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["build_id"] == _build_id()


def test_tui_detects_stale_server() -> None:
    import tui

    fresh = {"status": "online", "agent_profile": {}, "build_id": tui._local_build_id()}
    stale = {"status": "online", "agent_profile": {}, "build_id": "old-build-123"}

    with patch.object(tui, "_http_get", return_value=fresh):
        assert tui._check_stale_controller(8000) is True  # no prompt, no kill

    with patch.object(tui, "_http_get", return_value=stale), \
         patch.object(tui.Prompt, "ask", return_value="n"), \
         patch.object(tui, "terminate_pid") as kill_mock:
        assert tui._check_stale_controller(8000) is True
        kill_mock.assert_not_called()  # user said no -> nothing killed


def test_tui_stale_restart_kills_only_after_yes() -> None:
    import tui

    stale = {"status": "online", "agent_profile": {}, "build_id": "old-build-123"}
    with patch.object(tui, "_http_get", return_value=stale), \
         patch.object(tui.Prompt, "ask", return_value="y"), \
         patch.object(tui, "_pids_on_port", side_effect=[[4321], []]), \
         patch.object(tui, "terminate_pid") as kill_mock:
        tui._check_stale_controller(8000)
        kill_mock.assert_called_once_with(4321)


def test_html_plausibility_check() -> None:
    assert check_plausible_content("a.html", "") == "file is empty"
    assert check_plausible_content("a.html", "just some text") is not None
    assert check_plausible_content("a.html", "<!doctype html><html><head></head><body><p>ok page body</p></body></html>") is None
    assert check_plausible_content("a.json", "{bad json") is not None
    assert check_plausible_content("a.json", '{"ok": true}') is None
    assert check_plausible_content("a.css", "no braces here") is not None
    assert check_plausible_content("a.css", "body { color: red; }") is None


def test_scanner_ignores_junk_files(ws) -> None:
    root = Path(ws)
    for junk in ("tasks.md", "summary.txt", "task1_code.txt", "task12_code.txt"):
        (root / junk).write_text("junk", encoding="utf-8")
    (root / "app.py").write_text("X = 1\n", encoding="utf-8")

    assert is_junk_file("task3_code.txt")
    assert is_junk_file("tasks.md")
    assert is_junk_file("summary.txt")
    assert not is_junk_file("app.py")

    tree = WorkspaceScanner(root).scan_directory_tree()
    joined = "\n".join(tree)
    assert "app.py" in joined
    assert "tasks.md" not in joined
    assert "summary.txt" not in joined
    assert "task1_code.txt" not in joined


def test_plan_rejects_non_source_and_junk_targets() -> None:
    """run_pipeline's plan validation rejects junk artifacts and non-source
    targets (the retired parse_tasks_json used to do this for the old queue)."""
    bad = [
        {"path": "tasks.md", "action": "create",
         "description": "run artifact that must never be planned as a target"},
        {"path": "summary.txt", "action": "create",
         "description": "leftover summary artifact that must never be planned"},
        {"path": "task1_code.txt", "action": "create",
         "description": "leftover per-task code artifact that must not be planned"},
        {"path": "logo.png", "action": "create",
         "description": "binary image that is not a supported source target"},
    ]
    reasons = [_valid_plan_task(t) for t in bad]
    assert all(r is not None for r in reasons), reasons

    good = {"path": "app.py", "action": "create",
            "description": "real python application implementing the requested feature completely"}
    assert _valid_plan_task(good) is None
