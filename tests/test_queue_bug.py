"""Regression tests: plan->step queue persistence, stale-server detection,
non-Python targets, scanner junk filtering."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from main import _build_id
from sandbox import check_plausible_content
from scanner import WorkspaceScanner, is_junk_file

PLAN_JSON = '{"tasks":[{"id":1,"file":"index.html","action":"create","description":"x"}]}'


def _mock_planner() -> None:
    main.state.ollama.generate = AsyncMock(return_value=PLAN_JSON)


def _mock_coder(code: str = "<html><body>hi</body></html>") -> None:
    async def fake_stream(**kwargs):
        yield f"```html\n{code}\n```"

    main.state.ollama.stream_generate = MagicMock(side_effect=lambda **kw: fake_stream())


@pytest.fixture
def ws():
    with TemporaryDirectory() as tmp:
        main.state.rebind_workspace(tmp)
        yield tmp


def test_plan_then_step_same_workspace_keeps_queue(ws) -> None:
    client = TestClient(main.app)
    client.post("/workspace/init", json={"root_path": ws})
    _mock_planner()
    r = client.post("/agent/plan", json={"prompt": "make a page", "workspace_dir": ws})
    assert r.status_code == 200
    assert len(main.state.task_queue) == 1

    _mock_coder()
    r = client.post("/agent/step", json={"task_id": 1, "workspace_dir": ws})
    assert r.status_code == 200, r.text
    assert len(main.state.task_queue) == 1  # queue NOT wiped by same-workspace step


def test_different_workspace_resets_queue(ws) -> None:
    client = TestClient(main.app)
    client.post("/workspace/init", json={"root_path": ws})
    _mock_planner()
    client.post("/agent/plan", json={"prompt": "make a page", "workspace_dir": ws})
    assert len(main.state.task_queue) == 1

    with TemporaryDirectory() as other:
        client.post("/workspace/init", json={"root_path": other})
        assert len(main.state.task_queue) == 0  # intentional reset on real change


def test_step_unknown_task_id_clear_error(ws) -> None:
    client = TestClient(main.app)
    client.post("/workspace/init", json={"root_path": ws})
    _mock_planner()
    client.post("/agent/plan", json={"prompt": "p", "workspace_dir": ws})

    r = client.post("/agent/step", json={"task_id": 99, "workspace_dir": ws})
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "99" in detail and "known ids" in detail


def test_step_unknown_task_id_empty_queue_message(ws) -> None:
    client = TestClient(main.app)
    client.post("/workspace/init", json={"root_path": ws})
    r = client.post("/agent/step", json={"task_id": 5, "workspace_dir": ws})
    assert r.status_code == 404
    assert "EMPTY" in r.json()["detail"]


def test_step_with_task_object_empty_queue(ws) -> None:
    client = TestClient(main.app)
    client.post("/workspace/init", json={"root_path": ws})
    _mock_coder()
    task = {"id": 7, "file": "page.html", "action": "create", "description": "d"}
    r = client.post("/agent/step", json={"task": task, "workspace_dir": ws})
    assert r.status_code == 200, r.text


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


def test_html_step_stages_and_validates(ws) -> None:
    client = TestClient(main.app)
    client.post("/workspace/init", json={"root_path": ws})
    _mock_coder("<html><body>hi</body></html>")
    task = {"id": 1, "file": "index.html", "action": "create", "description": "page"}
    r = client.post("/agent/step", json={"task": task, "workspace_dir": ws})
    assert r.status_code == 200, r.text
    staged = Path(r.json()["staged_path"])
    assert staged.exists()
    assert not (Path(ws) / "index.html").exists()  # staged, not deployed


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


def test_plan_rejects_non_source_targets(ws) -> None:
    raw = json.dumps({"tasks": [
        {"id": 1, "file": "tasks.md", "action": "create", "description": "junk"},
        {"id": 2, "file": "logo.png", "action": "create", "description": "img"},
        {"id": 3, "file": "app.py", "action": "create", "description": "real"},
    ]})
    tasks = main.parse_tasks_json(raw)
    assert [t["file"] for t in tasks] == ["app.py"]
