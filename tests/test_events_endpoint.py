"""Phase 6 tests: events endpoint, cancel endpoint, rollback endpoint."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from fastapi.testclient import TestClient

import main
from agent_core.run_state import RUNS_DIR_NAME, RunJournal

pytestmark = pytest.mark.anyio


@pytest.fixture
def client(tmp_path: Path):
    main.state.rebind_workspace(tmp_path)
    return TestClient(main.app)


def _seed_run(workspace: Path, run_id: str | None = None) -> str:
    journal = RunJournal(workspace, run_id=run_id or "")
    journal.emit_event({"event": "run_started", "run_id": journal.run_id})
    journal.emit_event({"event": "plan_ready", "tasks": 1})
    journal.emit_event({"event": "run_finished", "status": "completed"})
    return journal.run_id


def test_events_endpoint_streams_sse(client, tmp_path: Path) -> None:
    run_id = _seed_run(tmp_path)
    response = client.get(f"/agent/events?run_id={run_id}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "data: " in body
    assert "run_started" in body
    assert "plan_ready" in body
    assert "run_finished" in body


def test_events_endpoint_latest_run_default(client, tmp_path: Path) -> None:
    # Two runs with unambiguous ordering; the default must pick the newer one.
    older = _seed_run(tmp_path, run_id="20261012T000000Z-old000")
    latest = _seed_run(tmp_path, run_id="20261013T000000Z-new999")
    response = client.get("/agent/events")
    assert response.status_code == 200
    # Exactly the latest run's events — the run id itself, no `or` fallback.
    assert latest in response.text
    assert "run_started" in response.text
    assert older not in response.text


def test_events_endpoint_404_without_runs(client, tmp_path: Path) -> None:
    response = client.get("/agent/events")
    assert response.status_code == 404


def test_cancel_endpoint_sets_engine_cancel(client, tmp_path: Path) -> None:
    from agent_core.engine import get_engine

    response = client.post("/agent/cancel", json={"workspace_dir": str(tmp_path)})
    assert response.status_code == 200
    assert response.json()["status"] == "cancel_requested"
    assert get_engine(tmp_path)._cancel_event.is_set()
    get_engine(tmp_path)._cancel_event.clear()


def test_rollback_endpoint_restores_backup(client, tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("BROKEN = True\n", encoding="utf-8")
    backup_dir = tmp_path / RUNS_DIR_NAME / "run-x" / "backup"
    backup_dir.mkdir(parents=True)
    (backup_dir / "app.py").write_text("BROKEN = False\n", encoding="utf-8")

    response = client.post("/agent/rollback", json={"run_id": "run-x"})
    assert response.status_code == 200
    assert response.json()["status"] == "restored"
    assert target.read_text(encoding="utf-8") == "BROKEN = False\n"


def test_rollback_endpoint_404_without_backups(client, tmp_path: Path) -> None:
    (tmp_path / RUNS_DIR_NAME).mkdir()
    response = client.post("/agent/rollback", json={})
    assert response.status_code == 404
