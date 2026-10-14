"""Phase 4 endpoint tests: the plan-confirm gate over HTTP and /router/classify.

The FastAPI pipeline runs against a fake gateway (no Ollama): generate calls
are intercepted at the AgentGateway level so no HTTP to Ollama is needed.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from fastapi.testclient import TestClient

import main
from tests.test_run_pipeline import (
    INDEX_HTML,
    SCRIPT_JS,
    STYLE_CSS,
    VALIDATE_OK,
    WEB_PLAN,
)


class _FakeGateway:
    """Mimics AgentGateway's surface used by /agent/run and /router/classify."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def think_supported(self, model: str):
        return None

    async def generate(self, role, prompt, *, system="", temperature=None,
                       cancel_event=None, keep_warm=True, think=None,
                       num_predict=None, _format_schema=None):
        import re

        self.calls.append({"role": role, "prompt": prompt})
        # The agent loop: write each web file, then finish.
        if "AGENT ACTION JSON:" in prompt:
            n = sum(1 for c in self.calls if "AGENT ACTION JSON:" in c["prompt"])
            actions = [
                {"action": "write_file", "path": "index.html", "text": "page"},
                {"action": "write_file", "path": "style.css", "text": "css"},
                {"action": "write_file", "path": "script.js", "text": "js"},
                {"action": "finish"},
            ]
            return json.dumps(actions[min(n - 1, len(actions) - 1)])
        target = re.search(r"TARGET FILE:\s*(\S+)", prompt) or re.search(r"Write\s+(\S+\.\w+)\.", prompt)
        if target:
            body = {"index.html": INDEX_HTML, "style.css": STYLE_CSS,
                    "script.js": SCRIPT_JS}[target.group(1)]
            return f"```\n{body}\n```"
        if "satisfies_request" in prompt:
            return VALIDATE_OK
        if "Describe only features visible" in prompt:
            return ""
        if "ROUTER INTENT JSON:" in prompt:
            return json.dumps({"intent": "QUESTION", "confidence": 0.9,
                               "deliverable": "page", "clarifying_question": ""})
        return "ok"

    async def structured(self, role, prompt, schema, *, cancel_event=None):
        raise RuntimeError("no structured in fake")  # forces the regex path


@pytest.fixture()
def workspace():
    with TemporaryDirectory() as tmp:
        old_root = main.state.workspace_root
        main.state.rebind_workspace(tmp)
        main.state._gateway = _FakeGateway()
        try:
            yield tmp
        finally:
            main.state._gateway = None
            main.state.rebind_workspace(old_root)


def test_run_without_confirm_flag_proceeds(workspace) -> None:
    client = TestClient(main.app)
    r = client.post("/agent/run", json={
        "prompt": "make a blog site in html5 css and js",
        "workspace_dir": workspace,
    })
    assert r.status_code == 200, r.text
    assert set(r.json()["staged_files"]) == {"index.html", "style.css", "script.js"}


def test_confirm_gate_pause_answer_approve(workspace) -> None:
    client = TestClient(main.app)
    result_holder: dict = {}

    def do_run():
        result_holder["r"] = client.post("/agent/run", json={
            "prompt": "make a blog site in html5 css and js",
            "workspace_dir": workspace,
            "confirm": True,
        })

    t = threading.Thread(target=do_run, daemon=True)
    t.start()
    # poll for the pending gate
    plan = None
    for _ in range(100):
        p = client.get("/agent/pending_confirm",
                       params={"workspace_dir": workspace}).json()
        if p.get("pending"):
            plan = p
            break
        time.sleep(0.05)
    assert plan is not None, "pipeline never paused for confirmation"
    assert "create index.html" in plan["intent"]
    assert {t_["path"] for t_ in plan["plan"]} == {"index.html", "style.css", "script.js"}

    a = client.post("/agent/confirm", json={"workspace_dir": workspace, "approved": True})
    assert a.json()["approved"] is True
    t.join(timeout=30)
    r = result_holder["r"]
    assert r.status_code == 200
    assert set(r.json()["staged_files"]) == {"index.html", "style.css", "script.js"}


def test_confirm_gate_decline_stages_nothing(workspace) -> None:
    client = TestClient(main.app)
    result_holder: dict = {}

    def do_run():
        result_holder["r"] = client.post("/agent/run", json={
            "prompt": "make a blog site in html5 css and js",
            "workspace_dir": workspace,
            "confirm": True,
        })

    t = threading.Thread(target=do_run, daemon=True)
    t.start()
    for _ in range(100):
        p = client.get("/agent/pending_confirm",
                       params={"workspace_dir": workspace}).json()
        if p.get("pending"):
            break
        time.sleep(0.05)
    client.post("/agent/confirm", json={"workspace_dir": workspace, "approved": False})
    t.join(timeout=30)
    r = result_holder["r"]
    assert r.status_code == 200
    assert r.json()["status"] == "declined"
    assert r.json()["staged_files"] == []
    assert not (Path(workspace) / ".agent_staging" / "index.html").exists()


def test_router_endpoint_is_gone(workspace) -> None:
    """The /router/classify endpoint was deleted with the rules router."""
    client = TestClient(main.app)
    r = client.post("/router/classify", json={"text": "hello"})
    assert r.status_code == 404


def test_pending_confirm_empty_when_no_run(workspace) -> None:
    client = TestClient(main.app)
    p = client.get("/agent/pending_confirm", params={"workspace_dir": workspace}).json()
    assert p == {"pending": False}


# --- Section 5: structured error envelope (never a bare 500) ------------------

def test_run_failure_returns_stage_detail_run_id(workspace) -> None:
    """A forced gateway exception -> {error, stage, detail, run_id} + a journal."""

    class _ExplodingGateway(_FakeGateway):
        async def generate(self, role, prompt, **kwargs):
            raise RuntimeError("synthetic GPU fault for the test")

    main.state._gateway = _ExplodingGateway()
    try:
        client = TestClient(main.app)
        r = client.post("/agent/run", json={
            "prompt": "make a blog site in html5 css and js",
            "workspace_dir": workspace,
        })
        assert r.status_code == 200, r.text  # never a bare 500
        data = r.json()
        assert data["error"] == "RuntimeError"
        assert data["stage"]  # the stage that was running when it blew
        assert "synthetic GPU fault" in data["detail"]
        assert data["run_id"]
        # journal exists for the failed turn too
        state_path = Path(workspace) / ".agent_runs" / data["run_id"] / "state.json"
        assert state_path.exists()
    finally:
        main.state._gateway = _FakeGateway()
