"""Phase 5 tests: /config + /config/role endpoints (live switching), /logs."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from fastapi.testclient import TestClient

import main
from agent_config import ROLE_CTX_CEILINGS, load_agent_config
from agent_core.run_state import RUNS_DIR_NAME, RunJournal
from ollama_client import OllamaSupervisorClient


class _TagsHandler(BaseHTTPRequestHandler):
    models = ["qwen2.5-coder:3b-instruct", "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest", "qwen2.5-coder:1.5b"]

    def do_GET(self) -> None:
        if self.path.endswith("/api/tags"):
            body = json.dumps({"models": [{"name": m} for m in type(self).models]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *_a: object) -> None:
        return


def _serve() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TagsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


@pytest.fixture
def ws():
    with TemporaryDirectory() as tmp:
        main.state.rebind_workspace(tmp)
        yield tmp


def test_get_config_returns_roles_and_router(ws) -> None:
    client = TestClient(main.app)
    r = client.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert set(data["roles"]) == {"thinker", "coder", "validator", "writer"}
    assert data["roles"]["coder"]["model"] == "qwen2.5-coder:3b-instruct"
    assert "min_confidence" in data["router"]


def test_config_role_updates_model_and_persists(ws) -> None:
    server, base = _serve()
    try:
        main.state.ollama = OllamaSupervisorClient(base_url=base)
        main.state._gateway = None  # rebuild against the fake
        client = TestClient(main.app)
        r = client.post("/config/role", json={
            "role": "coder", "model": "qwen2.5-coder:1.5b", "workspace_dir": ws,
        })
        assert r.status_code == 200, r.text
        assert r.json()["config"]["model"] == "qwen2.5-coder:1.5b"
        # persisted to .agent
        assert load_agent_config(ws).coder_model == "qwen2.5-coder:1.5b"
    finally:
        server.shutdown(); server.server_close()
        main.state.ollama = OllamaSupervisorClient()
        main.state._gateway = None


def test_config_role_rejects_uninstalled_model(ws) -> None:
    server, base = _serve()
    try:
        main.state.ollama = OllamaSupervisorClient(base_url=base)
        main.state._gateway = None
        client = TestClient(main.app)
        r = client.post("/config/role", json={
            "role": "coder", "model": "not-installed:9b", "workspace_dir": ws,
        })
        assert r.status_code == 400
        assert "ollama pull" in r.json()["detail"]
    finally:
        server.shutdown(); server.server_close()
        main.state.ollama = OllamaSupervisorClient()
        main.state._gateway = None


def test_config_role_clamps_ctx_to_ceiling(ws) -> None:
    server, base = _serve()
    try:
        main.state.ollama = OllamaSupervisorClient(base_url=base)
        main.state._gateway = None
        client = TestClient(main.app)
        r = client.post("/config/role", json={
            "role": "coder", "num_ctx": 99999, "workspace_dir": ws,
        })
        assert r.status_code == 200
        assert r.json()["config"]["num_ctx"] == ROLE_CTX_CEILINGS["coder"]
        assert load_agent_config(ws).get_role_config("coder")["num_ctx"] == ROLE_CTX_CEILINGS["coder"]
    finally:
        server.shutdown(); server.server_close()
        main.state.ollama = OllamaSupervisorClient()
        main.state._gateway = None


def test_config_role_unknown_role_rejected(ws) -> None:
    client = TestClient(main.app)
    r = client.post("/config/role", json={"role": "wizard", "model": "x"})
    assert r.status_code == 400
