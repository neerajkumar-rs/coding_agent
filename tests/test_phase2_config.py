"""Phase 2 tests: roles-as-source config, exact-match resolver, /models,
round-trip preservation of roles/keep_alive/allowed_commands/extra."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from agent_config import (
    AgentConfig,
    load_agent_config,
    parse_and_recalculate,
    save_agent_config,
)
from ollama_client import ModelNotInstalled, OllamaSupervisorClient


# --- config: roles are the source of truth ----------------------------------

def test_roles_defaults_and_ceilings() -> None:
    from agent_config import ROLE_CTX_CEILINGS

    c = AgentConfig()
    assert c.roles["thinker"]["model"] == "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
    assert c.roles["coder"]["model"] == "qwen2.5-coder:3b-instruct"
    assert c.roles["validator"]["model"] == "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
    # per-spec defaults
    assert c.roles["thinker"]["num_ctx"] == 8192
    assert c.roles["coder"]["num_ctx"] == 6144
    assert c.roles["validator"]["num_predict"] == 512
    assert c.roles["validator"]["temperature"] == 0.0
    # ceilings enforced via get_role_config
    big = AgentConfig(roles={"thinker": {"num_ctx": 99999}})
    assert big.get_role_config("thinker")["num_ctx"] == ROLE_CTX_CEILINGS["thinker"]
    bigc = AgentConfig(roles={"coder": {"num_ctx": 99999}})
    assert bigc.get_role_config("coder")["num_ctx"] == ROLE_CTX_CEILINGS["coder"]


def test_legacy_keys_fold_into_roles_on_load() -> None:
    with TemporaryDirectory() as tmp:
        (Path(tmp) / ".agent").write_text(json.dumps({
            "planner_model": "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest",
            "coder.5b_or_3b": "qwen2.5-coder:3b",
            "max_context_tokens": 4096,
        }), encoding="utf-8")
        c = load_agent_config(tmp)
        assert c.planner_model == "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
        assert c.coder_model == "qwen2.5-coder:3b"
        # Legacy max_context_tokens is IGNORED: role defaults stand (v4).
        # The compat property reports the thinker's window, not the legacy key.
        assert c.max_context_tokens == 8192
        assert c.roles["thinker"]["model"] == "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
        assert c.roles["coder"]["model"] == "qwen2.5-coder:3b"
        assert c.roles["thinker"]["num_ctx"] == 8192
        assert c.roles["coder"]["num_ctx"] == 6144


def test_recalculate_preserves_everything() -> None:
    with TemporaryDirectory() as tmp:
        c = AgentConfig(allowed_commands=["py_compile", "pytest"], extra={"k": 1})
        c.roles["thinker"]["keep_alive"] = "30m"
        c.roles["validator"]["temperature"] = 0.3
        save_agent_config(c, tmp)
        parse_and_recalculate([
            "--recalculate", "--workspace", tmp,
            "--coder-model", "qwen2.5-coder:3b", "--max-context-tokens", "8192",
        ])
        loaded = load_agent_config(tmp)
        assert loaded.extra == {"k": 1}
        assert loaded.allowed_commands == ["py_compile", "pytest"]
        assert loaded.roles["thinker"]["keep_alive"] == "30m"
        assert loaded.roles["validator"]["temperature"] == 0.3
        assert loaded.coder_model == "qwen2.5-coder:3b"
        assert loaded.planner_model == "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"


# --- exact-match resolver, never substitute ---------------------------------

class _TagsHandler(BaseHTTPRequestHandler):
    models = ["qwen2.5-coder:3b-instruct", "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"]

    def do_GET(self) -> None:
        if self.path.endswith("/api/tags"):
            body = json.dumps({"models": [{"name": m} for m in type(self).models]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *_a: object) -> None:
        return


def _serve() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TagsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_exact_match_resolver_no_substitution() -> None:
    server, base = _serve()
    try:
        client = OllamaSupervisorClient(base_url=base)
        # exact pass
        assert asyncio.run(client.resolve_model_name("erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest")) == "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
        # fuzzy request (old behavior would substitute 3b-instruct) now raises
        with pytest.raises(ModelNotInstalled) as ei:
            asyncio.run(client.resolve_model_name("qwen2.5-coder:3b"))
        assert "ollama pull qwen2.5-coder:3b" in str(ei.value)
        # untagged -> :latest normalization is allowed
        _TagsHandler.models = ["foo:latest"]
        client2 = OllamaSupervisorClient(base_url=base)
        assert asyncio.run(client2.resolve_model_name("foo")) == "foo:latest"
    finally:
        server.shutdown(); server.server_close()


# --- /models endpoint ---------------------------------------------------------

class _ModelsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.endswith("/api/tags"):
            body = json.dumps({"models": [
                {"name": "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest", "digest": "sha256:0a7c62f3bef49d9f1bef759b692335b2", "size": 1_100_000_000},
                {"name": "qwen2.5-coder:3b-instruct", "digest": "sha256:f72c60cabf6237b07f6e632b2c48d533", "size": 1_900_000_000},
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif self.path.endswith("/api/ps"):
            body = json.dumps({"models": [{"name": "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self) -> None:
        if self.path.endswith("/api/show"):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            name = payload.get("name", "")
            quant = "Q4_K_M" if "coder" in name else "Q4_0"
            body = json.dumps({"details": {"quantization_level": quant}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *_a: object) -> None:
        return


def test_models_endpoint_per_role() -> None:
    server, base = _serve_models()
    try:
        with TemporaryDirectory() as tmp:
            main.state.rebind_workspace(tmp)
            main.state.ollama = OllamaSupervisorClient(base_url=base)
            client = TestClient(main.app)
            r = client.get("/models")
            assert r.status_code == 200, r.text
            data = r.json()
            coder = data["roles"]["coder"]
            assert coder["model"] == "qwen2.5-coder:3b-instruct"
            assert coder["digest"] == "f72c60cabf62"  # 12 chars (the real digest)
            assert coder["quant"] == "Q4_K_M"
            assert coder["installed"] is True
            assert coder["resident"] is False
            thinker = data["roles"]["thinker"]
            assert thinker["resident"] is True
    finally:
        server.shutdown(); server.server_close()


def _serve_models() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_base_model_warning_heuristic() -> None:
    from main import _base_model_warning

    assert _base_model_warning("qwen2.5:3b", "Q4_K_M") is not None  # no instruct tag -> warn
    assert _base_model_warning("qwen2.5-coder:3b-instruct", "Q4_K_M") is None
    assert _base_model_warning("erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest", "Q4_0") is None  # r1 tag ok


def test_template_missing_warning() -> None:
    from main import _base_model_warning

    # empty template from a model that ANSWERED /api/show -> warn
    w = _base_model_warning("qwen2.5-coder:3b-instruct", "Q4_K_M", template="")
    assert w is not None and "no chat template" in w
    # template present -> no warning
    assert _base_model_warning("qwen2.5-coder:3b-instruct", "Q4_K_M", template="{{ .Prompt }}") is None
    # template None = /api/show never answered (model not installed) -> NO warning
    assert _base_model_warning("qwen2.5-coder:3b-instruct", "Q4_K_M", template=None) is None


def test_models_endpoint_template_warning() -> None:
    """The endpoint surfaces the missing-template warning per role card."""

    class _TemplateHandler(_ModelsHandler):
        def do_POST(self) -> None:
            if self.path.endswith("/api/show"):
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                name = payload.get("name", "")
                # coder gets a real template; thinker/validator get an empty one
                template = "{{ .Prompt }}" if "coder" in name else ""
                body = json.dumps({
                    "details": {"quantization_level": "Q4_K_M"},
                    "template": template,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
            else:
                self.send_response(404); self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), _TemplateHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with TemporaryDirectory() as tmp:
            main.state.rebind_workspace(tmp)
            main.state.ollama = OllamaSupervisorClient(
                base_url=f"http://127.0.0.1:{server.server_port}"
            )
            client = TestClient(main.app)
            data = client.get("/models").json()
            coder_warn = data["roles"]["coder"]["warning"]
            thinker_warn = data["roles"]["thinker"]["warning"]
            assert coder_warn is None  # template present, instruct tag present
            assert thinker_warn is not None and "no chat template" in thinker_warn
    finally:
        server.shutdown(); server.server_close()
