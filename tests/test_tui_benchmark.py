"""Tests for bench.py and the /benchmark TUI command (fake Ollama, offline)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

import bench
import tui
from agent_config import load_agent_config, save_agent_config, AgentConfig


class FakeOllama(BaseHTTPRequestHandler):
    """Extends the e2e pattern: /api/tags, /api/generate, /api/ps, /api/chat."""

    def do_GET(self) -> None:
        if self.path.endswith("/api/tags"):
            body = {"models": [
                {"name": "deepseek-r1:1.5b"},
                {"name": "qwen2.5-coder:3b"},
                {"name": "lfm2.5-thinking"},
            ]}
        elif self.path.endswith("/api/ps"):
            body = {"models": []}
        else:
            body = {"status": "ok"}
        self._json(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        # keep_alive=0 = unload request
        if payload.get("keep_alive") == 0:
            self._json({"done": True})
            return
        self._json({
            "response": "ok " * 40,
            "done": True,
            "done_reason": "stop",
            "eval_count": 40,
            "total_duration": 500_000_000,
        })

    def _json(self, body: dict) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture
def fake_ollama():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# bench.run() end to end
# ---------------------------------------------------------------------------

def test_bench_run_writes_reports(fake_ollama: str) -> None:
    with TemporaryDirectory() as tmp:
        save_agent_config(AgentConfig(), tmp)
        report, code = bench.run(["--workspace", tmp, "--quick", "--ollama-url", fake_ollama])
        assert code == bench.EXIT_OK
        assert report["results"], report
        bench_dir = Path(tmp) / ".agent_runs" / "bench"
        json_reports = list(bench_dir.glob("bench-*.json"))
        md_reports = list(bench_dir.glob("bench-*.md"))
        assert json_reports and md_reports
        data = json.loads(json_reports[0].read_text(encoding="utf-8"))
        assert data["results"]
        assert "planner" in data["best_per_role"]


def test_bench_run_missing_model_skips_with_hint(fake_ollama: str) -> None:
    with TemporaryDirectory() as tmp:
        cfg = AgentConfig(planner_model="nonexistent-model:9b")
        save_agent_config(cfg, tmp)
        report, code = bench.run([
            "--workspace", tmp, "--quick", "--roles", "planner",
            "--ollama-url", fake_ollama,
        ])
        assert code == bench.EXIT_OK
        planner = next(r for r in report["results"] if r["role"] == "planner")
        assert planner["ok"] is False
        assert "ollama pull nonexistent-model:9b" in planner["error"]


def test_bench_run_ollama_unreachable() -> None:
    with TemporaryDirectory() as tmp:
        report, code = bench.run([
            "--workspace", tmp, "--ollama-url", "http://127.0.0.1:9",
        ])
        assert code == bench.EXIT_OLLAMA_UNREACHABLE
        assert report["errors"]


def test_bench_compare_override(fake_ollama: str) -> None:
    with TemporaryDirectory() as tmp:
        save_agent_config(AgentConfig(), tmp)
        report, code = bench.run([
            "--workspace", tmp, "--quick",
            "--compare", "validator=deepseek-r1:1.5b,lfm2.5-thinking",
            "--roles", "validator",
            "--ollama-url", fake_ollama,
        ])
        assert code == bench.EXIT_OK
        validator = [r for r in report["results"] if r["role"] == "validator"]
        assert {r["model"] for r in validator} == {"deepseek-r1:1.5b", "lfm2.5-thinking"}
        assert all(r["ok"] for r in validator)


def test_bench_coexist_verdict(fake_ollama: str) -> None:
    with TemporaryDirectory() as tmp:
        save_agent_config(AgentConfig(), tmp)
        report, _ = bench.run([
            "--workspace", tmp, "--quick", "--coexist", "--ollama-url", fake_ollama,
        ])
        assert report["coexist_verdict"]


# ---------------------------------------------------------------------------
# TUI slash-command parsing
# ---------------------------------------------------------------------------

def test_slash_command_quoted_args() -> None:
    import shlex

    args = shlex.split('--compare planner=a,b --roles "my role"')
    assert args == ["--compare", "planner=a,b", "--roles", "my role"]


def test_gpu_lock_shared_between_submit_and_benchmark() -> None:
    assert hasattr(tui, "_get_gpu_lock")
    lock = tui._get_gpu_lock()
    assert lock is tui._get_gpu_lock()  # same lock instance


# ---------------------------------------------------------------------------
# /benchmark apply: roles written only after "y", other keys preserved
# ---------------------------------------------------------------------------

def test_apply_writes_roles_only_after_yes() -> None:
    import asyncio

    with TemporaryDirectory() as tmp:
        save_agent_config(AgentConfig(), tmp)
        tui._last_bench_report = {
            "best_per_role": {"planner": "lfm2.5-thinking", "coder": "qwen2.5-coder:3b"},
        }
        # Answer "n" -> nothing written
        with patch.object(tui.Prompt, "ask", return_value="n"):
            asyncio.run(tui._apply_bench_roles(tmp))
        data = json.loads((Path(tmp) / ".agent").read_text(encoding="utf-8"))
        assert "roles" not in data

        # Answer "y" -> roles written, other keys preserved
        with patch.object(tui.Prompt, "ask", return_value="y"):
            asyncio.run(tui._apply_bench_roles(tmp))
        data = json.loads((Path(tmp) / ".agent").read_text(encoding="utf-8"))
        assert data["roles"]["planner"]["model"] == "lfm2.5-thinking"
        assert data["planner_model"] == "deepseek-r1:1.5b"  # preserved
        assert data["max_context_tokens"] == 4096           # preserved


def test_config_roles_roundtrip() -> None:
    with TemporaryDirectory() as tmp:
        cfg = AgentConfig(roles={"planner": {"model": "x", "num_ctx": 2048}})
        save_agent_config(cfg, tmp)
        loaded = load_agent_config(tmp)
        assert loaded.roles == {"planner": {"model": "x", "num_ctx": 2048}}
        role = loaded.get_role_config("planner")
        assert role["model"] == "x"
        assert role["num_ctx"] == 2048
        assert role["keep_alive"] == "5m"
        # coder role falls back to defaults
        assert loaded.get_role_config("coder")["model"] == "qwen2.5-coder:3b"
