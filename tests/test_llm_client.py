"""Unit tests for the LLM layer: think-stripping, payload shape, truncation guard."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent_config import AgentConfig
from ollama_client import (
    OllamaEngine,
    OllamaTruncatedError,
    is_reasoning_model,
    strip_think_blocks,
)


# ---------------------------------------------------------------------------
# think-stripping
# ---------------------------------------------------------------------------

def test_strip_plain_text() -> None:
    reasoning, answer = strip_think_blocks("hello world")
    assert reasoning == ""
    assert answer == "hello world"


def test_strip_terminated_block() -> None:
    reasoning, answer = strip_think_blocks("<think>let me think</think>{\"ok\": true}")
    assert reasoning == "let me think"
    assert answer == '{"ok": true}'


def test_strip_multiple_blocks() -> None:
    reasoning, answer = strip_think_blocks(
        "<think>first</think>some text<think>second</think>final answer"
    )
    assert reasoning == "first\nsecond"
    assert answer == "some textfinal answer"


def test_strip_unterminated_block() -> None:
    reasoning, answer = strip_think_blocks(
        '{"partial": true}\n<think>this thought never ends and got cut off'
    )
    assert reasoning == "this thought never ends and got cut off"
    assert answer == '{"partial": true}'


def test_strip_only_think_block() -> None:
    reasoning, answer = strip_think_blocks("<think>nothing else</think>")
    assert reasoning == "nothing else"
    assert answer == ""


def test_strip_empty_and_none_like() -> None:
    assert strip_think_blocks("") == ("", "")


def test_reasoning_model_detection() -> None:
    assert is_reasoning_model("deepseek-r1:1.5b")
    assert is_reasoning_model("erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest")
    assert not is_reasoning_model("qwen2.5-coder:3b")


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------

def _engine(tmp: Path) -> OllamaEngine:
    cfg = AgentConfig(max_context_tokens=4096, keep_alive="5m")
    return OllamaEngine(base_url="http://127.0.0.1:1", config=cfg, workspace_root=tmp)


def test_engine_payload_options_are_valid() -> None:
    """keep_alive top-level only; options carry num_ctx/num_predict/temp/seed."""
    with TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        captured: dict = {}

        class Capture(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                captured.update(json.loads(self.rfile.read(length)))
                body = json.dumps({"response": "ok", "done_reason": "stop"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_a: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Capture)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            engine.base_url = f"http://127.0.0.1:{server.server_port}"
            engine.generate_url = f"{engine.base_url}/api/generate"
            assert engine.chat("qwen2.5-coder:3b", "ping") == "ok"
        finally:
            server.shutdown()
            server.server_close()

        assert "keep_alive" in captured and captured["keep_alive"] == "5m"
        options = captured["options"]
        assert "keep_alive" not in options
        assert "flash_attention" not in options
        assert options["num_ctx"] == 4096
        assert options["num_predict"] > 0
        assert options["temperature"] == 0.1
        assert "seed" in options


# ---------------------------------------------------------------------------
# Truncation handling (done_reason == "length")
# ---------------------------------------------------------------------------

class _TruncatingThenOK(BaseHTTPRequestHandler):
    """First call: cut off at num_predict. Second call (bigger budget): complete."""

    calls: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).calls.append(payload)
        num_predict = payload["options"]["num_predict"]
        if len(type(self).calls) == 1:
            body = {
                "response": "<think>partial",
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 100,
                "eval_count": num_predict,  # hit the num_predict ceiling
            }
        else:
            body = {
                "response": "<think>done</think>COMPLETE",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 100,
                "eval_count": 50,
            }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_a: object) -> None:
        return


class _ContextBoundTruncation(BaseHTTPRequestHandler):
    """Truncation where the context window (not num_predict) is the binding limit."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        num_predict = payload["options"]["num_predict"]
        body = {
            "response": "cut off",
            "done": True,
            "done_reason": "length",
            # prompt ate almost the whole window: no headroom for a bigger budget
            "prompt_eval_count": 4096 - num_predict,
            "eval_count": num_predict,
        }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_a: object) -> None:
        return


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_truncation_retries_with_bigger_budget() -> None:
    _TruncatingThenOK.calls = []
    server, base_url = _serve(_TruncatingThenOK)
    try:
        with TemporaryDirectory() as tmp:
            engine = OllamaEngine(
                base_url=base_url,
                config=AgentConfig(max_context_tokens=4096),
                workspace_root=tmp,
            )
            answer = engine.chat("qwen2.5-coder:3b", "ping", max_tokens=256)
        assert answer == "COMPLETE"
        assert len(_TruncatingThenOK.calls) == 2
        first_predict = _TruncatingThenOK.calls[0]["options"]["num_predict"]
        second_predict = _TruncatingThenOK.calls[1]["options"]["num_predict"]
        assert second_predict > first_predict
    finally:
        server.shutdown()
        server.server_close()


def test_context_bound_truncation_raises_retryable() -> None:
    server, base_url = _serve(_ContextBoundTruncation)
    try:
        with TemporaryDirectory() as tmp:
            engine = OllamaEngine(
                base_url=base_url,
                config=AgentConfig(max_context_tokens=4096),
                workspace_root=tmp,
            )
            with pytest.raises(OllamaTruncatedError) as excinfo:
                engine.chat("qwen2.5-coder:3b", "ping", max_tokens=256)
        assert excinfo.value.retryable is True
        assert excinfo.value.done_reason == "length"
    finally:
        server.shutdown()
        server.server_close()


def test_chat_strips_think_before_return() -> None:
    _TruncatingThenOK.calls = []
    server, base_url = _serve(_TruncatingThenOK)
    try:
        with TemporaryDirectory() as tmp:
            engine = OllamaEngine(
                base_url=base_url,
                config=AgentConfig(max_context_tokens=4096),
                workspace_root=tmp,
            )
            answer = engine.chat("deepseek-r1:1.5b", "ping", max_tokens=256)
        assert "<think>" not in answer
        assert answer == "COMPLETE"
    finally:
        server.shutdown()
        server.server_close()
