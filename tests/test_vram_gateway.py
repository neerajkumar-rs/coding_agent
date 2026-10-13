"""Phase 3 tests: single-lock AgentGateway + VRAM guard.

FakeOllama here tracks loaded models in memory: /api/ps reflects what is
resident and reports size/size_vram so the GPU-fraction guard can be driven.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent_config import AgentConfig
from ollama_client import (
    AgentGateway,
    OllamaSupervisorClient,
    VramGuardError,
)

MODEL = "qwen2.5-coder:3b-instruct"
THINKER = "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"


class FakeOllama(BaseHTTPRequestHandler):
    """In-memory residency model. `loaded` = {name: {"size":, "size_vram":}}."""

    loaded: dict[str, dict] = {}
    # Behavior knobs, set per-test:
    # how much of a freshly loaded model lands on GPU (0..1); callable(num_ctx)
    gpu_fraction = 1.0
    fail_times: int = 0          # fail the next N generates with HTTP 500
    oom_until_ctx: int = 0       # 500-OOM when num_ctx > this (0 = off)
    requests: list[dict] = []    # observed payloads
    # length_times: next N generates answer done_reason="length" (truncated);
    # reject_think: HTTP 400 on any payload carrying "think" (the probe's
    # supports_think_param=False path); response_text: canned answer override.
    # think_inert: accept think:false but answer WITH a <think> tag anyway
    # (the erwan2 distill's real behavior under Ollama 0.40, verified live).
    length_times: int = 0
    reject_think: bool = False
    think_inert: bool = False
    response_text: str = "OK"

    MODEL_SIZES = {MODEL: 1_900_000_000, THINKER: 1_100_000_000}

    @classmethod
    def reset(cls) -> None:
        cls.loaded = {}
        cls.gpu_fraction = 1.0
        cls.fail_times = 0
        cls.oom_until_ctx = 0
        cls.requests = []
        cls.length_times = 0
        cls.reject_think = False
        cls.think_inert = False
        cls.response_text = "OK"

    # -- helpers -------------------------------------------------------------

    @classmethod
    def _frac(cls, num_ctx: int) -> float:
        return cls.gpu_fraction(num_ctx) if callable(cls.gpu_fraction) else cls.gpu_fraction

    def do_GET(self) -> None:
        if self.path.endswith("/api/tags"):
            body = json.dumps({"models": [{"name": m} for m in type(self).MODEL_SIZES]}).encode()
        elif self.path.endswith("/api/ps"):
            models = [{"name": n, **v} for n, v in type(self).loaded.items()]
            body = json.dumps({"models": models}).encode()
        else:
            self.send_response(404); self.end_headers(); return
        self._send(body)

    def _do_show(self) -> None:
        body = json.dumps({
            "capabilities": ["completion"],
            "details": {"quantization_level": "Q4_K_M",
                        "digest": "sha256:abcdef1234567890"},
            "template": "{{ .Prompt }}",
        }).encode()
        self._send(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        model = payload.get("model", "")

        if self.path.endswith("/api/show"):
            self._do_show(); return

        type(self).requests.append(payload)

        if "think" in payload and type(self).reject_think:
            self._send(json.dumps({"error": "model does not support thinking"}).encode(), 400)
            return

        if payload.get("keep_alive") == 0 and not payload.get("prompt"):
            # Pure unload (no prompt). Real Ollama still GENERATES when a
            # prompt is present with keep_alive=0, then unloads right after.
            type(self).loaded.pop(model, None)
            self._send(b"{}"); return

        # a generate: simulate a load (evicting nothing; Ollama would, but our
        # gateway unloads first) then maybe fail
        num_ctx = payload.get("options", {}).get("num_ctx", 4096)
        if type(self).fail_times > 0:
            type(self).fail_times -= 1
            type(self).loaded[model] = {
                "size": self.MODEL_SIZES[model],
                "size_vram": int(self.MODEL_SIZES[model] * self._frac(num_ctx)),
            }
            self._send(json.dumps({"error": "out of memory (runner stopped)"}).encode(), 500)
            return
        if type(self).oom_until_ctx and num_ctx > type(self).oom_until_ctx:
            type(self).loaded[model] = {
                "size": self.MODEL_SIZES[model],
                "size_vram": int(self.MODEL_SIZES[model] * self._frac(num_ctx)),
            }
            self._send(json.dumps({"error": "out of memory"}).encode(), 500)
            return

        type(self).loaded[model] = {
            "size": self.MODEL_SIZES[model],
            "size_vram": int(self.MODEL_SIZES[model] * self._frac(num_ctx)),
        }
        if type(self).length_times > 0:
            type(self).length_times -= 1
            body = json.dumps({
                "response": type(self).response_text, "done": True,
                "done_reason": "length",
                "prompt_eval_count": 10, "eval_count": 2,
                "total_duration": 1_000_000, "eval_duration": 500_000,
            }).encode()
            self._send(body)
            return
        text = type(self).response_text
        if type(self).think_inert and "think" in payload:
            text = "<think>hmm</think>" + text
        body = json.dumps({
            "response": text, "done": True, "done_reason": "stop",
            "prompt_eval_count": 10, "eval_count": 2, "total_duration": 1_000_000,
            "eval_duration": 500_000,
        }).encode()
        self._send(body)
        if payload.get("keep_alive") == 0:
            # generated with keep_alive=0: real Ollama unloads right after
            type(self).loaded.pop(model, None)

    def _send(self, body: bytes, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a: object) -> None:
        return


def _serve() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _gateway(base: str, watchdog: bool = False) -> AgentGateway:
    sup = OllamaSupervisorClient(base_url=base)
    cfg = AgentConfig()
    return AgentGateway(sup, cfg, watchdog=watchdog)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset():
    FakeOllama.reset()
    yield
    FakeOllama.reset()


# --- 1. one resident model at a time + swap logging ---------------------------

def test_swap_unloads_other_model_first() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        # thinker resident, then generate with coder -> must evict thinker
        FakeOllama.loaded[THINKER] = {"size": 1_100_000_000, "size_vram": 1_100_000_000}
        out = _run(gw.generate("coder", "ping", keep_warm=False))
        assert out == "OK"
        # after generation (not keep_warm), nothing resident
        assert FakeOllama.loaded == {}
        # the coder was loaded at some point: a generate request went out
        assert any(r.get("model") == MODEL and r.get("prompt") == "ping"
                   for r in FakeOllama.requests)
        # and an unload of the thinker happened
        assert any(r.get("model") == THINKER and r.get("keep_alive") == 0
                   for r in FakeOllama.requests)
    finally:
        server.shutdown(); server.server_close()


# --- 2. 0.97 enforcement + step-down ------------------------------------------

def test_gpu_fraction_enforced_with_stepdown() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)

        # coder num_ctx=6144: passes only when ctx small enough
        def frac(num_ctx: int) -> float:
            return 0.99 if num_ctx <= 4096 else 0.60  # spills above 4096

        FakeOllama.gpu_fraction = frac
        out = _run(gw.generate("coder", "ping"))
        assert out == "OK"
        # Final successful generate used num_ctx 4096 (the passing rung)
        gen = [r for r in FakeOllama.requests if r.get("prompt") == "ping"]
        assert gen[-1]["options"]["num_ctx"] == 4096
    finally:
        server.shutdown(); server.server_close()


def test_vram_guard_raises_at_lowest_rung() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        FakeOllama.gpu_fraction = 0.5  # never fits, even at 4096
        with pytest.raises(VramGuardError):
            _run(gw.generate("coder", "ping"))
    finally:
        server.shutdown(); server.server_close()


# --- 3. OOM / runner-stopped -> same step-down retry --------------------------

def test_oom_triggers_stepdown_retry() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        # OOM whenever ctx > 4096; fits at 4096
        FakeOllama.oom_until_ctx = 4096
        out = _run(gw.generate("coder", "ping"))
        assert out == "OK"
        gen = [r for r in FakeOllama.requests if r.get("prompt") == "ping"]
        assert gen[-1]["options"]["num_ctx"] == 4096
        # at least two generate attempts happened (6144 OOM, then 4096 ok)
        assert len(gen) >= 2
    finally:
        server.shutdown(); server.server_close()


# --- 4. keep_warm batching: no unload between same-model calls -----------------

def test_keep_warm_skips_unload_between_calls() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        _run(gw.generate("coder", "one", keep_warm=True))
        # still resident after keep_warm
        assert MODEL in FakeOllama.loaded
        _run(gw.generate("coder", "two", keep_warm=True))
        assert MODEL in FakeOllama.loaded
        # final non-warm call unloads
        _run(gw.generate("coder", "three", keep_warm=False))
        assert FakeOllama.loaded == {}
    finally:
        server.shutdown(); server.server_close()


# --- 5. cancel unloads (cancel event set mid-call) -----------------------------

def test_cancel_unloads_and_stops() -> None:
    import threading as th

    server, base = _serve()
    try:
        gw = _gateway(base)
        cancel = th.Event()
        cancel.set()  # set before the call
        from ollama_client import OllamaCancelledError
        with pytest.raises(OllamaCancelledError):
            _run(gw.generate("coder", "ping", cancel_event=cancel))
        # nothing left resident
        assert FakeOllama.loaded == {}
    finally:
        server.shutdown(); server.server_close()


# --- 6. single shared lock between gateway and supervisor ---------------------

def test_gateway_and_supervisor_share_one_lock() -> None:
    server, base = _serve()
    try:
        sup = OllamaSupervisorClient(base_url=base)
        gw = AgentGateway(sup, AgentConfig(), watchdog=False)
        assert gw._lock is sup._vram_mutex  # exactly ONE lock in the process
    finally:
        server.shutdown(); server.server_close()


# --- 7. structured output goes through the same lock + guard -----------------

def test_structured_generate_uses_format_and_guard() -> None:
    from agent_core.schemas import ValidateResult

    server, base = _serve()
    try:
        # fake returns a JSON object for structured prompts
        class StructFake(FakeOllama):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                model = payload.get("model", "")
                if payload.get("keep_alive") == 0:
                    FakeOllama.loaded.pop(model, None)
                    self._send(b"{}"); return
                FakeOllama.loaded[model] = {
                    "size": self.MODEL_SIZES[model], "size_vram": self.MODEL_SIZES[model],
                }
                assert "format" in payload  # schema-constrained
                body = json.dumps({
                    "response": '{"satisfies_request": true, "problems": [], "summary": "ok"}',
                    "done": True, "done_reason": "stop",
                }).encode()
                self._send(body)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), StructFake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            gw = _gateway(f"http://127.0.0.1:{srv.server_port}")
            out = _run(gw.structured("validator", "review", ValidateResult))
            assert out.satisfies_request is True
            # structured goes through generate(keep_warm default True) -> stays warm
            assert THINKER in FakeOllama.loaded
        finally:
            srv.shutdown(); srv.server_close()
    finally:
        server.shutdown(); server.server_close()


# --- 8. phase batching: thinker phase -> coder phase -> thinker ---------------

def test_phase_batched_swaps_logged() -> None:
    """route/plan (thinker) -> code (coder) -> validate (thinker): each phase
    keeps its model warm; exactly two swaps (thinker->coder, coder->thinker),
    each logged with a duration; same-model calls never reload."""
    import logging

    server, base = _serve()
    try:
        gw = _gateway(base)
        records = []
        h = logging.Handler()
        h.emit = lambda r: records.append(r.getMessage())
        logging.getLogger("ollama_client").addHandler(h)
        logging.getLogger("ollama_client").setLevel(logging.INFO)

        async def go():
            # thinker phase (two calls, no reload between)
            await gw.generate("thinker", "route")
            await gw.generate("thinker", "plan")
            # coder phase (two calls, no reload between)
            await gw.generate("coder", "code1")
            await gw.generate("coder", "code2")
            # thinker phase again
            await gw.generate("thinker", "validate")

        _run(go())

        swaps = [m for m in records if "swap" in m and "took" in m]
        # exactly two model swaps: thinker->coder and coder->thinker
        assert len(swaps) == 2, swaps
        # the swap targets are correct
        assert "-> " + MODEL in swaps[0]
        assert "-> " + THINKER in swaps[1]
        # same-model second calls (plan, code2) did NOT trigger a swap: only 2
        gens = [r for r in FakeOllama.requests if r.get("prompt") in
                ("route", "plan", "code1", "code2", "validate")]
        assert len(gens) == 5  # each prompt generated once
    finally:
        server.shutdown(); server.server_close()


# --- 9. inline_system: thinker folds system into the prompt, coder doesn't ---

def test_inline_system_folds_system_into_prompt() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        _run(gw.generate("thinker", "do this", system="SYSTEM TEXT"))
        gen = [r for r in FakeOllama.requests if "SYSTEM TEXT" in str(r.get("prompt", ""))
               or r.get("system") == "SYSTEM TEXT"]
        assert gen, FakeOllama.requests
        payload = gen[-1]
        # thinker (inline_system=True): system folded into the user prompt,
        # and the system field itself is empty
        assert payload["system"] == ""
        assert payload["prompt"].startswith("SYSTEM TEXT\n\ndo this")
    finally:
        server.shutdown(); server.server_close()


def test_coder_sends_system_separately() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        _run(gw.generate("coder", "do this", system="SYSTEM TEXT"))
        payload = FakeOllama.requests[-1]
        assert payload["system"] == "SYSTEM TEXT"
        assert payload["prompt"] == "do this"
        # the coder never receives a think switch
        assert "think" not in payload
    finally:
        server.shutdown(); server.server_close()


# --- 10. capability probe: supports_think_param both ways, cached by digest ---

def test_probe_capabilities_accepts_think() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        entry = _run(gw.probe_capabilities(THINKER))
        assert entry["supports_think_param"] is True
        assert entry["digest12"] == "abcdef123456"
        assert entry["quant"] == "Q4_K_M"
        assert entry["template"]
        # cached under the digest12 key
        caps = gw.config.extra["capabilities"]
        assert "abcdef123456" in caps
        # think_supported finds it by model name via the entry scan
        assert gw.think_supported(THINKER) is True
    finally:
        server.shutdown(); server.server_close()


def test_probe_capabilities_think_rejected() -> None:
    server, base = _serve()
    try:
        FakeOllama.reject_think = True
        gw = _gateway(base)
        entry = _run(gw.probe_capabilities(THINKER))
        assert entry["supports_think_param"] is False
        assert gw.think_supported(THINKER) is False
    finally:
        server.shutdown(); server.server_close()


def test_probe_detects_accepted_but_ignored_think() -> None:
    """The live case: Ollama accepts think:false (HTTP 200) but the model
    answers with a <think> tag anyway — the switch is semantically inert, so
    supports_think_param must record False (budgets assume thinking)."""
    server, base = _serve()
    try:
        FakeOllama.think_inert = True
        gw = _gateway(base)
        entry = _run(gw.probe_capabilities(THINKER))
        assert entry["supports_think_param"] is False
        assert entry["think_ignored"] is True
        # and the budget therefore boosts (thinking will happen)
        _run(gw.generate("thinker", "hi"))
        gen = [r for r in FakeOllama.requests if r.get("prompt") == "hi"]
        assert "think" not in gen[-1]
        assert gen[-1]["options"]["num_predict"] == 2048 + 512
    finally:
        server.shutdown(); server.server_close()


# --- 11. truncation: one bigger-budget retry, then OllamaTruncatedError -------

def test_truncation_retries_once_with_bigger_budget() -> None:
    server, base = _serve()
    try:
        FakeOllama.length_times = 1  # first call cut off, second fine
        gw = _gateway(base)
        out = _run(gw.generate("coder", "ping"))
        assert out == "OK"
        gens = [r for r in FakeOllama.requests if r.get("prompt") == "ping"]
        assert len(gens) == 2  # exactly one retry
        first = gens[0]["options"]["num_predict"]
        second = gens[1]["options"]["num_predict"]
        assert second == min(first * 2, 4096)
    finally:
        server.shutdown(); server.server_close()


def test_truncation_twice_raises() -> None:
    from ollama_client import OllamaTruncatedError

    server, base = _serve()
    try:
        FakeOllama.length_times = 2  # both attempts cut off
        gw = _gateway(base)
        with pytest.raises(OllamaTruncatedError):
            _run(gw.generate("coder", "ping"))
    finally:
        server.shutdown(); server.server_close()


# --- 12. public unload(): the only sanctioned unload path ---------------------

def test_gateway_unload_public() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        _run(gw.generate("coder", "ping", keep_warm=True))
        assert MODEL in FakeOllama.loaded
        _run(gw.unload(MODEL))
        assert FakeOllama.loaded == {}
        assert gw._resident_model is None
    finally:
        server.shutdown(); server.server_close()


# --- 12b. only the gateway module posts to /api/generate ---------------------

def test_only_gateway_module_posts_to_generate() -> None:
    """Grep proof: no other repo module contains a literal /api/generate or
    /api/chat post target (bench, main, orchestrator, router all go through
    AgentGateway). Test files (fakes) are exempt."""
    import re
    from pathlib import Path as _P

    root = _P(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if any(part in {"env", ".git", "node_modules", "__pycache__",
                        "tests", ".agent_runs", ".agent_staging"}
               for part in path.parts):
            continue
        if path.name in {"ollama_client.py", "e2e_smoke_test.py",
                         "planner_eval.py"}:
            continue  # the gateway home, the mock server, dev tooling
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r'["\'][^"\']*/api/(?:generate|chat)["\']', text):
            offenders.append(f"{path.name}: {match.group(0)}")
    assert offenders == [], f"direct Ollama endpoints outside the gateway: {offenders}"


# --- 11b. think switch: default-off once the probe confirms support ----------

def test_think_defaults_false_once_probe_confirms_support() -> None:
    """Live finding: an R1 distill thinks by default when no think switch is
    sent, truncating the answer budget. Once the probe confirms support, a
    bare generate must SEND think=false and skip the think_budget boost."""
    server, base = _serve()
    try:
        gw = _gateway(base)
        _run(gw.probe_capabilities(THINKER))  # FakeOllama accepts think -> True
        _run(gw.generate("thinker", "hi"))
        gen = [r for r in FakeOllama.requests if r.get("prompt") == "hi"]
        assert gen[-1]["think"] is False
        # no think_budget boost: base 2048, not 2048+512
        assert gen[-1]["options"]["num_predict"] == 2048
    finally:
        server.shutdown(); server.server_close()


def test_think_explicit_true_keeps_budget_boost() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        _run(gw.probe_capabilities(THINKER))
        _run(gw.generate("thinker", "hi", think=True))
        gen = [r for r in FakeOllama.requests if r.get("prompt") == "hi"]
        assert gen[-1]["think"] is True
        assert gen[-1]["options"]["num_predict"] == 2048 + 512
    finally:
        server.shutdown(); server.server_close()


def test_think_unprobed_sends_no_switch_and_boosts_budget() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)  # never probed: think_supported is None
        _run(gw.generate("thinker", "hi"))
        gen = [r for r in FakeOllama.requests if r.get("prompt") == "hi"]
        assert "think" not in gen[-1]
        assert gen[-1]["options"]["num_predict"] == 2048 + 512
    finally:
        server.shutdown(); server.server_close()


def test_coder_never_gets_think_switch() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        _run(gw.generate("coder", "hi", think=None))
        gen = [r for r in FakeOllama.requests if r.get("prompt") == "hi"]
        assert "think" not in gen[-1]
        assert gen[-1]["options"]["num_predict"] == 2048  # no boost, ever
    finally:
        server.shutdown(); server.server_close()


# --- 12b. out_meta: callers can read Ollama meta without the on_call hook ----

def test_out_meta_delivers_eval_counts() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        box: list[dict] = []
        out = _run(gw.generate("coder", "ping", out_meta=box))
        assert out == "OK"
        assert len(box) == 1
        meta = box[0]
        assert meta["eval_count"] == 2
        assert meta["eval_duration"] == 500_000
        assert meta["done_reason"] == "stop"
        assert meta["num_ctx"] == 6144  # the rung actually used
        # failure leaves the box empty
        FakeOllama.fail_times = 99  # every generate 500s -> all rungs OOM
        box2: list[dict] = []
        try:
            _run(gw.generate("coder", "ping", out_meta=box2))
        except Exception:
            pass
        assert box2 == []
    finally:
        server.shutdown(); server.server_close()


# --- 12c. probe_ctx: one tiny generation at an exact ctx, no step-down -------

def test_probe_ctx_ok_and_fraction() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        out = _run(gw.probe_ctx("coder", 6144))
        assert out["ok"] is True and out["fraction"] == 1.0
        # a probe never keeps the model warm
        assert FakeOllama.loaded == {}
        # exactly one ping generate at the requested ctx
        gen = [r for r in FakeOllama.requests if r.get("prompt") == "ping"]
        assert gen[-1]["options"]["num_ctx"] == 6144
        assert gen[-1]["options"]["num_predict"] == 8
    finally:
        server.shutdown(); server.server_close()


def test_probe_ctx_fails_below_gpu_fraction() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        FakeOllama.gpu_fraction = 0.5
        out = _run(gw.probe_ctx("coder", 16384))
        assert out["ok"] is False and out["fraction"] == 0.5
        assert FakeOllama.loaded == {}
    finally:
        server.shutdown(); server.server_close()


# --- 13. on_call hook: full per-call record, and swap records -----------------

def test_on_call_hook_receives_full_record() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        records: list[dict] = []
        gw.on_call = records.append
        _run(gw.generate("coder", "ping", keep_warm=True))
        assert len(records) == 1
        rec = records[0]
        assert rec["role"] == "coder" and rec["model"] == MODEL
        assert rec["prompt"] == "ping" and rec["response"] == "OK"
        assert rec["done_reason"] == "stop"
        assert rec["eval_count"] == 2 and rec["eval_duration_ns"] == 500_000
        assert rec["num_ctx"] == 6144  # the coder's configured rung
        assert rec["size_vram"] == rec["size_total"]  # 100% GPU
        assert rec["swaps"] == []

        # A swap is included in the next call's record
        _run(gw.generate("thinker", "route", keep_warm=True))
        assert records[1]["swaps"][0]["from"] == MODEL
        assert records[1]["swaps"][0]["to"] == THINKER
        assert "seconds" in records[1]["swaps"][0]

        # Hook failure never breaks a call
        gw.on_call = lambda _r: 1 / 0
        assert _run(gw.generate("thinker", "still fine", keep_warm=False)) == "OK"
    finally:
        server.shutdown(); server.server_close()
