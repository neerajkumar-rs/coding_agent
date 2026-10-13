# CHANGELOG — v3 implementation (Phases 0–3 consolidation)

All changes offline-verified: **267 tests pass** (`pytest tests/ test_master_controller.py`),
24 test files, no live-Ollama dependency in the suite.

## What changed

### Quick fixes
- `tui.py`: the "wait for FastAPI" block sat after `return None` in
  `_next_free_port` and never ran. The `/health` wait (30 s, progress shown)
  now lives in `start_all_services` after the spawn; tests cover success,
  timeout-raises, and reuse-skips-wait.
- `test_terminate_pid_taskkill_success` asserts the taskkill command on every
  platform (patched `sys.platform`); a POSIX `os.kill` variant was added.
- `test_events_endpoint_latest_run_default` asserts the exact run id of the
  newer of two seeded runs — no `or` fallback.

### Phase B — one pipeline, for real
- `Engine.run` / `_run_task` / `_generate_task_files` / `_plan` and the
  module-level `run()` are deleted. `agent_core/engine.py` keeps only the
  budget dataclass/exceptions and a slim per-workspace cancel registry
  (`WorkspaceSession`, alias `Engine`, behind `get_engine`).
- Deleted dead modules: `agent_core/llm_gateway.py`, `agent_core/planner.py`,
  `agent_core/auditor.py`. Their behavior tests were ported onto
  `run_pipeline` (`tests/test_agent_core.py`: happy-path deploy + staged-only).
- `e2e_smoke_test.py` and `agent_core/planner_eval.py` now go through
  `run_pipeline` / `AgentGateway` respectively.

### Phase C — one logging path
- The retired workspace-level log directory is gone (name intentionally not
  spelled: a grep test in `test_run_logs.py` enforces zero references):
  removed from `workspace_staging.py` (the `log_*` matrix + dir creation),
  `sandbox.py` call sites, `tui.py` (service logs moved to `logs/`),
  skip/protect sets, `.gitignore`, README.
- `RunJournal.log_llm_call` records prompt + answer (20 KB cut) plus
  `think_chars`, `done_reason`, `eval_count`, tok/s (from `eval_duration`),
  num_ctx used, `size_vram/size`.
- `AgentGateway.on_call`: an optional per-run hook the server sets so every
  call is journaled exactly once with full Ollama meta. `run_pipeline` gained
  `log_llm_calls=False` for that path (tests/headless keep pipeline-side
  logging). Hook exceptions never break a run.
- `state.json` is enriched by the server with intent, per-role cards (model,
  digest12, `supports_think_param`), swaps with seconds, and per-call num_ctx.
- CHECK outcomes (per-file + verifier steps + infra errors) are written to the
  journal's `sandbox` channel. VALIDATE prompt-path calls log under the
  `validator` role.

### Phase D — /logs, plug-and-play
- `agent_core.run_state.LOG_CHANNELS` is the single registry mapping channel →
  filename; `RunJournal.role_log_path`, `tail_run_logs`, `format_run_logs` and
  both TUIs' `/logs` all go through it. Adding a channel (a fourth role, a
  bench log) is a one-line entry; the registry test iterates it directly.
- `/logs [thinker|coder|validator|sandbox|events] [N]` in both TUIs: latest
  run of the ACTIVE workspace, N **lines** (default 60), and the exact path
  searched when no run exists. Works from workspaces outside the repo.

### Phase E — gateway alignment
- `AgentGateway.generate(..., schema=...)` alias for structured output.
- Truncation: `done_reason=length` retries ONCE with doubled `num_predict`
  (bounded by `MAX_NUM_PREDICT`) at the same ctx rung, then raises
  `OllamaTruncatedError`. (The locked supervisor path already did this.)
- Capability probe cache is keyed by digest12 (fallback: model name);
  `capability_entry()`/`think_supported()` find entries by name scan.
- `AgentGateway.unload(model)` is public; the server's cancel path uses it.
  Grep test: nothing outside `ollama_client.py` posts to `/api/generate` or
  `/api/chat`.
- `bench.py` probes go through the `AgentGateway` (one lock + VRAM guard);
  `--compare` models use a deepcopy-safe config shim. The old direct
  `_probe_once`/`_unload_models` transports are deleted.
- New fake-Ollama coverage: `inline_system` on/off, `supports_think_param`
  both ways (400 vs accept), truncation retry/raise, public unload, on_call
  record fields, swap records.

### Phase F — seams for later phases
- `agent_config.ceiling_for(role)` — every num_ctx clamp goes through it
  (including the `max_context_tokens` compat property). A calibration phase
  swaps in measured values here without touching callers.
- `agent_config.RoleRegistry` — validated get/set over roles, persisted to
  `.agent`; `/config/role` (and the Textual `/model` `/ctx` commands) run
  through it. Reads happen at call time under the gateway lock, so changes
  apply on the next call, never mid-generation.

## Deviations (with reasons)

- **Router v2, `/router`, confirm gate, eval-set gate shipped ahead of
  schedule** (v3 defers them to a later phase). They were already built and
  green under the earlier v2.1 prompt; removing them would have been pure
  waste. The v3 hotfix rules are a subset of what shipped.
- **TUI `_gpu_lock` kept as a UX mutex** (submit vs in-TUI bench in the same
  session). True GPU safety is the gateway lock + VRAM guard; a cross-process
  lock (TUI bench vs server run) is impossible without a server-side bench
  endpoint, which is deferred with the bench rewrite.
- **VALIDATE's schema-constrained path logs under `thinker`** when the
  validator shares the thinker model (the default) — only the prompt fallback
  path carries the explicit `role="validator"` hint. With a distinct validator
  model configured, `_role_of` already maps it correctly.
- **`/logs run <id>` selection dropped** per v3 deferral (channel-only now).

## Not verified on this machine (needs live Ollama + the RTX 3050)

- Whether `erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest` accepts `think: false`
  (the probe path is implemented and tested against a fake 400; real model
  untested).
- How Ollama `format=` constrained decoding interacts with the thinker's
  `<think>` output (fallback path exists; interaction unmeasured).
- The `think_budget` (512) that actually avoids mid-answer truncation.
- Which `num_ctx` rungs pass the 0.97 GPU-fraction check per model.
- Real nvidia-smi readings during a run, real swap durations.
- `scripts/live_smoke.py` end-to-end against the real server.
