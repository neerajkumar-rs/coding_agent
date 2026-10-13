# Migration Guide — Architecture

This document describes the CURRENT architecture, what changed, and how to adapt.

## The single pipeline (current)

`POST /agent/run` runs **one** pipeline, `orchestrator.run_pipeline`:

```
ROUTE -> PLAN -> CODE -> CHECK -> VALIDATE -> SUMMARY
```

- **ROUTE** is the router v2: T0/T1 word-boundary rules first (no model),
  then the T2 grey-zone classifier (one thinker call) only when the rules are
  unsure. Intents: `CHAT` (canned reply, zero model calls), `COMMAND`,
  `QUESTION`, `FILE_QUESTION`, `CODE_TASK`, `UNCLEAR` (one clarifying
  question). There is no CODE_TASK fall-through — unclassifiable is UNCLEAR.
- **PLAN** produces tasks in the single unified format `{path, action, description}`.
  With a `structured_generate` seam it uses Ollama `format` + pydantic
  (`agent_core.schemas.PipelinePlan`) first, then falls back to the JSON-regex path.
  Plan validation rejects placeholder filenames, junk artifacts, unsupported
  types, sub-8-word descriptions, language mismatches, and incomplete web plans.
- **CODE** runs each task through the coder with per-task CHECK (max 2 retries).
- **CHECK** is in-process (`compile()` for Python, plausibility for the rest)
  PLUS the deterministic `agent_core.verifier` chain on staged Python files
  (py_compile → ruff* → import smoke → pytest*; *skipped when unavailable).
- **VALIDATE** = one reviewer call (schema-constrained when available) + one fix pass.
- **SUMMARY** = a deterministic skeleton (files built/failed+reasons, how-to-run,
  total time) + 2–3 guarded model sentences.

The retired `POST /agent/plan` + `POST /agent/step` queue is gone; `/agent/run`
supersedes it. `run_pipeline` enforces budgets (900 s wall / 40 LLM calls /
2 attempts), the no-progress rule (identical diagnostic twice => marked
needs-human), and per-run observability via `RunJournal`
(`.agent_runs/<run_id>/state.json` + `events.jsonl` + per-channel logs).

### Per-run logs (current)

Every run writes one block per LLM call into the channel logs
(`thinker.log` / `coder.log` / `validator.log`) and one per CHECK into
`sandbox.log`, driven by the `LOG_CHANNELS` registry in
`agent_core.run_state` (adding a channel is a one-line entry). Call records
carry prompt, answer, `think_chars`, `done_reason`, `eval_count`, tok/s,
num_ctx actually used, and `size_vram/size`. `state.json` is enriched by the
server with the intent, per-role cards (model, digest12,
supports_think_param), swaps with seconds, and per-call num_ctx.
Both TUIs read them with `/logs [thinker|coder|validator|sandbox|events] [N]`
(default: latest run, 60 lines). The retired workspace-level log dir is gone
(a grep test enforces it).

### Cancellation

One cancel event per workspace (the engine's `_cancel_event`, set by
`POST /agent/cancel`). The pipeline checks it before every LLM call and between
stages; the supervisor races the in-flight HTTP request against the event and
closes the connection on cancel, then the caller unloads the resident model.
A cancelled run returns `status: "cancelled"` and stages nothing further.

### Lock count (honest, current)

There is exactly **ONE process-wide asyncio lock** on the hot path:
`AgentGateway._lock` in `ollama_client.py`. The supervisor's `_vram_mutex`
is the *same object* (injected by the gateway), and bench probes go through
the same gateway. The retired sync `OllamaEngine._sync_lock` and the
`agent_core.llm_gateway` process lock are gone with the retired Engine
pipeline. The TUI keeps a client-side UI mutex (`_gpu_lock`) so an in-TUI
benchmark never overlaps a TUI-submitted run — it is UX serialization, not
GPU safety; GPU safety is the gateway lock + the VRAM guard, and it cannot
serialize a headless bench against a server run (cross-process; the guard's
unload-others step is the honest mechanism there).

## For users

### What you will notice

- **No more junk files.** `tasks.md`, `summary.txt`, `taskN_code.txt` and stub
  files are gone from your workspace. Run metadata now lives under
  `.agent_runs/<run_id>/` (git-ignored): `state.json`, `events.jsonl`, `tasks.md`,
  `backup/`.
- **Failed code is loud, not silent.** When the coder produces unusable output,
  the attempt FAILS with a clear reason. Nothing is scaffolded into your tree.
- **Runs stop on their own.** Budgets: 900 s wall time, 40 LLM calls, 2 attempts
  per task. A task repeating the identical error twice is marked `needs_human`
  instead of looping forever.
- **Editing, not rewriting.** The coder edits existing files with
  SEARCH/REPLACE blocks; only new files are written whole.
- **The TUI no longer murders Ollama.** `tui.py` reuses a running Ollama and
  only manages processes it spawned itself.
- **Server binds to localhost.** `HOST` now defaults to `127.0.0.1`. Set
  `HOST=0.0.0.0` explicitly if you need LAN access.

### New capabilities

| Feature | How |
|---|---|
| One-shot autonomous run | `POST /agent/run` (or `/run` in the TUI) |
| Live progress | `GET /agent/events` (SSE) or the TUI task table |
| Cancel a run | `POST /agent/cancel` (or `/cancel` in the TUI) |
| Undo a deploy | `POST /agent/rollback` (restores `.agent_runs/<id>/backup/`) |
| Planner health check | `python -m agent_core.planner_eval --model erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest` |

### .agent profile additions

```json
{
  "roles": {
    "thinker":   {"model": "...", "num_ctx": 8192, "num_predict": 2048,
                  "temperature": 0.2, "keep_alive": "5m",
                  "inline_system": true,  "think_budget": 512},
    "coder":     {"model": "...", "num_ctx": 6144, "num_predict": 2048,
                  "temperature": 0.1, "keep_alive": "5m",
                  "inline_system": false, "think_budget": 0},
    "validator": {"model": "...", "num_ctx": 8192, "num_predict": 512,
                  "temperature": 0.0, "keep_alive": "5m",
                  "inline_system": true,  "think_budget": 512}
  },
  "allowed_commands": ["py_compile"],
  "router": {"min_confidence": 0.7, "chat_use_llm": false}
}
```

- `roles` is the single source of truth. Legacy top-level keys
  (`planner_model`, `coder.5b_or_3b`, `max_context_tokens`, `keep_alive`)
  fold into `roles` on load and are never read elsewhere.
- Every `num_ctx` is clamped through `agent_config.ceiling_for(role)`
  (static ceilings today; a calibration phase swaps in measured values later).
- Live switching: `POST /config/role` (or `/model` `/ctx` in the Textual TUI)
  goes through `RoleRegistry` — validated, persisted, applied on the next
  gateway call.
- `allowed_commands` gates the sandbox: any custom validation command must be
  one of these **names** (free-form shell is rejected). Add `"pytest"` to enable
  the pytest vector.

## For developers

### orchestrator.run_pipeline is the entry point

The `VibeOrchestrator` shim is retired. New code calls the single pipeline:

```python
from orchestrator import run_pipeline

result = await run_pipeline(
    "make a blog site in html5 css and js",
    workspace,
    generate=my_generate,          # async (model, prompt, system=, temperature=) -> str
    planner_model="erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest",
    coder_model="qwen2.5-coder:3b",
)
result["staged_files"], result["failed_tasks"], result["trace"], result["run_id"]
```

`agent_core.engine` no longer contains a pipeline. It holds the shared budget
dataclass/exceptions (`RunBudgets`, `BudgetExhausted`, `RunCancelled`) and a
slim per-workspace session object (`WorkspaceSession`, alias `Engine`) behind
`get_engine` — the only cross-request state is the cancel event set by
`POST /agent/cancel`. The old `Engine.run` loop, `agent_core.llm_gateway`,
`agent_core.planner` and `agent_core.auditor` are deleted; their behavior
tests were ported onto `run_pipeline` (`tests/test_agent_core.py`).

### The pipeline contract

1. `context.py` — `ContextBudget` enforces ≤85% of num_ctx for prompt inputs.
2. PLAN/CODE/VALIDATE go through the injected `generate` seam (all LLM access).
3. `staging.py`/`workspace_staging.py` write to `.agent_staging/` only;
   protected paths (`.git`, `.agent*`, `env/`, `.venv`) and escapes are rejected.
4. `verifier.py` is the deterministic Python gate; the in-process plausibility
   checks cover non-Python. The validator LLM never blocks a deterministic pass.
5. Retry feedback goes straight to the coder, not through the planner.
6. Deploy happens only via `/agent/deploy` (with backups for `/agent/rollback`).

### Ollama payload rules (corrected)

- `keep_alive` is a **top-level** payload field (default `"5m"`).
- `options` contains exactly `num_ctx`, `num_predict`, `temperature`, `seed`.
- `flash_attention` / model residency are **server** settings:
  `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_MAX_LOADED_MODELS=1`.
- `done_reason == "length"` is retryable: retried once with a doubled
  `num_predict` when output-bound; otherwise the caller retries with a smaller
  prompt. Truncated output is never judged.

## Test suite map

| Suite | Covers |
|---|---|
| `tests/test_llm_client.py` | think-stripping, truncation retry, in-flight cancel |
| `tests/test_run_pipeline.py` | ROUTE/PLAN/CODE/CHECK/VALIDATE/SUMMARY, journal, cancel, verifier gate |
| `tests/test_orchestrator.py` | run_pipeline budgets, no-progress, no-junk regression |
| `tests/test_router.py` | router v2 rules + T2 battery + guards |
| `tests/test_router_eval.py` | 62-phrase eval set, accuracy gate |
| `tests/test_confirm_gate.py` | plan-confirm endpoints (timeout, decline, approve) |
| `tests/test_plan_quality.py` | plan validation rules, fallback |
| `tests/test_summary.py` | summary skeleton + guarded prose, UNVERIFIED |
| `tests/test_save_flow.py` | save prompt, --yes, config fallback, no removed-model strings |
| `tests/test_check_sandbox.py` | in-process CHECK, infra vs content errors, plausibility |
| `tests/test_queue_bug.py` | build_id stale-server guard, junk filter, target rejection |
| `tests/test_agent_core.py` | schemas, coder parsing, session registry (ported off Engine.run) |
| `tests/test_edits.py` | SEARCH/REPLACE applier, token budgeter, repo map |
| `tests/test_verifier.py` | deterministic step chain |
| `tests/test_safety.py` | path escape, protected paths, sandbox allowlist, rollback |
| `tests/test_events_endpoint.py` | SSE events, cancel, rollback endpoints |
| `tests/test_run_logs.py` | /logs readers, LOG_CHANNELS registry, retired-dir grep test |
| `tests/test_vram_gateway.py` | single lock, VRAM guard, swaps, probe, truncation, hook |
| `tests/test_role_registry.py` | ceiling_for, RoleRegistry validate/clamp/persist |
| `tests/test_phase2_config.py` | roles defaults, exact-match resolver, round-trip |
| `tests/test_phase5_live_switching.py` | /config/role endpoint (model + num_ctx) |
| `tests/test_tui_textual.py` | Textual UI (sidebar, save prompt, wordmark) |
| `tests/test_tui_startup.py` | TUI startup, workspace picker, CLI args, health wait |
| `tests/test_tui_benchmark.py` | benchmark via gateway, config roles, apply flow |
| `test_master_controller.py` | legacy FastAPI lifecycle (via /agent/run) |
| `e2e_smoke_test.py` | full run_pipeline against a mock Ollama |
