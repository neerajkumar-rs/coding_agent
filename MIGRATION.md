# Migration Guide — Architecture

This document describes the CURRENT architecture, what changed, and how to adapt.

## The single pipeline (current)

`POST /agent/run` runs **one** pipeline, `orchestrator.run_pipeline`:

```
ROUTE -> PLAN -> CODE -> CHECK -> VALIDATE -> SUMMARY
```

- **ROUTE** is rules-only (no model): `QUESTION` / `FILE_QUESTION` / `CODE_TASK` / `UNCLEAR`.
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
(`.agent_runs/<run_id>/state.json` + `events.jsonl`).

### Cancellation

One cancel event per workspace (the engine's `_cancel_event`, set by
`POST /agent/cancel`). The pipeline checks it before every LLM call and between
stages; the supervisor races the in-flight HTTP request against the event and
closes the connection on cancel, then the caller unloads the resident model.
A cancelled run returns `status: "cancelled"` and stages nothing further.

### Lock count (honest, pre-Phase-3)

Today there is ONE async lock on the hot path: the supervisor's `_vram_mutex`
(`/agent/run` goes through it). The sync `OllamaEngine._sync_lock`, the
`agent_core.llm_gateway` process lock, and the TUI `_gpu_lock` still exist and
are consolidated into a single gateway lock in Phase 3.

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
  "keep_alive": "5m",
  "allowed_commands": ["py_compile"]
}
```

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

`agent_core.engine.Engine` still exists as the per-workspace session object
behind `get_engine` / `POST /agent/cancel` / `POST /agent/rollback` /
`GET /agent/events`. Its internal `run()` loop (structured planner, auditor)
is retained for those consumers but is NOT the `/agent/run` path.

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
| `tests/test_router.py` | rules-only classify_request |
| `tests/test_plan_quality.py` | plan validation rules, fallback |
| `tests/test_summary.py` | summary skeleton + guarded prose, UNVERIFIED |
| `tests/test_save_flow.py` | save prompt, --yes, config fallback, no removed-model strings |
| `tests/test_check_sandbox.py` | in-process CHECK, infra vs content errors, plausibility |
| `tests/test_queue_bug.py` | build_id stale-server guard, junk filter, target rejection |
| `tests/test_agent_core.py` | structured outputs, planner fallback, engine sessions |
| `tests/test_edits.py` | SEARCH/REPLACE applier, token budgeter, repo map |
| `tests/test_verifier.py` | deterministic step chain |
| `tests/test_safety.py` | path escape, protected paths, sandbox allowlist, rollback |
| `tests/test_events_endpoint.py` | SSE events, cancel, rollback endpoints |
| `tests/test_tui_textual.py` | Textual UI (sidebar, save prompt, wordmark) |
| `tests/test_tui_startup.py` | TUI startup, workspace picker, CLI args |
| `tests/test_tui_benchmark.py` | benchmark run/main, gpu lock, config roles |
| `test_master_controller.py` | legacy FastAPI lifecycle (now via /agent/run) |
| `e2e_smoke_test.py` | full Engine pipeline against a mock Ollama |
