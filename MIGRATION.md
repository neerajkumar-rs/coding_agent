# Migration Guide — Phase 1→6 Refactor

This document covers what changed, why, and how to adapt.

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
| Planner health check | `python -m agent_core.planner_eval --model deepseek-r1:1.5b` |

### .agent profile additions

```json
{
  "keep_alive": "5m",
  "allowed_commands": ["py_compile"]
}
```

- `allowed_commands` gates the sandbox: `/agent/step`'s `custom_command` must be
  one of these **names** (free-form shell is rejected). Add `"pytest"` to enable
  the pytest vector.

## For developers

### orchestrator.py is now a shim

`VibeOrchestrator` still works (same constructor, same `run()` signature, same
`RunResult` shape) but delegates everything to `agent_core.engine.Engine`.
New code should use the engine directly:

```python
from agent_core.engine import Engine, RunRequest

report = Engine(workspace=".").run(RunRequest(prompt="add a /health endpoint"))
report.status, report.files, report.deployed_files
```

### The pipeline contract

1. `context.py` builds a token-budgeted repo map + snippets (≤85% of num_ctx).
2. `planner.py` produces tasks (structured output, pydantic-validated, one
   retry; markdown fallback for weak models).
3. `coder.py` produces full blocks (new files) or SEARCH/REPLACE edits
   (existing files). Edit application is exact-first, whitespace-fuzzy second,
   and fails loudly on no-match/ambiguity.
4. `staging.py` writes to `.agent_staging/` only; protected paths
   (`.git`, `.agent*`, `env/`, `.venv`) and escapes are rejected.
5. `verifier.py` is the ONLY pass/fail gate: py_compile → ruff* → import smoke
   → pytest* (*skipped when unavailable).
6. `auditor.py` reviews the diff and NEVER blocks a deterministic pass.
7. Retry feedback goes straight to the coder (failing verifier output ≈1500
   chars + staged file state), not through the planner.
8. Deploy happens on success (with backups) or explicitly via `/agent/deploy`.

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
| `tests/test_llm_client.py` | think-stripping, truncation retry |
| `tests/test_orchestrator.py` | budgets, no-progress, no-junk regression |
| `tests/test_agent_core.py` | structured outputs, planner fallback, sessions |
| `tests/test_edits.py` | SEARCH/REPLACE applier, token budgeter, repo map |
| `tests/test_verifier.py` | deterministic step chain |
| `tests/test_safety.py` | path escape, protected paths, sandbox allowlist, rollback |
| `tests/test_events_endpoint.py` | SSE events, cancel, rollback endpoints |
| `test_master_controller.py` | legacy FastAPI lifecycle |
| `e2e_smoke_test.py` | full pipeline against a mock Ollama |
