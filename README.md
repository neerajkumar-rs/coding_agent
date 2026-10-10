# Local LLM Master Controller — Ollama Native

A GPU-constrained agentic coding pipeline designed for a 4 GB RTX 3050.
All LLM inference runs through a local [Ollama](https://ollama.ai) instance at
`http://localhost:11434`. A pure-Python FastAPI supervisor (0 MB VRAM) routes
work through a **single** `agent_core` engine — one pipeline, one process-wide
LLM lock, deterministic verification.

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Architecture

```
FastAPI Master Controller (main.py)   ←  0 MB VRAM deterministic supervisor
        │  /agent/run · /agent/plan · /agent/step · /agent/deploy
        │  /agent/events (SSE) · /agent/cancel · /agent/rollback
        ▼
agent_core.engine.Engine              ←  THE pipeline (orchestrator.py is a shim)
        │
        ├── context.py     AST repo map + token budgeter (chars/3.5, ≤85% of num_ctx)
        ├── planner.py     Plan via Ollama structured outputs (pydantic-validated,
        │                  one retry) with markdown fallback
        ├── coder.py       New files: full fenced block. Edits: SEARCH/REPLACE
        │                  (exact → whitespace-normalized, unique-only)
        ├── verifier.py    DETERMINISTIC: py_compile → ruff* → import smoke → pytest*
        │                  (*skipped when tool absent; the only pass/fail gate)
        ├── auditor.py     LLM review of the DIFF only; verdict pass/concerns;
        │                  NEVER blocks a deterministic pass
        ├── staging.py     .agent_staging/ → verify → deploy (backups to
        │                  .agent_runs/<id>/backup/), path-escape + protected paths
        └── run_state.py   .agent_runs/<run_id>/{state.json,events.jsonl,tasks.md}
        │
        ▼  one process-wide lock (sync threading.Lock + async gateway)
Local Ollama  http://localhost:11434
        ├── deepseek-r1:1.5b          ←  Planner / Auditor
        └── qwen2.5-coder:3b          ←  Coder
```

**Run budgets** (per run): 900 s wall time, 40 LLM calls, 2 attempts per task.
Two consecutive attempts with an identical diagnostic hash → task `needs_human`.
Run statuses: `completed` | `failed` | `needs_human` | `budget_exhausted` | `cancelled`.

---

## Models

| Role | Model | Used by |
|---|---|---|
| Planner / Auditor | `deepseek-r1:1.5b` | planner, auditor, spec refinement |
| Coder | `qwen2.5-coder:3b` (or `1.5b`) | code generation / SEARCH/REPLACE edits |

```powershell
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5-coder:3b
```

Check whether the planner model conforms to the Plan schema reliably:

```powershell
.\env\Scripts\python.exe -m agent_core.planner_eval --model deepseek-r1:1.5b
# >20% schema failure  ->  use --planner-model qwen2.5-coder:3b
```

---

## Quick Start

### 1. Prerequisites

- Ollama running locally: `ollama serve` (default port 11434). If it is already
  running, `tui.py` **reuses** it — it never kills ollama.exe processes it did
  not spawn.
- Python environment: `.\env\Scripts\activate`

### 2. Initialize the `.agent` profile

```powershell
# Auto-benchmark hardware and write profile (RTX 3050 → 4096 tokens)
.\env\Scripts\python.exe main.py --recalculate

# Manual bounds
.\env\Scripts\python.exe main.py --recalculate --max-context-tokens 8192 --coder-model qwen2.5-coder:1.5b
```

`.agent` (workspace root, git-ignored):

```json
{
  "planner_model": "deepseek-r1:1.5b",
  "coder.5b_or_3b": "qwen2.5-coder:3b",
  "max_context_tokens": 4096,
  "keep_alive": "5m",
  "allowed_commands": ["py_compile"]
}
```

Every Ollama payload enforces exactly:

```json
{
  "keep_alive": "5m",                  // top-level only
  "options": { "num_ctx": 4096, "num_predict": 2048, "temperature": 0.1, "seed": 42 }
}
```

(`flash_attention` is a **server** setting — see Environment Variables.)

### 3. Run

```powershell
# TUI: reuses running Ollama, starts FastAPI, /run for autonomous runs
.\env\Scripts\python.exe tui.py

# Or just the controller (localhost by default)
.\env\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8000
```

---

## API Reference

All endpoints accept and return JSON.

### `POST /agent/run` — autonomous pipeline (the one you want)

```json
{ "prompt": "Create a greeting CLI in generated.py", "auto_deploy": true }
```

Runs the full engine: context scan → plan → per task [coder → stage →
deterministic verify → audit (diff only)] → deploy on success with backups.
Response carries `status`, `run_id`, `files`, `deployed_files`, `attempts`,
`llm_calls`, `errors`, `needs_human`, `audits`.

### `POST /agent/plan` / `POST /agent/step`

Interactive flow: plan a task queue, then execute tasks one at a time into
`.agent_staging/`, each followed by a sandbox validation. `/agent/step` accepts
`custom_command` only as an **allowlisted name** from `.agent`
(`allowed_commands`, default `["py_compile"]`); free-form shell is rejected.

### `POST /workspace/status`

`difflib.unified_diff` between `.agent_staging/` and the active workspace.

### `POST /agent/deploy`

Atomic deploy of staged files into the workspace, then flushes staging.
Overwritten files are backed up to `.agent_runs/<latest>/backup/` first.

### `POST /agent/rollback`

```json
{ "run_id": "20261010T120000Z-ab12cd" }   // or omit for the latest backup
```

Restores files from a run's deploy backup.

### `GET /agent/events?workspace_dir=…&run_id=…&follow=true`

Server-sent events stream of a run's `events.jsonl` (replays history;
`follow=true` tails for live updates).

### `POST /agent/cancel`

Requests cancellation of the in-progress run for a workspace.

---

## Project Structure

```
e:\coding_agent\
├── agent_core/            ← THE pipeline
│   ├── engine.py          ← Engine.run(request): budgets, no-progress, sessions
│   ├── llm_gateway.py     ← structured outputs (format=schema + pydantic, 1 retry)
│   ├── schemas.py         ← Plan / AuditResult pydantic models
│   ├── planner.py         ← structured plan + markdown fallback
│   ├── coder.py           ← full blocks + SEARCH/REPLACE edits
│   ├── verifier.py        ← py_compile → ruff → import smoke → pytest
│   ├── auditor.py         ← diff review, never blocks a deterministic pass
│   ├── staging.py         ← staging + path safety + backups/rollback
│   ├── run_state.py       ← .agent_runs journal (state/events/tasks)
│   ├── context.py         ← AST repo map + token budgeter
│   └── planner_eval.py    ← planner schema-conformance benchmark
│
├── main.py                ← FastAPI controller (all endpoints)
├── orchestrator.py        ← COMPAT SHIM over Engine (legacy RunResult)
├── agent_config.py        ← .agent profile + --recalculate CLI
├── ollama_client.py       ← think-stripping + truncation-guarded LLM layer
├── workspace_staging.py   ← staging manager + difflib + atomic deploy
├── sandbox.py             ← allowlisted validation commands (.ps1/.sh)
├── scanner.py             ← workspace scanner (interactive endpoints)
├── tui.py                 ← Rich TUI (reuses running Ollama)
│
├── tests/                 ← pytest suite (61 tests, offline)
├── test_master_controller.py  ← legacy lifecycle suite (offline)
├── e2e_smoke_test.py          ← end-to-end with mock Ollama server
└── test_live_ollama.py        ← live Ollama connection check
```

Workspace-side artifacts (all git-ignored):

```
.agent             profile        .agent_logs/     role logs
.agent_staging/    code cache     .agent_runs/<id>/ state.json · events.jsonl · tasks.md · backup/
```

The workspace root itself only ever receives the **requested target files**
(after deploy). No `tasks.md`/`summary.txt`/`taskN_code.txt`/stub files.

---

## Testing

```powershell
# Unit tests (61, offline)
.\env\Scripts\python.exe -m pytest tests/ -q

# Legacy lifecycle suite (offline)
.\env\Scripts\python.exe test_master_controller.py

# End-to-end with a mock Ollama server (offline)
.\env\Scripts\python.exe e2e_smoke_test.py

# Live Ollama check (needs ollama serve)
.\env\Scripts\python.exe test_live_ollama.py
```

---

## VRAM Safety Model (RTX 3050 4 GB)

1. **One model at a time** — a single process-wide lock serializes every
   generation across all workspaces and endpoints.
2. **Server-side residency** — when the TUI spawns `ollama serve` it sets
   `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_MAX_LOADED_MODELS=1`
   (server env vars — NOT payload fields). `keep_alive: "5m"` keeps the model
   warm between calls via the payload's top-level field.
3. **Context discipline** — `num_ctx=4096`; prompt inputs are capped at 85% of
   the window (tokens ≈ chars/3.5) minus the output reserve.

---

## Benchmarking

Compare candidate models per role (planner / coder / validator) on real probes.
Reports land in `<workspace>\.agent_runs\bench\` (JSON + Markdown).

```powershell
# Headless from the shell (starts/reuses Ollama, NO FastAPI):
.\env\Scripts\python.exe tui.py --benchmark --quick
.\env\Scripts\python.exe tui.py --workspace E:\proj --benchmark --quick --coexist
.\env\Scripts\python.exe tui.py --benchmark --compare validator=deepseek-r1:1.5b,phi4-mini --repeat 3

# Or directly:
.\env\Scripts\python.exe bench.py --workspace . --quick
```

Inside the TUI (GPU-locked against plan/step so a benchmark never overlaps
inference on the 4 GB card):

```
/benchmark                                  all roles from .agent
/benchmark --quick
/benchmark --roles validator --repeat 3
/benchmark --compare planner=deepseek-r1:1.5b,phi4-mini
/benchmark --plan-mode two-stage
/benchmark --coexist
/benchmark last                             print the newest report
/benchmark apply                            write recommended models into
                                            .agent "roles" (asks y/N first)
```

Missing models are skipped with an `ollama pull <name>` hint. Models are
unloaded (`keep_alive=0`) when the benchmark finishes. Ctrl-C cancels the
benchmark and returns to the prompt without touching services.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_FLASH_ATTENTION` | `1` (set by TUI) | Server: flash attention |
| `OLLAMA_MAX_LOADED_MODELS` | `1` (set by TUI) | Server: one resident model |
| `VIBE_MAX_WALL_SECONDS` | `900` | Run wall-time budget |
| `VIBE_MAX_LLM_CALLS` | `40` | Run LLM-call budget |
| `VIBE_VALIDATION_TIMEOUT_SECONDS` | `300` | Sandbox subprocess timeout |
| `VIBE_PORT` | `8000` | FastAPI port (TUI) |
| `PORT` | `8000` | FastAPI listen port |
| `HOST` | `127.0.0.1` | FastAPI listen host (set `0.0.0.0` to expose) |
