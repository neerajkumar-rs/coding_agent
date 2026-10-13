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
        │  /agent/run (THE pipeline) · /agent/deploy
        │  /agent/events (SSE) · /agent/cancel · /agent/rollback
        ▼
orchestrator.run_pipeline             ←  ROUTE → PLAN → CODE → CHECK → VALIDATE → SUMMARY
        │  (agent_core is the library it calls)
        ├── agent_core.context.py    ContextBudget: prompts capped at ≤85% of num_ctx
        ├── agent_core.schemas.py    PipelinePlan / ValidateResult (Ollama format +
        │                            pydantic, 1 retry) with JSON-regex fallback
        ├── agent_core.verifier.py   DETERMINISTIC Python gate: py_compile → ruff* →
        │                            import smoke → pytest* (*skipped when tool absent)
        ├── agent_core.staging.py    path-escape + protected paths, backups/rollback
        ├── agent_core.run_state.py  RunJournal: .agent_runs/<run_id>/{state.json,
        │                            events.jsonl}
        ├── workspace_staging.py     .agent_staging/ → difflib → atomic deploy
        └── sandbox.py               in-process plausibility CHECK (non-Python)
        │
        ▼  one async lock on the hot path (OllamaSupervisorClient._vram_mutex)
Local Ollama  http://localhost:11434
        ├── erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest  ←  Thinker / Planner / Validator / Router T2
        └── qwen2.5-coder:3b-instruct                    ←  Coder / Answers / Summaries
```

**Run budgets** (per run): 900 s wall time, 40 LLM calls, 2 attempts per task.
Two consecutive attempts with an identical diagnostic hash → task `needs_human`.
Run statuses: `completed` | `failed` | `needs_human` | `budget_exhausted` | `cancelled`.

---

## Models

| Role | Model | Used by |
|---|---|---|
| Thinker / Planner / Validator / Router T2 | `erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest` | routing grey zone, planning, validation, Q&A |
| Coder | `qwen2.5-coder:3b-instruct` | code generation / SEARCH/REPLACE edits |

```powershell
ollama pull erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest
ollama pull qwen2.5-coder:3b-instruct
```

Model names are matched **exactly** against `/api/tags` (the thinker name
carries a namespace and a `:latest` tag — never stripped or normalized). A
missing model raises with the exact `ollama pull <name>` hint.

Check whether the planner model conforms to the Plan schema reliably:

```powershell
.\env\Scripts\python.exe -m agent_core.planner_eval --model erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest
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
  "roles": {
    "thinker":   {"model": "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest", "num_ctx": 8192, "num_predict": 2048, "temperature": 0.2, "keep_alive": "5m", "inline_system": true,  "think_budget": 512},
    "coder":     {"model": "qwen2.5-coder:3b-instruct",                    "num_ctx": 6144, "num_predict": 2048, "temperature": 0.1, "keep_alive": "5m", "inline_system": false, "think_budget": 0},
    "validator": {"model": "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest", "num_ctx": 8192, "num_predict": 512,  "temperature": 0.0, "keep_alive": "5m", "inline_system": true,  "think_budget": 512}
  },
  "allowed_commands": ["py_compile"],
  "router": {"min_confidence": 0.7, "chat_use_llm": false}
}
```

Legacy keys (`planner_model`, `coder.5b_or_3b`, `max_context_tokens`,
`keep_alive`) are migration inputs only: on load they fold into `roles` and
are never read elsewhere.

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

Runs the single pipeline `orchestrator.run_pipeline`: ROUTE → PLAN → CODE →
CHECK → VALIDATE → SUMMARY. Stages CHECK-passed files (never auto-deploys).
Response carries `summary`, `staged_files`, `failed_tasks`, `trace`, `run_id`
for code tasks, or `answer`, `trace`, `run_id` for questions, plus `status`
on cancel/budget stop.

`POST /agent/plan` and `POST /agent/step` were **retired** — `/agent/run`
supersedes them. `/agent/run` never auto-deploys: it stages CHECK-passed files
and returns `{summary, staged_files, failed_tasks, trace, run_id}` (or
`{answer, trace, run_id}` for questions). Deploy is explicit:

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
├── orchestrator.py        ← run_pipeline: THE single pipeline behind /agent/run
├── agent_config.py        ← .agent profile + --recalculate CLI
├── ollama_client.py       ← think-stripping + truncation-guarded LLM layer
├── workspace_staging.py   ← staging manager + difflib + atomic deploy
├── sandbox.py             ← in-process plausibility CHECK + allowlisted commands
├── scanner.py             ← workspace scanner (file questions)
├── tui.py                 ← Rich TUI (reuses running Ollama)
├── tui_textual.py         ← Textual TUI (two-column, sidebar, save prompt)
│
├── tests/                 ← pytest suite (160 tests, offline)
├── test_master_controller.py  ← legacy lifecycle suite (offline, via /agent/run)
├── e2e_smoke_test.py          ← end-to-end with mock Ollama server
└── scripts/live_smoke.py      ← live Ollama connection check (not pytest)
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
# Unit tests (160, offline)
.\env\Scripts\python.exe -m pytest tests/ -q

# Legacy lifecycle suite (offline)
.\env\Scripts\python.exe test_master_controller.py

# End-to-end with a mock Ollama server (offline)
.\env\Scripts\python.exe e2e_smoke_test.py

# Live Ollama check (needs ollama serve)
.\env\Scripts\python.exe scripts/live_smoke.py
```

---

## VRAM Safety Model (RTX 3050 4 GB)

**The honest guarantee:** at most ONE model is resident in VRAM, and after
every load at least **97% of that model's weights sit on the GPU**
(`size_vram / size >= 0.97`). The OS may still page the model file into cache
while loading — that is normal and not a VRAM leak.

1. **One process-wide lock** — `AgentGateway._lock` (in `ollama_client.py`)
   serializes every generation on the `/agent/run` path. It is the only lock
   on the hot path (the supervisor's `_vram_mutex` is the *same* object,
   injected by the gateway).
2. **VRAM guard (per load of model M)** — `/api/ps` first: every other resident
   model is unloaded (`keep_alive=0`) and polled empty (15 s timeout). After the
   load, `size_vram/size >= 0.97` is required; otherwise M is unloaded, num_ctx
   steps down one rung of [4096, 6144, 8192, 12288, 16384], and the load is
   retried. At the lowest rung it raises `VramGuardError` — it never falls back
   to CPU silently. OOM / "runner stopped" / HTTP 500 trigger the same
   step-down retry.
3. **Server-side residency** — when the TUI spawns `ollama serve` it sets
   `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_MAX_LOADED_MODELS=1`,
   `OLLAMA_NUM_PARALLEL=1` (server env vars, NOT payload fields). Reusing an
   already-running Ollama prints a warning that these cannot be verified.
   `keep_alive: "5m"` is the payload's top-level field.
4. **Context discipline** — per-role `num_ctx` (thinker 8192, coder 6144 by
   default); prompt inputs capped at 85% of the window minus the output reserve.
5. **nvidia-smi watchdog** — logs `memory.used` around each load; warns when
   free VRAM < 300 MB; silent when nvidia-smi is absent. On Windows it prints
   once: set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback".

---

## Benchmarking

Compare candidate models per role (planner / coder / validator) on real probes.
Reports land in `<workspace>\.agent_runs\bench\` (JSON + Markdown).

```powershell
# Headless from the shell (starts/reuses Ollama, NO FastAPI):
.\env\Scripts\python.exe tui.py --benchmark --quick
.\env\Scripts\python.exe tui.py --workspace E:\proj --benchmark --quick --coexist
.\env\Scripts\python.exe tui.py --benchmark --compare validator=erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest --repeat 3

# Or directly:
.\env\Scripts\python.exe bench.py --workspace . --quick
```

Inside the TUI (GPU-locked against plan/step so a benchmark never overlaps
inference on the 4 GB card):

```
/benchmark                                  all roles from .agent
/benchmark --quick
/benchmark --roles validator --repeat 3
/benchmark --compare planner=erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest
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
