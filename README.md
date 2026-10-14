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
        │                            events.jsonl, thinker/coder/validator/sandbox.log}
        │                            + LOG_CHANNELS registry (the plug-and-play
        │                            log channel map — adding a channel is 1 line)
        ├── workspace_staging.py     .agent_staging/ → difflib → atomic deploy
        └── sandbox.py               in-process plausibility CHECK (non-Python)
        │
        ▼  ONE process-wide lock (AgentGateway._lock — the supervisor's
           _vram_mutex is the same object, injected by the gateway)
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

Runs the single pipeline `orchestrator.run_pipeline`: fast paths (zero model)
→ CLASSIFY → VERIFY → PLAN → CODE → CHECK → VALIDATE → SUMMARY. CLASSIFY is
one tiny schema-constrained call (no previous-turn text — a "Last turn" line
biased labels toward `code`). A `code|both` label is re-checked by VERIFY
(one `yes|no` call: "create or change a file?"); `no` downgrades to `reply`
before anything is planned. A plan may legitimately return `tasks: []` +
`reason` (→ reply fallback), and a plan whose only task is an unrelated
invented file (no web/page wording in the request) is rejected in code —
retried once, then `reply`. Replies are grounded in a Python-built FACTS
block: the file table plus the CONTENT of any file the message names (cap 3
files, 6000 chars, read-only), with a Python listing skeleton for list/show
asks (a contradicting model reply is replaced by the table). Stages
CHECK-passed files (never auto-deploys). Response carries `summary`,
`staged_files`, `failed_tasks`, `trace`, `run_id` for code tasks, or
`answer`, `trace`, `run_id` for questions, plus `status` on cancel/budget
stop.

Every LLM call is recorded in the run journal
(`.agent_runs/<run_id>/{thinker,coder,validator}.log` + `sandbox.log` for
CHECK output) with prompt, answer, `think_chars`, `done_reason`, `eval_count`,
tok/s, the num_ctx actually used, and `size_vram/size`; `state.json` carries
the intent, per-role cards (model, digest12, supports_think_param), swaps with
seconds, and per-call num_ctx.

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

### `GET /agent/pending_confirm` · `POST /agent/confirm`

The plan-confirm gate: when a run is started with `"confirm": true`, it pauses
after PLAN; the client polls `/agent/pending_confirm` and answers via
`/agent/confirm` (`{"approved": true|false}`). 180 s timeout declines.

### `POST /router/classify`

`/router <text>` backend: tier, matched verb/deliverable, scores, whether the
T2 grey-zone classifier ran, and the raw classifier JSON. T0/T1 rules never
touch a model; T2 goes through the same gateway lock.

### `GET /config` · `POST /config/role`

The live per-role config (thinker/coder/validator/writer: model, num_ctx,
num_predict, temperature, keep_alive, inline_system, think_budget) and the
validated update path. Changes persist to `.agent` and apply on the NEXT
gateway call, never mid-generation. num_ctx is clamped through
`ceiling_for(role)`.

### Machine calibration — `/setup` or `python calibrate.py`

Probes each role model at the context rungs against the 0.97 GPU guard and
writes measured ceilings to `calibration.json` (repo root, git-ignored,
fingerprint-gated to this machine). y/N gated — never auto-runs; atomic write
after each role so a cancel keeps completed roles. `ceiling_for` prefers
measured values immediately.

---

## Project Structure

```
e:\coding_agent\
├── agent_core/            ← the LIBRARY run_pipeline is built from
│   ├── engine.py          ← budgets + the per-workspace cancel registry
│   │                          (the Engine.run pipeline is retired)
│   ├── schemas.py         ← PipelinePlan / ValidateResult / AuditResult
│   ├── coder.py           ← full blocks + SEARCH/REPLACE edits, output parsing
│   ├── verifier.py        ← py_compile → ruff → import smoke → pytest
│   ├── staging.py         ← staging + path safety + backups/rollback
│   ├── run_state.py       ← .agent_runs journal + LOG_CHANNELS registry
│   ├── context.py         ← AST repo map + token budgeter
│   └── planner_eval.py    ← planner schema-conformance benchmark (dev tool)
│
├── main.py                ← FastAPI controller (all endpoints)
├── orchestrator.py        ← run_pipeline: THE single pipeline behind /agent/run
├── router.py              ← router v2: T0/T1 rules + T2 grey-zone classifier
├── agent_config.py        ← .agent profile + RoleRegistry + ceiling_for
├── ollama_client.py       ← AgentGateway (THE lock + VRAM guard + capability
│                              probe) + think-stripping/truncation guards
├── workspace_staging.py   ← staging manager + difflib + atomic deploy
├── sandbox.py             ← in-process plausibility CHECK + allowlisted commands
├── scanner.py             ← workspace scanner (file questions)
├── tui.py                 ← Rich TUI (reuses running Ollama)
├── tui_textual.py         ← Textual TUI (two-column, sidebar, save prompt)
│
├── tests/                 ← pytest suite (offline) + classify_eval.jsonl and
│                              classify_holdout.jsonl (held-back label sets —
│                              never used as CLASSIFY prompt examples)
├── test_master_controller.py  ← legacy lifecycle suite (offline, via /agent/run)
├── e2e_smoke_test.py          ← end-to-end with mock Ollama server
└── scripts/live_smoke.py      ← live Ollama connection check (not pytest)
```

Workspace-side artifacts (all git-ignored):

```
.agent             profile        .agent_runs/<id>/ state.json · events.jsonl · role logs · backup/
.agent_staging/    code cache     (one run dir per run: thinker/coder/validator/sandbox logs)
```

The workspace root itself only ever receives the **requested target files**
(after deploy). No `tasks.md`/`summary.txt`/`taskN_code.txt`/stub files.

---

## Testing

```powershell
# Unit tests (292 incl. the root lifecycle suite, offline)
.\env\Scripts\python.exe -m pytest tests/ test_master_controller.py -q

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
   default); prompt inputs capped at 85% of the window minus the output
   reserve. **One num_ctx per model**: roles resolving to the same model name
   send the same `num_ctx` on every call
   (`effective_ctx_for_model = min(ceiling, max(role num_ctx))`), so a context
   change never forces a hidden reload between phases; the journal flags
   `reload_suspected` on a call event when `size_total` changed.
5. **nvidia-smi watchdog** — logs `memory.used` around each load; warns when
   free VRAM < 300 MB; silent when nvidia-smi is absent. On Windows it prints
   once: set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback".

---

## Benchmarking

Quality-scored role benchmark (planner / coder / validator / writer). Every
probe answer passes deterministic per-role checks — planner: JSON task array
with files+spec; coder: fenced `def greet(` returning a string; validator:
PASS/CONCERNS verdict line; writer: clean prose, no think tags. A role is
"ok" only at quality 100. **Score = quality × tok/s** (tok/s from Ollama's
eval counts), so a fast wrong answer never beats a slower right one.
Reports land in `<workspace>\.agent_runs\bench\` (JSON + Markdown).

```powershell
# Headless from the shell (starts/reuses Ollama, NO FastAPI):
.\env\Scripts\python.exe tui.py --benchmark --quick
.\env\Scripts\python.exe tui.py --workspace E:\proj --benchmark --quick --coexist
.\env\Scripts\python.exe tui.py --benchmark --compare validator=erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest --repeat 3

# Or directly:
.\env\Scripts\python.exe bench.py --workspace . --quick
```

Inside the TUI (GPU-locked against runs so a benchmark never overlaps
inference on the 4 GB card):

```
/benchmark                                  all roles from .agent
/benchmark --quick
/benchmark --roles writer --repeat 3
/benchmark --compare coder=qwen2.5-coder:3b-instruct,qwen2.5-coder:1.5b
/benchmark --coexist
/benchmark last                             print the newest report
/benchmark apply                            write recommended models into
                                            .agent "roles" (asks y/N first)
```

Missing models are skipped with an `ollama pull <name>` hint. All probes go
through the single AgentGateway (one lock + VRAM guard); models are unloaded
when the benchmark finishes. Ctrl-C cancels the benchmark and returns to the
prompt without touching services.

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
