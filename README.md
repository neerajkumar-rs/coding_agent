# Local LLM Master Controller — Ollama Native

A high-performance, GPU-constrained agentic coding pipeline designed for a 4 GB RTX 3050.  
All LLM inference is routed through a local [Ollama](https://ollama.ai) instance at `http://localhost:11434`.  
A pure-Python FastAPI supervisor (0 MB VRAM) orchestrates quantized models **sequentially** — never simultaneously — to enforce the VRAM ceiling.

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Architecture

```
FastAPI Master Controller (main.py)   ←  0 MB VRAM deterministic supervisor
        │
        │  asyncio.Lock  (one model at a time)
        │
        ▼
Local Ollama  http://localhost:11434
        │
        ├── deepseek-r1:1.5b          ←  Planner / Auditor  (THINKER)
        └── qwen2.5-coder:3b          ←  Code generator     (CODER)
                │
                ▼
        .agent_staging/               ←  Isolated staging cache (never touches production)
        .agent_logs/                  ←  Discrete per-role session logs
                │
                ▼  POST /agent/deploy
        Active Workspace              ←  Atomic deployment after validation
```

---

## Models

| Role | Model | When used |
|---|---|---|
| Planner / Auditor | `deepseek-r1:1.5b` | `/agent/plan`, LLM validation loop |
| Code Generator | `qwen2.5-coder:3b` (or `1.5b`) | `/agent/step` |

Pull both models once:

```powershell
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5-coder:3b
```

---

## Quick Start

### 1. Prerequisites

- Ollama running locally: `ollama serve` (default port 11434)
- Python virtual environment activated: `.\env\Scripts\activate`

### 2. Initialize the `.agent` profile

The profile is written to your workspace root as `.agent` (JSON). It controls `num_ctx`, model names, and is read by every Ollama API payload.

```powershell
# Auto-benchmark hardware and write profile (RTX 3050 → 4096 tokens)
.\env\Scripts\python.exe main.py --recalculate

# Or set manual bounds
.\env\Scripts\python.exe main.py --recalculate --max-context-tokens 8192 --coder-model qwen2.5-coder:1.5b
```

Profile written to `e:\coding_agent\.agent`:

```json
{
  "planner_model": "deepseek-r1:1.5b",
  "coder.5b_or_3b": "qwen2.5-coder:3b",
  "max_context_tokens": 4096
}
```

Every Ollama payload enforces these options:

```json
{
  "keep_alive": "10m",
  "options": {
    "num_ctx": 4096,
    "flash_attention": true,
    "keep_alive": "10m"
  }
}
```

### 3. Start the FastAPI Master Controller

```powershell
.\env\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8000
```

### 4. Launch the TUI (optional)

The Rich TUI starts FastAPI automatically and connects to the Ollama endpoint:

```powershell
.\env\Scripts\python.exe tui.py
```

---

## API Reference

All endpoints accept and return JSON.

### `POST /workspace/init`

Initialize or parse `.agent` configuration, create `.agent_staging/` and `.agent_logs/`.

```json
{ "root_path": "E:/my_project", "coder_model": "qwen2.5-coder:3b" }
```

### `POST /agent/plan`

Query the Planner model (`deepseek-r1:1.5b`). Returns a strict task queue. Raw `<think>` reasoning is logged to `.agent_logs/planner.log`.

```json
{ "prompt": "Add a FastAPI health endpoint to app.py" }
```

Response schema:

```json
{
  "tasks": [
    { "id": 1, "file": "app.py", "action": "edit", "description": "Add GET /health returning {'status': 'ok'}" }
  ],
  "model_used": "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
}
```

### `POST /agent/step`

Execute one task from the queue. The Coder model streams a patch into `.agent_staging/`. Then a native OS validation script runs in a subprocess boundary.

- **Windows**: Generates a PowerShell `.ps1` and runs it via `pwsh.exe`.
- **Linux/macOS**: Generates a Bash `.sh` and runs it.
- `stdout` / `stderr` are trapped and returned for the evaluation loop.
- Logged to `.agent_logs/coder.log` (code) and `.agent_logs/sandbox.log` (output).

```json
{ "task_id": 1 }
```

### `POST /workspace/status`

Computes `difflib.unified_diff` between `.agent_staging/` and the active workspace. Returns a line-by-line git-style diff object for TUI/GUI change previews.

### `POST /agent/deploy`

Atomic deployment: maps files from `.agent_staging/` into active workspace paths, then flushes the staging cache. **Only triggered explicitly** — active workspace is never modified during active pipeline cycles.

---

## Project Structure

```
e:\coding_agent\
│
├── main.py               ← FastAPI Master Controller (5 endpoints)
├── agent_config.py       ← Dynamic System Engine: .agent profile + --recalculate CLI
├── ollama_client.py      ← Async Ollama supervisor with asyncio.Lock VRAM fencing
├── workspace_staging.py  ← Deferred staging cache + difflib status + atomic deploy
├── sandbox.py            ← OS-aware .ps1/.sh validation script generator & executor
├── scanner.py            ← Workspace directory scanner + prompt context injection
├── orchestrator.py       ← Four-role retry pipeline (Thinker→Coder→Writer→Validator)
├── tui.py                ← Rich CLI: starts FastAPI, streams plan→step→deploy cycle
│
├── .agent                ← Persistent profile (auto-created, git-ignored)
├── .agent_staging/       ← Isolated code generation cache (git-ignored)
├── .agent_logs/          ← Discrete operational logs (git-ignored)
│   ├── planner.log       ← Raw reasoning + task queue arrays
│   ├── coder.log         ← Generated source code blocks
│   └── sandbox.log       ← Subprocess stdout/stderr + exit codes
│
├── test_master_controller.py  ← Full lifecycle test suite (no live Ollama needed)
├── e2e_smoke_test.py          ← End-to-end test with mock Ollama server
├── test_live_ollama.py        ← Live Ollama connection verification
└── requirements.txt
```

---

## Testing

### Offline tests (mock Ollama server — no GPU required)

```powershell
# Unit + integration: staging, difflib, sandbox, FastAPI endpoints
.\env\Scripts\python.exe test_master_controller.py

# End-to-end: full Thinker→Coder→Write→Compile pipeline
.\env\Scripts\python.exe e2e_smoke_test.py
```

### Live Ollama verification

```powershell
.\env\Scripts\python.exe test_live_ollama.py
```

---

## VRAM Safety Model

The RTX 3050's 4 GB hard ceiling is enforced at three levels:

1. **Sequential execution** — `asyncio.Lock` in `OllamaSupervisorClient` ensures only one model inference is in-flight at any time. Thinker and Coder never run in parallel.
2. **Strict `keep_alive: 10m`** — Models stay resident in VRAM between back-to-back calls to eliminate 0ms reload lag on rapid step execution.
3. **`num_ctx` cap** — Set to 4096 tokens for 3B models (leaves ~1.8 GB headroom for activations). Configurable via `--recalculate` for 1.5B models (8192 tokens).

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `VIBE_API_URL` | `http://127.0.0.1:8000` | FastAPI controller URL (TUI) |
| `VIBE_VALIDATION_TIMEOUT_SECONDS` | `300` | Sandbox subprocess timeout |
| `PORT` | `8000` | FastAPI listen port |
| `HOST` | `0.0.0.0` | FastAPI listen host |
