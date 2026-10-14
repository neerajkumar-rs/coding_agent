# CHANGELOG

## v5 — classify-then-act + history/debug export (2026-10-14)

All changes offline-verified: **336 tests pass** (`pytest tests/ test_master_controller.py`).
Live verification on this machine (RTX 3050 4GB, Ollama 0.40.3) — see below.

### Classify-then-act (replaces the one-call Understand step)

The single Understand call (intent + tasks + assumptions + reply + question in
one JSON schema) was too big for a 3B model — observed in real sessions:
"list all the files" → "I couldn't understand that", "bro list ... size in kb"
→ a useless rephrase question, "make list of file ... use cli tools" → a
shell-command tutorial. Classification itself is easy; the call was too big.
Split it:

- **`classify_mode`** — ONE tiny call, `format` schema `{"mode": enum}`,
  temperature 0, `num_predict <= 12`, instructions in the user turn. Output is
  always reply|code|both (label regex + JSON extraction; junk → safe `reply`).
  There is no "couldn't understand" path. Infra errors (missing model, VRAM
  guard, timeout) propagate to the structured error envelope — never a rephrase
  message.
- **`reply_step`** — ONE call over a Python-built FACTS block: file table with
  KB sizes computed in Python (never by the model), extensionless-file flags,
  last-run trace, last_result (1500 chars), last 3 turns. Instructions: answer
  from the facts only; never explain shell commands; not in the facts → say so
  in one sentence. Empty workspace → one plain sentence. No question field.
- **`plan_tasks`** — ONE call, small schema `{tasks, assumptions}` (no question
  field; missing details become listed assumptions). Post-validated in code
  (`_valid_plan_task`); one retry with errors appended → plain JSON →
  `PlanFailed` → `needs_human` with the reason (no files, no loop).
- **Fast paths** (slash commands, whole-message greetings/thanks/acks/openers
  English + Hinglish, word-boundary regexes) stay zero-model, zero-files.
- `both` runs reply first (printed), then plan. `uses_last_result` attaches the
  previous turn's result to the coder prompt. Pending questions merge into the
  next turn; never two questions in a row.
- **Trace/UI**: only stages that actually ran appear (`CLASSIFY ok 0.4s
  (reply) | ANSWER ok 2.1s`); never a synthetic SKIPPED row; red only for
  failed stages. Sidebar LAST RUN uses the same rule.
- **Markers**: `CLASSIFY MODE:` / `PLAN TASKS JSON:` (never collide with other
  pipeline markers — tested).
- `ALLOWED_TARGET_SUFFIXES` widened to `{.py,.html,.css,.js,.json,.md,.txt,.rs}`
  so "store this table in dir_table.md" no longer crashes `safe_target`.
- `run_pipeline` honors `roles.understand.model` for the understand role
  (was hardcoded to the coder model).

### History + one-command debug export

- **`history.py`** — append-only `.agent_runs/history.jsonl`, one entry per
  turn (fast paths, replies, code runs, needs_human, cancelled, errors;
  state-changing slash commands; pure display commands excluded): `id`
  (monotonic, user-visible), `ts` (ISO), `text` (4 KB cap), `path`, `label`,
  `run_id`, `models`, `trace` (stages + model + seconds), `llm_calls`,
  `seconds`, `outcome`, `error {stage, detail}`, `files_planned`,
  `files_staged`, `answer_preview` (500 chars), `state_snapshot`. Atomic
  appends under a per-workspace lock (Textual workers + controller both write);
  corrupt/truncated lines skipped + counted; rotation past 20 MB / 1000
  entries (readers span rotated files); a write failure never raises into the
  user flow. Every turn — including a pipeline exception (traceback lands in
  events.jsonl) — gets an entry with a `run_id` whose folder has at least
  state.json + events.jsonl.
- **`export.py`** — pure functions on paths: `do_export(workspace, target,
  --full/--zip/--fixture/--no-redact)`. Target: `last` (default), an id, N
  (last N), `a-b`, `all`. Outputs only under `.agent_runs/exports/`,
  timestamped, never overwrite. Report layout: Summary table + fact-derived
  "Likely cause" (missing model + pull command, VRAM guard, truncation,
  validation reason, exception stage), Environment (build_id, OS, Python,
  Ollama version/reachability, `/api/ps` snapshot, nvidia-smi GPU, per-role
  model+ctx+installed), per-turn (user text, collapsed state snapshot, trace
  table, per-call role/model/tokens/tok/s/done_reason with prompt+response
  fenced at 4 KB pointing at the full file, validation errors, plan/files,
  answer, error + traceback), events tail (40), Reproduce (`/retry <id>`).
  Report capped at 200 KB keeping Summary + Environment + the failing turn.
  `--zip` adds history-slice + linked run folders + config-snapshot +
  environment.json. `--fixture` appends eval-format lines (`text`, `state`,
  `expected`, `observed_ok: null`) for paste-into-`classify_eval.jsonl`.
  Redaction on by default (workspace/home/user paths → `<WORKSPACE>`/`~`,
  sk-/ghp_/AKIA/Bearer/key=token=secret=password assignments, private-key
  blocks, .env-style KEY=value) across the report, slice, zip logs, fixtures;
  `--no-redact` skips it with a printed warning.
- **Commands** (both TUIs via shared functions): `/history [N]|<id>|search
  <text>` (failed turns in red), `/retry <id>` (re-sends the text as a NEW
  turn, says old state is not restored), `/export ...` (prints path + size,
  copies the report to the clipboard when available — never fails on clipboard
  errors). Up/Down in the input bar recalls previous inputs (shell-style, per
  workspace, persisted from history; survives restart); Esc clears.
- **Endpoints**: `GET /agent/history?limit=&before=` (paging) and
  `GET /agent/export?target=&format=md|zip|fixture`.
- **CLI**: `python tui.py --workspace <dir> --export last|<id>|all [--zip]
  [--fixture] [--full]` works with Ollama/FastAPI down; exit 1 + a clear
  message when there is no history.
- Grep tests: no retired log-dir references; history/export make no
  gateway/Ollama calls; exports never land outside `.agent_runs/exports/`.

### Verified live on this machine (2026-10-14)

- **Classifier A/B** (`--classify-ab`, eval set held back from the prompt):
  coder 83.9% accuracy (p50 0.50s, p95 0.54s, 0 shell-command leaks on reply
  lines); thinker 55.4% (truncation errors at the classify budget — a
  reasoning model cannot answer in `num_predict<=12`). **Coder stays the
  understand default** (accuracy first, latency second). The 95% gate is NOT
  met by either model on this adversarial set — the 3B ceiling on held-back
  phrases; the failure modes that mattered (rephrase requests, shell
  tutorials, "couldn't understand", bare 500s) are all gone.
- **Acceptance session** (7 messages, live controller): greeting → reply
  grounded; "list all the files" → reply with the real table ("The workspace
  is empty"); "what is your readme" → reply grounded; "make some random file
  with some random data" → code, `random_data.txt` staged; "format this data
  ... dir_table.md" → code, `dir_table.md` staged. One known miss: "bro list
  ... size in kb" classified `both` (3B limit) — it answered the listing first,
  then failed safely through the verifier (nothing staged).
- **History/export acceptance**: 5-turn session (greeting, listing, vague code,
  cancelled code, forced failure) — `/history` shows all five with correct
  path/outcome; `/export all --zip` writes report + slice + run folders +
  config + environment; redaction masks a planted `sk-test...` key and the
  workspace path; `--no-redact` shows them with the warning; fixture lines load
  with the eval loader; CLI export works with the stack down; `ollama ps`
  during a code run shows exactly one resident model at 100% GPU.

### Not verified here

- Whether the thinker accepts `think:false` *semantically* (probed live
  earlier: the switch is accepted but ignored — the budget always includes
  `think_budget` for this model).
- `format` enum decoding internals per model (used by the classifier; behavior
  verified, Ollama internals not).
- Per-model classifier latency beyond the A/B numbers above.
- Clipboard behavior per terminal (Textual `copy_to_clipboard` availability
  varies; the command never fails on it — prints the path instead).
- nvidia-smi availability off this machine (the Environment probe degrades to
  "n/a").

## v3 implementation (Phases 0–3 consolidation + backlog 1–6)

All changes offline-verified: **292 tests pass** (`pytest tests/ test_master_controller.py`),
26 test files. Live Ollama verification done on this machine (see below).

## Bench rewrite (2026-10-13, item 7)

- **Quality-scored probes**: deterministic per-role checks on the actual
  answer — planner (JSON task array + files + spec), coder (fenced `def
  greet(` returning a string), validator (PASS/CONCERNS first line), writer
  (clean prose, no think tags). A role is "ok" only at quality 100.
- **Score = quality × tok/s** (tok/s from Ollama `eval_count/eval_duration`,
  chars/s fallback). A fast wrong answer can never beat a slower right one.
- New plumbing: `AgentGateway.generate(..., out_meta=[])` hands the Ollama
  meta dict to callers that need it (bench, future tooling) without the
  per-run journal hook.
- `--roles` and `--compare` accept the writer role; reports (JSON/MD/console)
  carry quality, tok/s, per-check detail.
- Live run on this machine: planner/validator quality 100 @ 97.7 tok/s,
  coder quality 100 @ 51.4 tok/s.
- The repo's own `.agent` profile was still pre-rename legacy keys; rewritten
  to current role defaults.
- Dropped the vestigial `--plan-mode` flag (the two-stage planner concept
  died with the retired Engine pipeline).

## Backlog round (2026-10-13)

- **Textual hygiene grep test** — AST scan of `tui_textual.py`: no
  `console.print` / `Prompt.ask` / `input()` / `breakpoint()` inside the app.
- **`/models` template warning** — a role whose model answers `/api/show`
  with an empty template is flagged ("no chat template"); models that never
  answered (not installed) are not flagged.
- **`/logs run <id>`** — both TUIs select a run by exact id or unique prefix
  (`/logs <run> [channel] [N]`), with the searched path reported on no match.
- **Writer role (4th role)** — `roles.writer` (summary/docs prose; thinker
  model by default, warmer temperature, 1024 predict), its own
  `writer.log` channel, registry/`/config/role`-switchable. Zero extra VRAM
  until reconfigured. (v5: the writer is PARKED — inert, hidden from the
  status bar; the summary prose runs on the thinker.)
- **`/setup` machine calibration** — `calibrate.py` + `/setup` in both TUIs
  (y/N gated, never auto-runs): binary search over the ctx rungs per role via
  `AgentGateway.probe_ctx` (one tiny generation at an exact ctx, no
  step-down), atomic write to `calibration.json` after EACH role
  (cancel/crash keeps completed roles), machine fingerprint (TOTAL vram/ram +
  os + cpu) gating. `ceiling_for(role)` prefers measured ceilings when the
  fingerprint matches — no caller changes.
- **Live-found think fix** — the thinker accepts but ignores `think:false`;
  the probe now detects it semantically, and `generate()` only sends
  `think=False` when the probe confirmed a REAL switch (otherwise the budget
  boost applies). Four new gateway tests pin the behavior.

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

## Verified live on this machine (RTX 3050 4GB, Ollama 0.40.2, 2026-10-13)

- Both reference models installed with the exact expected digests
  (`0a7c62f3bef4`, `f72c60cabf62`).
- **`think:false` is accepted by Ollama but IGNORED by the erwan2 R1
  distill** — it emits `<think>` regardless. The capability probe now checks
  the OUTPUT for a think tag (not just the HTTP status) and records
  `supports_think_param=false, think_ignored=true`; budgets always include
  `think_budget` for this model. (A trivial prompt spent 108–311 think tokens;
  `think_budget=512` is right-sized.)
- Live generates through the gateway: exactly one resident model at every
  `/api/ps` sample, clean thinker→coder swap, `size_vram/size = 1.000`.
- **Measured ctx rungs (0.97 guard):** thinker passes 8192/12288/16384;
  coder passes 6144/8192 and FAILS 12288 (0.833). The static coder ceiling
  (12288) is too optimistic on this card — run `/setup` (or
  `python calibrate.py --yes`) to write measured ceilings into
  `calibration.json`; `ceiling_for` then prefers them automatically.

## Still not verified here

- How Ollama `format=` constrained decoding interacts with the thinker's
  `<think>` output (fallback path exists; interaction unmeasured live).
- `scripts/live_smoke.py` end-to-end against the real server.
- nvidia-smi watchdog readings during a loaded run (GPU busy mid-run).
