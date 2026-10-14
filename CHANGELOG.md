# CHANGELOG

## v6 — classifier accuracy, reply evidence, one-ctx-per-model, honest summaries (2026-10-14)

Source of truth: a real 4-turn debug export from the previous build (labels
wrong, replies ungrounded, hidden reloads, a dead validator, dishonest
summaries). Offline: **378 tests pass** (`pytest tests/ test_master_controller.py`).
Live numbers measured on this machine (RTX 3050 4GB, Ollama 0.40.3) below.

### 1. Classifier accuracy (side effects verified)

- **CLASSIFY prompt rewritten** (861 chars incl. examples, target <900): the
  `Last turn: ...` line is REMOVED (a previous "Built: index.html" result
  biased labels toward `code`); it survives only as `LEGACY_LAST_TURN_LINE`,
  used solely by bench's A/B arm. The prompt now says "If unsure, answer
  reply" and carries generic examples covering messy phrasing and tool words
  (never the holdout phrases).
- **`verify_side_effect`** — after a `code|both` label, ONE more tiny call
  (`VERIFY SIDE EFFECT:` marker, enum `yes|no`, temp 0, `num_predict<=6`, no
  thinking) confirms the user really asked to create/change a file; `no`
  downgrades to `reply` BEFORE anything is planned or written. Both calls are
  journaled (`VERIFY` trace stage + `verify_downgrade` event). Unparseable
  output keeps the classifier's label (constrained decoding makes junk
  practically impossible; dropping a real create on transport junk is worse).
- **Plan step**: may return `{"tasks": [], "reason": ...}` when no file the
  user asked for can be named or inferred → the pipeline falls back to the
  reply step (no files, no needs_human). PLAN prompt: "Never invent a file the
  user did not ask for (no index.html unless the request is a web page)."
  `invented_file_rejection` (plan-level): a plan whose ONLY task is an
  unrelated invented file (name absent from the request, no web/page wording)
  is rejected in code, retried once, then reply.
- **Confirm gate**: shows file names, the assumptions, and "(overwrites
  existing file)" per path that already exists (both TUIs render the marker).
- **Hold-out eval** `tests/classify_holdout.jsonl` (71 lines; the four
  evidence messages, Hinglish, typos, tool-phrased asks, greetings) with
  per-line `files_created`; tests pin size, labels, the evidence messages,
  no-overlap with the prompt, and fast-path 100%.
- **Live A/B** (`bench.py --classify-ab`): coder vs thinker, each with and
  without the retired Last-turn line, over the holdout WITH the verify pass
  and the production plan validators. Metrics: label accuracy after verify,
  false-`code` rate, `overcreate` (files planned on a reply line — the real
  harm), verify downgrades, classify p50/p95.

### 2. Replies need evidence (attach, do not guess)

- `build_reply_facts(workspace, message, facts)`: the Python file table PLUS
  the CONTENT of any workspace file the message names (case-insensitive exact
  name or relative path; extensionless names like readme/license/dockerfile
  map by stem) — cap 3 files, 6000 chars each, read-only via
  `safe_read_target` (new read-side variant of `safe_target`: containment +
  protection, no write-suffix gate), binary/large files skipped with a note.
  Named-but-absent files get a `NOT-IN-WORKSPACE: <name>` marker the reply
  prompt maps to "<name> is not in this workspace". The content sits BEFORE
  the conversation tail (small models weigh the prompt's end heaviest).
- `build_file_facts` also returns `all_files` (every relative path, binaries
  included) so "named but binary" is never misread as "absent".
- **Listing asks** (`is_listing_request`): the answer skeleton is built in
  Python from the table (`build_listing_skeleton` — names/types/sizes); the
  model may add one short sentence; `reply_contradicts_table` detects a file
  name absent from the table and the Python table REPLACES the reply. An
  empty workspace answers one deterministic sentence with zero model calls.
- Reply prompt: ignore CLI tool words; quote/summarize FILE CONTENT (with a
  concrete good-answer example — a 3B coder model needs the anchor); never
  explain how to run ls/cat/dir.

### 3. Hidden model reload (ctx mismatch)

- **One num_ctx per model**: `effective_ctx_for_model(config, model) =
  min(ceiling, max(num_ctx of roles using that model))` — the gateway starts
  every role at the model's effective rung. The legacy defect (coder sent at
  4096 while understand used 6144 on the SAME model → a reload every
  classify→code transition) is gone; the role defaults stand at thinker 8192 /
  coder 6144 / validator follows thinker, and the legacy `max_context_tokens`
  key is ignored with a one-line note (already the case; now pinned by tests).
- The journal flags `reload_suspected: true` on an `llm_call` event when
  `size_total` changed versus the previous call to the same model; the export
  surfaces it as a "Likely cause" line and on the per-call record.
- Offline fake-Ollama test (`tests/test_same_ctx.py`): a full code turn
  (classify, verify, plan, code, validate, summary) loads each model EXACTLY
  once and every coder call carries num_ctx 6144 on the wire, even with the
  defective 4096/6144 config pair. `e2e_smoke_test.py` rewritten onto the
  current pipeline API with the same one-load assertion.

### 4. Dead validator + honest summaries

- Exact reference names only; `deepseek-r1:1.5b` survives ONLY in
  `main.RETIRED_MODEL_NAMES` (the detection registry) and the migration tests
  that exercise it. Generic test fixtures no longer use it.
- **Migration root cause found + fixed**: the startup audit lived only in the
  Rich TUI — the Textual TUI never ran it, so a saved `deepseek-r1:1.5b`
  validator was never offered for migration. The audit is now factored into
  `tui.audit_configured_models` (detection) + `tui.apply_model_migration`
  (persist); the Rich TUI keeps its [y/N] prompt; the Textual TUI runs the
  audit at startup and asks through its inline y/N gate (the app's modal).
  Tests: migration y persists, N keeps + prints the missing-model table,
  Ollama-down stays silent, NO prompt when already correct, and a saved
  `roles.validator.model = deepseek-r1:1.5b` directly (not via legacy keys)
  is detected.
- Status bar + `/models` show the real configured names with digest12
  (already true; now covered by the audit tests).
- **Summary text**: when VALIDATE fails or is skipped, line one is
  `UNVERIFIED - validation could not run: <reason>. Fix: <command>` (the
  `ollama pull <name>` is lifted out of a ModelNotInstalled detail), and
  "Known issues: none" is NEVER printed in that case (the validation problem
  is the known issue). A clean run distinguishes "Checks passed
  (syntax/links); validated against your request" from a syntax-only pass —
  a CHECK pass is never reported as "verified".
- **History outcome vocabulary**: `ok | unverified | partial | needs_human |
  failed | cancelled | error`. `unverified` renders yellow in `/history` and
  the sidebar (never green). Trace rows for CODE/VALIDATE/SUMMARY show the
  model; CHECK shows `-` (deterministic).
- Before the confirm gate of a code run, one warning line when the validator
  model is unavailable (with the exact pull command; `/agent/run` computes it
  from the installed list and passes it down).

### 5. History/export gaps

- **`calls/<n>.json`** per LLM call in the run folder (full prompt, RAW
  response, think text separate, num_ctx, prompt/output tokens, tok/s,
  done_reason, size_vram/size, reload_suspected) — the export's "LLM calls"
  section finally has its data source. The gateway's on_call record now
  carries `raw_response`, `think_text`, `prompt_eval_count`.
- "Likely cause" from facts only: missing model (+pull command),
  reload_suspected, done_reason=length (incl. per-call events), validation
  failure reason, classifier/verify downgrade, stage exception with traceback.
- Environment: real `build_id` (computed by export itself when the caller
  doesn't pass one), every role with exact name + digest12 + installed yes/no
  + num_ctx configured vs effective + num_predict + temperature + keep_alive,
  `/api/ps` snapshot, GPU as `name, total MiB, used MiB, free MiB` labeled,
  timestamps in UTC AND local time with offset.
- **`state_snapshot` is the PRE-turn state** (captured right after
  ConversationState.load) — turn 2's snapshot no longer contains turn 2's own
  result. Fixtures depend on this; tested.
- **`llm_calls` counts real model calls only**: `RunJournal.llm_call_count`
  increments exactly once per `llm_call` event; state.json and history report
  it (the local counter stays the budget tick). Test: `llm_calls == number of
  llm_call events == number of calls/*.json records`.
- `/export --fixture` lines now match the holdout shape (`text`, `state`,
  `expected`, `files_created`, `observed_ok`, `note`) — paste-compatible with
  `tests/classify_holdout.jsonl`; tested.

### Live verification on this machine (2026-10-14)

- **Classifier A/B with verify + plan safety net** (holdout, 62 model lines):
  coder/clean **acc 91.9%, false_code 2, overcreate 0, downgrades 2,
  p50 0.45s / p95 0.48s**; coder/legacy acc 93.5% / overcreate 1;
  thinker/clean acc 56.5% / overcreate 4; thinker/legacy acc 50.0%.
  The thinker (R1 distill) cannot classify at `num_predict<=12` — it thinks
  past the budget; the coder stays the understand default.
  **Gate verdict: accuracy 91.9% misses the 95% target** (3B ceiling on an
  adversarial holdout), but **overcreate = 0 on reply lines** — the real harm
  metric — and p95 0.48s beats the 1.5s latency gate. The two label misses
  ("write code", a pasted Traceback) are caught downstream: the planner names
  no file (`tasks: []` / invented-file rejection) and nothing is created.
- **Acceptance session** (fresh workspace, only `index.html` with an
  `<h1>File List</h1>`, live controller + models, in order):
  `make readme.md` → **code**, verify confirmed, 5 calls, 6.0 s (first call
  2.17 s cold), readme.md staged; `let list all the file in this dir` →
  **reply**, the Python table with `index.html | html | 2 lines | 0.1 KB`,
  1.3 s; `ok use cli tool and list all file and and there types` → classify
  said `code` but the **verify pass downgraded to reply** (no side effect),
  the same Python table, 2.1 s; `what is in index.html use cat to see` →
  **reply**: "index.html contains the heading \"File List\"." (the file
  content reached the model and the answer quotes it), 1.0 s;
  `make a notes.md with my todo list` → **code**, verify confirmed,
  notes.md staged, 7 calls, 14.9 s. User-visible labels: **5/5**
  (code, reply, reply, reply, code). First-call latency per turn once a model
  is resident: 0.27–0.29 s (target <1.5 s ✓).
- `/models`: real configured names with digests `0a7c62f3bef4` /
  `f72c60cabf62`, all installed=yes. `/export last`: likely-cause line,
  per-call prompts + raw responses + think text, labeled environment
  (build_id, per-role digest/installed/configured-vs-effective ctx/keep_alive,
  `gpu: NVIDIA GeForce RTX 3050 A Laptop GPU, total 4094 MiB, used 1375 MiB,
  free 2719 MiB`), UTC + local timestamps, pre-turn snapshot. `ollama ps`
  after the session: exactly one resident model, `size_vram/size = 1.00`.

### Deviations (with reasons)

- **The written gate "0 false-code on reply lines" is gated on `overcreate`**
  (files actually planned on a reply line) instead of the raw label: a wrong
  `code` label that produces zero files is a latency cost, not harm. The
  label-level `false_code` is still measured and printed.
- **Verify junk default keeps the classifier's label** (documented in
  `verify_side_effect`): with the enum schema the yes/no output is
  constrained; erroring/downgrading on transport junk would break exactly the
  create requests the pass is meant to protect.
- **Textual migration prompt is the app's inline y/N gate**, not a separate
  ModalScreen class — the same pattern the plan-confirm gate already uses;
  a Rich prompt cannot run inside the fullscreen app.

### Not verified here

- Whether the verify pass lowers false-`code` on the THINKER as well as the
  coder (the thinker truncates at the classify/verify budgets — its arms
  measure mostly truncation noise).
- Whether the thinker accepts `think: false` semantically (earlier probe: the
  switch is accepted but ignored — `think_ignored=true`).
- Measured classify latency under the same-ctx rule beyond the numbers above
  (the fake-Ollama test proves ONE load per model per turn; the resident
  first-call latency target (<1.5 s) is confirmed at p95 0.48s).

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
