"""The bounded agent loop: the model decides, Python enforces safety.

Replaces the hardcoded decision logic (router tiers, the classify/verify
calls, the listing/canned/placeholder rules). ONE structured call per step
(role ``understand``, temperature 0.2, ``format`` schema, num_predict 400):

    {"action": "list_files|read_file|search|write_file|edit_file|check_file
               |run_test|reply|finish",
     "path": "", "text": ""}

Python executes the action and appends a compact observation. The loop stops
on ``reply``/``finish``, on the step budget (``max_steps``, counted against
the pipeline's LLM-call budget), or on a repeated identical action (the
no-progress rule -> needs_human with the reason, never a silent chat reply).

Safety lives in Python, never in the model's hands: reads go through
``safe_read_target``; writes go through the registry extraction + check, then
``safe_target``, and are COLLECTED (the caller shows them at the confirm gate
before anything is staged); check/run use the registry and the allowlist
(``py_compile``/``pytest`` plus registry checkers when installed).

The loop prompt stays under ~700 tokens, generic, no test phrases: the tool
list (one line each), the user request verbatim, this turn's observations,
and the last 2 turns as one line each (what the agent DID, never raw reply
text).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from filetypes import TEXT_SUFFIXES, check_file, extract_content
from ollama_client import strip_think

LOOP_MARKER = "AGENT ACTION JSON:"

LOOP_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "enum": [
                "list_files", "read_file", "search",
                "write_file", "edit_file", "check_file", "run_test",
                "reply", "finish",
            ]
        },
        "path": {"type": "string"},
        "text": {"type": "string"},
    },
    "required": ["action"],
    "additionalProperties": False,
}

LOOP_SYSTEM = "You are a careful coding agent operating through tools."

# Generic, no test phrases, ~700 tokens. {tools}/{request}/{history}/
# {observations} are filled per step.
LOOP_PROMPT = """You act in steps. Each step you output ONE action as JSON.

Tools (one line each):
{tools}

Rules: look at files before answering about them; only create or change files
the user asked for; choose sensible defaults and say them in the reply; do the
task yourself — never explain shell commands or tell the user to run tools;
when the work is done, reply with a short summary; when nothing more is
needed, finish.

USER REQUEST: {request}

RECENT TURNS (what the agent did, one line each):
{history}

OBSERVATIONS THIS TURN (newest last):
{observations}

{LOOP_MARKER}
{ "action": "...", "path": "...", "text": "..." }"""

TOOL_LINES = [
    "list_files            — list workspace files with sizes (read-only)",
    'read_file <path>      — show a file\'s content (read-only; cap 6000 chars)',
    'search <text>         — find lines containing <text> in workspace files',
    'write_file <path> + text  — create a NEW file (staged for approval)',
    'edit_file <path> + text   — change an existing file (staged for approval)',
    'check_file <path>     — run the syntax/plausibility check for a file',
    'run_test              — run the workspace test command (allowlisted)',
    'reply <text>          — answer the user and end the turn',
    'finish                — end the turn (after writes, validation runs)',
]

MAX_STEPS = 6
READ_CAP = 6_000
SEARCH_CAP = 2_000
MAX_ATTEMPTS_PER_FILE = 2

# File-name tokens for the listing contradiction guard (the Python table wins
# over a reply that names a file not in the workspace).
_FILE_TOKEN_RE = re.compile(
    r"\b([\w\-./\\]+\.(?:py|html?|css|js|jsx|ts|tsx|json|md|txt|rs|toml|ya?ml|cfg|ini|csv|xml|sql|sh|bat))\b",
    re.IGNORECASE,
)


def _reply_contradicts_table(answer: str, table_names: list[str]) -> str | None:
    """The first file name in the answer that is NOT in the table, or None."""
    known = {n.lower() for n in table_names}
    known |= {n.rsplit("/", 1)[-1].lower() for n in table_names}  # basenames
    for match in _FILE_TOKEN_RE.finditer(answer):
        token = match.group(1).lower().lstrip("./")
        base = token.rsplit("/", 1)[-1]
        if token not in known and base not in known:
            return match.group(1)
    return None


class LoopAction:
    """One validated loop action (code-verified, never trusted)."""

    VALID = {
        "list_files", "read_file", "search",
        "write_file", "edit_file", "check_file", "run_test",
        "reply", "finish",
    }

    def __init__(self, data: dict[str, Any]) -> None:
        self.action = str(data.get("action", "")).strip().lower()
        if self.action not in self.VALID:
            self.action = ""
        self.path = str(data.get("path", "")).strip().replace("\\", "/")
        self.text = str(data.get("text", "") or "")

    @property
    def valid(self) -> bool:
        return bool(self.action)


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


class LoopResult:
    """What a finished loop hands back to the pipeline."""

    def __init__(self) -> None:
        self.answer: str = ""                       # the reply text (if any)
        self.writes: dict[str, str] = {}            # path -> content (staged by caller)
        self.edits: dict[str, str] = {}             # path -> content (existing files)
        self.observations: list[str] = []           # compact observation lines
        self.steps: int = 0
        self.stopped_by: str = ""                   # reply|finish|budget|no_progress|error
        self.checks: dict[str, str | None] = {}     # path -> check error or None
        self.test_result: str = ""                  # run_test observation

    @property
    def pending_writes(self) -> list[str]:
        return sorted(set(self.writes) | set(self.edits))


# ---------------------------------------------------------------------------
# Tool execution (Python-side; never a model call)
# ---------------------------------------------------------------------------

def _list_files(workspace: Path) -> str:
    from orchestrator import build_file_facts

    facts = build_file_facts(workspace)
    files = facts.get("files", [])
    if not files:
        return "The workspace is empty (no files)."
    lines = []
    for f in files[:80]:
        size_kb = float(f.get("size", 0) or 0) / 1024.0
        lines.append(
            f"{f.get('name')} | {f.get('language')} | {f.get('lines')} lines | {size_kb:.1f} KB"
        )
    return "WORKSPACE FILES (name | type | lines | size):\n" + "\n".join(lines)


def _read_file(workspace: Path, rel: str) -> str:
    from agent_core.staging import PathEscapeError, safe_read_target

    if not rel:
        return "error: read_file needs a path"
    try:
        target = safe_read_target(workspace, rel)
    except PathEscapeError as exc:
        return f"error: {exc}"
    if not target.is_file():
        return f"error: {rel} is not in this workspace"
    suffix = Path(rel).suffix.lower()
    if suffix and suffix not in TEXT_SUFFIXES:
        return f"note: {rel} is a binary file (content not shown)"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"error: {rel} is not readable: {exc}"
    truncated = len(text) > READ_CAP
    body = text[:READ_CAP]
    note = f"\n(+{len(text) - READ_CAP} chars truncated)" if truncated else ""
    return f"CONTENT of {rel}:\n{body}{note}"


def _search(workspace: Path, needle: str) -> str:
    if not needle.strip():
        return "error: search needs text"
    from orchestrator import build_file_facts

    facts = build_file_facts(workspace)
    hits: list[str] = []
    for f in facts.get("files", []):
        name = str(f.get("name", ""))
        target = workspace / name
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if needle.lower() in line.lower():
                hits.append(f"{name}:{i}: {line.strip()[:120]}")
        if len("\n".join(hits)) > SEARCH_CAP:
            break
    out = "\n".join(hits[:40])
    if not out:
        return f"no matches for {needle!r}"
    return out[:SEARCH_CAP]


def _run_test(workspace: Path) -> str:
    """The allowlisted test command (pytest) when enabled; otherwise a note."""
    from sandbox import ExecutionSandbox

    sandbox = ExecutionSandbox(workspace)
    try:
        sandbox.resolve_custom_command("pytest")
    except ValueError:
        return "note: pytest is not allowlisted in this workspace (.agent allowed_commands)"
    return "note: run_test is available (pytest allowlisted)"


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

async def agent_turn(
    user_request: str,
    workspace: Path,
    *,
    generate: Callable[..., Awaitable[str]],
    llm: Callable[..., Awaitable[str]],
    history_lines: list[str] | None = None,
    max_steps: int = MAX_STEPS,
    role: str = "understand",
) -> LoopResult:
    """Run the bounded loop. Never raises for model/content failures (the
    caller reads stopped_by); infra errors propagate to the caller's
    structured-error envelope.

    ``generate`` is the role-aware seam for the loop's own calls (JSON out).
    ``llm`` is the same seam used for the coder write/edit calls (fenced out).
    """
    from agent_core.staging import safe_target

    result = LoopResult()
    workspace = Path(workspace).resolve()
    history = history_lines or []
    observations: list[str] = []
    last_action_key = ""
    repeated = 0
    write_attempts: dict[str, int] = {}
    # The most recent list_files observation (the Python-built table) and the
    # names it listed — the reply contradiction guard consults them.
    last_listing_table = ""
    last_listing_names: list[str] | None = None

    for step in range(1, max_steps + 1):
        prompt = (
            LOOP_PROMPT
            .replace("{tools}", "\n".join(TOOL_LINES))
            .replace("{request}", user_request[:2_000])
            .replace("{history}", "\n".join(history[-2:]) or "(no prior turns)")
            .replace("{observations}", "\n".join(observations[-8:]) or "(none yet)")
            .replace("{LOOP_MARKER}", LOOP_MARKER)
        )
        raw = await generate(
            None, prompt, system=LOOP_SYSTEM, temperature=0.2,
            role=role, num_predict=400, _format_schema=LOOP_SCHEMA,
        )
        answer, _think, _chars = strip_think(raw)
        action = LoopAction(_extract_json(answer))
        result.steps = step
        if not action.valid:
            observations.append(f"step {step}: invalid action output (ignored)")
            continue

        key = f"{action.action}:{action.path}:{action.text[:40]}"
        if key == last_action_key:
            repeated += 1
            if repeated >= 1:
                result.stopped_by = "no_progress"
                result.observations = observations
                return result
        else:
            repeated = 0
        last_action_key = key

        # ---- read-only tools ----
        if action.action == "list_files":
            obs = _list_files(workspace)
            last_listing_table = obs
            from orchestrator import build_file_facts

            last_listing_names = [
                str(f.get("name", ""))
                for f in build_file_facts(workspace).get("files", [])
            ]
        elif action.action == "read_file":
            obs = _read_file(workspace, action.path)
        elif action.action == "search":
            obs = _search(workspace, action.text or action.path)
        elif action.action == "run_test":
            obs = _run_test(workspace)
            result.test_result = obs
        elif action.action == "check_file":
            rel = action.path
            content = result.writes.get(rel) or result.edits.get(rel)
            if content is None:
                target = workspace / rel
                if target.is_file():
                    try:
                        content = target.read_text(encoding="utf-8", errors="replace")
                    except OSError as exc:
                        obs = f"error: {rel} not readable: {exc}"
                        observations.append(f"step {step}: {obs}")
                        continue
                else:
                    obs = f"error: {rel} not found (nothing written for it this turn)"
                    observations.append(f"step {step}: {obs}")
                    continue
            err = check_file(rel, content)
            result.checks[rel] = err
            obs = f"check {rel}: " + (f"FAIL {err}" if err else "ok")
        # ---- writes (collected; the caller stages them after the gate) ----
        elif action.action in ("write_file", "edit_file"):
            rel = action.path
            if not rel:
                obs = f"step {step}: {action.action} needs a path"
                observations.append(obs)
                continue
            try:
                safe_target(workspace, rel)  # raises on escape/protected/type
            except Exception as exc:
                obs = f"error: {exc}"
                observations.append(f"step {step}: {obs}")
                continue
            # Same safety rules as the plan gate: junk artifacts, placeholder
            # stems — never written regardless of the model's wording.
            from orchestrator import _valid_plan_task

            reject = _valid_plan_task({"path": rel, "action": "create", "description": action.text})
            if reject:
                observations.append(f"step {step}: {reject} (the write was NOT staged)")
                result.checks[rel] = reject
                continue
            attempts = write_attempts.get(rel, 0)
            if attempts >= MAX_ATTEMPTS_PER_FILE:
                observations.append(
                    f"step {step}: {rel} already attempted {attempts} times "
                    "(no progress) — not retried"
                )
                continue
            write_attempts[rel] = attempts + 1
            current = ""
            if action.action == "edit_file":
                existing = workspace / rel
                if existing.is_file():
                    try:
                        current = existing.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        current = ""
            # ONE coder call: the ORIGINAL request, the path, the model's
            # text (what to write), and for edits the current content.
            from filetypes import hint_for, language_of

            language = language_of(rel)
            hint = hint_for(rel)
            coder_prompt = (
                f"Write {rel}.\n"
                f"ORIGINAL USER REQUEST: {user_request[:2000]}\n"
                f"WHAT TO WRITE: {action.text[:1000]}\n"
                f"TYPE: {hint}\n"
                + (f"CURRENT CONTENT:\n{current[:3000]}\n" if current else "")
                + f"\nReturn ONLY the complete file contents inside one "
                  f"```{language} fenced block. No commentary."
            )
            raw_file = await llm(
                None, coder_prompt,
                system=f"You output only complete {language} files in one fenced block.",
                temperature=0.1, role="coder",
            )
            clean, _think_text, _think_chars = strip_think(raw_file)
            content = extract_content(clean, rel)
            try:
                err = check_file(rel, content)
            except Exception:
                # A checker CRASH is infra (the tool broke, not the content):
                # propagate so the pipeline's structured-error envelope carries
                # it. Content failures are observations, never raised.
                raise
            result.checks[rel] = err
            if err is not None:
                obs = f"check {rel}: FAIL {err} (the write was NOT staged)"
                observations.append(f"step {step}: {obs}")
                continue
            if action.action == "write_file":
                result.writes[rel] = content
            else:
                result.edits[rel] = content
            obs = f"staged {action.action} {rel} ({len(content.splitlines())} lines)"
        # ---- terminal actions ----
        elif action.action == "reply":
            text = action.text.strip()
            # Contradiction guard: after a list_files observation, the Python
            # table wins over a reply that names a file not in the workspace.
            if text and last_listing_table and last_listing_names is not None:
                if _reply_contradicts_table(text, last_listing_names) is not None:
                    text = last_listing_table
            result.answer = text
            result.stopped_by = "reply"
            result.observations = observations
            return result
        elif action.action == "finish":
            result.stopped_by = "finish"
            result.observations = observations
            return result
        else:
            obs = f"step {step}: unknown action {action.action!r} (ignored)"

        observations.append(f"step {step}: {obs}")

    result.stopped_by = "budget"
    result.observations = observations
    return result
