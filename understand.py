"""Classify-then-act routing: tiny classifier, then one call per label.

Replaces the single one-call "Understand" step (too big for a 3B model).
Three steps, each ONE gateway call at most:

1. ``classify_mode`` — ``CLASSIFY MODE:`` marker, ``format`` = JSON schema
   ``{"mode": {"enum": [...]}}``, temperature 0, ``num_predict`` <= 12.
   Output is always one of reply|code|both; there is no "couldn't
   understand" path. The prompt carries NO previous-turn text (a "Last turn"
   line biased labels toward `code`; it survives only as a bench A/B knob).
1b. ``verify_side_effect`` — after a code|both label, ONE more tiny call
   (``VERIFY SIDE EFFECT:`` marker, enum yes|no, ``num_predict`` <= 6)
   confirms the user really asked to create/change a file; ``no`` downgrades
   the turn to ``reply`` before anything is planned or written.
2. ``reply_step`` — answers from a Python-built FACTS block only (file
   table with KB sizes computed in Python, flags, last-run trace,
   last_result, last 3 turns). Never a question, never a rephrase request,
   never shell-command tutorials.
3. ``plan_tasks`` — ``PLAN TASKS JSON:`` marker, small schema
   ``{"tasks": [...], "assumptions": [...]}`` (no question field). Missing
   details become defaults listed in assumptions. Post-validated in code.

Fast paths (slash commands, whole-message greetings/thanks/acks/openers in
English + Hinglish) never touch a model and never create files.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from ollama_client import strip_think

CLASSIFY_MARKER = "CLASSIFY MODE:"
PLAN_MARKER = "PLAN TASKS JSON:"

# ---------------------------------------------------------------------------
# Fast paths (no model call, no files — ever)
# ---------------------------------------------------------------------------

_SLASH_RE = re.compile(r"^/\w")
_GREETING_RE = re.compile(
    r"^(?:"
    r"(?:h+i+|hello|hey+y?|howdy|good\s+(?:morning|afternoon|evening|night)|yo|sup|hiya)\b"
    r"(?:[\s,!.]+(?:there|everyone|all|friend|buddy|bro|sir|madam|ji))*\s*[!.]*"
    r"|(?:namaste|namaskar|vanakkam|salaam|adaab|sat\s*sri\s*akal)\b\s*[!.]*"
    r"|(?:shukriya|dhanyavad|dhanyawaad|thank\s*(?:you|u)(?:\s*so\s*much)?"
    r"|thanks(?:\s*a\s*lot)?(?:\s*so\s*much)?|thx)\b"
    r"(?:\s+(?:so\s+much|a\s+lot|for\s+[\w\s]+?))?\s*[!.]*"
    r"|(?:(?:ok(?:ay)?|theek\s*hai|accha|sahi|great|nice|cool|perfect|awesome)\b[\s,!.]*)+"
    r"|(?:(?:chalo|chalu|shuru)\s+)+(?:karte|karen|kar)?\s*(?:hain|hai|h)?\b\s*[!.]*"
    r")$",
    re.IGNORECASE,
)
_WORK_OPENER_RE = re.compile(
    r"^(?:h+i+|hello|hey+y?|namaste)\b"
    r"(?:[\s,!.]+(?:there|everyone|all|ji))*[\s,!.]*"
    r"(?:let'?s|let\s+us|chalo|aao)\s+"
    r"(?:get\s+started|start|begin|work\s+together|work|shuru\s+karte\s+hain)\b\s*[!.]*$",
    re.IGNORECASE,
)


def fast_path(text: str) -> str | None:
    """Return the fast-path kind for a message, or None.

    Fast paths: slash commands (COMMAND) and whole-message greetings/thanks/
    acks/work-openers (CHAT, English + Hinglish, word-boundary regexes).
    A fast-path message never creates files and never calls a model.
    """
    stripped = text.strip()
    if not stripped:
        return None
    if _SLASH_RE.match(stripped):
        return "COMMAND"
    if _GREETING_RE.match(stripped) or _WORK_OPENER_RE.match(stripped):
        return "CHAT"
    return None


# ---------------------------------------------------------------------------
# Step 1: the classifier (tiny, always valid)
# ---------------------------------------------------------------------------

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {"mode": {"enum": ["reply", "code", "both"]}},
    "required": ["mode"],
    "additionalProperties": False,
}

CLASSIFY_PROMPT = """Classify the message. One word: reply, code, or both.

code = asks to CREATE, CHANGE, FIX, ADD, UPDATE or SAVE a concrete file or
app, naming what to build (notes.md, app.py, "a flask app"). Polite forms
and Hinglish count. both = a question AND ALSO a file request.
reply = everything else: listing, showing, reading, counting or explaining
existing files, questions, errors, follow-ups, chat - even with tool words
("use cli", "with cat") or typos. Vague fragments with no clear artifact
("make somthing", "creat a pag") are reply.
If unsure, answer reply (reply never creates a file; a wrong code does).

Files in workspace: {files}

"show me the files here" -> reply
"tell me whats inside config.json" -> reply
"list what types of files are here" -> reply
"make a notes.md with my todo list" -> code
"write a readme" -> code

Message: "{text}"

{CLASSIFY_MARKER}"""

# The retired "Last turn: ..." line. Kept ONLY for bench.py's A/B arm -- the
# production prompt above never carries previous-turn result text, which
# biased the label toward `code`.
LEGACY_LAST_TURN_LINE = "Last turn: {last_turn}\n"

_LABEL_RE = re.compile(r"\b(reply|code|both)\b")


async def classify_mode(
    text: str,
    *,
    generate: Callable[..., Awaitable[str]],
    file_names: list[str],
    last_turn: str = "",
    role: str = "understand",
) -> str:
    """Classify into reply|code|both with ONE tiny call. Always valid.

    Constrained decoding (``format`` schema, temperature 0, num_predict<=12)
    plus a label regex over the think-stripped output; any non-label junk
    falls back to the first recognised label, default ``reply``.
    Infra errors (model missing, VRAM guard, timeout) propagate to the
    caller's structured-error envelope — never a rephrase message.

    ``last_turn`` is a bench-only A/B knob: when non-empty, the retired
    "Last turn: ..." line is inserted (measuring the old bias). Production
    callers never pass it — previous-turn context goes to reply/plan only.
    """
    prompt = (
        CLASSIFY_PROMPT
        .replace("{files}", ", ".join(file_names[:40]) or "(empty workspace)")
        .replace("{text}", text[:2_000])
        .replace("{CLASSIFY_MARKER}", CLASSIFY_MARKER)
    )
    if last_turn:
        prompt = prompt.replace(
            "\nMessage:", "\n" + LEGACY_LAST_TURN_LINE.replace("{last_turn}", last_turn) + "\nMessage:", 1
        )
    # Never send `think` here: the understand role defaults to the coder
    # model (an instruct model — think is meaningless), and the gateway
    # itself sends think=False for a reasoning model whose probe confirmed
    # support. This layer stays think-free.
    raw = await generate(
        None, prompt, system="", temperature=0.0,
        role=role, num_predict=32, _format_schema=CLASSIFY_SCHEMA,
    )
    answer, _think, _chars = strip_think(raw)
    match = _LABEL_RE.search(answer.lower())
    if match:
        return match.group(1)
    # The JSON-schema path returns {"mode": "..."} — extract via json too.
    try:
        data = json.loads(answer)
        mode = str(data.get("mode", "")).lower()
        if mode in ("reply", "code", "both"):
            return mode
    except (json.JSONDecodeError, AttributeError):
        pass
    return "reply"  # safe default: no files, a grounded answer


# ---------------------------------------------------------------------------
# Step 1b: side-effect verification (one tiny yes/no call before any file work)
# ---------------------------------------------------------------------------

VERIFY_MARKER = "VERIFY SIDE EFFECT:"

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"side_effect": {"enum": ["yes", "no"]}},
    "required": ["side_effect"],
    "additionalProperties": False,
}

VERIFY_PROMPT = """{VERIFY_MARKER} Does the user ask to create or change a file?

yes = make, create, write, add, fix, change, update, build or save a file or
app — even politely ("can you make...") or with typos.
no = only listing, showing, reading, counting or explaining files that already
exist — nothing is created or changed.

"add a goodbye function to hello.py" -> yes
"build a home page for my bakery" -> yes
"fix the crash in parser.py" -> yes
"list all the files here" -> no
"show me what index.html contains" -> no

Message: "{text}"
"""

_YES_NO_RE = re.compile(r"\b(yes|no)\b")


async def verify_side_effect(
    text: str,
    *,
    generate: Callable[..., Awaitable[str]],
    role: str = "understand",
) -> bool:
    """True only when the user really asks to create/change a file.

    Runs after a ``code``/``both`` classify, before ANY side effect: one tiny
    call (temperature 0, num_predict <= 6, enum schema, no thinking). A ``no``
    downgrades the turn to ``reply`` at the caller. Unparseable output keeps
    the classifier's label (constrained decoding makes junk practically
    impossible; dropping a real create request on transport junk is worse).
    Infra errors propagate like the classifier's.
    """
    prompt = (
        VERIFY_PROMPT
        .replace("{VERIFY_MARKER}", VERIFY_MARKER)
        .replace("{text}", text[:2_000])
    )
    raw = await generate(
        None, prompt, system="", temperature=0.0,
        role=role, num_predict=32, _format_schema=VERIFY_SCHEMA,
    )
    answer, _think, _chars = strip_think(raw)
    match = _YES_NO_RE.search(answer.lower())
    if match:
        return match.group(1) == "yes"
    try:
        data = json.loads(answer)
        side = str(data.get("side_effect", "")).lower()
        if side in ("yes", "no"):
            return side == "yes"
    except (json.JSONDecodeError, AttributeError):
        pass
    return True  # junk: keep the classifier's label (see docstring)


# ---------------------------------------------------------------------------
# Step 2a: the reply call (facts only, no questions, no tutorials)
# ---------------------------------------------------------------------------

REPLY_SYSTEM = (
    "You answer questions about the user's workspace from facts. Be short."
)

REPLY_PROMPT = """Answer using ONLY the facts below. Copy file names, sizes and line counts
exactly from the table. Do the task yourself; never explain how the user
could do it with shell commands or tools. If the message mentions CLI tools
(cat, ls, dir), ignore the tool words and answer from the facts.

Rules:
- FILE CONTENT below is the evidence: quote or summarize WHAT IS IN the file
  (headings, text, functions). Example: for a page whose content shows
  "<h1>File List</h1>", a good answer is: it contains the heading "File List".
  Never answer with just the file name, and never say a file "is in this
  workspace" when its content is shown.
- Only when a named file has a NOT-IN-WORKSPACE note, answer exactly:
  "<name> is not in this workspace".
- If the workspace is empty, answer in one plain sentence.
- When an ANSWER SKELETON is given, keep its names, types and sizes exactly;
  you may add one short sentence.
- If the answer is not in the facts, say so in one sentence. Be short.
- Write the answer in GitHub-flavored markdown. Short. File names in
  backticks. Code in fenced blocks with the language. A table only for lists
  of files. No headings for answers under 5 lines.

MESSAGE: {message}

FACTS:
{facts}
{skeleton}"""


async def reply_step(
    message: str,
    *,
    generate: Callable[..., Awaitable[str]],
    facts_block: str,
    skeleton: str = "",
    table_names: list[str] | None = None,
    role: str = "understand",
) -> str:
    """One reply call grounded in the Python-built FACTS block. Never raises
    (infra errors propagate; an empty reply becomes a one-sentence fallback).

    ``skeleton`` is the Python-built listing answer (names/types/sizes). When
    the model's reply contradicts the table (names a file that is not in
    ``table_names``), the skeleton replaces the reply — the table wins.
    """
    skeleton_block = (
        "\nANSWER SKELETON (keep its names, types and sizes exactly; you may "
        "add one short sentence):\n" + skeleton
        if skeleton else ""
    )
    prompt = (
        REPLY_PROMPT
        .replace("{message}", message[:4_000])
        .replace("{facts}", facts_block)
        .replace("{skeleton}", skeleton_block)
    )
    raw = await generate(
        None, prompt, system=REPLY_SYSTEM, temperature=0.0,
        role=role, num_predict=400,
    )
    answer, _think, _chars = strip_think(raw)
    answer = answer.strip()
    if not answer:
        return "I don't know — that is not in the workspace facts."
    if skeleton and table_names:
        bad = reply_contradicts_table(answer, table_names)
        if bad is not None:
            return skeleton  # the Python table wins over an invented name
    return answer


def reply_contradicts_table(answer: str, table_names: list[str]) -> str | None:
    """The first file name in the answer that is NOT in the table, or None."""
    known = {n.lower() for n in table_names}
    known |= {n.rsplit("/", 1)[-1].lower() for n in table_names}  # basenames
    for match in _FILE_TOKEN_RE.finditer(answer):
        token = match.group(1).lower().lstrip("./")
        base = token.rsplit("/", 1)[-1]
        if token not in known and base not in known:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Step 2b: the planning call (small schema, no question field)
# ---------------------------------------------------------------------------

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "action": {"enum": ["create", "edit"]},
                    "description": {"type": "string"},
                    "uses_last_result": {"type": "boolean"},
                },
                "required": ["path", "action", "description"],
                "additionalProperties": False,
            },
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        # Set when no file the user asked for can be named or inferred:
        # {"tasks": [], "reason": "one line"} -> the caller falls back to reply.
        "reason": {"type": "string"},
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

PLAN_SYSTEM = (
    "You are a software planner. Output only the JSON plan."
)

PLAN_PROMPT = """Plan the files for this request. One task = one file, ordered, 1..6 tasks.
When details are missing, choose sensible defaults and list them in
assumptions — never ask a question. Real file names only, never placeholder
names. Never invent a file the user did not ask for (no index.html unless
the request is a web page). When no file the user asked for can be named or
inferred, return {"tasks": [], "reason": "one line"} instead. Descriptions:
at least 8 words, matching the file's language. Respect explicit constraints
in the request ("single html file" = exactly one file).

REQUEST: {message}

WORKSPACE FILES: {files}

{PLAN_MARKER}
{ "tasks": [ { "path": "...", "action": "create"|"edit",
               "description": "...", "uses_last_result": false } ],
  "assumptions": ["..."], "reason": "" }"""


class PlanResult:
    """Validated plan output. All fields code-verified, never trusted.

    ``tasks == []`` with a non-empty ``reason`` is a VALID outcome: no file
    the user asked for could be named or inferred — the caller falls back to
    the reply step instead of creating anything."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.assumptions = [
            str(a) for a in data.get("assumptions", []) if isinstance(a, str)
        ][:6]
        self.reason = str(data.get("reason", "")).strip()
        self.tasks: list[dict[str, Any]] = []
        for task in (data.get("tasks") or [])[:6]:
            if not isinstance(task, dict):
                continue
            path = str(task.get("path", "")).strip().replace("\\", "/")
            if not path:
                continue
            action = str(task.get("action", "create")).lower()
            self.tasks.append({
                "path": path,
                "action": action if action in ("create", "edit") else "create",
                "description": str(task.get("description", "")).strip(),
                "uses_last_result": bool(task.get("uses_last_result", False)),
            })

    def validate(self) -> list[str]:
        reasons: list[str] = []
        if not self.tasks and not self.reason:
            reasons.append("plan requires at least one task")
        return reasons

    def apply_task_rules(
        self, task_validator: Callable[[dict[str, Any]], str | None]
    ) -> list[str]:
        """Run the pipeline's per-task SAFETY rules over every task. A rejected
        task is never silently dropped: its reason is collected (the caller
        retries once with the reasons appended, then PlanFailed)."""
        kept: list[dict[str, Any]] = []
        rejected: list[str] = []
        for task in self.tasks:
            reason = task_validator(task)
            if reason is None:
                kept.append(task)
            else:
                rejected.append(reason)
        self.tasks = kept
        return rejected


def _extract_json(text: str) -> dict[str, Any]:
    """Plain-JSON fallback: first {...} block."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


class PlanFailed(Exception):
    """Raised when the plan call fails validation after retry + plain JSON.
    Carries the reasons for the needs_human result."""


async def plan_tasks(
    message: str,
    *,
    generate: Callable[..., Awaitable[str]],
    file_names: list[str],
    role: str = "understand",
    task_validator: Callable[[dict[str, Any]], str | None] | None = None,
    plan_validator: Callable[[list[dict[str, Any]], str], str | None] | None = None,
) -> PlanResult:
    """One planning call with the small schema; one retry with the errors
    appended; then the plain-JSON path; then PlanFailed (no files, no loop).

    Three exits:
    - a valid plan (``tasks`` non-empty),
    - an EMPTY plan with ``reason`` (the model's own ``tasks: []`` or a
      plan_validator rejection that survives the retry) — the caller falls
      back to the reply step, no files, no needs_human,
    - PlanFailed for unparseable junk after the retry (needs_human).

    Infra errors propagate to the caller's structured-error envelope.
    """
    prompt = (
        PLAN_PROMPT
        .replace("{message}", message[:4_000])
        .replace("{files}", ", ".join(file_names[:40]) or "(empty workspace)")
        .replace("{PLAN_MARKER}", PLAN_MARKER)
    )
    last_reasons: list[str] = []
    plan_level_rejection = ""
    for attempt in range(2):
        eff_prompt = prompt
        if last_reasons:
            eff_prompt += (
                "\n\nYour previous plan was rejected: "
                + "; ".join(last_reasons)
                + "\nReturn ONLY a corrected JSON plan."
            )
        raw = await generate(
            None, eff_prompt, system=PLAN_SYSTEM, temperature=0.0,
            role=role, num_predict=700,
            _format_schema=PLAN_SCHEMA if attempt == 0 else None,
        )
        answer, _think, _chars = strip_think(raw)
        result = PlanResult(_extract_json(answer))
        # Safety rules: rejected tasks collect their reasons (never silently
        # dropped). Non-empty tasks ALWAYS win over a model-supplied reason;
        # when every task was rejected the plan retries once with the reasons,
        # even when the model also filled `reason` (a rejected task is a fixable
        # failure, not a "nothing to do").
        task_rejects = (
            result.apply_task_rules(task_validator) if task_validator is not None else []
        )
        last_reasons = list(task_rejects)
        if result.tasks:
            result.reason = ""  # non-empty tasks win over a model reason
        last_reasons.extend(result.validate())
        plan_level_rejection = ""
        if not last_reasons and plan_validator is not None and result.tasks:
            # Plan-level rules (e.g. invented-file rejection). A rejection is
            # retried once; if it survives, it becomes an empty plan -> reply.
            plan_level_rejection = plan_validator(result.tasks, message) or ""
            if plan_level_rejection:
                last_reasons = [plan_level_rejection]
        if not last_reasons:
            return result
    if plan_level_rejection:
        # The planner insists on a file the request never named: fall back to
        # reply (harmless) instead of needs_human.
        empty = PlanResult({})
        empty.reason = plan_level_rejection
        return empty
    raise PlanFailed("; ".join(last_reasons) or "planner returned no valid plan")


# ---------------------------------------------------------------------------
# FACTS block builder (Python-side; sizes in KB computed here, never by LLM)
# ---------------------------------------------------------------------------

def build_facts_block(
    facts: dict[str, Any],
    *,
    turns: list[dict[str, Any]] | None = None,
    last_result: dict[str, Any] | None = None,
    last_trace: list[dict[str, Any]] | None = None,
    pending_question: str = "",
    max_chars: int = 3_000,
) -> str:
    """The FACTS block for reply_step: file table (path, language, lines,
    size in KB with one decimal — computed here), extensionless-file flags,
    last-run trace, last_result (capped), last 3 turns, pending question."""
    lines: list[str] = []
    files = facts.get("files", [])
    if not files:
        lines.append("The workspace is empty (no files).")
    else:
        lines.append("FILE TABLE (path | language | lines | size KB):")
        for f in files[:60]:
            # build_file_facts writes "size" (bytes); some callers write
            # "bytes". Read both — never 0.0 KB for a real file.
            size_kb = float(f.get("size", f.get("bytes", 0)) or 0) / 1024.0
            lines.append(
                f"- {f.get('name', '?')} | {f.get('language', '?')} | "
                f"{f.get('lines', 0)} lines | {size_kb:.1f} KB"
            )
        # Extensionless flags (readme/license/dockerfile/makefile/requirements)
        # are computed HERE from the file table — nothing upstream produces a
        # "flags" key.
        bases = {str(f.get("name", "")).rsplit("/", 1)[-1].lower() for f in files}
        stems = {b.rsplit(".", 1)[0] for b in bases}
        wanted = ("readme", "license", "dockerfile", "makefile", "requirements")
        present = [w for w in wanted if w in stems or any(b.startswith(w) for b in bases)]
        lines.append(
            "Extensionless files present: " + (", ".join(present) if present else "none")
        )
    if last_trace:
        trace_line = " ".join(
            f"{t.get('stage', '?')}:{t.get('status', '?')}:{t.get('seconds', 0)}s"
            for t in last_trace[-8:]
        )
        lines.append(f"last_run trace: {trace_line}")
    if last_result:
        text = str(last_result.get("text", ""))[:1_500]
        lines.append(f"last_result [{last_result.get('type', '?')}]: {text}")
        files_prev = ", ".join(str(f) for f in last_result.get("files", [])[:10])
        if files_prev:
            lines.append(f"last_result files: {files_prev}")
    for turn in (turns or [])[-3:]:
        lines.append(
            f"- user: {turn.get('user', '')[:200]} -> {turn.get('mode', '')} "
            f"({turn.get('result', '')[:120]})"
        )
    if pending_question:
        lines.append(f"pending_question: {pending_question}")
    return "\n".join(lines)[:max_chars]


# ---------------------------------------------------------------------------
# Reply evidence: named-file resolution + content (the future tool backend)
# ---------------------------------------------------------------------------

# Filename-like tokens, optionally a relative path. Text extensions AND the
# common binary ones (a named binary must resolve to a "skipped (binary)"
# note, not an "absent" one).
_FILE_TOKEN_RE = re.compile(
    r"\b([\w\-./\\]+\.(?:py|html?|css|js|jsx|ts|tsx|json|md|txt|rs|toml|ya?ml|cfg|ini|csv|xml|sql|sh|bat|bin|png|jpe?g|gif|ico|pdf|zip|gz|exe|dll|so|dylib|woff2?|ttf|otf|mp[34]|wav|mov|webm))\b",
    re.IGNORECASE,
)
# Extensionless names that map to the matching file when present.
_EXTENSIONLESS_NAMES = (
    "readme", "license", "licence", "dockerfile", "makefile",
    "changelog", "contributing", "authors", "notice", "copying",
)
# The one file-type registry (filetypes.py); TEXT_SUFFIXES is derived there.
from filetypes import TEXT_SUFFIXES as _TEXT_SUFFIXES
REPLY_MAX_FILES = 3
REPLY_FILE_CHARS = 6_000
REPLY_LARGE_BYTES = 512 * 1024  # bigger files are noted, not dumped


def find_named_files(
    message: str, all_files: list[str]
) -> tuple[list[str], list[str]]:
    """Files the message names, matched against the workspace listing.

    Returns (found, absent): found = relative paths present in all_files
    (case-insensitive exact name or relative-path match; extensionless names
    like readme/license/dockerfile map by stem). absent = extension-ful names
    the message mentions but the workspace does not have.
    """
    by_lower = {str(n).lower(): str(n) for n in all_files}
    by_base = {str(n).rsplit("/", 1)[-1].lower(): str(n) for n in all_files}
    by_stem = {}
    for n in all_files:
        base = str(n).rsplit("/", 1)[-1].lower()
        stem = base.rsplit(".", 1)[0] if "." in base else base
        by_stem.setdefault(stem, str(n))

    found: list[str] = []
    absent: list[str] = []
    for match in _FILE_TOKEN_RE.finditer(message):
        token = match.group(1).replace("\\", "/").lstrip("./").lower()
        hit = by_lower.get(token) or by_base.get(token)
        if hit is not None:
            if hit not in found:
                found.append(hit)
        else:
            if token not in absent:
                absent.append(match.group(1))
    lower_msg = message.lower()
    for name in _EXTENSIONLESS_NAMES:
        if re.search(rf"\b{name}\b", lower_msg) and name in by_stem:
            hit = by_stem[name]
            if hit not in found:
                found.append(hit)
    return found, absent


def read_named_file_contents(
    workspace: Any, rel_paths: list[str],
    *, max_files: int = REPLY_MAX_FILES, max_chars: int = REPLY_FILE_CHARS,
) -> tuple[list[str], list[str]]:
    """Read named workspace files for the reply FACTS block.

    Read-only via ``safe_read_target``; capped at max_files / max_chars each;
    binary or very large files are skipped with a note. Returns
    (content_sections, notes).
    """
    from pathlib import Path as _Path

    from agent_core.staging import PathEscapeError, safe_read_target

    ws = _Path(workspace).resolve()
    sections: list[str] = []
    notes: list[str] = []
    for rel in rel_paths[:max_files]:
        try:
            target = safe_read_target(ws, rel)
        except PathEscapeError:
            notes.append(f"{rel}: skipped (unsafe path)")
            continue
        suffix = _Path(rel).suffix.lower()
        if suffix and suffix not in _TEXT_SUFFIXES:
            notes.append(f"{rel}: skipped (binary file)")
            continue
        try:
            size = target.stat().st_size
        except OSError:
            notes.append(f"{rel}: not readable")
            continue
        if size > REPLY_LARGE_BYTES:
            notes.append(f"{rel}: skipped (large file, {size // 1024} KB)")
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            notes.append(f"{rel}: skipped (binary file)")
            continue
        except OSError:
            notes.append(f"{rel}: not readable")
            continue
        sections.append(f"FILE CONTENT: {rel}\n{text[:max_chars]}")
    for extra in rel_paths[max_files:]:
        notes.append(f"{extra}: skipped (cap of {max_files} files)")
    return sections, notes


def build_reply_facts(
    workspace: Any,
    message: str,
    facts: dict[str, Any],
    *,
    turns: list[dict[str, Any]] | None = None,
    last_result: dict[str, Any] | None = None,
    last_trace: list[dict[str, Any]] | None = None,
    pending_question: str = "",
    max_chars: int = 8_000,
) -> str:
    """The reply FACTS block: the Python file table PLUS the content of any
    file the message names (and notes for absent/binary/large ones). Built in
    Python before the model call; kept separate so it can become the backend
    of a real read-only tool layer later.

    Layout: the file table first, then the NAMED FILE CONTENT (the evidence),
    and only then the conversation tail — small models weigh the end of the
    prompt heaviest, and the content must outweigh the history."""
    table = build_facts_block(
        {"files": facts.get("files", []), "flags": facts.get("flags", {})},
        max_chars=3_000,
    )
    all_files = [str(f.get("name", "")) for f in facts.get("files", [])]
    all_files += [n for n in facts.get("all_files", []) if n not in all_files]
    found, absent = find_named_files(message, all_files)
    parts = [table]
    for name in absent[:3]:
        parts.append(f"NOT-IN-WORKSPACE: {name}")
    if found:
        sections, notes = read_named_file_contents(workspace, found)
        parts.extend(sections)
        parts.extend(f"NOTE: {n}" for n in notes)
    # Conversation tail LAST and small: context for pronouns, never the answer.
    tail = build_facts_block(
        {"files": [], "flags": {}},
        turns=turns,
        last_result=last_result,
        last_trace=last_trace,
        pending_question=pending_question,
        max_chars=800,
    )
    tail_lines = [
        ln for ln in tail.splitlines()
        if ln.strip() and "workspace is empty" not in ln
    ]
    if tail_lines:
        parts.append("CONTEXT (recent turns, for reference only):")
        parts.extend(tail_lines)
    return "\n".join(parts)[:max_chars]


# Listing intent: the answer is a Python-built table the model may only frame.
_LISTING_RE = re.compile(
    r"\b(?:list|show|display|all\s+(?:the\s+)?files?|what\s+files?|which\s+files?"
    r"|file\s+(?:names?|types?|sizes?)|types?\s+of\s+files?|how\s+many\s+files?"
    r"|what(?:'s|\s+is|\s+are)?\s+in\s+(?:this|the|my)\s+(?:dir|directory|folder|workspace)"
    r"|contain)\b",
    re.IGNORECASE,
)


def is_listing_request(message: str) -> bool:
    return bool(_LISTING_RE.search(message))


def build_listing_skeleton(facts: dict[str, Any]) -> str:
    """The deterministic listing answer: names, types, sizes from the table.
    The model may frame it; it may not change the numbers."""
    files = facts.get("files", [])
    if not files:
        return "The workspace is empty — there are no files to list."
    lines = ["Files in this workspace (name | type | lines | size):"]
    for f in files[:60]:
        size_kb = float(f.get("bytes", f.get("size", 0))) / 1024.0
        lines.append(
            f"- {f.get('name', '?')} | {f.get('language', '?')} | "
            f"{f.get('lines', 0)} lines | {size_kb:.1f} KB"
        )
    return "\n".join(lines)
