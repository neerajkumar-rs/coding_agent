"""Classify-then-act routing: tiny classifier, then one call per label.

Replaces the single one-call "Understand" step (too big for a 3B model).
Three steps, each ONE gateway call at most:

1. ``classify_mode`` — ``CLASSIFY MODE:`` marker, ``format`` = JSON schema
   ``{"mode": {"enum": [...]}}``, temperature 0, ``num_predict`` <= 12.
   Output is always one of reply|code|both; there is no "couldn't
   understand" path.
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

CLASSIFY_PROMPT = """Classify the message. Answer with one word: reply, code, or both.
reply = user wants information, a list, an explanation, or chat. No file is created or changed.
code  = user wants a file created or changed.
both  = user wants something explained AND a file created or changed.
Files in workspace: {files}
Last turn: {last_turn}

"hi bro" -> reply
"how many lines in index.html" -> reply
"show me the file sizes" -> reply
"create a script that renames files" -> code

Message: "{text}"

{CLASSIFY_MARKER}"""

_LABEL_RE = re.compile(r"\b(reply|code|both)\b")


async def classify_mode(
    text: str,
    *,
    generate: Callable[..., Awaitable[str]],
    file_names: list[str],
    last_turn: str,
    role: str = "understand",
) -> str:
    """Classify into reply|code|both with ONE tiny call. Always valid.

    Constrained decoding (``format`` schema, temperature 0, num_predict<=12)
    plus a label regex over the think-stripped output; any non-label junk
    falls back to the first recognised label, default ``reply``.
    Infra errors (model missing, VRAM guard, timeout) propagate to the
    caller's structured-error envelope — never a rephrase message.
    """
    prompt = (
        CLASSIFY_PROMPT
        .replace("{files}", ", ".join(file_names[:40]) or "(empty workspace)")
        .replace("{last_turn}", last_turn or "none")
        .replace("{text}", text[:2_000])
        .replace("{CLASSIFY_MARKER}", CLASSIFY_MARKER)
    )
    # Never send `think` here: the understand role defaults to the coder
    # model (an instruct model — think is meaningless), and the gateway
    # itself sends think=False for a reasoning model whose probe confirmed
    # support. This layer stays think-free.
    raw = await generate(
        None, prompt, system="", temperature=0.0,
        role=role, num_predict=12, _format_schema=CLASSIFY_SCHEMA,
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
# Step 2a: the reply call (facts only, no questions, no tutorials)
# ---------------------------------------------------------------------------

REPLY_SYSTEM = (
    "You answer questions about the user's workspace from facts. Be short."
)

REPLY_PROMPT = """Answer using ONLY the facts below. Copy file names, sizes and line counts
exactly from the table. Do the task yourself; never explain how the user
could do it with shell commands or tools. If the answer is not in the facts,
say so in one sentence. Be short.

MESSAGE: {message}

FACTS:
{facts}"""


async def reply_step(
    message: str,
    *,
    generate: Callable[..., Awaitable[str]],
    facts_block: str,
    role: str = "understand",
) -> str:
    """One reply call grounded in the Python-built FACTS block. Never raises
    (infra errors propagate; an empty reply becomes a one-sentence fallback)."""
    prompt = (
        REPLY_PROMPT
        .replace("{message}", message[:4_000])
        .replace("{facts}", facts_block)
    )
    raw = await generate(
        None, prompt, system=REPLY_SYSTEM, temperature=0.0,
        role=role, num_predict=400,
    )
    answer, _think, _chars = strip_think(raw)
    answer = answer.strip()
    return answer or "I don't know — that is not in the workspace facts."


# ---------------------------------------------------------------------------
# Step 2b: the planning call (small schema, no question field)
# ---------------------------------------------------------------------------

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
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
names. Descriptions: at least 8 words, matching the file's language. Respect
explicit constraints in the request ("single html file" = exactly one file).

REQUEST: {message}

WORKSPACE FILES: {files}

{PLAN_MARKER}
{{ "tasks": [ {{ "path": "...", "action": "create"|"edit",
                "description": "...", "uses_last_result": false }} ],
   "assumptions": ["..."] }}"""


class PlanResult:
    """Validated plan output. All fields code-verified, never trusted."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.assumptions = [
            str(a) for a in data.get("assumptions", []) if isinstance(a, str)
        ][:6]
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
        if not self.tasks:
            reasons.append("plan requires at least one task")
        return reasons

    def apply_task_rules(self, task_validator: Callable[[dict[str, Any]], str | None]) -> None:
        """Run the pipeline's per-task rules (placeholder/junk/extension)
        over every task, dropping rejected ones. Code-side, never the model."""
        self.tasks = [t for t in self.tasks if task_validator(t) is None]


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
) -> PlanResult:
    """One planning call with the small schema; one retry with the errors
    appended; then the plain-JSON path; then PlanFailed (no files, no loop).

    Infra errors propagate to the caller's structured-error envelope.
    """
    prompt = (
        PLAN_PROMPT
        .replace("{message}", message[:4_000])
        .replace("{files}", ", ".join(file_names[:40]) or "(empty workspace)")
        .replace("{PLAN_MARKER}", PLAN_MARKER)
    )
    last_reasons: list[str] = []
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
        if task_validator is not None:
            result.apply_task_rules(task_validator)
        last_reasons = result.validate()
        if not last_reasons:
            return result
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
            size_kb = float(f.get("bytes", 0)) / 1024.0
            lines.append(
                f"- {f.get('name', '?')} | {f.get('language', '?')} | "
                f"{f.get('lines', 0)} lines | {size_kb:.1f} KB"
            )
        flags = facts.get("flags", {})
        if flags:
            present = [k for k, v in flags.items() if v]
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
