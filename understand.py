"""The Understand step: LLM-first routing, replacing the rules router.

One schema-constrained gateway call (role ``understand``, coder model by
default) reads the user message + workspace FACTS + CONVERSATION STATE and
returns:

    { "mode": "reply" | "code" | "code_and_reply",
      "reply": string|null,
      "tasks": [ { "path", "action": "create"|"edit", "description",
                   "uses_last_result": bool } ],
      "assumptions": [string],
      "question": string|null }

Post-validation lives here in code (never trust the model): schema + mode
consistency, one retry with the errors appended, then a plain-JSON regex
fallback, then a graceful "couldn't understand" reply. Fast paths (slash
commands, whole-message greetings/thanks/acks/openers in English + Hinglish)
never touch the model and never create files.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from ollama_client import strip_think

UNDERSTAND_MARKER = "UNDERSTAND JSON:"

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
    r"|(?:ok(?:ay)?|theek\s*hai|accha|sahi|great|nice|cool|perfect|awesome)\b\s*[!.]*"
    r"|(?:chalo|chalu|shuru)\s+(?:karte|karen|kar)\s+(?:hain|hai|h)\b\s*[!.]*"
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
    """Return the canned reply for a fast-path message, or None.

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
# The Understand call
# ---------------------------------------------------------------------------

UNDERSTAND_SYSTEM = (
    "You are the routing brain of a local coding agent. Read the message, the "
    "workspace FACTS and the CONVERSATION STATE, then output ONE JSON object. "
    "Rules: act with stated assumptions instead of asking (list them in "
    "assumptions); ask at most ONE question and only when truly blocked, with "
    "a proposed default in the question; one task = one file; real file names "
    "only, never placeholder names; never use the request text as file "
    "content; mode reply uses ONLY FACTS and CONVERSATION STATE (say 'not in "
    "this workspace' or 'I don't know' otherwise); questions about what the "
    "agent did are answered from the LAST RUN TRACE; respect explicit "
    "constraints ('single html file' = exactly one file). Output JSON only."
)

UNDERSTAND_PROMPT = """MESSAGE: {message}

FACTS (workspace file table — path, language, lines, size; flags for common
extensionless files present: readme/license/dockerfile/makefile/requirements):
{facts}

CONVERSATION STATE (last turns one-liners, last_result, last-run trace of
stages/models/files/seconds, pending_question):
{state}

{UNDERSTAND_MARKER}
{{ "mode": "reply" | "code" | "code_and_reply",
  "reply": string|null,
  "tasks": [ {{ "path": string, "action": "create"|"edit",
               "description": string, "uses_last_result": bool }} ],
  "assumptions": [string],
  "question": string|null }}"""


class UnderstandResult:
    """Validated Understand output. All fields code-verified, never trusted."""

    def __init__(self, data: dict[str, Any]) -> None:
        mode = str(data.get("mode", ""))
        self.mode = mode if mode in ("reply", "code", "code_and_reply") else "reply"
        self.reply = data.get("reply") if isinstance(data.get("reply"), str) else None
        self.assumptions = [
            str(a) for a in data.get("assumptions", []) if isinstance(a, str)
        ][:6]
        self.question = (
            data.get("question") if isinstance(data.get("question"), str) else None
        )
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
        """Mode-consistency rules. Returns rejection reasons."""
        reasons: list[str] = []
        if self.mode in ("code", "code_and_reply") and not self.tasks:
            reasons.append(f"mode {self.mode} requires at least one task")
        if self.mode == "reply" and not (self.reply or self.question):
            reasons.append("mode reply requires a reply or a question")
        return reasons

    def apply_task_rules(self, task_validator: Callable[[dict[str, Any]], str | None]) -> None:
        """Run the pipeline's per-task rules (placeholder/junk/extension)
        over every task, dropping rejected ones. Code-side, never the model."""
        kept: list[dict[str, Any]] = []
        for task in self.tasks:
            if task_validator(task) is None:
                kept.append(task)
        self.tasks = kept


def _extract_json(text: str) -> dict[str, Any]:
    """Plain-JSON regex fallback: first balanced {...} block."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


async def understand(
    message: str,
    *,
    generate: Callable[..., Awaitable[str]],
    facts_table: str,
    state_block: str,
    role: str = "understand",
    task_validator: Callable[[dict[str, Any]], str | None] | None = None,
) -> UnderstandResult:
    """One Understand call with validation + one retry + regex fallback.

    Never raises. On total failure returns a graceful 'couldn't understand'
    reply result (no files, no loop).
    """
    # Plain replacement, never str.format: the template embeds a literal JSON
    # example whose braces would be eaten by format().
    prompt = (
        UNDERSTAND_PROMPT
        .replace("{message}", message)
        .replace("{facts}", facts_table)
        .replace("{state}", state_block)
        .replace("{UNDERSTAND_MARKER}", UNDERSTAND_MARKER)
    )
    raw = ""
    last_reasons: list[str] = []
    for attempt in range(2):
        eff_prompt = prompt
        if last_reasons:
            eff_prompt += (
                "\n\nYour previous reply failed validation: "
                + "; ".join(last_reasons)
                + "\nReturn ONLY a corrected JSON object."
            )
        # Infra/transport exceptions (model down, VRAM guard, GPU fault)
        # propagate to the caller's structured-error envelope — the retry
        # loop exists for VALIDATION failures (bad content), not dead models.
        raw = await generate(
            None, eff_prompt, system=UNDERSTAND_SYSTEM,
            temperature=0.0, role=role, num_predict=700,
        )
        answer, _think, _chars = strip_think(raw)
        data = _extract_json(answer)
        result = UnderstandResult(data)
        if task_validator is not None:
            result.apply_task_rules(task_validator)
        last_reasons = result.validate()
        if not last_reasons:
            return result
    # Total failure: graceful reply, no files, no loop.
    return UnderstandResult({
        "mode": "reply",
        "reply": "I couldn't understand that — could you rephrase it?",
        "tasks": [], "assumptions": [], "question": None,
    })


# ---------------------------------------------------------------------------
# Conversation-state block builder (compact, ContextBudget-capped)
# ---------------------------------------------------------------------------

def build_state_block(
    turns: list[dict[str, Any]],
    last_result: dict[str, Any],
    last_trace: list[dict[str, Any]],
    pending_question: str,
    max_chars: int = 3_000,
) -> str:
    """One compact CONVERSATION STATE block for the Understand prompt."""
    lines: list[str] = []
    for turn in turns[-3:]:
        lines.append(
            f"- user: {turn.get('user', '')[:200]} -> {turn.get('mode', '')} "
            f"({turn.get('result', '')[:120]})"
        )
    if last_result:
        text = str(last_result.get("text", ""))[:1_500]
        files = ", ".join(str(f) for f in last_result.get("files", [])[:10])
        lines.append(f"last_result [{last_result.get('type', '?')}]: {text}")
        if files:
            lines.append(f"last_result files: {files}")
    if last_trace:
        trace_line = " ".join(
            f"{t.get('stage', '?')}:{t.get('status', '?')}:{t.get('seconds', 0)}s"
            for t in last_trace[-8:]
        )
        lines.append(f"last_run trace: {trace_line}")
    if pending_question:
        lines.append(f"pending_question: {pending_question}")
    block = "\n".join(lines) or "(no prior conversation)"
    return block[:max_chars]
