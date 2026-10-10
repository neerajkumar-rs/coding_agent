"""Planner: turns a messy request into an ordered list of atomic tasks.

Structured-first when the LLM supports Ollama schema-constrained decoding
(pydantic-validated, one retry inside the gateway); falls back to a markdown
prompt with tolerant parsing so the pipeline keeps working with weaker or
legacy chat-only models.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from agent_core.schemas import Plan
from ollama_client import TEMPERATURE_PLAN, OllamaTruncatedError

MAX_TASKS = 12
MAX_PLAN_TOKENS = 900

MARKDOWN_PLAN_PROMPT = (
    "You are the thinker and task planner. Break the messy request into "
    "small ordered atomic tasks. Return ONLY Markdown using this exact shape:\n"
    "## TASK 1: short task name\n"
    "- files: relative/file.py\n"
    "- logic: signatures, parameters, and rules\n"
    "Create no more than 12 tasks and keep every task under 120 words. "
    "Do not generate code. Each task must be independently codable and testable.\n\n"
    "WORKSPACE:\n{workspace}\n\nEXISTING FILE CONTEXT:\n{context}\n\nREQUEST:\n{request}"
)

STRUCTURED_PLAN_PROMPT = (
    "You are the task planner. Break the request into small ordered atomic "
    "tasks (max 12). Each task: a short name, the relative target files it "
    "touches, and a precise spec (signatures, parameters, rules; under 120 "
    "words). Do not write code.\n\n"
    "WORKSPACE:\n{workspace}\n\nEXISTING FILE CONTEXT:\n{context}\n\nREQUEST:\n{request}"
)


class LLM(Protocol):
    supports_structured: bool

    def text(self, model: str, prompt: str, **kwargs: Any) -> str: ...

    def structured(self, model: str, prompt: str, schema: type[Plan], **kwargs: Any) -> Plan: ...


def parse_markdown_tasks(plan: str) -> list[dict[str, Any]]:
    """Tolerant parser for the markdown task shape (legacy fallback)."""
    matches = re.findall(
        r"##\s*TASK\s*(\d+)\s*:\s*([^\r\n]+)\r?\n(.*?)(?=##\s*TASK|\Z)",
        plan,
        re.IGNORECASE | re.DOTALL,
    )
    tasks: list[dict[str, Any]] = []
    for number, name, body in matches[:MAX_TASKS]:
        if not body.strip():
            continue
        files_match = re.search(r"-\s*files\s*:\s*([^\r\n]+)", body, re.IGNORECASE)
        target_files = (
            [f.strip() for f in files_match.group(1).split(",") if f.strip()]
            if files_match
            else []
        )
        tasks.append(
            {
                "number": number,
                "name": name.strip(),
                "spec": body.strip(),
                "target_files": target_files,
                "status": "pending",
            }
        )
    if tasks:
        return tasks
    return [
        {
            "number": "1",
            "name": "primary implementation",
            "spec": plan.strip(),
            "target_files": [],
            "status": "pending",
        }
    ]


def plan_tasks(
    llm: LLM,
    planner_model: str,
    request: str,
    workspace_context: str,
    workspace: str,
) -> tuple[list[dict[str, Any]], str]:
    """Return (tasks, raw_plan_text). Structured path first when supported."""
    if getattr(llm, "supports_structured", False):
        prompt = STRUCTURED_PLAN_PROMPT.format(
            workspace=workspace, context=workspace_context, request=request
        )
        try:
            plan = llm.structured(
                planner_model,
                prompt,
                Plan,
                max_tokens=MAX_PLAN_TOKENS,
                temperature=TEMPERATURE_PLAN,
            )
            tasks = [
                {
                    "number": str(t.id),
                    "name": t.name,
                    "spec": t.spec,
                    "target_files": [f.replace("\\", "/").strip() for f in t.target_files],
                    "status": "pending",
                }
                for t in plan.tasks
            ]
            raw = "\n".join(f"## TASK {t['number']}: {t['name']}\n{t['spec']}" for t in tasks)
            return tasks, raw
        except Exception:  # noqa: BLE001 - deliberate fallback
            # Any structured failure (validation, truncation, transport)
            # degrades to the markdown path below.
            pass

    prompt = MARKDOWN_PLAN_PROMPT.format(
        workspace=workspace, context=workspace_context, request=request
    )
    raw = llm.text(
        planner_model, prompt, max_tokens=MAX_PLAN_TOKENS, temperature=TEMPERATURE_PLAN
    )
    return parse_markdown_tasks(raw), raw
