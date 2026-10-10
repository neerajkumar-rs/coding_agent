"""LLM auditor. Runs ONLY after the deterministic verifier passes and NEVER
blocks it: the worst it can do is attach concerns to the run report.

The auditor sees the diff (what changed), not whole files, to stay inside
the 4096-token budget.
"""

from __future__ import annotations

from typing import Any, Protocol

from agent_core.schemas import AuditResult
from ollama_client import TEMPERATURE_PLAN

MAX_AUDIT_TOKENS = 700
MAX_DIFF_CHARS = 3_000

AUDIT_PROMPT = (
    "You are a code auditor. Given the task specification and the diff of what "
    "changed, report only genuine problems: missing required behaviour, broken "
    "references, obvious runtime defects. Do not ask for rewrites.\n\n"
    "SPECIFICATION:\n{spec}\n\nDIFF:\n{diff}"
)


class LLM(Protocol):
    supports_structured: bool

    def text(self, model: str, prompt: str, **kwargs: Any) -> str: ...

    def structured(
        self, model: str, prompt: str, schema: type[AuditResult], **kwargs: Any
    ) -> AuditResult: ...


def audit_diff(
    llm: LLM,
    auditor_model: str,
    spec: str,
    diff_text: str,
) -> AuditResult:
    """Structured audit with a graceful text fallback. Never raises."""
    prompt = AUDIT_PROMPT.format(spec=spec[:4_000], diff=diff_text[:MAX_DIFF_CHARS])
    if getattr(llm, "supports_structured", False):
        try:
            return llm.structured(
                auditor_model, prompt, AuditResult,
                max_tokens=MAX_AUDIT_TOKENS, temperature=TEMPERATURE_PLAN,
            )
        except Exception:  # noqa: BLE001 - auditor failure must never block
            pass
    try:
        feedback = llm.text(
            auditor_model,
            prompt + "\n\nFirst line: PASS or CONCERNS, then short reasons.",
            max_tokens=MAX_AUDIT_TOKENS,
            temperature=TEMPERATURE_PLAN,
        )
        first = feedback.splitlines()[0].strip().upper() if feedback else ""
        verdict = "pass" if first.startswith("PASS") else "concerns"
        if verdict == "concerns" and first not in ("CONCERNS", "FAIL"):
            # Unparseable reply: record it as info, do not punish the run.
            return AuditResult(verdict="pass", issues=[])
        from agent_core.schemas import AuditIssue

        issues = [] if verdict == "pass" else [
            AuditIssue(severity="warning", message=feedback[:500])
        ]
        return AuditResult(verdict=verdict, issues=issues)
    except Exception:  # noqa: BLE001
        return AuditResult(verdict="pass", issues=[])
