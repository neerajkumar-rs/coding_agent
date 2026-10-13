"""Pydantic schemas for structured LLM outputs (planner and auditor)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlanTask(BaseModel):
    """One atomic, independently-codable unit of work."""

    id: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=120)
    target_files: list[str] = Field(min_length=1, max_length=8)
    spec: str = Field(min_length=1, max_length=2_000)


class Plan(BaseModel):
    """Structured planner output: an ordered list of small tasks."""

    tasks: list[PlanTask] = Field(min_length=1, max_length=12)


# --- run_pipeline plan shape (path/action/description) -------------------------

class PipelineTask(BaseModel):
    """One run_pipeline task: the single unified plan format."""

    path: str = Field(min_length=1, max_length=200)
    action: str = Field(default="create", pattern="^(create|edit)$")
    description: str = Field(min_length=1, max_length=2_000)


class PipelinePlan(BaseModel):
    """run_pipeline planner output: an ordered list of file tasks."""

    tasks: list[PipelineTask] = Field(min_length=1, max_length=12)


class ValidateResult(BaseModel):
    """Validator output: does the generated set satisfy the request?"""

    satisfies_request: bool
    problems: list[str] = Field(default_factory=list, max_length=20)
    summary: str = Field(default="", max_length=500)


class RouterIntent(BaseModel):
    """T2 router classifier output."""

    intent: str = Field(pattern="^(CHAT|CODE_TASK|QUESTION|FILE_QUESTION|COMMAND|UNCLEAR)$")
    confidence: float = Field(ge=0.0, le=1.0)
    deliverable: str | None = None
    clarifying_question: str = ""


class AuditIssue(BaseModel):
    severity: str = Field(pattern="^(info|warning|critical)$")
    message: str = Field(min_length=1, max_length=500)


class AuditResult(BaseModel):
    """Auditor output. Warnings never block a deterministic verifier pass."""

    verdict: str = Field(pattern="^(pass|concerns)$")
    issues: list[AuditIssue] = Field(default_factory=list, max_length=10)
