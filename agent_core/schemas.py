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


class AuditIssue(BaseModel):
    severity: str = Field(pattern="^(info|warning|critical)$")
    message: str = Field(min_length=1, max_length=500)


class AuditResult(BaseModel):
    """Auditor output. Warnings never block a deterministic verifier pass."""

    verdict: str = Field(pattern="^(pass|concerns)$")
    issues: list[AuditIssue] = Field(default_factory=list, max_length=10)
