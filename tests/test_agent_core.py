"""Phase 2 tests: structured outputs, planner fallback, engine sessions."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent_core.engine import Engine, RunRequest, get_engine
from agent_core.llm_gateway import LegacyLLMAdapter, StructuredOutputError
from agent_core.planner import parse_markdown_tasks, plan_tasks
from agent_core.schemas import AuditResult, Plan


class FakeStructuredLLM:
    """Scriptable llm with native structured support (like LLMGateway)."""

    supports_structured = True

    def __init__(self, structured_payloads=None, text_responses=None, fail_structured=0):
        self.structured_payloads = list(structured_payloads or [])
        self.text_responses = list(text_responses or [])
        self.fail_structured = fail_structured
        self.structured_calls = 0
        self.text_calls = 0

    def text(self, model, prompt, **kwargs):
        self.text_calls += 1
        return self.text_responses.pop(0) if self.text_responses else "PASS\nok"

    def structured(self, model, prompt, schema, **kwargs):
        self.structured_calls += 1
        if self.fail_structured > 0:
            self.fail_structured -= 1
            raise StructuredOutputError("bad schema", retryable=False)
        payload = self.structured_payloads.pop(0)
        return schema.model_validate(payload)


PLAN_PAYLOAD = {
    "tasks": [
        {
            "id": 1,
            "name": "create greeting",
            "target_files": ["generated.py"],
            "spec": "def greet(name: str) -> str returning a greeting",
        }
    ]
}
GOOD_CODE = "```FILE: generated.py\n\ndef greet(name: str) -> str:\n    return f'Hello, {name}!'\n```"
AUDIT_PAYLOAD = {"verdict": "pass", "issues": []}


def test_plan_schema_validation() -> None:
    plan = Plan.model_validate(PLAN_PAYLOAD)
    assert plan.tasks[0].name == "create greeting"
    with pytest.raises(Exception):
        Plan.model_validate({"tasks": []})  # min_length=1


def test_audit_schema_validation() -> None:
    ok = AuditResult.model_validate({"verdict": "concerns", "issues": [
        {"severity": "warning", "message": "missing docstring"}]})
    assert ok.verdict == "concerns"
    with pytest.raises(Exception):
        AuditResult.model_validate({"verdict": "maybe", "issues": []})


def test_structured_planner_path_used_when_supported() -> None:
    llm = FakeStructuredLLM(structured_payloads=[PLAN_PAYLOAD])
    tasks, raw = plan_tasks(llm, "planner", "make a greeter", "(empty)", "/tmp/ws")
    assert llm.structured_calls == 1 and llm.text_calls == 0
    assert tasks[0]["name"] == "create greeting"
    assert tasks[0]["target_files"] == ["generated.py"]


def test_planner_falls_back_to_markdown_on_structured_failure() -> None:
    llm = FakeStructuredLLM(
        fail_structured=1,
        text_responses=["## TASK 1: greeter\n- files: generated.py\n- logic: greet func"],
    )
    tasks, _ = plan_tasks(llm, "planner", "make a greeter", "(empty)", "/tmp/ws")
    assert llm.structured_calls == 1 and llm.text_calls == 1
    assert tasks[0]["name"] == "greeter"
    assert tasks[0]["target_files"] == ["generated.py"]


def test_markdown_task_parser_minimal_fallback() -> None:
    tasks = parse_markdown_tasks("some prose with no task headers")
    assert len(tasks) == 1
    assert tasks[0]["name"] == "primary implementation"


def test_engine_happy_path_via_structured_llm() -> None:
    with TemporaryDirectory() as tmp:
        llm = FakeStructuredLLM(
            structured_payloads=[PLAN_PAYLOAD, AUDIT_PAYLOAD],
            text_responses=[GOOD_CODE],
        )
        engine = Engine(workspace=tmp, llm=llm)
        report = engine.run(RunRequest(prompt="Create a greeting function"))
        assert report.status == "completed", report
        assert (Path(tmp) / "generated.py").exists()
        assert report.llm_calls == 3  # plan + code + audit
        # Staging flushed after auto-deploy
        assert not list((Path(tmp) / ".agent_staging").rglob("*"))


def test_engine_without_auto_deploy_keeps_staging() -> None:
    with TemporaryDirectory() as tmp:
        llm = FakeStructuredLLM(
            structured_payloads=[PLAN_PAYLOAD, AUDIT_PAYLOAD],
            text_responses=[GOOD_CODE],
        )
        engine = Engine(workspace=tmp, llm=llm)
        report = engine.run(RunRequest(prompt="g", auto_deploy=False))
        assert report.status == "completed"
        assert not (Path(tmp) / "generated.py").exists()
        assert (Path(tmp) / ".agent_staging" / "generated.py").exists()


def test_legacy_adapter_routes_chat_and_blocks_structured() -> None:
    class Stub:
        def chat(self, model, prompt, system="", max_tokens=None, temperature=0.1):
            return "raw text"

    adapter = LegacyLLMAdapter(Stub())
    assert adapter.supports_structured is False
    assert adapter.text("m", "p") == "raw text"
    with pytest.raises(StructuredOutputError):
        adapter.structured("m", "p", Plan)


def test_session_registry_is_per_workspace() -> None:
    with TemporaryDirectory() as a, TemporaryDirectory() as b:
        ea, eb = get_engine(a), get_engine(b)
        assert ea is not eb
        assert get_engine(a) is ea  # same workspace -> same session
        assert get_engine(Path(a) / ".") is ea
