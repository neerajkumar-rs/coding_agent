"""Router eval gate (tests/router_eval.jsonl, >= 60 phrases).

Rules:
- Lines with tier_max <= 1 must classify correctly with NO model object at
  all (rules-only path; a generate that raises proves it is never called).
- tier_max == 2 lines run the T2 classifier against a fake generate that
  returns the line's recorded classifier JSON.
- Gate: >= 95% accuracy overall; ZERO CODE_TASK for any greeting / ack /
  thanks / opener / Hinglish line.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from router import ROUTER_MARKER, route, route_rules_only

EVAL_PATH = Path(__file__).parent / "router_eval.jsonl"

# Groups that must NEVER be CODE_TASK.
_NEVER_CODE_PREFIXES = ()  # intent-based check below instead


def _lines() -> list[dict]:
    rows: list[dict] = []
    for line in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _boom(model, prompt, **kwargs):
    raise AssertionError("a model was called for a rules-only eval line")


def test_eval_set_size() -> None:
    assert len(_lines()) >= 60


def test_router_eval_accuracy_gate() -> None:
    rows = _lines()
    failures: list[str] = []
    social_intents = {"CHAT"}
    social_zero_code = 0

    for row in rows:
        text, expected, tier_max = row["text"], row["expected"], row["tier_max"]
        if tier_max <= 1:
            got = route_rules_only(text).intent  # no model involved, by construction
        else:
            payload = row["t2"]

            async def fake(model, prompt, system="", temperature=0.0, **kwargs):
                assert ROUTER_MARKER in prompt
                return json.dumps(payload)

            got = asyncio.run(route(text, generate=fake, thinker_model="fake-thinker")).intent
        if expected in social_intents and got == "CODE_TASK":
            social_zero_code += 1
        if got != expected:
            failures.append(f"{text!r}: want {expected}, got {got}")

    accuracy = (len(rows) - len(failures)) / len(rows)
    assert social_zero_code == 0, f"CODE_TASK for social lines: {failures}"
    assert accuracy >= 0.95, f"accuracy {accuracy:.1%} < 95%:\n" + "\n".join(failures)


def test_rules_tier_lines_never_construct_a_model_call() -> None:
    """Every tier_max <= 1 line resolves without any model object: the full
    async route() is driven with a generate that raises if ever called."""
    rows = [r for r in _lines() if r["tier_max"] <= 1]
    assert rows
    for row in rows:
        result = asyncio.run(route(row["text"], generate=_boom, thinker_model="x"))
        assert result.intent == row["expected"], (
            f"{row['text']!r}: want {row['expected']}, got {result.intent}"
        )
