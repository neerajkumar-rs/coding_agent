"""Planner model schema-conformance benchmark.

Runs N probe prompts through the single AgentGateway (role config, VRAM
guard, structured-output path) and reports the failure rate. If the
configured thinker (default erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest)
fails schema validation on more than PLANNER_FAILURE_THRESHOLD of the probes,
the recommended fallback is qwen2.5-coder:3b-instruct.

Usage:
    .\\env\\Scripts\\python.exe -m agent_core.planner_eval [--model NAME] [--probes 5]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from agent_core.schemas import Plan
from ollama_client import AgentGateway, OllamaSupervisorClient

PLANNER_FAILURE_THRESHOLD = 0.20
FALLBACK_PLANNER = "qwen2.5-coder:3b-instruct"

PROBES = [
    "Create a Python CLI that greets a user by name.",
    "Add a dark-mode toggle button to index.html.",
    "Build a small Flask app with one /health endpoint returning JSON.",
    "Write a function that parses CSV rows into dicts, with tests.",
    "Create styles.css for a two-column responsive layout.",
]


class _ProbeConfig:
    """Minimal config shim: lets a probe model override the thinker role."""

    def __init__(self, base, model: str) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_base"), name)

    def __deepcopy__(self, memo):
        return _ProbeConfig(self._base, self._model)

    def get_role_config(self, role: str) -> dict:
        cfg = dict(self._base.get_role_config(role))
        if role in ("thinker", "planner"):
            cfg["model"] = self._model
        return cfg


async def _evaluate(model: str, probes: int) -> dict:
    gateway = AgentGateway(OllamaSupervisorClient())
    if model:
        gateway.config = _ProbeConfig(gateway.config, model)
    failures = 0
    for probe in PROBES[: max(1, probes)]:
        try:
            await gateway.structured(
                "thinker", f"Plan this request:\n{probe}", Plan,
            )
        except Exception:  # noqa: BLE001 - any failure counts against the model
            failures += 1
    rate = failures / max(1, min(probes, len(PROBES)))
    return {
        "model": model or gateway.config.get_role_config("thinker")["model"],
        "probes": min(probes, len(PROBES)),
        "failures": failures,
        "failure_rate": round(rate, 3),
        "ok": rate <= PLANNER_FAILURE_THRESHOLD,
    }


def evaluate_planner(model: str, probes: int = len(PROBES)) -> dict:
    """Return {'model', 'probes', 'failures', 'failure_rate', 'ok'}."""
    return asyncio.run(_evaluate(model, probes))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Planner schema-conformance benchmark")
    parser.add_argument("--model", default="")
    parser.add_argument("--probes", type=int, default=len(PROBES))
    args = parser.parse_args(argv)

    result = evaluate_planner(args.model, args.probes)
    print(
        f"planner eval: model={result['model']} probes={result['probes']} "
        f"failures={result['failures']} rate={result['failure_rate']}"
    )
    if not result["ok"]:
        print(
            f"FAIL: schema failure rate above {PLANNER_FAILURE_THRESHOLD:.0%}. "
            f"Recommended planner: {FALLBACK_PLANNER}."
        )
        return 1
    print("PASS: model conforms to the Plan schema reliably enough.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
