"""Planner model schema-conformance benchmark.

Runs N probe prompts against a candidate planner model through Ollama's
structured-output path and reports the failure rate. If the configured
planner (default deepseek-r1:1.5b) fails schema validation on more than
PLANNER_FAILURE_THRESHOLD of the probes, the recommended fallback is
qwen2.5-coder:3b.

Usage:
    .\\env\\Scripts\\python.exe -m agent_core.planner_eval [--model NAME] [--probes 5]
"""

from __future__ import annotations

import argparse
import sys

from agent_core.llm_gateway import LLMGateway, StructuredOutputError
from agent_core.schemas import Plan
from ollama_client import OllamaEngine, TEMPERATURE_PLAN

PLANNER_FAILURE_THRESHOLD = 0.20
FALLBACK_PLANNER = "qwen2.5-coder:3b"

PROBES = [
    "Create a Python CLI that greets a user by name.",
    "Add a dark-mode toggle button to index.html.",
    "Build a small Flask app with one /health endpoint returning JSON.",
    "Write a function that parses CSV rows into dicts, with tests.",
    "Create styles.css for a two-column responsive layout.",
]


def evaluate_planner(model: str, probes: int = len(PROBES)) -> dict:
    """Return {'model', 'probes', 'failures', 'failure_rate', 'ok'}."""
    gateway = LLMGateway(OllamaEngine())
    failures = 0
    for probe in PROBES[: max(1, probes)]:
        try:
            gateway.structured(
                model, f"Plan this request:\n{probe}", Plan,
                max_tokens=900, temperature=TEMPERATURE_PLAN,
            )
        except (StructuredOutputError, Exception):  # noqa: BLE001
            failures += 1
    rate = failures / max(1, min(probes, len(PROBES)))
    return {
        "model": model,
        "probes": min(probes, len(PROBES)),
        "failures": failures,
        "failure_rate": round(rate, 3),
        "ok": rate <= PLANNER_FAILURE_THRESHOLD,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Planner schema-conformance benchmark")
    parser.add_argument("--model", default="deepseek-r1:1.5b")
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
            f"Recommended planner: {FALLBACK_PLANNER} "
            f"(set planner_model in .agent or pass --planner-model)."
        )
        return 1
    print("PASS: model conforms to the Plan schema reliably enough.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
