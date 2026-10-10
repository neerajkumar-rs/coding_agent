"""Model role benchmark for the local Ollama pipeline.

Headless and TUI-embeddable: run(argv) does the work and returns
(report_dict, exit_code) without sys.exit; main() is a thin wrapper.

Exit codes: 0 = ok, 2 = Ollama unreachable.

Reports are written to <workspace>/.agent_runs/bench/ (JSON + Markdown).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from agent_config import AgentConfig, load_agent_config

DEFAULT_OLLAMA = "http://localhost:11434"
EXIT_OK = 0
EXIT_OLLAMA_UNREACHABLE = 2

# Role -> probe prompt. Small, schema-shaped where relevant.
ROLE_PROBES: dict[str, str] = {
    "planner": (
        "Break this request into atomic tasks as JSON: "
        '{"tasks": [{"id": 1, "name": "...", "target_files": ["..."], "spec": "..."}]}. '
        "Request: create a Python CLI that greets the user by name."
    ),
    "coder": (
        "Return one ```FILE: hello.py``` block containing a function "
        "def greet(name: str) -> str that returns a greeting."
    ),
    "validator": (
        "Reply PASS or CONCERNS on the first line, then one short reason. "
        "Diff: +def greet(name): return f'Hello, {name}!'. Spec: greet by name."
    ),
}


@dataclass
class RoleResult:
    role: str
    model: str
    ok: bool
    seconds: float = 0.0
    output_chars: int = 0
    error: str = ""
    score: float = 0.0  # higher = better (throughput proxy)


@dataclass
class BenchReport:
    workspace: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    results: list[RoleResult] = field(default_factory=list)
    best_per_role: dict[str, str] = field(default_factory=dict)
    coexist_verdict: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "started_at": self.started_at,
            "results": [r.__dict__ for r in self.results],
            "best_per_role": self.best_per_role,
            "coexist_verdict": self.coexist_verdict,
            "errors": self.errors,
        }


def _role_models(config: AgentConfig, args: argparse.Namespace) -> dict[str, list[str]]:
    """Map each role to the model(s) to probe."""
    roles: dict[str, list[str]] = {
        "planner": [config.planner_model],
        "coder": [config.coder_model],
        "validator": [config.planner_model],
    }
    for override in args.compare or []:
        if "=" in override:
            role, models = override.split("=", 1)
            role = role.strip()
            if role in roles:
                roles[role] = [m.strip() for m in models.split(",") if m.strip()]
    if args.roles:
        keep = {r.strip() for r in args.roles.split(",") if r.strip()}
        roles = {r: m for r, m in roles.items() if r in keep}
    return roles


def _probe_once(
    client: httpx.Client, base: str, model: str, prompt: str, num_predict: int
) -> tuple[bool, float, int, str]:
    """One /api/generate probe. Returns (ok, seconds, output_chars, error)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    start = time.monotonic()
    try:
        resp = client.post(f"{base}/api/generate", json=payload, timeout=300.0)
        if resp.status_code != 200:
            return False, 0.0, 0, f"HTTP {resp.status_code}"
        data = resp.json()
        seconds = round(time.monotonic() - start, 3)
        return True, seconds, len(data.get("response", "")), ""
    except httpx.HTTPError as exc:
        return False, round(time.monotonic() - start, 3), 0, str(exc)


def _installed_models(client: httpx.Client, base: str) -> list[str]:
    try:
        resp = client.get(f"{base}/api/tags", timeout=10.0)
        if resp.status_code == 200:
            return [m.get("name", "") for m in resp.json().get("models", [])]
    except httpx.HTTPError:
        pass
    return []


def _unload_models(client: httpx.Client, base: str, models: list[str]) -> None:
    """Best-effort unload (keep_alive=0) after benchmarking."""
    for model in models:
        try:
            client.post(
                f"{base}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
                timeout=15.0,
            )
        except httpx.HTTPError:
            pass


def _write_reports(report: BenchReport, workspace: Path) -> tuple[Path, Path]:
    bench_dir = workspace / ".agent_runs" / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = bench_dir / f"bench-{stamp}.json"
    md_path = bench_dir / f"bench-{stamp}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    lines = [
        f"# Benchmark report — {report.started_at}",
        f"Workspace: {report.workspace}",
        "",
        "| Role | Model | OK | Seconds | Chars | Score | Error |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report.results:
        lines.append(
            f"| {r.role} | {r.model} | {'yes' if r.ok else 'no'} | "
            f"{r.seconds:.2f} | {r.output_chars} | {r.score:.1f} | {r.error} |"
        )
    lines.append("")
    lines.append("## Best per role")
    for role, model in report.best_per_role.items():
        lines.append(f"- **{role}**: `{model}`")
    if report.coexist_verdict:
        lines.append(f"\nCoexist verdict: {report.coexist_verdict}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _print_table(report: BenchReport) -> None:
    print("\nBenchmark results")
    print(f"{'Role':<12}{'Model':<30}{'OK':<5}{'Secs':>8}{'Chars':>8}{'Score':>8}  Error")
    for r in report.results:
        print(
            f"{r.role:<12}{r.model:<30}{str(r.ok):<5}{r.seconds:>8.2f}"
            f"{r.output_chars:>8}{r.score:>8.1f}  {r.error}"
        )
    print("\nBest per role:")
    for role, model in report.best_per_role.items():
        print(f"  {role}: {model}")
    if report.coexist_verdict:
        print(f"\nCoexist verdict: {report.coexist_verdict}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="Model role benchmark")
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--quick", action="store_true", help="1 repeat, small probes")
    parser.add_argument("--roles", default=None, help="comma list: planner,coder,validator")
    parser.add_argument(
        "--compare", action="append", default=None,
        help="role=model1,model2 (repeatable)",
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--plan-mode", default="single", choices=["single", "two-stage"])
    parser.add_argument("--coexist", action="store_true", help="test model coexistence")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    return parser


def run(argv: list[str] | None = None) -> tuple[dict[str, Any], int]:
    """Execute the benchmark. Prints progress; returns (report_dict, exit_code)."""
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace).resolve()
    config = load_agent_config(workspace)
    report = BenchReport(workspace=str(workspace))

    client = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
    installed = _installed_models(client, args.ollama_url)
    if not installed:
        msg = f"Ollama unreachable at {args.ollama_url}"
        print(f"ERROR: {msg}")
        report.errors.append(msg)
        return report.to_dict(), EXIT_OLLAMA_UNREACHABLE

    repeat = 1 if args.quick else max(1, args.repeat)
    num_predict = 256 if args.quick else 512
    role_models = _role_models(config, args)
    probed_models: list[str] = []

    for role, models in role_models.items():
        prompt = ROLE_PROBES.get(role, ROLE_PROBES["coder"])
        for model in models:
            if model not in installed:
                hint = f"missing model {model} — run: ollama pull {model}"
                print(f"SKIP {role}/{model}: {hint}")
                report.results.append(
                    RoleResult(role=role, model=model, ok=False, error=hint)
                )
                continue
            probed_models.append(model)
            best_score = -1.0
            total_seconds = 0.0
            ok_any = False
            error = ""
            for _ in range(repeat):
                ok, seconds, chars, error = _probe_once(
                    client, args.ollama_url, model, prompt, num_predict
                )
                ok_any = ok_any or ok
                total_seconds += seconds
                if ok and seconds > 0:
                    best_score = max(best_score, chars / seconds)
            avg = round(total_seconds / repeat, 3)
            result = RoleResult(
                role=role, model=model, ok=ok_any,
                seconds=avg, output_chars=0, error="" if ok_any else error,
                score=round(best_score, 1) if best_score > 0 else 0.0,
            )
            report.results.append(result)
            print(f"{role}/{model}: {'ok' if ok_any else 'FAIL'} ({avg}s avg)")

    # Best per role = highest throughput score among ok results
    for role in role_models:
        candidates = [r for r in report.results if r.role == role and r.ok]
        if candidates:
            report.best_per_role[role] = max(candidates, key=lambda r: r.score).model

    if args.coexist:
        # With OLLAMA_MAX_LOADED_MODELS=1 the expectation is sequential residency.
        report.coexist_verdict = (
            "sequential residency enforced (OLLAMA_MAX_LOADED_MODELS=1); "
            "models swap rather than coexist"
        )

    json_path, md_path = _write_reports(report, workspace)
    _print_table(report)
    print(f"\nReports: {json_path}\n         {md_path}")
    print("Note: benchmark models are unloaded when done (keep_alive=0).")

    _unload_models(client, args.ollama_url, probed_models)
    return report.to_dict(), EXIT_OK


def main(argv: list[str] | None = None) -> int:
    _, exit_code = run(argv)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
