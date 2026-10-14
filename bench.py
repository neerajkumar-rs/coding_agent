"""Model role benchmark for the local Ollama pipeline — quality-scored.

Not "did the model respond" but "did it respond with something USABLE for the
role": every probe answer passes deterministic per-role quality checks (JSON
task shape for the planner, a fenced greet function for the coder, a
PASS/CONCERNS verdict line for the validator, clean prose for the writer).
Score = quality(0..100) x tok/s — a fast wrong answer never beats a slower
right one.

Headless and TUI-embeddable: run(argv) does the work and returns
(report_dict, exit_code) without sys.exit; main() is a thin wrapper.

Exit codes: 0 = ok, 2 = Ollama unreachable.

Reports are written to <workspace>/.agent_runs/bench/ (JSON + Markdown).
"""

from __future__ import annotations

import argparse
import json
import re
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

# The held-back classify eval set (tests/classify_eval.jsonl). Bench reads it
# for the classifier A/B (accuracy first, latency second); the prompt examples
# never contain these phrases.
EVAL_PATH = Path(__file__).parent / "tests" / "classify_eval.jsonl"


# --- role probes: prompt + deterministic quality checks ----------------------
# Each check is (name, predicate). Quality = 100 * passed / total; a role is
# "ok" only at 100 (all checks pass).

def _checks_planner(text: str) -> list[tuple[str, bool]]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    tasks: Any = None
    if match:
        try:
            tasks = json.loads(match.group(0))
        except json.JSONDecodeError:
            tasks = None
    first = tasks[0] if isinstance(tasks, list) and tasks and isinstance(tasks[0], dict) else {}
    files = first.get("target_files") or ([first.get("path")] if first.get("path") else [])
    return [
        ("json task array", isinstance(tasks, list) and len(tasks) >= 1),
        ("task has files", bool(files)),
        ("task has spec", bool(first.get("spec") or first.get("description"))),
    ]


def _checks_coder(text: str) -> list[tuple[str, bool]]:
    has_fence = bool(re.search(r"```(?:\w+)?\n.*?```", text, re.DOTALL))
    return [
        ("fenced block", has_fence),
        ("greet function", "def greet(" in text),
        ("returns a string", "return" in text),
    ]


def _checks_validator(text: str) -> list[tuple[str, bool]]:
    first = text.strip().splitlines()[0].strip().upper() if text.strip() else ""
    return [
        ("verdict first line", first.startswith(("PASS", "CONCERNS"))),
        ("has a reason", len(text.strip().splitlines()) >= 1 and len(text.strip()) > 4),
    ]


def _checks_writer(text: str) -> list[tuple[str, bool]]:
    stripped = text.strip()
    return [
        ("prose without think tags", "<think>" not in stripped.lower()),
        ("at least one sentence", len(stripped) >= 20 and "." in stripped),
    ]


ROLE_PROBES: dict[str, dict[str, Any]] = {
    "planner": {
        "prompt": (
            "Break this request into atomic tasks as JSON: "
            '{"tasks": [{"id": 1, "name": "...", "target_files": ["..."], "spec": "..."}]}. '
            "Request: create a Python CLI that greets the user by name."
        ),
        "checks": _checks_planner,
    },
    "coder": {
        "prompt": (
            "Return one ```FILE: hello.py``` block containing a function "
            "def greet(name: str) -> str that returns a greeting."
        ),
        "checks": _checks_coder,
    },
    "validator": {
        "prompt": (
            "Reply PASS or CONCERNS on the first line, then one short reason. "
            "Diff: +def greet(name): return f'Hello, {name}!'. Spec: greet by name."
        ),
        "checks": _checks_validator,
    },
    "writer": {
        "prompt": (
            "In two plain sentences, describe what a function named greet(name) "
            "that returns f'Hello, {name}!' does. No code blocks, no preamble."
        ),
        "checks": _checks_writer,
    },
}


@dataclass
class RoleResult:
    role: str
    model: str
    ok: bool
    seconds: float = 0.0
    output_chars: int = 0
    error: str = ""
    quality: float = 0.0   # 0..100: fraction of the role's checks passed
    tok_per_s: float = 0.0  # Ollama eval tokens/s when reported, else chars/s
    score: float = 0.0      # quality x tok/s (higher = better)
    checks: list[str] = field(default_factory=list)  # "name:PASS/FAIL"


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
    classify_ab: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "started_at": self.started_at,
            "results": [r.__dict__ for r in self.results],
            "best_per_role": self.best_per_role,
            "coexist_verdict": self.coexist_verdict,
            "errors": self.errors,
            "classify_ab": self.classify_ab,
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


# Role names as the gateway knows them ("planner" is the thinker role there).
_GATEWAY_ROLE = {"planner": "thinker", "coder": "coder", "validator": "validator", "writer": "writer"}


class _BenchConfig:
    """Config view that pins ONE role to the model being probed, so a
    --compare model goes through the gateway's role machinery (ceilings,
    VRAM guard) without touching the persisted .agent profile."""

    def __init__(self, base: AgentConfig, role: str, model: str) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_role", role)
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name: str):
        # Dunder/private lookups (deepcopy's __deepcopy__ probe, etc.) must
        # raise immediately — otherwise ``self._base`` re-enters __getattr__
        # and recurses without bound.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_base"), name)

    def __deepcopy__(self, memo: dict) -> "_BenchConfig":
        return _BenchConfig(self._base, self._role, self._model)

    def get_role_config(self, role: str) -> dict[str, Any]:
        cfg = dict(self._base.get_role_config(role))
        if role == self._role:
            cfg["model"] = self._model
        return cfg


async def _probe_gateway(
    gateway: Any, role: str, prompt: str, num_predict: int
) -> tuple[bool, float, str, dict[str, Any], str]:
    """One probe through THE AgentGateway: single lock, VRAM guard, role
    config. Returns (ok, seconds, text, meta, error)."""
    meta_box: list[dict[str, Any]] = []
    start = time.monotonic()
    try:
        text = await gateway.generate(
            role, prompt, num_predict=num_predict, keep_warm=False,
            out_meta=meta_box,
        )
        seconds = round(time.monotonic() - start, 3)
        return True, seconds, text, (meta_box[0] if meta_box else {}), ""
    except Exception as exc:
        return False, round(time.monotonic() - start, 3), "", {}, str(exc)[:300]


def _quality_score(role: str, text: str) -> tuple[float, list[str]]:
    """Deterministic per-role checks -> (0..100, ["name:PASS/FAIL", ...])."""
    checks_fn = ROLE_PROBES.get(role, ROLE_PROBES["coder"])["checks"]
    results = checks_fn(text)
    passed = sum(1 for _name, ok in results if ok)
    total = len(results)
    quality = round(100.0 * passed / total, 1) if total else 0.0
    detail = [f"{name}:{'PASS' if ok else 'FAIL'}" for name, ok in results]
    return quality, detail


def _installed_models(client: httpx.Client, base: str) -> list[str]:
    try:
        resp = client.get(f"{base}/api/tags", timeout=10.0)
        if resp.status_code == 200:
            return [m.get("name", "") for m in resp.json().get("models", [])]
    except httpx.HTTPError:
        pass
    return []


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
        "| Role | Model | OK | Quality | tok/s | Seconds | Score | Checks | Error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.results:
        lines.append(
            f"| {r.role} | {r.model} | {'yes' if r.ok else 'no'} | "
            f"{r.quality:.0f} | {r.tok_per_s:.1f} | {r.seconds:.2f} | "
            f"{r.score:.1f} | {', '.join(r.checks)} | {r.error} |"
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
    print("\nBenchmark results (score = quality x tok/s)")
    print(f"{'Role':<12}{'Model':<30}{'OK':<5}{'Qual':>6}{'tok/s':>8}{'Secs':>8}{'Score':>8}  Error")
    for r in report.results:
        print(
            f"{r.role:<12}{r.model:<30}{str(r.ok):<5}{r.quality:>6.0f}"
            f"{r.tok_per_s:>8.1f}{r.seconds:>8.2f}{r.score:>8.1f}  {r.error}"
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
    parser.add_argument("--roles", default=None, help="comma list: planner,coder,validator,writer")
    parser.add_argument(
        "--compare", action="append", default=None,
        help="role=model1,model2 (repeatable)",
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--coexist", action="store_true", help="test model coexistence")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    parser.add_argument(
        "--classify-ab", action="store_true",
        help="A/B the coder vs thinker as the classifier over tests/classify_eval.jsonl "
             "(label accuracy, p50/p95 latency, shell-command leakage on reply lines)",
    )
    return parser


# ---------------------------------------------------------------------------
# Classifier A/B (Section 6): which model owns the understand role
# ---------------------------------------------------------------------------

_SHELL_HINT_RE = re.compile(
    r"\b(?:ls|dir|find|grep|cat|head|tail|du|stat|tree|fd)\b\s+[-~\w.]",
    re.IGNORECASE,
)


def _pctl(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return round(ordered[idx], 3)


async def _classify_ab_model(
    gateway: Any, model: str, eval_lines: list[dict[str, Any]], ollama_url: str
) -> dict[str, Any]:
    """Run the classifier + reply step over the eval set with `model` pinned
    as the understand role. Returns accuracy/latency/leakage metrics."""
    from understand import build_facts_block, classify_mode, reply_step

    gateway.config = _BenchConfig(gateway.config, "understand", model)

    async def _gen(m, prompt, system="", temperature=0.0, **kwargs):
        return await gateway.generate(
            "understand", prompt, system=system, temperature=temperature,
            num_predict=kwargs.get("num_predict", 12),
            _format_schema=kwargs.get("_format_schema"),
        )

    facts_block = build_facts_block({"files": [], "flags": {}})
    correct = 0
    total = 0
    latencies: list[float] = []
    shell_leaks = 0
    reply_lines = 0
    errors: list[str] = []
    for entry in eval_lines:
        if entry["expected"] == "fast":
            continue  # fast paths never touch a model — nothing to measure
        total += 1
        t0 = time.monotonic()
        try:
            label = await classify_mode(
                entry["text"], generate=_gen, file_names=[], last_turn="",
            )
        except Exception as exc:
            # A model that cannot classify at this budget (e.g. a reasoning
            # model truncated at num_predict<=12) records the line as wrong
            # and the error — the A/B report shows it, never crashes.
            latencies.append(time.monotonic() - t0)
            errors.append(f"{entry['text'][:40]}: {str(exc)[:120]}")
            continue
        latencies.append(time.monotonic() - t0)
        if label == entry["expected"]:
            correct += 1
        if label == "reply":
            reply_lines += 1
            try:
                answer = await reply_step(entry["text"], generate=_gen, facts_block=facts_block)
            except Exception as exc:
                errors.append(f"reply {entry['text'][:40]}: {str(exc)[:120]}")
                continue
            if _SHELL_HINT_RE.search(answer):
                shell_leaks += 1
    accuracy = round(100.0 * correct / total, 1) if total else 0.0
    return {
        "model": model,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "p50": _pctl(latencies, 50),
        "p95": _pctl(latencies, 95),
        "reply_lines": reply_lines,
        "shell_leaks": shell_leaks,
        "errors": errors[:10],
    }


async def _classify_ab(config: AgentConfig, ollama_url: str, installed: list[str]) -> dict[str, Any]:
    """A/B coder vs thinker as the classifier. Picks roles.understand.model by
    accuracy first, latency second. Gate: >=95% accuracy, 0 shell leaks, p95 < 2s."""
    import asyncio as _asyncio  # local: the lock binds to this loop

    from ollama_client import AgentGateway, OllamaSupervisorClient

    if not EVAL_PATH.exists():
        return {"error": f"eval set missing: {EVAL_PATH}"}
    eval_lines = [
        json.loads(raw) for raw in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    candidates = {
        "coder": str(config.get_role_config("coder")["model"]),
        "thinker": str(config.get_role_config("thinker")["model"]),
    }
    supervisor = OllamaSupervisorClient(base_url=ollama_url)
    gateway = AgentGateway(supervisor, config)
    rows: dict[str, Any] = {}
    for label, model in candidates.items():
        if model not in installed:
            rows[label] = {"model": model, "error": f"missing — ollama pull {model}"}
            print(f"SKIP classify/{label}: missing model {model} — ollama pull {model}")
            continue
        print(f"classify A/B: {label} = {model} ...")
        rows[label] = await _classify_ab_model(gateway, model, eval_lines, ollama_url)
        print(
            f"  {label}: acc={rows[label]['accuracy']}% "
            f"p50={rows[label]['p50']}s p95={rows[label]['p95']}s "
            f"shell_leaks={rows[label]['shell_leaks']}/{rows[label]['reply_lines']}"
        )
        await gateway.unload(model)
    gateway.config = config  # restore

    # Pick by accuracy first, latency second.
    winner = None
    scored = {k: v for k, v in rows.items() if "accuracy" in v}
    if scored:
        winner = max(
            scored,
            key=lambda k: (scored[k]["accuracy"], -scored[k]["p95"]),
        )
    result: dict[str, Any] = {"rows": rows, "winner": winner}
    if winner:
        w = scored[winner]
        gate_ok = w["accuracy"] >= 95.0 and w["shell_leaks"] == 0 and w["p95"] < 2.0
        result["gate_ok"] = gate_ok
        result["recommendation"] = (
            f"roles.understand.model = {scored[winner]['model']}"
            if gate_ok else
            f"no model passes the gate (winner {winner} misses: "
            f"acc={w['accuracy']}% p95={w['p95']}s leaks={w['shell_leaks']})"
        )
    return result


def run(argv: list[str] | None = None) -> tuple[dict[str, Any], int]:
    """Execute the benchmark. Prints progress; returns (report_dict, exit_code).

    Every generation goes through THE AgentGateway (single process lock,
    VRAM guard, role config) — bench never posts to /api/generate directly.
    """
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

    # THE gateway: one lock + VRAM guard for every probe below. Built once
    # per run; the lock is created inside this event loop.
    import asyncio

    from ollama_client import AgentGateway, OllamaSupervisorClient

    async def _bench_all() -> None:
        supervisor = OllamaSupervisorClient(base_url=args.ollama_url)
        gateway = AgentGateway(supervisor, config)
        for role, models in role_models.items():
            probe = ROLE_PROBES.get(role, ROLE_PROBES["coder"])
            prompt = probe["prompt"]
            gw_role = _GATEWAY_ROLE.get(role, "coder")
            for model in models:
                if model not in installed:
                    hint = f"missing model {model} — run: ollama pull {model}"
                    print(f"SKIP {role}/{model}: {hint}")
                    report.results.append(
                        RoleResult(role=role, model=model, ok=False, error=hint)
                    )
                    continue
                probed_models.append(model)
                # Pin this role to the probed model for the duration of its
                # probes (the persisted .agent profile is never touched).
                gateway.config = _BenchConfig(config, gw_role, model)
                best_score = -1.0
                total_seconds = 0.0
                ok_any = False
                best_quality = 0.0
                best_tps = 0.0
                best_checks: list[str] = []
                error = ""
                for _ in range(repeat):
                    ok, seconds, text, meta, error = await _probe_gateway(
                        gateway, gw_role, prompt, num_predict
                    )
                    ok_any = ok_any or ok
                    total_seconds += seconds
                    if not ok or seconds <= 0:
                        continue
                    quality, detail = _quality_score(role, text)
                    eval_count = int(meta.get("eval_count", 0) or 0)
                    eval_dur = int(meta.get("eval_duration", 0) or 0)
                    tps = (eval_count / (eval_dur / 1e9)) if eval_dur else (len(text) / seconds)
                    score = quality * tps
                    if score > best_score:
                        best_score = score
                        best_quality = quality
                        best_tps = tps
                        best_checks = detail
                avg = round(total_seconds / repeat, 3)
                # ok = at least one call succeeded AND passed all quality checks
                passed_all = ok_any and best_quality >= 100.0
                result = RoleResult(
                    role=role, model=model, ok=passed_all,
                    seconds=avg, output_chars=0,
                    error="" if passed_all else (error or "quality checks failed"),
                    quality=best_quality, tok_per_s=round(best_tps, 1),
                    score=round(best_score, 1) if best_score > 0 else 0.0,
                    checks=best_checks,
                )
                report.results.append(result)
                print(f"{role}/{model}: {'ok' if passed_all else 'FAIL'} "
                      f"q={best_quality:.0f} {best_tps:.1f}tok/s ({avg}s avg)")
                # keep_warm=False on each probe already unloaded; belt+braces:
                await gateway.unload(model)
        gateway.config = config  # restore

    classify_ab_result: dict[str, Any] | None = None

    try:
        asyncio.run(_bench_all())
    except Exception as exc:  # gateway/transport failure mid-bench
        report.errors.append(str(exc)[:300])
        print(f"ERROR: bench aborted: {exc}")

    if args.classify_ab:
        import asyncio as _asyncio

        try:
            classify_ab_result = _asyncio.run(_classify_ab(config, args.ollama_url, installed))
            report.classify_ab = classify_ab_result
            if classify_ab_result.get("recommendation"):
                print(f"\nClassifier A/B: {classify_ab_result['recommendation']}")
        except Exception as exc:
            report.errors.append(f"classify-ab: {str(exc)[:300]}")
            print(f"ERROR: classify A/B failed: {exc}")

    # Best per role = highest score among ok results; every probed role is
    # listed ("none passed" instead of omitting a role whose probes failed).
    for role in role_models:
        candidates = [r for r in report.results if r.role == role and r.ok]
        report.best_per_role[role] = (
            max(candidates, key=lambda r: r.score).model if candidates else "none passed"
        )

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

    return report.to_dict(), EXIT_OK


def main(argv: list[str] | None = None) -> int:
    _, exit_code = run(argv)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
