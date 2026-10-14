"""One-command debug export: a shareable Markdown report (or zip, or eval
fixtures) for one or more turns.

Pure functions on paths (testable without the UI): the report, the zip and
the fixture lines are built from history.jsonl + the linked run folders.
No gateway/Ollama calls here except clearly-marked, failure-tolerant
environment probes (ollama version, /api/ps, nvidia-smi) used for the
Environment section; each degrades to "n/a".

Redaction is ON by default: user home/workspace paths, the user name in
paths, likely secrets (sk-..., ghp_..., AKIA..., Bearer ..., key/token/
secret/password assignments, private-key blocks, .env-style KEY=value) are
masked in the report, the history slice, run logs inside the zip, and
fixtures. --no-redact skips it (with a printed warning).

Outputs land only under <workspace>/.agent_runs/exports/ via timestamped
names (never overwrite).
"""

from __future__ import annotations

import json
import logging
import platform
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_core.run_state import RUNS_DIR_NAME
from history import get_turn, read_history

logger = logging.getLogger("export")

REPORT_CAP = 200 * 1024  # 200 KB
PROMPT_CAP = 4 * 1024    # 4 KB per fenced block
EVENTS_TAIL = 40

FULL_WARNING = "This export may contain code or secrets; review before sharing."


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

_SECRET_RES = [
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{10,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*['\"]?[^\s'\"]{4,}"
    ),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(r"(?m)^([A-Z][A-Z0-9_]{2,})=(\S+)$"),  # .env-style KEY=value
]


def _mask_secrets(text: str) -> str:
    out = text
    for pattern in _SECRET_RES:
        if pattern.pattern.startswith("(?m)^"):
            out = pattern.sub(lambda m: f"{m.group(1)}=<redacted>", out)
        elif "PRIVATE KEY" in pattern.pattern:
            out = pattern.sub("<redacted private key>", out)
        elif "api[_-]?key" in pattern.pattern or "token" in pattern.pattern:
            out = pattern.sub(lambda m: m.group(0).split("=")[0].split(":")[0] + "=<redacted>"
                              if ("=" in m.group(0) or ":" in m.group(0)) else "<redacted>", out)
        else:
            out = pattern.sub("<redacted>", out)
    return out


def redact(text: str, workspace: str | Path) -> str:
    """Mask the workspace path, the user home, the user name, and secrets."""
    out = text
    ws = str(Path(workspace).resolve())
    home = str(Path.home())
    if ws:
        out = out.replace(ws, "<WORKSPACE>")
    if home and home != ws:
        out = out.replace(home, "~")
    # user name in paths (C:\Users\<name>\... or /home/<name>/...)
    try:
        user = Path.home().name
        if user:
            out = re.sub(
                rf"(?i)(Users|home)[/\\]{re.escape(user)}\b",
                r"\1\<user>", out,
            )
    except Exception:
        pass
    return _mask_secrets(out)


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def select_turns(workspace: str | Path, target: str) -> list[dict[str, Any]]:
    """target: 'last' (default), a turn id, 'N' (last N), a range 'a-b', 'all'."""
    target = (target or "last").strip().lower()
    entries, _skipped = read_history(workspace)
    if not entries:
        return []
    if target == "last":
        return entries[-1:]
    if target == "all":
        return entries
    if re.fullmatch(r"\d+-\d+", target):
        a, b = (int(x) for x in target.split("-", 1))
        return [e for e in entries if a <= int(e.get("id", -1)) <= b]
    if re.fullmatch(r"\d+", target):
        n = int(target)
        by_id = get_turn(workspace, n)
        if by_id is not None:
            return [by_id]
        return entries[-n:]  # N = last N turns
    return entries[-1:]


# ---------------------------------------------------------------------------
# Environment (best-effort probes; each degrades to "n/a")
# ---------------------------------------------------------------------------

def _probe(cmd: list[str], timeout: float = 3.0) -> str:
    import subprocess

    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "n/a"


def build_environment(workspace: str | Path, config: Any | None = None) -> dict[str, Any]:
    """Environment section data. No gateway calls; probes never raise."""
    import ollama_client  # for the default host only

    env: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "workspace": str(Path(workspace).resolve()),
    }
    base = getattr(ollama_client, "DEFAULT_OLLAMA_HOST", "http://localhost:11434")
    env["ollama_url"] = base
    # ollama --version + reachability + /api/ps (all optional)
    env["ollama_version"] = _probe(["ollama", "--version"])
    try:
        import urllib.request

        with urllib.request.urlopen(f"{base}/api/ps", timeout=3) as r:
            ps = json.loads(r.read())
        env["ollama_reachable"] = True
        env["ps"] = [
            {"name": m.get("name"), "size": m.get("size"), "size_vram": m.get("size_vram")}
            for m in ps.get("models", [])
        ]
    except Exception:
        env["ollama_reachable"] = False
        env["ps"] = []
    env["gpu"] = _probe(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
         "--format=csv,noheader"]
    )
    if config is not None:
        try:
            from agent_config import ROLE_NAMES

            env["roles"] = {
                role: config.get_role_config(role) for role in ROLE_NAMES
            }
        except Exception:
            pass
    return env


# ---------------------------------------------------------------------------
# Likely-cause derivation (facts only)
# ---------------------------------------------------------------------------

def likely_cause(turn: dict[str, Any], run_dir: Path | None) -> str | None:
    """A 'Likely cause' line only when derivable from facts."""
    error = turn.get("error") or {}
    detail = str(error.get("detail", ""))
    stage = str(error.get("stage", ""))
    if not error:
        return None
    if "model not installed" in detail or "ModelNotInstalled" in detail:
        match = re.search(r"ollama pull (\S+)", detail)
        return f"missing model — run: ollama pull {match.group(1) if match else '?'}"
    if "VramGuardError" in detail or "GPU fraction" in detail:
        return f"VRAM guard: model did not fit at any context rung (stage {stage})"
    if "done_reason" in detail and "length" in detail:
        return "response truncated (done_reason=length) — prompt too large"
    if "failed validation" in detail or "valid plan" in detail:
        return f"validation failure at {stage}: {detail[:160]}"
    return f"{stage}: {detail[:160]}" if stage else detail[:160]


# ---------------------------------------------------------------------------
# The Markdown report
# ---------------------------------------------------------------------------

def _run_dir(workspace: str | Path, run_id: str | None) -> Path | None:
    if not run_id:
        return None
    path = Path(workspace).resolve() / RUNS_DIR_NAME / run_id
    return path if path.is_dir() else None


def _cap(text: str, limit: int, full_path: Path | None = None) -> str:
    if len(text) <= limit:
        return text
    note = f"\n(+{len(text) - limit} chars truncated"
    if full_path is not None:
        note += f", full text in {full_path}"
    return text[:limit] + note + ")"


def _turn_section(workspace: str | Path, turn: dict[str, Any], *, redact_on: bool,
                  full: bool) -> str:
    ws = Path(workspace).resolve()
    run_dir = _run_dir(ws, turn.get("run_id"))
    lines: list[str] = []
    lines.append(f"\n## Turn {turn.get('id', '?')} — {turn.get('path', '?')}"
                 f" ({turn.get('outcome', '?')})")
    lines.append(f"- time: {turn.get('ts', '?')}")
    lines.append(f"- label: {turn.get('label') or '—'}")
    lines.append(f"- seconds: {turn.get('seconds', '?')}  llm_calls: {turn.get('llm_calls', '?')}")
    lines.append(f"- run_id: {turn.get('run_id', '?')}"
                 + (f"  (folder: {run_dir})" if run_dir else ""))
    if turn.get("models"):
        models = "; ".join(f"{r}={m}" for r, m in turn["models"].items())
        lines.append(f"- models: {models}")
    lines.append("\n**User text**\n")
    text = str(turn.get("text", ""))
    lines.append("```\n" + _cap(text, PROMPT_CAP) + "\n```")

    snap = turn.get("state_snapshot") or {}
    if snap:
        lines.append("\n<details><summary>State snapshot</summary>\n")
        lines.append("```json\n" + _cap(json.dumps(snap, indent=2), PROMPT_CAP) + "\n```\n</details>")

    trace = turn.get("trace") or []
    if trace:
        lines.append("\n**Trace**\n")
        lines.append("| stage | status | seconds | model |")
        lines.append("|---|---|---|---|")
        for t in trace:
            lines.append(
                f"| {t.get('stage', '?')} | {t.get('status', '?')} | "
                f"{t.get('seconds', '?')} | {t.get('model', '')} |"
            )

    # Per-call detail from the journal (role logs: prompt + response + meta)
    if run_dir is not None:
        calls_dir = run_dir / "calls"
        call_files = sorted(calls_dir.glob("*.json")) if calls_dir.is_dir() else []
        if call_files:
            lines.append("\n**LLM calls**\n")
            for call_file in call_files:
                try:
                    call = json.loads(call_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                lines.append(
                    f"- `{call.get('role', '?')}` model={call.get('model', '?')} "
                    f"num_ctx={call.get('num_ctx', '?')} "
                    f"prompt_tokens={call.get('prompt_eval_count', '?')} "
                    f"output_tokens={call.get('eval_count', '?')} "
                    f"done_reason={call.get('done_reason', '?')} "
                    f"seconds={call.get('seconds', '?')}"
                )
                prompt_text = str(call.get("prompt", ""))
                response_text = str(call.get("response", ""))
                lines.append("  prompt:\n  ```\n" + _cap(prompt_text, PROMPT_CAP, call_file) + "\n  ```")
                lines.append("  response:\n  ```\n" + _cap(response_text, PROMPT_CAP, call_file) + "\n  ```")

    if turn.get("files_planned"):
        lines.append("\n**Plan**: " + ", ".join(turn["files_planned"]))
    if turn.get("files_staged"):
        lines.append("**Staged**: " + ", ".join(turn["files_staged"]))
    if turn.get("answer_preview"):
        lines.append("\n**Answer**\n")
        lines.append("```\n" + _cap(str(turn["answer_preview"]), PROMPT_CAP) + "\n```")
    if turn.get("error"):
        err = turn["error"]
        lines.append(f"\n**Error** — stage `{err.get('stage', '?')}`: {err.get('detail', '')}")

    # Events tail + traceback from the run folder
    if run_dir is not None:
        events_path = run_dir / "events.jsonl"
        if events_path.exists():
            try:
                raw = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except OSError:
                raw = []
            tail = raw[-EVENTS_TAIL:]
            if tail:
                lines.append(f"\n<details><summary>Events tail ({len(tail)})</summary>\n")
                lines.append("```jsonl\n" + "\n".join(tail) + "\n```\n</details>")
            for ln in raw:
                try:
                    ev = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if ev.get("traceback"):
                    lines.append("\n**Traceback**\n")
                    lines.append("```\n" + _cap(str(ev["traceback"]), PROMPT_CAP) + "\n```")
                    break
    section = "\n".join(lines)
    if redact_on:
        section = redact(section, ws)
    return section


def build_report(
    workspace: str | Path,
    turns: list[dict[str, Any]],
    *,
    redact_on: bool = True,
    full: bool = False,
    environment: dict[str, Any] | None = None,
) -> str:
    """The Markdown debug report (short first; capped at REPORT_CAP keeping
    the Summary, Environment and the failing turn in full)."""
    ws = Path(workspace).resolve()
    parts: list[str] = []
    parts.append("# Debug report\n")
    parts.append(f"generated: {datetime.now(timezone.utc).isoformat()}\n")

    # 1. Summary
    parts.append("## Summary\n")
    parts.append("| id | time | path | label | outcome | seconds | error |")
    parts.append("|---|---|---|---|---|---|---|")
    for t in turns:
        err = t.get("error") or {}
        parts.append(
            f"| {t.get('id', '?')} | {t.get('ts', '?')} | {t.get('path', '?')} | "
            f"{t.get('label') or '—'} | {t.get('outcome', '?')} | "
            f"{t.get('seconds', '?')} | {err.get('stage', '') or '—'} |"
        )
    for t in turns:
        cause = likely_cause(t, _run_dir(ws, t.get("run_id")))
        if cause:
            parts.append(f"\n**Likely cause (turn {t.get('id', '?')}):** {cause}")

    # 2. Environment
    env = environment or {}
    parts.append("\n## Environment\n")
    parts.append(f"- build_id: {env.get('build_id', 'n/a')}")
    parts.append(f"- OS: {env.get('os', 'n/a')}   python: {env.get('python', 'n/a')}")
    parts.append(f"- ollama: {env.get('ollama_version', 'n/a')} "
                 f"(reachable: {env.get('ollama_reachable', 'n/a')})")
    ps = env.get("ps") or []
    if ps:
        for m in ps:
            parts.append(f"  - resident: {m.get('name')} size={m.get('size')} size_vram={m.get('size_vram')}")
    parts.append(f"- gpu: {env.get('gpu', 'n/a')}")
    roles = env.get("roles") or {}
    if roles:
        parts.append("- roles:")
        for role, cfg in roles.items():
            parts.append(
                f"  - {role}: {cfg.get('model')} num_ctx={cfg.get('num_ctx')} "
                f"num_predict={cfg.get('num_predict')} temp={cfg.get('temperature')}"
            )
    parts.append(f"- workspace: {env.get('workspace', str(ws))}")

    # 3+. Per turn (failing turns first so they survive the cap)
    ordered = sorted(turns, key=lambda t: 0 if t.get("error") else 1)
    body = "".join(
        _turn_section(ws, t, redact_on=redact_on, full=full) for t in ordered
    )

    # 5. Reproduce
    repro_lines = ["\n## Reproduce\n"]
    for t in turns:
        repro_lines.append(
            f"- turn {t.get('id', '?')}: /retry {t.get('id', '?')}  "
            f"(re-sends the text as a NEW turn; old state is not restored)"
        )
    repro = "\n".join(repro_lines)

    report = "\n".join(parts) + "\n" + body + "\n" + repro + "\n"
    if redact_on:
        report = redact(report, ws)
    if len(report) > REPORT_CAP:
        head = "\n".join(parts)
        failing = "".join(
            _turn_section(ws, t, redact_on=redact_on, full=full)
            for t in ordered if t.get("error")
        )
        report = (head + "\n" + failing + "\n"
                  + f"\n_(trimmed to {REPORT_CAP // 1024} KB: Summary, Environment and the "
                    "failing turn kept in full; other turns trimmed)_\n")
    return report


# ---------------------------------------------------------------------------
# Outputs (report / zip / fixture) — only under .agent_runs/exports/
# ---------------------------------------------------------------------------

def exports_dir(workspace: str | Path) -> Path:
    path = Path(workspace).resolve() / RUNS_DIR_NAME / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def write_report(workspace: str | Path, report: str) -> Path:
    path = exports_dir(workspace) / f"debug-{_stamp()}.md"
    path.write_text(report, encoding="utf-8")
    return path


def build_fixture_lines(turns: list[dict[str, Any]], *, redact_on: bool = True,
                        workspace: str | Path = ".") -> list[str]:
    """One JSON line per turn in the eval format (text, state, expected,
    observed_ok: null) for paste-into-classify_eval.jsonl."""
    lines: list[str] = []
    for t in turns:
        entry = {
            "text": t.get("text", ""),
            "state": t.get("state_snapshot") or {},
            "expected": t.get("label") or t.get("path"),
            "observed_ok": None,
        }
        line = json.dumps(entry, ensure_ascii=False)
        lines.append(redact(line, workspace) if redact_on else line)
    return lines


def write_fixtures(workspace: str | Path, turns: list[dict[str, Any]],
                   *, redact_on: bool = True) -> Path:
    path = exports_dir(workspace) / f"fixtures-{_stamp()}.jsonl"
    lines = build_fixture_lines(turns, redact_on=redact_on, workspace=workspace)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def do_export(
    workspace: str | Path,
    target: str = "last",
    *,
    full: bool = False,
    zip_out: bool = False,
    fixture: bool = False,
    no_redact: bool = False,
    config: Any | None = None,
    build_id: str = "",
) -> dict[str, Any]:
    """The shared export entry: selects turns, builds the output, writes it
    under .agent_runs/exports/. Returns {path, kind, size, warning?, turns}.
    Never raises on missing history (returns {"error": ...} instead)."""
    ws = Path(workspace).resolve()
    turns = select_turns(ws, target)
    if not turns:
        return {"error": f"no history for target {target!r} (no turns recorded yet)",
                "path": None}
    redact_on = not no_redact
    warning = FULL_WARNING if (full or no_redact) else ""
    env = build_environment(ws, config)
    if build_id:
        env["build_id"] = build_id

    if fixture:
        path = write_fixtures(ws, turns, redact_on=redact_on)
        return {
            "path": str(path), "kind": "fixture",
            "size": path.stat().st_size, "turns": len(turns),
            "warning": warning,
            "note": ("Review each line, set observed_ok true/false, and paste "
                     "the corrected lines into tests/classify_eval.jsonl."),
        }

    report = build_report(ws, turns, redact_on=redact_on, full=full, environment=env)
    path = write_report(ws, report)
    result: dict[str, Any] = {
        "path": str(path), "kind": "report",
        "size": path.stat().st_size, "turns": len(turns),
        "report": report,  # for clipboard copy at the caller
        "warning": warning,
    }
    if zip_out:
        zip_path = write_zip(ws, turns, report, redact_on=redact_on, environment=env)
        result["zip_path"] = str(zip_path)
        result["zip_size"] = zip_path.stat().st_size
    return result


def write_zip(workspace: str | Path, turns: list[dict[str, Any]], report: str,
              *, redact_on: bool = True,
              environment: dict[str, Any] | None = None) -> Path:
    """report + history slice + each linked run folder + config snapshot +
    environment.json, zipped under exports/."""
    ws = Path(workspace).resolve()
    path = exports_dir(ws) / f"debug-{_stamp()}.zip"
    slice_lines = []
    for t in turns:
        line = json.dumps(t, ensure_ascii=False)
        slice_lines.append(redact(line, ws) if redact_on else line)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.md", report)
        zf.writestr("history-slice.jsonl", "\n".join(slice_lines) + "\n")
        zf.writestr("environment.json", json.dumps(environment or {}, indent=2))
        config_path = ws / ".agent"
        if config_path.exists():
            try:
                cfg_text = config_path.read_text(encoding="utf-8")
                zf.writestr("config-snapshot.json",
                            redact(cfg_text, ws) if redact_on else cfg_text)
            except OSError:
                pass
        for t in turns:
            run_dir = _run_dir(ws, t.get("run_id"))
            if run_dir is None:
                continue
            for file in run_dir.rglob("*"):
                if not file.is_file():
                    continue
                try:
                    text = file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                arc = f"runs/{run_dir.name}/{file.relative_to(run_dir).as_posix()}"
                zf.writestr(arc, redact(text, ws) if redact_on else text)
    return path
