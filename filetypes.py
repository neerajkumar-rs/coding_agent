"""The ONE file-type registry: extension -> {language, text, prose, hint, check}.

Every module that maps a file extension to a language, decides whether a file
is text, picks a coder hint, or runs a per-type check imports from here. Never
a second table (the old five copies drifted: a .md write passed the plan gate
but crashed safe_target).

``check(rel_path, content)`` returns an error string or None. It NEVER blocks
a write on its own — it informs the caller (the pipeline stages the file and
reports the error). Checkers that need a binary (node, rustc, bash) report
"checker not installed" — a skipped check, never a failure — when the binary
is not on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Per-type checks (each returns an error string or None)
# ---------------------------------------------------------------------------

# Alias so tests can patch the compile call at filetypes.compile (the registry
# runs the .py check now; orchestrator.compile is no longer the seam).
compile = compile  # type: ignore[assignment]  # re-exported for the patch seam


def _check_py(rel_path: str, content: str) -> str | None:
    if not content.strip():
        return "file is empty"
    try:
        compile(content, rel_path, "exec")
    except SyntaxError as exc:
        return f"syntax error: {exc.msg} (line {exc.lineno})"
    return None


def _check_json(rel_path: str, content: str) -> str | None:
    if not content.strip():
        return "file is empty"
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return f"invalid JSON: {exc}"
    return None


def _check_toml(rel_path: str, content: str) -> str | None:
    if not content.strip():
        return "file is empty"
    try:
        import tomllib  # py3.11+
    except ImportError:
        return None  # no parser in this Python: the check is skipped, not failed
    try:
        tomllib.loads(content)
    except Exception as exc:
        return f"invalid TOML: {exc}"
    return None


def _check_yaml(rel_path: str, content: str) -> str | None:
    if not content.strip():
        return "file is empty"
    try:
        import yaml  # type: ignore
    except ImportError:
        return None  # PyYAML absent: skipped, not failed
    try:
        yaml.safe_load(content)
    except Exception as exc:
        return f"invalid YAML: {exc}"
    return None


def _fence_lines(content: str) -> int:
    return sum(1 for ln in content.splitlines() if ln.strip().startswith("```"))


def _check_prose(rel_path: str, content: str) -> str | None:
    """md/txt: non-empty + balanced fences (an odd fence count means the model
    was cut off mid-block). Code fences are LEGAL in prose (a README with a
    ```bash section must pass)."""
    if not content.strip():
        return "file is empty"
    if _fence_lines(content) % 2:
        return "unbalanced markdown fences (odd ``` count — truncated output?)"
    return None


def _check_html(rel_path: str, content: str) -> str | None:
    if not content.strip():
        return "file is empty"
    if "```" in content:
        return "leftover markdown fences (```) in file content"
    lower = content.lower()
    if "<html" not in lower and "<!doctype" not in lower:
        return "HTML file contains neither <html> nor <!doctype"
    if "<body" not in lower:
        return "HTML file has no <body>"
    if len(content.strip()) < 40:
        return "HTML file is too short to be a real page"
    return None


def _check_css(rel_path: str, content: str) -> str | None:
    if not content.strip():
        return "file is empty"
    if "```" in content:
        return "leftover markdown fences (```) in file content"
    if "{" not in content or "}" not in content:
        return "CSS file contains no rule blocks"
    return None


def _binary_check(binary: str, args: list[str], rel_path: str, content: str) -> str | None:
    """Run a language checker binary when on PATH; otherwise a skipped note
    (never a failure)."""
    if not content.strip():
        return "file is empty"
    if "```" in content:
        return "leftover markdown fences (```) in file content"
    if shutil.which(binary) is None:
        return None  # checker not installed: the check is skipped
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=Path(rel_path).suffix, delete=False, encoding="utf-8"
        ) as fh:
            fh.write(content)
            tmp = fh.name
        out = subprocess.run(
            [binary, *args, tmp], capture_output=True, text=True, timeout=30,
            check=False,
        )
        if out.returncode != 0:
            return f"{binary} check failed: {(out.stderr or out.stdout).strip()[:300]}"
        return None
    except Exception as exc:
        return f"{binary} check could not run: {exc}"
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except OSError:
            pass


def _check_js(rel_path: str, content: str) -> str | None:
    if not content.strip():
        return "file is empty"
    if "```" in content:
        return "leftover markdown fences (```) in file content"
    plausible = ("function", "=>", "const ", "let ", "var ", "document.", "module")
    if not any(token in content for token in plausible):
        return "JS file has no recognizable JavaScript constructs"
    if shutil.which("node"):
        return _binary_check("node", ["--check"], rel_path, content)
    return None


def _check_sh(rel_path: str, content: str) -> str | None:
    if shutil.which("bash"):
        return _binary_check("bash", ["-n"], rel_path, content)
    return _check_prose(rel_path, content)


def _check_rs(rel_path: str, content: str) -> str | None:
    if not content.strip():
        return "file is empty"
    if shutil.which("rustc"):
        return _binary_check("rustc", ["--emit=metadata"], rel_path, content)
    return None  # checker not installed


def _check_generic(rel_path: str, content: str) -> str | None:
    """Fallback for unregistered suffixes: non-empty + no stray wrapping fence."""
    if not content.strip():
        return "file is empty"
    if _fence_lines(content) % 2:
        return "unbalanced markdown fences (odd ``` count — truncated output?)"
    return None


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

# extension -> {language, text, prose, hint, check}
# text = safe to read into prompts; prose = the md/txt policy (balanced fences,
# code fences legal); hint = the one-line coder hint for the file type.
FILE_TYPES: dict[str, dict[str, Any]] = {
    ".py":   {"language": "python",     "text": True,  "prose": False,
              "hint": "a complete Python module",            "check": _check_py},
    ".js":   {"language": "javascript", "text": True,  "prose": False,
              "hint": "complete JavaScript",                  "check": _check_js},
    ".ts":   {"language": "typescript", "text": True,  "prose": False,
              "hint": "complete TypeScript",                  "check": _check_generic},
    ".html": {"language": "html",       "text": True,  "prose": False,
              "hint": "a complete HTML page (doctype + body)","check": _check_html},
    ".css":  {"language": "css",        "text": True,  "prose": False,
              "hint": "a complete stylesheet",                "check": _check_css},
    ".json": {"language": "json",       "text": True,  "prose": False,
              "hint": "valid JSON only",                      "check": _check_json},
    ".md":   {"language": "markdown",   "text": True,  "prose": True,
              "hint": "markdown prose; code blocks allowed",  "check": _check_prose},
    ".txt":  {"language": "text",       "text": True,  "prose": True,
              "hint": "plain text",                           "check": _check_prose},
    ".rs":   {"language": "rust",       "text": True,  "prose": False,
              "hint": "a complete Rust file",                 "check": _check_rs},
    ".toml": {"language": "toml",       "text": True,  "prose": False,
              "hint": "valid TOML only",                      "check": _check_toml},
    ".yaml": {"language": "yaml",       "text": True,  "prose": False,
              "hint": "valid YAML only",                      "check": _check_yaml},
    ".yml":  {"language": "yaml",       "text": True,  "prose": False,
              "hint": "valid YAML only",                      "check": _check_yaml},
    ".sh":   {"language": "bash",       "text": True,  "prose": False,
              "hint": "a complete bash script",               "check": _check_sh},
    ".csv":  {"language": "csv",        "text": True,  "prose": False,
              "hint": "CSV rows only",                        "check": _check_generic},
    ".xml":  {"language": "xml",        "text": True,  "prose": False,
              "hint": "well-formed XML only",                 "check": _check_generic},
    ".sql":  {"language": "sql",        "text": True,  "prose": False,
              "hint": "SQL statements only",                  "check": _check_generic},
}

# The suffixes a generated file may have (the plan gate + safe_target use this;
# it is derived from the registry, never a second list).
ALLOWED_SUFFIXES: set[str] = set(FILE_TYPES)
# Text suffixes (safe to read into prompts), derived from the registry.
TEXT_SUFFIXES: set[str] = {ext for ext, e in FILE_TYPES.items() if e["text"]}
# Prose suffixes (the md/txt extraction + check policy).
PROSE_SUFFIXES: set[str] = {ext for ext, e in FILE_TYPES.items() if e["prose"]}


def entry_for(path: str) -> dict[str, Any]:
    """The registry entry for a path's suffix, or a generic fallback."""
    return FILE_TYPES.get(Path(path).suffix.lower(), {
        "language": "text", "text": True, "prose": False,
        "hint": "plain text", "check": _check_generic,
    })


def language_of(path: str) -> str:
    return str(entry_for(path)["language"])


def hint_for(path: str) -> str:
    """The one-line coder hint for the file type."""
    return str(entry_for(path)["hint"])


def check_file(rel_path: str, content: str) -> str | None:
    """The per-type check: an error string, or None when OK (or when the
    checker binary is not installed — a skipped check, never a failure)."""
    check = entry_for(rel_path)["check"]
    return check(rel_path, content)


def extract_content(raw: str, rel_path: str) -> str:
    """Per-type extraction from a model reply.

    Prose (.md/.txt): remove ONE wrapping fence only when the whole response is
    a single fenced block; inner fences stay. Everything else: the longest
    fenced block wins, or the stripped raw text when no fences exist.
    """
    import re

    if Path(rel_path).suffix.lower() in PROSE_SUFFIXES:
        lines = raw.strip().splitlines()
        if (
            len(lines) >= 2
            and lines[0].strip().startswith("```")
            and lines[-1].strip() == "```"
        ):
            return "\n".join(lines[1:-1]).rstrip() + "\n"
        return raw.strip() + "\n"
    matches = re.findall(r"```(?:\w+)?\n(.*?)```", raw, re.DOTALL)
    if matches:
        return max(matches, key=len).rstrip() + "\n"
    return raw.strip() + "\n"
