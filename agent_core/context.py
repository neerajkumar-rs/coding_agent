"""Workspace context: AST repo map + token budgeter.

Budget rule: rough estimate ``tokens = chars / 3.5``; prompt inputs must stay
within 85% of ``num_ctx`` (reserve the rest for output). The estimator is a
single constant so it can be swapped for a real tokenizer later.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

WORKSPACE_FILE_SUFFIXES = {".py", ".html", ".css", ".js", ".json", ".md", ".txt"}
MAX_WORKSPACE_FILES = 60
MAX_WORKSPACE_CONTEXT_CHARS = 12_000
MAX_FILE_SNIPPET_CHARS = 3_000
MAX_REPO_MAP_CHARS = 6_000

CHARS_PER_TOKEN = 3.5
CONTEXT_SAFETY_FRACTION = 0.85


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 3.5). Replaceable estimator."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


class ContextBudget:
    """Enforces the 85% rule: inputs may use at most 85% of num_ctx, minus
    the tokens reserved for the model's reply."""

    def __init__(self, num_ctx: int, reserve_output_tokens: int = 0) -> None:
        self.num_ctx = num_ctx
        self.input_token_limit = int(num_ctx * CONTEXT_SAFETY_FRACTION) - reserve_output_tokens

    @property
    def input_char_limit(self) -> int:
        return max(0, int(self.input_token_limit * CHARS_PER_TOKEN))

    def fits(self, text: str) -> bool:
        return estimate_tokens(text) <= self.input_token_limit

    def trim(self, text: str) -> str:
        """Hard-trim text to the input budget (char-accurate, tail-marked)."""
        limit = self.input_char_limit
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 40)] + "\n...[trimmed to fit context]"


def build_repo_map(workspace: Path, max_chars: int = MAX_REPO_MAP_CHARS) -> str:
    """AST-based map of Python files: top-level classes/functions/signatures.

    Cheaper and denser than raw snippets for orienting the planner.
    """
    entries: list[str] = []
    for path in list_workspace_files(workspace):
        if path.suffix.lower() != ".py":
            continue
        relative = path.relative_to(workspace).as_posix()
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="replace"), filename=relative
            )
        except (SyntaxError, OSError):
            continue
        lines: list[str] = [f"{relative}:"]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = ", ".join(a.arg for a in node.args.args)
                lines.append(f"  def {node.name}({args})  # line {node.lineno}")
            elif isinstance(node, ast.ClassDef):
                bases = ", ".join(
                    ast.unparse(b) if hasattr(ast, "unparse") else "?" for b in node.bases
                )
                lines.append(f"  class {node.name}({bases})  # line {node.lineno}")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = ", ".join(a.arg for a in item.args.args)
                        lines.append(f"    def {item.name}({args})  # line {item.lineno}")
        if len(lines) > 1:
            entries.append("\n".join(lines))
    return "\n".join(entries)[:max_chars]


def list_workspace_files(workspace: Path) -> list[Path]:
    files: list[Path] = []
    for path in workspace.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in WORKSPACE_FILE_SUFFIXES
            and not any(
                part.startswith(".") for part in path.relative_to(workspace).parts
            )
        ):
            files.append(path)
    return sorted(files)


def build_workspace_context(
    workspace: Path,
    request_text: str,
    report: Callable[[str], None] | None = None,
    budget: ContextBudget | None = None,
) -> str:
    """Repo map + bounded relevant snippets, trimmed to the token budget."""
    files = list_workspace_files(workspace)
    if not files:
        return "(workspace is empty)"

    char_limit = (
        min(budget.input_char_limit, MAX_WORKSPACE_CONTEXT_CHARS)
        if budget
        else MAX_WORKSPACE_CONTEXT_CHARS
    )
    sections: list[str] = [f"FILES ({len(files)} found):"]
    used = len(sections[0])

    repo_map = build_repo_map(workspace)
    if repo_map:
        section = f"\nREPO MAP (Python symbols):\n{repo_map}"
        if used + len(section) <= char_limit:
            sections.append(section)
            used += len(section)

    request_lower = request_text.lower()
    ranked = sorted(
        files,
        key=lambda path: (
            0 if path.name.lower() in request_lower else 1,
            0 if path.suffix.lower() in {".py", ".html", ".js", ".css"} else 1,
            str(path),
        ),
    )
    for path in ranked[:MAX_WORKSPACE_FILES]:
        relative = path.relative_to(workspace)
        header = f"- {relative}"
        sections.append(header)
        used += len(header) + 1
        if used >= char_limit:
            break
        try:
            if report:
                report(f"tool • filesystem.read_snippet: {path}")
            snippet = path.read_text(encoding="utf-8", errors="replace")[
                :MAX_FILE_SNIPPET_CHARS
            ]
        except OSError:
            continue
        if snippet:
            section = f"\n[{relative}]\n{snippet}"
            allowed = char_limit - used
            sections.append(section[:allowed])
            used += min(len(section), allowed)
    return "\n".join(sections)[:char_limit]
