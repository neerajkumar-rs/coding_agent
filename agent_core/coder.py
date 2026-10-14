"""Coder: turns one task spec into concrete file contents.

Parses the model reply as either a JSON object ({"filename", "code"} or
{"files": [...]}) or ```FILE: blocks. Anything else is unusable and raises
ValueError — the caller FAILS the attempt; no scaffolds, no stubs.
(Phase 3 adds SEARCH/REPLACE edit blocks for editing existing files.)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ollama_client import TEMPERATURE_CODE

# Suffixes a generated file may have. Derived from the one file-type registry
# (filetypes.py): the plan gate, safe_target and the coder parser all enforce
# exactly this set — never a second list.
from filetypes import ALLOWED_SUFFIXES as ALLOWED_TARGET_SUFFIXES
MAX_CODE_TOKENS = 1800

CODER_PROMPT_TEMPLATE = (
    "You are a pure code generator. Use ONLY this specification. Return "
    "one JSON object with this shape: {\"filename\": \"relative/path.ext\", "
    "\"description\": \"brief purpose\", \"code\": \"complete file contents\"}. "
    "If multiple files are required, return {\"files\": [objects with those "
    "three fields]}. Return JSON only, use relative paths, and do not invent "
    "requirements.\n\nSPECIFICATION:\n"
)


class LLM(Protocol):
    def text(self, model: str, prompt: str, **kwargs: Any) -> str: ...


# ----------------------------------------------------------------------
# SEARCH/REPLACE edit blocks (aider-style)
# ----------------------------------------------------------------------

SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"


class EditApplyError(ValueError):
    """A SEARCH/REPLACE block could not be applied (no match or ambiguous)."""


@dataclass
class FileEdit:
    """Parsed coder output for one file: either full content or edit blocks."""

    path: str
    full_content: str | None = None
    edits: list[tuple[str, str]] = field(default_factory=list)


def _norm_lines(text: str) -> list[str]:
    """Whitespace-normalized lines for fuzzy matching."""
    return [line.strip() for line in text.strip().splitlines()]


def apply_search_replace(original: str, search: str, replace: str) -> str:
    """Apply one SEARCH/REPLACE block. Exact match first, then a
    whitespace-normalized match. Raises EditApplyError on no match or on
    more than one candidate location."""
    if not search.strip():
        raise EditApplyError("empty SEARCH block")

    # 1) Exact match — must be unique.
    count = original.count(search)
    if count == 1:
        return original.replace(search, replace, 1)
    if count > 1:
        raise EditApplyError(
            f"SEARCH block is ambiguous ({count} exact matches); add more context"
        )

    # 2) Whitespace-normalized fuzzy match over line spans.
    orig_lines = original.splitlines()
    search_lines = _norm_lines(search)
    n = len(search_lines)
    norm_orig = [line.strip() for line in orig_lines]
    matches: list[int] = []
    for start in range(0, len(orig_lines) - n + 1):
        if norm_orig[start : start + n] == search_lines:
            matches.append(start)
    if len(matches) > 1:
        raise EditApplyError(
            f"SEARCH block is ambiguous ({len(matches)} fuzzy matches); add more context"
        )
    if not matches:
        raise EditApplyError("SEARCH block not found in the target file")

    start = matches[0]
    # Preserve the indentation of the first matched line on the replacement.
    indent = orig_lines[start][: len(orig_lines[start]) - len(orig_lines[start].lstrip())]
    replace_lines = replace.strip("\n").splitlines()
    reindented = [
        (indent + line if line.strip() else line) for line in replace_lines
    ]
    new_lines = orig_lines[:start] + reindented + orig_lines[start + n :]
    trailing_nl = "\n" if original.endswith("\n") else ""
    return "\n".join(new_lines) + trailing_nl


def apply_file_edit(original: str | None, file_edit: FileEdit) -> str:
    """Materialize a FileEdit into final content."""
    if file_edit.full_content is not None:
        return file_edit.full_content
    if original is None:
        raise EditApplyError(
            f"edit blocks target a file that does not exist yet: {file_edit.path}; "
            "create it with a full file block instead"
        )
    content = original
    for search, replace in file_edit.edits:
        content = apply_search_replace(content, search, replace)
    return content if content.endswith("\n") else content + "\n"


def parse_edit_blocks(content: str) -> list[tuple[str, str]]:
    """Extract (search, replace) pairs from a FILE block body."""
    edits: list[tuple[str, str]] = []
    pattern = re.compile(
        re.escape(SEARCH_MARKER)
        + r"\r?\n(.*?)\r?\n?"
        + re.escape(DIVIDER_MARKER)
        + r"\r?\n(.*?)\r?\n?"
        + re.escape(REPLACE_MARKER),
        re.DOTALL,
    )
    for search, replace in pattern.findall(content):
        edits.append((search, replace))
    return edits


def _check_path(name: str) -> str:
    clean = name.replace("\\", "/").strip()
    if not clean or clean.startswith("/") or ".." in clean.split("/"):
        raise ValueError(f"unsafe or duplicate generated file path: {name}")
    if Path(clean).suffix.lower() not in ALLOWED_TARGET_SUFFIXES:
        raise ValueError(f"unsupported generated file type: {name}")
    return clean


def parse_coder_files(response: str) -> dict[str, str]:
    """Extract {relative_path: content} from a coder reply or raise ValueError."""
    try:
        parsed = json.loads(response.strip().strip("`"))
        objects = (
            parsed.get("files", [])
            if isinstance(parsed, dict) and "files" in parsed
            else [parsed]
        )
        if objects and all(isinstance(item, dict) for item in objects):
            json_files: dict[str, str] = {}
            for item in objects:
                name = str(item.get("filename", ""))
                content = item.get("code")
                if not name or not isinstance(content, str):
                    raise ValueError("Coder JSON objects require filename and code")
                clean = _check_path(name)
                if clean in json_files:
                    raise ValueError(f"unsafe or duplicate generated file path: {name}")
                json_files[clean] = content if content.endswith("\n") else content + "\n"
            if json_files:
                return json_files
    except json.JSONDecodeError:
        pass

    matches = re.findall(
        r"```FILE:\s*([^\r\n`]+?)(?:```|\r?\n)(.*?)(?:```|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if not matches:
        raise ValueError(
            "coder returned no usable files (no JSON object or ```FILE: blocks found); "
            f"response excerpt: {response[:200]!r}"
        )
    files: dict[str, str] = {}
    for name, content in matches:
        # Some models attach the closing fence to the filename; never treat
        # fence markup as part of the path.
        safe_name = name.split("```", 1)[0].strip().strip("`").replace("\\", "/")
        safe_name = re.sub(r"^FILE:\s*", "", safe_name, flags=re.IGNORECASE)
        safe_name = safe_name.rstrip("`").strip()
        clean = _check_path(safe_name)
        if clean in files:
            raise ValueError(f"unsafe or duplicate generated file path: {name}")
        if not content.strip():
            raise ValueError(f"coder produced an empty file: {safe_name}")
        files[clean] = content.strip() + "\n"
    return files


def generate_files(llm: LLM, coder_model: str, spec: str) -> dict[str, str]:
    response = llm.text(
        coder_model,
        CODER_PROMPT_TEMPLATE + spec,
        max_tokens=MAX_CODE_TOKENS,
        temperature=TEMPERATURE_CODE,
    )
    return parse_coder_files(response)


# ----------------------------------------------------------------------
# Edit-aware generation: existing files get SEARCH/REPLACE blocks,
# new files get a single full fenced block.
# ----------------------------------------------------------------------

EDIT_PROMPT_TEMPLATE = (
    "You are a precise code editor. Implement the specification.\n"
    "For EVERY file output one block:\n"
    "```FILE: relative/path.ext\n"
    "...content...\n```\n"
    "- For an EXISTING file (contents provided below), the block must contain "
    "one or more SEARCH/REPLACE edits in exactly this shape:\n"
    "<<<<<<< SEARCH\n"
    "exact lines copied from the current file\n"
    "=======\n"
    "replacement lines\n"
    ">>>>>>> REPLACE\n"
    "Copy the SEARCH lines verbatim and include enough surrounding context to "
    "make the match unique.\n"
    "- For a NEW file, the block contains the complete file contents.\n"
    "Return only these blocks, no commentary.\n\n"
    "SPECIFICATION:\n"
)


def parse_coder_output(response: str) -> list[FileEdit]:
    """Parse coder output into FileEdits (full content or SEARCH/REPLACE)."""
    edits: list[FileEdit] = []
    matches = re.findall(
        r"```FILE:\s*([^\r\n`]+?)(?:```|\r?\n)(.*?)(?:```|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if matches:
        for name, content in matches:
            safe_name = name.split("```", 1)[0].strip().strip("`").replace("\\", "/")
            safe_name = re.sub(r"^FILE:\s*", "", safe_name, flags=re.IGNORECASE).rstrip("`").strip()
            clean = _check_path(safe_name)
            pairs = parse_edit_blocks(content)
            body = content.strip()
            # Strip the edit markers themselves when deciding on full content.
            body_without_edits = re.sub(
                re.escape(SEARCH_MARKER) + r".*?" + re.escape(REPLACE_MARKER),
                "",
                body,
                flags=re.DOTALL,
            ).strip()
            if pairs and not body_without_edits:
                edits.append(FileEdit(path=clean, edits=pairs))
            else:
                if not body:
                    raise ValueError(f"coder produced an empty file: {safe_name}")
                edits.append(FileEdit(path=clean, full_content=body + "\n"))
        return edits

    # JSON replies are always full files.
    json_files = parse_coder_files(response)
    return [FileEdit(path=name, full_content=content) for name, content in json_files.items()]


RETRY_FEEDBACK_CHARS = 1_500


def generate_file_edits(
    llm: LLM,
    coder_model: str,
    spec: str,
    existing_files: dict[str, str] | None = None,
    retry_feedback: str | None = None,
) -> list[FileEdit]:
    """Edit-aware coder call. ``existing_files`` (bounded by the caller's
    token budget) is shown to the model so it can emit SEARCH/REPLACE.

    On retry, ``retry_feedback`` carries the failing verifier output (~1500
    chars) plus the staged file state — straight back to the coder, NOT via
    the planner.
    """
    prompt = EDIT_PROMPT_TEMPLATE + spec
    if retry_feedback:
        prompt += (
            "\n\nPREVIOUS ATTEMPT FAILED. Fix exactly this:\n"
            + retry_feedback[:RETRY_FEEDBACK_CHARS]
        )
    if existing_files:
        prompt += "\n\nEXISTING FILES:\n"
        for name, content in existing_files.items():
            prompt += f"\n[{name}]\n```\n{content}\n```\n"
    response = llm.text(
        coder_model, prompt, max_tokens=MAX_CODE_TOKENS, temperature=TEMPERATURE_CODE
    )
    return parse_coder_output(response)
