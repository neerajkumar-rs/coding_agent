"""Coder: turns one task spec into concrete file contents.

Parses the model reply as either a JSON object ({"filename", "code"} or
{"files": [...]}) or ```FILE: blocks. Anything else is unusable and raises
ValueError — the caller FAILS the attempt; no scaffolds, no stubs.
(Phase 3 adds SEARCH/REPLACE edit blocks for editing existing files.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from ollama_client import TEMPERATURE_CODE

ALLOWED_TARGET_SUFFIXES = {".py", ".html", ".css", ".js"}
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
