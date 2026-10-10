"""Four-role, retry-bounded vibe-coding pipeline."""

from __future__ import annotations

import os
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ollama_client import OllamaEngine


# Native local Ollama models supporting strict VRAM limits (RTX 3050 4GB ceiling)
THINKER_MODEL = "deepseek-r1:1.5b"
DEBUGGER_MODEL = THINKER_MODEL
CODER_MODEL = "qwen2.5-coder:3b"
MAX_RETRIES = 3
MAX_SPEC_WORDS = 300
MAX_INPUT_CHARS = 120_000
THINKER_INPUT_CHARS = 16_000
CODER_SPEC_CHARS = 8_000
MAX_LOG_CHARS = 8_000
MAX_GENERATED_CODE_CHARS = 100_000
ALLOWED_TARGET_SUFFIXES = {".py", ".html", ".css", ".js"}
WORKSPACE_FILE_SUFFIXES = ALLOWED_TARGET_SUFFIXES | {".json", ".md", ".txt"}
MAX_WORKSPACE_FILES = 60
MAX_WORKSPACE_CONTEXT_CHARS = 12_000
MAX_FILE_SNIPPET_CHARS = 3_000
TASK_PLAN_FILENAME = "tasks.md"
SUMMARY_FILENAME = "summary.txt"
MAX_TASKS = 12
VALIDATION_TIMEOUT_SECONDS = int(os.getenv("VIBE_VALIDATION_TIMEOUT_SECONDS", "300"))


@dataclass
class RunResult:
    status: str
    target_file: str = ""
    files: list[str] = field(default_factory=list)
    specification: str = ""
    code: str = ""
    attempts: int = 0
    stdout: str = ""
    stderr: str = ""
    llm_validation: str = ""
    errors: list[str] = field(default_factory=list)


class VibeOrchestrator:
    def __init__(
        self,
        engine: OllamaEngine | None = None,
        workspace: Path | str = ".",
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.engine = engine or OllamaEngine(workspace_root=self.workspace)
        self.max_retries = max(1, min(max_retries, MAX_RETRIES))

    def run(
        self,
        user_request: str,
        target_file: str | None = None,
        test_command: list[str] | None = None,
        progress: Callable[[str], None] | None = None,
        workspace_dir: str | Path | None = None,
    ) -> RunResult:
        def report(message: str) -> None:
            if progress:
                progress(message)

        if not user_request.strip():
            raise ValueError("request must not be empty")
        report("preparing workspace and safety checks")
        self._set_workspace(workspace_dir)
        report("tool • workspace.list_files: scanning the selected workspace")
        workspace_context = self._workspace_context(user_request, report)
        report("tool • workspace.list_files: scan complete")
        report("stage 1/4 • thinker: breaking request into sequential tasks")
        plan = self._think(user_request[:THINKER_INPUT_CHARS], workspace_context)
        tasks = self._parse_tasks(plan)
        plan_path = self.workspace / TASK_PLAN_FILENAME
        self._write_task_plan(plan_path, tasks)
        report(f"tool • filesystem.write_file: {plan_path} ({len(tasks)} tasks)")
        result = RunResult("running", specification=plan)
        all_files: list[str] = []

        for task_number, task in enumerate(tasks, 1):
            task_spec = task["spec"]
            task_name = task["name"]
            report(f"task {task_number}/{len(tasks)} • {task_name}")
            task_passed = False
            for attempt in range(1, self.max_retries + 1):
                result.attempts += 1
                report(
                    f"task {task_number} • stage 2/4 coder • attempt {attempt}"
                )
                files = self._code(task_spec)
                total_size = sum(len(content) for content in files.values())
                if total_size > MAX_GENERATED_CODE_CHARS:
                    raise ValueError("generated task output exceeds the configured safety limit")
                report(f"task {task_number} • stage 3/4 writer • writing generated files")
                artifact_name, artifact_path = self._write_task_outputs(
                    task_number, task_name, files, report
                )
                report(f"task {task_number} • fetcher: retrieving {artifact_path}")
                fetched_artifact = self._fetch_file(artifact_path)
                report(f"task {task_number} • stage 4/4 validator: reading task files and summary")
                task_files = self._read_generated_files(files, report)
                summary = self._read_summary(report)
                empty_files = [
                    name for name, content in task_files.items() if not content.strip()
                ]
                summary_valid = self._summary_has_files(summary, files)
                if empty_files:
                    local_passed = False
                    stdout = ""
                    stderr = "FILE EMPTY: " + ", ".join(empty_files)
                    llm_passed = False
                    llm_feedback = "FAIL\nExplicit file empty signal: " + ", ".join(empty_files)
                elif not summary_valid:
                    local_passed = False
                    stdout = ""
                    stderr = f"{SUMMARY_FILENAME} does not list every generated file"
                    llm_passed = False
                    llm_feedback = "FAIL\nWriter summary does not match generated files."
                else:
                    local_passed, stdout, stderr = self._validate(
                        task_files, test_command
                    )
                    validation_files = dict(task_files)
                    validation_files[artifact_name] = fetched_artifact
                    validation_files[SUMMARY_FILENAME] = summary
                    llm_passed, llm_feedback = self._llm_validate(
                        task_spec, validation_files
                    )
                result.llm_validation = llm_feedback
                result.files = list(
                    dict.fromkeys(all_files + list(files) + [artifact_name, SUMMARY_FILENAME])
                )
                result.code = "\n\n".join(task_files.values())
                result.stdout, result.stderr = stdout, stderr
                if local_passed and llm_passed:
                    task_passed = True
                    all_files.extend(list(files) + [artifact_name, SUMMARY_FILENAME])
                    task["status"] = "complete"
                    self._write_task_plan(plan_path, tasks)
                    report(f"task {task_number} complete • moving to next task")
                    break
                diagnostics = (
                    stdout + "\n" + stderr + "\nLLM VALIDATION:\n" + llm_feedback
                ).strip()[-MAX_LOG_CHARS:]
                result.errors.append(f"Task {task_number}: {diagnostics}")
                task["status"] = "failed"
                self._write_task_plan(plan_path, tasks)
                if attempt < self.max_retries:
                    report(f"task {task_number} failed • thinker is refining this task")
                    task_spec = self._debug(task_spec, diagnostics)
                    task["spec"] = task_spec
                    self._write_task_plan(plan_path, tasks)
            if not task_passed:
                result.status = "failed"
                result.specification = "\n\n".join(t["spec"] for t in tasks)
                return result

        result.specification = "\n\n".join(t["spec"] for t in tasks)
        result.status = "completed"
        report("all tasks complete • validated code is ready")
        return result

    def _think(self, request_text: str, workspace_context: str) -> str:
        prompt = (
            "You are the thinker and task planner. Break the messy request into "
            "small ordered atomic tasks. Return ONLY Markdown using this exact shape:\n"
            "## TASK 1: short task name\n"
            "- files: relative/file.py\n"
            "- logic: signatures, parameters, and rules\n"
            "Create no more than 12 tasks and keep every task under 120 words. "
            "Do not generate code. Each task must be independently codable and testable.\n\n"
            f"WORKSPACE:\n{self.workspace}\n\nEXISTING FILE CONTEXT:\n"
            f"{workspace_context}\n\nREQUEST:\n{request_text}"
        )
        spec = self.engine.chat(THINKER_MODEL, prompt, max_tokens=900)
        return spec[:CODER_SPEC_CHARS]

    def _parse_tasks(self, plan: str) -> list[dict[str, str]]:
        matches = re.findall(
            r"##\s*TASK\s*(\d+)\s*:\s*([^\r\n]+)\r?\n(.*?)(?=##\s*TASK|\Z)",
            plan,
            re.IGNORECASE | re.DOTALL,
        )
        tasks = [
            {"number": number, "name": name.strip(), "spec": body.strip(), "status": "pending"}
            for number, name, body in matches[:MAX_TASKS]
            if body.strip()
        ]
        if tasks:
            return tasks
        return [{"number": "1", "name": "primary implementation", "spec": plan, "status": "pending"}]

    def _write_task_plan(self, path: Path, tasks: list[dict[str, str]]) -> None:
        lines = ["# Vibe coding task plan", "", "Tasks are processed sequentially.", ""]
        for task in tasks:
            lines.extend(
                [
                    f"## TASK {task['number']}: {task['name']} [{task['status']}]",
                    task["spec"][:CODER_SPEC_CHARS],
                    "",
                ]
            )
        self._write_file(path, "\n".join(lines))

    def _read_generated_files(
        self, files: dict[str, str], report: Callable[[str], None]
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in files:
            path = self._safe_target(name)
            report(f"tool • filesystem.read_file: {path}")
            result[name] = path.read_text(encoding="utf-8", errors="replace")[:12000]
        return result

    def _write_task_outputs(
        self,
        task_number: int,
        task_name: str,
        files: dict[str, str],
        report: Callable[[str], None],
    ) -> tuple[str, Path]:
        """Writer role: persist Coder output and append a machine-readable summary."""
        for relative_name, content in files.items():
            target = self._safe_target(relative_name)
            report(f"tool • filesystem.write_file: {target}")
            self._write_file(target, content)

        artifact_name = f"task{task_number}_code.txt"
        artifact_path = self.workspace / artifact_name
        report(f"tool • filesystem.write_file: {artifact_path}")
        self._write_file(artifact_path, self._task_artifact(task_number, task_name, files))

        summary_path = self.workspace / SUMMARY_FILENAME
        entries: list[dict[str, object]] = []
        if summary_path.exists():
            try:
                existing = summary_path.read_text(encoding="utf-8")
                entries = [json.loads(line) for line in existing.splitlines() if line.strip()]
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read existing {SUMMARY_FILENAME}: {exc}") from exc
        entries.append(
            {
                "task": task_number,
                "summary": f"Generated files: {', '.join(files)}",
            }
        )
        report(f"tool • filesystem.write_file: {summary_path}")
        self._write_file(
            summary_path,
            "\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n",
        )
        return artifact_name, artifact_path

    def _read_summary(self, report: Callable[[str], None]) -> str:
        summary_path = self.workspace / SUMMARY_FILENAME
        report(f"tool • filesystem.read_file: {summary_path}")
        try:
            lines = [
                json.loads(line)
                for line in summary_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid {SUMMARY_FILENAME}: {exc}") from exc
        if not lines:
            raise ValueError(f"{SUMMARY_FILENAME} is empty")
        return "\n".join(json.dumps(line, sort_keys=True) for line in lines)

    def _summary_has_files(self, summary: str, files: dict[str, str]) -> bool:
        """Ensure Validator is checking the same files that Writer reported."""
        try:
            entries = [json.loads(line) for line in summary.splitlines() if line.strip()]
        except json.JSONDecodeError:
            return False
        if not entries or not isinstance(entries[-1], dict):
            return False
        latest_summary = str(entries[-1].get("summary", ""))
        return all(name in latest_summary for name in files)

    def _workspace_context(
        self, request_text: str, report: Callable[[str], None]
    ) -> str:
        """Find relevant bounded snippets for the thinker, never the coder."""
        files = self._list_workspace_files()
        if not files:
            return "(workspace is empty)"
        request_lower = request_text.lower()
        ranked = sorted(
            files,
            key=lambda path: (
                0 if path.name.lower() in request_lower else 1,
                0 if path.suffix.lower() in {".py", ".html", ".js", ".css"} else 1,
                str(path),
            ),
        )
        sections: list[str] = [f"FILES ({len(files)} found):"]
        remaining = MAX_WORKSPACE_CONTEXT_CHARS
        for path in ranked[:MAX_WORKSPACE_FILES]:
            relative = path.relative_to(self.workspace)
            sections.append(f"- {relative}")
            if remaining <= 0:
                break
            try:
                report(f"tool • filesystem.read_snippet: {path}")
                snippet = self._read_file_snippet(path)
            except OSError:
                continue
            if snippet:
                section = f"\n[{relative}]\n{snippet}"
                sections.append(section[:remaining])
                remaining -= len(section)
        return "\n".join(sections)[:MAX_WORKSPACE_CONTEXT_CHARS]

    def _list_workspace_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.workspace.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in WORKSPACE_FILE_SUFFIXES
                and not any(part.startswith(".") for part in path.relative_to(self.workspace).parts)
            ):
                files.append(path)
        return sorted(files)

    def _read_file_snippet(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_SNIPPET_CHARS]

    def _write_file(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def _fetch_file(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")[:MAX_GENERATED_CODE_CHARS]

    def _task_artifact(
        self, task_number: int, task_name: str, files: dict[str, str]
    ) -> str:
        sections = [
            f"# Task {task_number}: {task_name}",
            "",
            "Generated files for Writer output. Fetcher retrieved this file for validation.",
            "",
        ]
        for name, content in files.items():
            sections.extend([f"## FILE: {name}", "```", content, "```", ""])
        return "\n".join(sections)

    def _code(self, specification: str) -> dict[str, str]:
        prompt = (
            "You are a pure code generator. Use ONLY this specification. Return "
            "one JSON object with this shape: {\"filename\": \"relative/path.ext\", "
            "\"description\": \"brief purpose\", \"code\": \"complete file contents\"}. "
            "If multiple files are required, return {\"files\": [objects with those "
            "three fields]}. Return JSON only, use relative paths, and do not invent "
            "requirements.\n\nSPECIFICATION:\n"
            + specification[:CODER_SPEC_CHARS]
        )
        response = self.engine.chat(CODER_MODEL, prompt, max_tokens=1800)
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
                    name = str(item.get("filename", "")).replace("\\", "/").strip()
                    content = item.get("code")
                    if not name or not isinstance(content, str):
                        raise ValueError("Coder JSON objects require filename and code")
                    if name in json_files or name.startswith("/") or ".." in name.split("/"):
                        raise ValueError(f"unsafe or duplicate generated file path: {name}")
                    if Path(name).suffix.lower() not in ALLOWED_TARGET_SUFFIXES:
                        raise ValueError(f"unsupported generated file type: {name}")
                    json_files[name] = content if content.endswith("\n") else content + "\n"
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
            return self._fallback_files(specification)
        files: dict[str, str] = {}
        for name, content in matches:
            # Some models accidentally attach the closing fence to the
            # filename (`FILE:pages/index.html```). Never treat fence markup
            # as part of the path.
            safe_name = name.split("```", 1)[0].strip().strip("`").replace("\\", "/")
            safe_name = re.sub(r"^FILE:\s*", "", safe_name, flags=re.IGNORECASE)
            safe_name = safe_name.rstrip("`").strip()
            if safe_name in files or safe_name.startswith("/") or ".." in safe_name.split("/"):
                raise ValueError(f"unsafe or duplicate generated file path: {name}")
            if Path(safe_name).suffix.lower() not in ALLOWED_TARGET_SUFFIXES:
                raise ValueError(f"unsupported generated file type: {name}")
            files[safe_name] = content.strip() + "\n" if content.strip() else self._scaffold(safe_name)
        return files

    def _fallback_files(self, specification: str) -> dict[str, str]:
        paths = re.findall(
            r"(?:files?|path|output)\s*:\s*([^\r\n,]+)",
            specification,
            re.IGNORECASE,
        )
        safe_paths = [path.strip().strip("`") for path in paths]
        safe_paths = [
            path for path in safe_paths if Path(path).suffix.lower() in ALLOWED_TARGET_SUFFIXES
        ]
        if not safe_paths:
            safe_paths = ["generated.py"]
        return {path: self._scaffold(path) for path in safe_paths[:8]}

    def _scaffold(self, relative_name: str) -> str:
        suffix = Path(relative_name).suffix.lower()
        if suffix == ".html":
            return (
                "<!doctype html>\n<html lang=\"en\">\n<head>\n"
                "  <meta charset=\"utf-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                "  <title>Generated page</title>\n</head>\n<body>\n"
                "  <main><h1>Generated page</h1></main>\n</body>\n</html>\n"
            )
        if suffix == ".css":
            return "/* Generated stylesheet */\n:root { color-scheme: light; }\n"
        if suffix == ".js":
            return "\"use strict\";\n// Generated script\n"
        return '"""Generated module."""\n'

    def _debug(self, specification: str, diagnostics: str) -> str:
        prompt = (
            "You are a debugging context parser. Update the Markdown specification "
            "under 300 words using ONLY the existing specification and test logs. "
            "Keep signatures and logic rules; state the minimal correction. No code.\n\n"
            f"SPECIFICATION:\n{specification}\n\nTEST LOG:\n{diagnostics}"
        )
        return " ".join(self.engine.chat(DEBUGGER_MODEL, prompt, max_tokens=900).split()[:MAX_SPEC_WORDS])

    def _llm_validate(
        self, specification: str, files: dict[str, str]
    ) -> tuple[bool, str]:
        bundled_files = "\n\n".join(
            f"FILE: {name}\n{content[:12000]}" for name, content in files.items()
        )[:24000]
        prompt = (
            "You are the validation agent. Compare the generated files against "
            "the specification. Check required behavior, syntax-level problems, "
            "file references, and obvious runtime defects. Do not rewrite code. "
            "Respond exactly with PASS or FAIL on the first line, followed by a "
            "short reason. Choose FAIL for any required behavior that is missing.\n\n"
            f"SPECIFICATION:\n{specification}\n\nGENERATED FILES:\n{bundled_files}"
        )
        feedback = self.engine.chat(DEBUGGER_MODEL, prompt, max_tokens=700)
        first_line = feedback.splitlines()[0].strip().upper() if feedback else ""
        return first_line == "PASS", feedback[-MAX_LOG_CHARS:]

    def _safe_target(self, target_file: str) -> Path:
        candidate = (self.workspace / target_file).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise ValueError("target_file must remain inside the workspace")
        if candidate.suffix.lower() not in ALLOWED_TARGET_SUFFIXES:
            raise ValueError("target_file must end in .py, .html, .css, or .js")
        return candidate

    def _set_workspace(self, workspace_dir: str | Path | None) -> None:
        if workspace_dir is None:
            return
        raw_path = str(workspace_dir).strip().strip('"')
        candidate = Path(raw_path).expanduser().resolve()
        if not candidate.exists():
            raise ValueError(f"workspace directory does not exist: {candidate}")
        if not candidate.is_dir():
            raise ValueError(f"workspace path is not a directory: {candidate}")
        self.workspace = candidate

    def _validate(
        self, files: dict[str, str], test_command: list[str] | None
    ) -> tuple[bool, str, str]:
        if test_command is None:
            python_files = [self._safe_target(name) for name in files if name.endswith(".py")]
            if python_files:
                command = [sys.executable, "-m", "py_compile", *map(str, python_files)]
            elif not all(content.strip() for content in files.values()):
                return False, "", "generated file is empty"
            else:
                return True, "", ""
        else:
            command = test_command
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("test_command must be a non-empty list of strings")
        try:
            completed = subprocess.run(
                command, cwd=self.workspace, capture_output=True, text=True,
                timeout=VALIDATION_TIMEOUT_SECONDS, shell=False, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = str(exc.stdout or "")[-MAX_LOG_CHARS:]
            stderr = str(exc.stderr or "")[-MAX_LOG_CHARS:]
            return False, stdout, (
                f"validation timed out after {VALIDATION_TIMEOUT_SECONDS} seconds\n{stderr}"
            )
        return completed.returncode == 0, completed.stdout[-MAX_LOG_CHARS:], completed.stderr[-MAX_LOG_CHARS:]
