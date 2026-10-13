"""Dynamic System Engine: Persistent .agent profile and hardware bounds calculator."""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent_config")

# Reference models (exact names; the thinker carries a namespace + :latest tag —
# never strip, lowercase or "normalize" for matching).
DEFAULT_PLANNER_MODEL = "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
DEFAULT_CODER_MODEL = "qwen2.5-coder:3b-instruct"
DEFAULT_CONTEXT_TOKENS = 4096
DEFAULT_KEEP_ALIVE = "5m"
RTX_3050_VRAM_CEILING_GB = 4.0

# Role ceilings (num_ctx). The effective num_ctx is always clamped to these
# (and further by calibration in Phase 6).
ROLE_CTX_CEILINGS = {"thinker": 16384, "coder": 12288, "validator": 16384}

# Canonical default roles: the single source of truth for model + sampling.
# thinker = planner/Q&A, coder = code, validator = review (thinker model+ctx).
def _default_roles() -> dict[str, dict[str, Any]]:
    return {
        "thinker": {
            "model": DEFAULT_PLANNER_MODEL,
            "num_ctx": 8192,
            "num_predict": 2048,
            "temperature": 0.2,
            "keep_alive": DEFAULT_KEEP_ALIVE,
            # Reasoning distill: DeepSeek advises instructions in the user turn.
            "inline_system": True,
            # Answer budget gets this think allowance on top when thinking
            # cannot be disabled (num_predict is shared between think+answer).
            "think_budget": 512,
        },
        "coder": {
            "model": DEFAULT_CODER_MODEL,
            "num_ctx": 6144,
            "num_predict": 2048,
            "temperature": 0.1,
            "keep_alive": DEFAULT_KEEP_ALIVE,
            "inline_system": False,
            "think_budget": 0,  # instruct model: no reasoning mode, never send think
        },
        "validator": {
            "model": DEFAULT_PLANNER_MODEL,
            "num_ctx": 8192,
            "num_predict": 512,
            "temperature": 0.0,
            "keep_alive": DEFAULT_KEEP_ALIVE,
            "inline_system": True,
            "think_budget": 512,
        },
    }


@dataclass(init=False)
class AgentConfig:
    # ``roles`` is the single source of truth: {thinker, coder, validator},
    # each {model, num_ctx, num_predict, temperature, keep_alive}.
    roles: dict[str, dict[str, Any]]
    # Sandbox allowlist: only these named command vectors may run for
    # validation. Free-form shell is never executed.
    allowed_commands: list[str] | None  # defaults to ["py_compile"]
    # Unknown keys from .agent are preserved verbatim on save (round-trip).
    extra: dict[str, Any] | None

    _KNOWN_KEYS = {
        "planner_model", "coder.5b_or_3b", "coder_5b_or_3b", "coder_model",
        "max_context_tokens", "keep_alive", "allowed_commands", "roles",
        "router",
    }

    def __init__(
        self,
        roles: dict[str, dict[str, Any]] | None = None,
        allowed_commands: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        router: dict[str, Any] | None = None,
        # Legacy constructor inputs (migration only; folded into roles below):
        planner_model: str | None = None,
        coder_5b_or_3b: str | None = None,
        max_context_tokens: int | None = None,
        keep_alive: str | None = None,
        **legacy: Any,
    ) -> None:
        import re as _re

        removed_pattern = _re.compile(r"l[fm]{2}|li[q]uid", _re.IGNORECASE)
        # coder may also arrive under the alternate legacy spellings.
        if coder_5b_or_3b is None:
            coder_5b_or_3b = legacy.pop("coder_model", None) or legacy.pop("coder.5b_or_3b", None)

        merged = _default_roles()
        saved = roles or {}
        if "planner" in saved and "thinker" not in saved:
            saved = {**saved, "thinker": saved["planner"]}
        for role, overrides in saved.items():
            if role in merged and isinstance(overrides, dict):
                merged[role] = {**merged[role], **{k: v for k, v in overrides.items() if v is not None}}

        def _clean(model: Any) -> str | None:
            if model and removed_pattern.search(str(model)):
                return None
            return str(model) if model else None

        # Fold legacy ctor inputs into roles (only where the role did not
        # already set the key, so an explicit saved role always wins).
        if (p := _clean(planner_model)) and "model" not in saved.get("thinker", {}):
            merged["thinker"]["model"] = p
            merged["validator"]["model"] = p
        if (c := _clean(coder_5b_or_3b)) and "model" not in saved.get("coder", {}):
            merged["coder"]["model"] = c
        if max_context_tokens is not None:
            for role in merged:
                if "num_ctx" not in saved.get(role, {}):
                    merged[role]["num_ctx"] = int(max_context_tokens)
        if keep_alive:
            for role in merged:
                if "keep_alive" not in saved.get(role, {}):
                    merged[role]["keep_alive"] = str(keep_alive)

        self.roles = merged
        self.allowed_commands = allowed_commands
        self.extra = extra
        # Router v2 settings. min_confidence: below it, the T2 classifier's
        # answer is treated as UNCLEAR (never guessed).
        self.router: dict[str, Any] = {"min_confidence": 0.7, "chat_use_llm": False}
        if isinstance(router, dict):
            self.router.update(router)

    # --- compat views: read from roles (migration inputs, never the source) ---

    @property
    def planner_model(self) -> str:
        return str(self.roles["thinker"]["model"])  # type: ignore[index]

    @property
    def coder_model(self) -> str:
        return str(self.roles["coder"]["model"])  # type: ignore[index]

    @property
    def coder_5b_or_3b(self) -> str:
        return self.coder_model

    @property
    def max_context_tokens(self) -> int:
        # The thinker context window is the pipeline's default budget ceiling.
        return int(self.roles["thinker"]["num_ctx"])  # type: ignore[index]

    @property
    def keep_alive(self) -> str:
        return str(self.roles["thinker"]["keep_alive"])  # type: ignore[index]

    def get_role_config(self, role: str) -> dict[str, Any]:
        """Effective settings for a role, num_ctx clamped to the role ceiling."""
        # Legacy alias: "planner" is the thinker role.
        canonical = "thinker" if role == "planner" else role
        cfg = dict(self.roles.get(canonical, self.roles["thinker"]))  # type: ignore[arg-type]
        ceiling = ROLE_CTX_CEILINGS.get(canonical)
        if ceiling is not None:
            cfg["num_ctx"] = min(int(cfg.get("num_ctx", ceiling)), ceiling)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = dict(self.extra or {})
        data["roles"] = self.roles
        data["allowed_commands"] = self.allowed_commands or ["py_compile"]
        data["router"] = self.router
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        # Delegate to the constructor: it folds legacy top-level keys into roles.
        raw_allowed = data.get("allowed_commands")
        allowed = [str(a) for a in raw_allowed] if isinstance(raw_allowed, list) else None
        extra = {k: v for k, v in data.items() if k not in cls._KNOWN_KEYS}
        return cls(
            roles=data.get("roles") if isinstance(data.get("roles"), dict) else None,
            allowed_commands=allowed,
            extra=extra or None,
            router=data.get("router") if isinstance(data.get("router"), dict) else None,
            planner_model=data.get("planner_model"),
            coder_5b_or_3b=(
                data.get("coder.5b_or_3b")
                or data.get("coder_5b_or_3b")
                or data.get("coder_model")
            ),
            max_context_tokens=data.get("max_context_tokens"),
            keep_alive=data.get("keep_alive"),
        )


def detect_system_ram_gb() -> float:
    """Read available host RAM without third-party dependencies."""
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.avail_phys / (1024**3), 2)
        elif sys.platform == "linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = float(line.split()[1])
                        return round(kb / (1024**2), 2)
        elif sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(out.strip()) / (1024**3), 2)
    except Exception as exc:
        logger.warning(f"Could not read host RAM: {exc}")
    return 8.0  # safe fallback


def detect_gpu_vram_gb() -> float:
    """Probe live NVIDIA VRAM or default to the RTX 3050 4GB ceiling."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            total_mb = float(result.stdout.strip().splitlines()[0])
            return round(total_mb / 1024, 2)
    except Exception:
        pass
    # RTX 3050 4GB hard ceiling specification
    return RTX_3050_VRAM_CEILING_GB


def benchmark_and_calculate_tokens(
    coder_model: str,
    manual_context: int | None = None,
) -> int:
    """Benchmark system resources against 4GB RTX 3050 ceiling to calculate context tokens."""
    if manual_context is not None and manual_context > 0:
        return manual_context

    vram_gb = detect_gpu_vram_gb()
    ram_gb = detect_system_ram_gb()
    logger.info(f"System benchmark: Detected {vram_gb} GB VRAM, {ram_gb} GB Free RAM")

    # Quantized 3B models consume ~2.2GB base weights in VRAM.
    # On a 4GB ceiling, KV-cache must be constrained to 4096 tokens.
    # Quantized 1.5B models consume ~1.0GB base weights; 8192 tokens can fit if VRAM allows.
    is_3b = "3b" in coder_model.lower()
    if is_3b or vram_gb <= RTX_3050_VRAM_CEILING_GB:
        return 4096
    return 8192


def get_agent_config_path(workspace_root: str | Path | None = None) -> Path:
    root = Path(workspace_root).resolve() if workspace_root else Path.cwd()
    return root / ".agent"


def load_agent_config(workspace_root: str | Path | None = None) -> AgentConfig:
    """Load persistent .agent profile or create default if not present."""
    path = get_agent_config_path(workspace_root)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AgentConfig.from_dict(data)
        except Exception as exc:
            logger.warning(f"Error loading {path}, regenerating defaults: {exc}")

    config = AgentConfig()
    save_agent_config(config, workspace_root)
    return config


def save_agent_config(config: AgentConfig, workspace_root: str | Path | None = None) -> Path:
    """Save persistent .agent profile to the workspace root."""
    path = get_agent_config_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    return path


def parse_and_recalculate(args: list[str] | None = None) -> AgentConfig | None:
    """Handle CLI invocation, --recalculate flag, and manual bounds."""
    parser = argparse.ArgumentParser(
        description="Dynamic System Engine: Configure and benchmark .agent profile."
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help="Dynamically benchmark system resources to rewrite max_context_tokens in .agent",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(Path.cwd()),
        help="Workspace root path",
    )
    parser.add_argument(
        "--planner-model",
        type=str,
        default=None,
        help="Planner model (e.g., erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest)",
    )
    parser.add_argument(
        "--coder-model",
        type=str,
        default=None,
        help="Coder model (e.g., qwen2.5-coder:1.5b or qwen2.5-coder:3b)",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Manual token ceiling (e.g. 4096 or 8192)",
    )

    parsed, _ = parser.parse_known_args(args)

    if parsed.recalculate or parsed.coder_model or parsed.planner_model or parsed.max_context_tokens:
        workspace_dir = Path(parsed.workspace).resolve()
        existing = load_agent_config(workspace_dir)

        coder = parsed.coder_model or existing.coder_model
        tokens = benchmark_and_calculate_tokens(
            coder_model=coder,
            manual_context=parsed.max_context_tokens,
        )

        # Copy-on-write: never rebuild AgentConfig from scratch (that would drop
        # roles, keep_alive, allowed_commands and extra). Update roles in place
        # on a copy of the existing config.
        updated = copy.deepcopy(existing)
        if parsed.planner_model:
            updated.roles["thinker"]["model"] = parsed.planner_model
            updated.roles["validator"]["model"] = parsed.planner_model
        if parsed.coder_model:
            updated.roles["coder"]["model"] = parsed.coder_model
        # Recalculated token ceiling applies to every role (clamped per role).
        for role, cfg in updated.roles.items():
            ceiling = ROLE_CTX_CEILINGS.get(role, tokens)
            cfg["num_ctx"] = min(tokens, ceiling)

        saved_path = save_agent_config(updated, workspace_dir)
        print(f"[Dynamic System Engine] Profile written to: {saved_path}")
        print(json.dumps(updated.to_dict(), indent=2))
        return updated

    return None


if __name__ == "__main__":
    parse_and_recalculate(sys.argv[1:])
