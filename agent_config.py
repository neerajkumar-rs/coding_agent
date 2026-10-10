"""Dynamic System Engine: Persistent .agent profile and hardware bounds calculator."""

from __future__ import annotations

import argparse
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

DEFAULT_PLANNER_MODEL = "deepseek-r1:1.5b"
DEFAULT_CODER_MODEL = "qwen2.5-coder:3b"
DEFAULT_CONTEXT_TOKENS = 4096
DEFAULT_KEEP_ALIVE = "5m"
RTX_3050_VRAM_CEILING_GB = 4.0


@dataclass
class AgentConfig:
    planner_model: str = DEFAULT_PLANNER_MODEL
    coder_5b_or_3b: str = DEFAULT_CODER_MODEL  # Serialized as "coder.5b_or_3b"
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS
    keep_alive: str = DEFAULT_KEEP_ALIVE  # Top-level Ollama keep_alive (model residency)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_model": self.planner_model,
            "coder.5b_or_3b": self.coder_5b_or_3b,
            "max_context_tokens": self.max_context_tokens,
            "keep_alive": self.keep_alive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        coder = (
            data.get("coder.5b_or_3b")
            or data.get("coder_5b_or_3b")
            or data.get("coder_model")
            or DEFAULT_CODER_MODEL
        )
        planner = data.get("planner_model") or DEFAULT_PLANNER_MODEL
        ctx = int(data.get("max_context_tokens", DEFAULT_CONTEXT_TOKENS))
        keep_alive = str(data.get("keep_alive") or DEFAULT_KEEP_ALIVE)
        return cls(
            planner_model=planner,
            coder_5b_or_3b=coder,
            max_context_tokens=ctx,
            keep_alive=keep_alive,
        )

    @property
    def coder_model(self) -> str:
        return self.coder_5b_or_3b


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
        help="Planner model (e.g., deepseek-r1:1.5b)",
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

        planner = parsed.planner_model or existing.planner_model
        coder = parsed.coder_model or existing.coder_5b_or_3b
        tokens = benchmark_and_calculate_tokens(
            coder_model=coder,
            manual_context=parsed.max_context_tokens,
        )

        updated = AgentConfig(
            planner_model=planner,
            coder_5b_or_3b=coder,
            max_context_tokens=tokens,
        )
        saved_path = save_agent_config(updated, workspace_dir)
        print(f"[Dynamic System Engine] Profile written to: {saved_path}")
        print(json.dumps(updated.to_dict(), indent=2))
        return updated

    return None


if __name__ == "__main__":
    parse_and_recalculate(sys.argv[1:])
