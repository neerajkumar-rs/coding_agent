"""Machine-level calibration: measure the highest num_ctx rung per role that
passes the VRAM guard (>= 97% of the model on the GPU) on THIS machine.

Never auto-runs. Triggered by the `/setup` TUI command or the CLI:

    .\\env\\Scripts\\python.exe calibrate.py --yes        # calibrate all roles
    .\\env\\Scripts\\python.exe calibrate.py --roles thinker,coder

Writes ``calibration.json`` (repo root, git-ignored) atomically after EACH
role, so a cancel/crash keeps every completed role. ``ceiling_for(role)`` in
agent_config swaps in the measured ceiling when the machine fingerprint
matches — callers never change.

The probe goes through ``AgentGateway.probe_ctx``: one tiny generation at an
exact num_ctx (no step-down, no retry), then /api/ps for size_vram/size.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent_config import (
    CALIBRATION_PATH,
    ROLE_NAMES,
    load_calibration,
    machine_fingerprint,
)
from agent_core.run_state import utc_now_iso

# The rungs probed, ascending. Matches the gateway's CTX_RUNGS.
RUNGS = [4096, 6144, 8192, 12288, 16384]


def write_calibration_atomic(data: dict[str, Any], path: Path | None = None) -> None:
    path = path or CALIBRATION_PATH
    """Atomic write: temp file + os.replace (never a half-written file)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# A probe: (role, num_ctx) -> {"ok": bool, "fraction": float}. Injected in
# tests; the real one lives in run_calibration and uses gateway.probe_ctx.
ProbeFn = Callable[[str, int], Awaitable[dict[str, Any]]]


async def calibrate_role(
    role: str,
    probe: ProbeFn,
    rungs: list[int] | None = None,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    """Binary search for the highest rung that passes the probe.

    Cancel-safe: checks ``cancel_event`` before every probe and returns the
    partial result (``cancelled: True``) with the best rung found so far.
    Never raises on a failed probe — a failed rung just narrows the search.
    """
    ladder = sorted(rungs or RUNGS)
    lo, hi = 0, len(ladder) - 1
    best: int | None = None
    fractions: dict[str, float] = {}
    probes = 0
    cancelled = False

    while lo <= hi:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        mid = (lo + hi) // 2
        rung = ladder[mid]
        result = await probe(role, rung)
        probes += 1
        fractions[str(rung)] = round(float(result.get("fraction", 0.0)), 3)
        if result.get("ok"):
            best = rung
            lo = mid + 1
        else:
            hi = mid - 1

    return {
        "ceiling": best,
        "fractions": fractions,
        "probes": probes,
        "cancelled": cancelled,
        "probed_at": utc_now_iso(),
    }


async def run_calibration(
    gateway: Any,
    roles: list[str] | None = None,
    *,
    path: Path | None = None,
    cancel_event: Any | None = None,
    on_role_done: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Calibrate each role through the gateway and persist per role.

    The file is written atomically after EACH role, so a cancel or crash
    keeps every completed role's measurement.
    """
    async def _probe(role: str, num_ctx: int) -> dict[str, Any]:
        return await gateway.probe_ctx(role, num_ctx)

    out_path = path or CALIBRATION_PATH
    data: dict[str, Any] = load_calibration(out_path) or {}
    data["fingerprint"] = machine_fingerprint()
    data.setdefault("roles", {})

    for role in (roles or list(ROLE_NAMES)):
        if cancel_event is not None and cancel_event.is_set():
            break
        result = await calibrate_role(role, _probe, cancel_event=cancel_event)
        if result["ceiling"] is None and not result["cancelled"]:
            # Nothing passed at any rung: keep the lowest rung as the ceiling
            # so the role still runs (the guard will step down as needed).
            result["ceiling"] = RUNGS[0]
        # A calibration made on CPU (every probe showed size_vram == 0) is
        # unusable: refuse to write it and say so.
        fractions = result.get("fractions") or {}
        if fractions and all(v == 0.0 for v in fractions.values()):
            print(
                f"  {role}: every probe ran on CPU (size_vram=0) — this "
                "calibration is unusable; not written. Fix the Ollama GPU "
                "setup first."
            )
            continue
        data["roles"][role] = result
        write_calibration_atomic(data, out_path)
        if on_role_done:
            on_role_done(role, result)
        if result["cancelled"]:
            break
    return data


def _print_report(data: dict[str, Any]) -> None:
    print("\nCalibration results (ceiling = highest num_ctx passing the 0.97 GPU guard):")
    for role, entry in (data.get("roles") or {}).items():
        ceil_ = entry.get("ceiling")
        note = " (cancelled)" if entry.get("cancelled") else ""
        print(f"  {role}: {ceil_}{note}  fractions={entry.get('fractions')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Machine calibration probe")
    parser.add_argument("--roles", default=None, help="comma list (default: all roles)")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args(argv)

    if not args.yes:
        answer = input(
            "This probes every role model at several context sizes against the "
            "GPU guard (loads/unloads models, a few minutes). Continue? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborted — nothing changed.")
            return 0

    from ollama_client import AgentGateway, OllamaSupervisorClient

    roles = [r.strip() for r in args.roles.split(",") if r.strip()] if args.roles else None

    async def _go() -> dict[str, Any]:
        gateway = AgentGateway(OllamaSupervisorClient())
        return await run_calibration(gateway, roles)

    try:
        data = asyncio.run(_go())
    except KeyboardInterrupt:
        print("\nCancelled — completed roles were kept.")
        return 130
    _print_report(data)
    print(f"\nWritten to {CALIBRATION_PATH}")
    print("Effective immediately: ceiling_for(role) prefers these measured values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
