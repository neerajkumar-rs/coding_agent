"""Single-command launcher: reuses a running Ollama when one is already up
(never kills ollama.exe processes it did not spawn), starts the FastAPI
controller as an owned child, runs a safety preflight, then opens the
interactive chat TUI.

Usage:
    python tui.py

On exit (Ctrl-C or /quit) only child processes spawned by this TUI are
terminated.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

try:
    from rich import box
    from rich.console import Console
    from rich.json import JSON
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError as exc:
    raise SystemExit("Run:  pip install rich") from exc


# ---------------------------------------------------------------------------
# Config — override via environment variables
# ---------------------------------------------------------------------------
FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = int(os.getenv("VIBE_PORT", "8000"))  # may fall forward 8001..8010
FASTAPI_BASE = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"


def _set_fastapi_port(port: int) -> None:
    """Repoint every FastAPI URL at *port* (busy-port fallback)."""
    global FASTAPI_PORT, FASTAPI_BASE, HEALTH_URL
    global STATUS_URL, DEPLOY_URL, INIT_URL, RUN_URL, CANCEL_URL, EVENTS_URL
    FASTAPI_PORT = port
    FASTAPI_BASE = f"http://{FASTAPI_HOST}:{port}"
    HEALTH_URL = FASTAPI_BASE + "/health"
    STATUS_URL = FASTAPI_BASE + "/workspace/status"
    DEPLOY_URL = FASTAPI_BASE + "/agent/deploy"
    INIT_URL = FASTAPI_BASE + "/workspace/init"
    RUN_URL = FASTAPI_BASE + "/agent/run"
    CANCEL_URL = FASTAPI_BASE + "/agent/cancel"
    EVENTS_URL = FASTAPI_BASE + "/agent/events"

OLLAMA_EXE = shutil.which("ollama") or str(
    Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
)

REQUIRED_MODELS = [
    "deepseek-r1:1.5b",   # planner  — resolved via fuzzy match
    "qwen2.5-coder:3b",   # coder    — exact or fuzzy
]

OLLAMA_TAGS_URL = OLLAMA_BASE + "/api/tags"
_set_fastapi_port(FASTAPI_PORT)  # initializes HEALTH_URL/RUN_URL/... consistently

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / ".agent_logs"
LOG_DIR.mkdir(exist_ok=True)

console = Console()

# Registry of child processes owned by this TUI session
_owned_procs: list[subprocess.Popen] = []
_log_handles: list               = []
_shutdown_done                   = False
_ollama_reused                   = False  # True when Ollama was already running
_cli_workspace: str | None       = None   # --workspace skips the picker
_cli_yes: bool                   = False  # --yes auto-answers prompts
# GPU safety: one lock shared by submit() and /benchmark — a benchmark never
# overlaps a plan/step on the 4 GB GPU.
_gpu_lock: asyncio.Lock | None   = None


def _get_gpu_lock() -> asyncio.Lock:
    global _gpu_lock
    if _gpu_lock is None:
        _gpu_lock = asyncio.Lock()
    return _gpu_lock


# ---------------------------------------------------------------------------
# Process utilities
# ---------------------------------------------------------------------------

# PIDs that must never be terminated: 0 (System Idle), 4 (System), ourselves.
_NEVER_KILL_PIDS = {0, 4}


def _parse_netstat_listeners(output: str, port: int) -> list[int]:
    """Parse `netstat -ano` output: only LISTENING rows whose LOCAL address
    ends in exactly :<port> (':80001' must not match 8000). PID 0/4 filtered."""
    pids: set[int] = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() not in ("TCP", "TCP6"):
            continue
        local, state, pid_str = parts[1], parts[3].upper(), parts[-1]
        if state != "LISTENING":
            continue
        # Local address is the LAST :port segment (handles 127.0.0.1:8000,
        # 0.0.0.0:8000, [::]:8000) with an exact port match.
        if local.rsplit(":", 1)[-1] != str(port):
            continue
        if pid_str.isdigit():
            pid = int(pid_str)
            if pid not in _NEVER_KILL_PIDS and pid != os.getpid():
                pids.add(pid)
    return sorted(pids)


def _pids_on_port(port: int) -> list[int]:
    """Return PIDs LISTENING on *port* (psutil if available, else netstat /
    lsof / ss). Never returns 0, 4, or our own PID."""
    try:
        import psutil  # optional dependency

        found: set[int] = set()
        for conn in psutil.net_connections(kind="tcp"):
            if (
                conn.status == "LISTEN"
                and conn.laddr
                and conn.laddr.port == port
                and conn.pid
                and conn.pid not in _NEVER_KILL_PIDS
                and conn.pid != os.getpid()
            ):
                found.add(conn.pid)
        return sorted(found)
    except ImportError:
        pass
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "tcp"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            )
        except Exception:
            return []
        return _parse_netstat_listeners(out, port)

    # POSIX: lsof (listeners only), then ss
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", "-iTCP", f":{port}", "-sTCP:LISTEN", "-t"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        return sorted(
            {int(p) for p in out.split() if p.strip().isdigit()
             and int(p) not in _NEVER_KILL_PIDS and int(p) != os.getpid()}
        )
    except Exception:
        pass
    pids: set[int] = set()
    try:
        out = subprocess.check_output(
            ["ss", "-ltnp"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[3].rsplit(":", 1)[-1] == str(port) and "pid=" in line:
                for token in line.split(","):
                    if token.startswith("pid="):
                        pid = int(token.split("=")[1])
                        if pid not in _NEVER_KILL_PIDS and pid != os.getpid():
                            pids.add(pid)
    except Exception:
        pass
    return sorted(pids)


def terminate_pid(pid: int) -> bool:
    """Best-effort terminate. NEVER raises; returns True if a kill was issued.

    Order: psutil -> taskkill (Windows) -> os.kill. FileNotFoundError
    (missing powershell/taskkill), PermissionError and OSError are caught.
    """
    if pid in _NEVER_KILL_PIDS or pid == os.getpid() or pid <= 0:
        console.print(f"  [yellow]!! refusing to terminate PID {pid}[/yellow]")
        return False
    try:
        import psutil

        try:
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                proc.kill()
            return True
        except psutil.NoSuchProcess:
            return True  # already gone
    except ImportError:
        pass
    except Exception as exc:
        console.print(f"  [yellow]!! psutil terminate failed for PID {pid}: {exc}[/yellow]")

    if sys.platform == "win32":
        taskkill = shutil.which("taskkill") or str(
            Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "taskkill.exe"
        )
        try:
            result = subprocess.run(
                [taskkill, "/PID", str(pid), "/T", "/F"],
                capture_output=True, check=False, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, PermissionError, OSError) as exc:
            console.print(f"  [yellow]!! taskkill failed for PID {pid}: {exc}[/yellow]")
    else:
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except (PermissionError, OSError) as exc:
            console.print(f"  [yellow]!! os.kill failed for PID {pid}: {exc}[/yellow]")
    return False


def _port_owner_is_our_controller(port: int) -> bool:
    """GET /health on *port*: True when it answers as THIS controller."""
    try:
        data = _http_get(f"http://{FASTAPI_HOST}:{port}/health", timeout=2)
        return data.get("status") == "online" and "agent_profile" in data
    except Exception:
        return False


def _local_build_id() -> str:
    """Same hash main.py exposes via /health (mtime+size of main.py)."""
    import hashlib

    main_path = PROJECT_ROOT / "main.py"
    try:
        st = main_path.stat()
        raw = f"{main_path.name}:{st.st_mtime_ns}:{st.st_size}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
    except OSError:
        return "unknown"


def _check_stale_controller(port: int, assume_yes: bool = False) -> bool:
    """Compare the running controller's build_id with main.py on disk.

    Returns True when it is safe to proceed (fresh server, or user declined a
    restart and we moved on). Offers — never forces — a restart of a stale
    controller: the old process is only killed after explicit confirmation,
    and PID 0/4 are never touched.
    """
    try:
        data = _http_get(f"http://{FASTAPI_HOST}:{port}/health", timeout=2)
    except Exception:
        return True  # nothing running — fine
    if data.get("status") != "online" or "agent_profile" not in data:
        return True  # not our controller — _kill_port handles it
    server_build = data.get("build_id")
    local_build = _local_build_id()
    if not server_build or server_build == local_build:
        return True  # fresh
    console.print(
        f"[yellow]!! The running controller on :{port} is STALE "
        f"(server build {server_build} ≠ disk build {local_build}). "
        "main.py changed since it started.[/yellow]"
    )
    restart = assume_yes or Prompt.ask(
        "Restart the controller with the current code?", choices=["y", "n"], default="y"
    ) == "y"
    if not restart:
        console.print("[dim]Keeping the stale controller; behaviour may be outdated.[/dim]")
        return True
    for pid in _pids_on_port(port):
        if pid in (0, 4) or pid == os.getpid():
            continue
        terminate_pid(pid)
    for _ in range(10):
        if not _pids_on_port(port):
            break
        time.sleep(0.5)
    return True


def _pid_image_name(pid: int) -> str:
    """Best-effort process image name (for the kill-confirmation prompt)."""
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            )
            return out.split('","')[0].strip('"')
        except Exception:
            pass
    return "unknown"


def _kill_port(port: int, label: str, assume_yes: bool = False) -> bool:
    """Handle foreign PIDs LISTENING on *port*.

    Returns True when the port is (or became) free, or is owned by our own
    already-running controller (which we then reuse). Never kills Ollama;
    never kills PID 0/4; asks y/N (default N) before killing anything else.
    """
    pids = _pids_on_port(port)
    if not pids:
        return True

    # Our own controller already serving? Reuse it like the running Ollama.
    if label == "FastAPI" and _port_owner_is_our_controller(port):
        console.print(
            f"  [bright_green]✓ Reusing already-running controller on :{port} "
            f"(PIDs {pids}; not owned, not killed)[/bright_green]"
        )
        return True

    for pid in pids:
        name = _pid_image_name(pid)
        if name.lower().startswith("ollama"):
            console.print(
                f"  [yellow]!! :{port} is held by Ollama (PID {pid}); will NOT kill it[/yellow]"
            )
            return False
        kill = assume_yes
        if not assume_yes:
            try:
                kill = Prompt.ask(
                    f"  [yellow]Port :{port} is used by {name} (PID {pid}). Kill it?[/yellow]",
                    choices=["y", "n"], default="n",
                ) == "y"
            except Exception:
                kill = False
        if kill:
            terminate_pid(pid)

    for _ in range(10):
        if not _pids_on_port(port):
            return True
        time.sleep(0.5)
    return not _pids_on_port(port)


def _spawn(args: list[str], log_name: str, **popen_kwargs) -> subprocess.Popen:
    """Spawn a child process owned by this TUI and redirect its output to a log file."""
    log_path = LOG_DIR / log_name
    fh = log_path.open("a", encoding="utf-8")
    _log_handles.append(fh)
    proc = subprocess.Popen(
        args,
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=os.environ.copy(),
        **popen_kwargs,
    )
    _owned_procs.append(proc)
    return proc


def _wait_for_http(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """Poll *url* until it returns HTTP 200 or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def _shutdown(reason: str = "exit") -> None:
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True
    console.print(f"\n[dim][STOP] Shutdown ({reason}) -- terminating child processes...[/dim]")
    for proc in _owned_procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in _owned_procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                terminate_pid(proc.pid)
    _owned_procs.clear()
    for fh in _log_handles:
        try:
            fh.close()
        except Exception:
            pass
    _log_handles.clear()


atexit.register(_shutdown, "atexit")

try:
    signal.signal(signal.SIGTERM, lambda *_: (_shutdown("SIGTERM"), sys.exit(0)))
except (OSError, ValueError):
    pass  # Not supported on all platforms / in threads


# ---------------------------------------------------------------------------
# Preflight safety checks
# ---------------------------------------------------------------------------

class CheckResult(NamedTuple):
    label: str
    ok: bool
    detail: str


def _check_ollama_binary() -> CheckResult:
    if OLLAMA_EXE and Path(OLLAMA_EXE).exists():
        return CheckResult("Ollama binary", True, OLLAMA_EXE)
    found = shutil.which("ollama")
    if found:
        return CheckResult("Ollama binary", True, found)
    return CheckResult("Ollama binary", False, "not found on PATH — install from https://ollama.ai")


def _check_ollama_running() -> CheckResult:
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=3) as r:
            if r.status == 200:
                return CheckResult("Ollama API", True, OLLAMA_TAGS_URL)
    except Exception:
        pass
    return CheckResult("Ollama API", False, f"{OLLAMA_TAGS_URL} not responding")


def _check_required_models() -> CheckResult:
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=5) as r:
            data = json.loads(r.read())
    except Exception as exc:
        return CheckResult("Required models", False, f"cannot query tags: {exc}")

    installed = [m.get("name", "") for m in data.get("models", [])]
    missing: list[str] = []
    for req in REQUIRED_MODELS:
        # fuzzy: req prefix must appear in any installed name
        req_base = req.split(":")[0].lower()
        if not any(req_base in m.lower() for m in installed):
            missing.append(req)

    if missing:
        pull_cmds = "  ".join(f"ollama pull {m}" for m in missing)
        return CheckResult(
            "Required models", False,
            f"Missing: {missing}\n     Pull with:  {pull_cmds}"
        )
    return CheckResult("Required models", True, f"installed: {installed}")


def _check_fastapi_port_free() -> CheckResult:
    pids = _pids_on_port(FASTAPI_PORT)
    if pids:
        return CheckResult(
            f"Port {FASTAPI_PORT} (FastAPI)", False,
            f"occupied by PIDs {pids} — will be killed"
        )
    return CheckResult(f"Port {FASTAPI_PORT} (FastAPI)", True, "free")


def _run_preflight_table() -> bool:
    """Run all checks, render a Rich table, return True only if all pass."""
    checks = [
        _check_ollama_binary(),
        _check_ollama_running(),
        _check_required_models(),
        _check_fastapi_port_free(),
    ]
    table = Table(title="Preflight checks", box=box.ROUNDED, border_style="bright_blue")
    table.add_column("Check", style="bright_cyan", width=26)
    table.add_column("Status", width=8)
    table.add_column("Detail", style="dim")

    all_ok = True
    for c in checks:
        icon   = "[bold bright_green]  PASS[/bold bright_green]" if c.ok else "[bold red]  FAIL[/bold red]"
        detail = c.detail if c.ok else f"[red]{c.detail}[/red]"
        table.add_row(c.label, icon, detail)
        if not c.ok:
            all_ok = False

    console.print(table)
    return all_ok


# ---------------------------------------------------------------------------
# Service startup
# ---------------------------------------------------------------------------

def _start_ollama() -> subprocess.Popen:
    exe = OLLAMA_EXE if (OLLAMA_EXE and Path(OLLAMA_EXE).exists()) else shutil.which("ollama")
    if not exe:
        raise RuntimeError("ollama executable not found — cannot start")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{OLLAMA_HOST}:{OLLAMA_PORT}"
    # 4GB VRAM ceiling: flash attention + single loaded model.
    env.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    env.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
    console.print(f"  [dim]→ ollama serve  (port {OLLAMA_PORT})[/dim]")
    return _spawn([exe, "serve"], "ollama_service.log", env=env)


def _start_fastapi() -> subprocess.Popen:
    console.print(f"  [dim]→ uvicorn main:app  (port {FASTAPI_PORT})[/dim]")
    return _spawn(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", FASTAPI_HOST, "--port", str(FASTAPI_PORT)],
        "fastapi_service.log",
    )


def start_all_services(assume_yes: bool = False) -> None:
    """Reuse a running Ollama (never kill ollama.exe we did not spawn), reuse
    an already-running controller, else start FastAPI as an owned child.
    Busy FastAPI port falls forward to the next free port (8001..8010)."""
    console.print("\n[bold bright_cyan]Starting services...[/bold bright_cyan]")

    # 1. Ollama: reuse if the API already answers; otherwise start our own.
    global _ollama_reused
    _ollama_reused = _check_ollama_running().ok
    if _ollama_reused:
        console.print("  [bright_green]✓ Reusing already-running Ollama (not owned by this TUI)[/bright_green]")
    else:
        _start_ollama()
        with Progress(
            SpinnerColumn(), TextColumn("[dim]Waiting for Ollama API..."), transient=True, console=console
        ) as prog:
            prog.add_task("", total=None)
            ok = _wait_for_http(OLLAMA_TAGS_URL, timeout=20)
        if not ok:
            _shutdown("Ollama failed to start")
            raise RuntimeError("Ollama did not respond within 20 s")
        console.print("  [bright_green]✓ Ollama API ready[/bright_green]")

    # 2. FastAPI: reuse our controller if it answers (and is not stale);
    #    otherwise reclaim the port or fall forward to a free one.
    _check_stale_controller(FASTAPI_PORT, assume_yes=assume_yes)
    if _port_owner_is_our_controller(FASTAPI_PORT):
        console.print(
            f"  [bright_green]✓ Reusing already-running controller on :{FASTAPI_PORT}[/bright_green]"
        )
        return
    if not _kill_port(FASTAPI_PORT, "FastAPI", assume_yes=assume_yes):
        fallback = _next_free_port(FASTAPI_PORT)
        if fallback is None:
            raise RuntimeError(
                f"Ports {FASTAPI_PORT}..8010 are all busy; free one or use --port"
            )
        console.print(
            f"  [yellow]!! :{FASTAPI_PORT} busy — falling forward to :{fallback}[/yellow]"
        )
        _set_fastapi_port(fallback)
    _start_fastapi()


def _next_free_port(start: int) -> int | None:
    """First port in start..8010 with no LISTENING owner."""
    for candidate in range(start + 1, 8011):
        if not _pids_on_port(candidate):
            return candidate
    return None

    # 6. Wait for FastAPI
    with Progress(
        SpinnerColumn(), TextColumn("[dim]Waiting for FastAPI..."), transient=True, console=console
    ) as prog:
        prog.add_task("", total=None)
        ok = _wait_for_http(HEALTH_URL, timeout=20)
    if not ok:
        _shutdown("FastAPI failed to start")
        raise RuntimeError("FastAPI did not respond within 20 s")
    console.print("  [bright_green]✓ FastAPI controller ready[/bright_green]")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _http_post(url: str, payload: dict, timeout: float = 600) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


async def get_json(url: str) -> dict:
    return await asyncio.to_thread(_http_get, url)


async def post_json(url: str, payload: dict) -> dict:
    return await asyncio.to_thread(_http_post, url, payload)


# ---------------------------------------------------------------------------
# Workspace picker + persisted TUI state
# ---------------------------------------------------------------------------

TUI_STATE_PATH = Path(__file__).resolve().parent / ".agent_tui_state.json"


def _load_tui_state() -> dict:
    """Tolerate missing/corrupt state file."""
    try:
        data = json.loads(TUI_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_tui_state(state: dict) -> None:
    """Atomic write: tmp file then replace."""
    tmp = TUI_STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(TUI_STATE_PATH)


def _remember_workspace(path: str) -> None:
    state = _load_tui_state()
    state["last_workspace"] = path
    recent = [path] + [p for p in state.get("recent", []) if p != path]
    state["recent"] = recent[:5]
    _save_tui_state(state)


def _valid_workspace(path: str | None) -> str | None:
    """Resolve a candidate; return the resolved path only if it is a dir."""
    if not path:
        return None
    try:
        resolved = Path(str(path).strip().strip('"')).expanduser().resolve()
        return str(resolved) if resolved.is_dir() else None
    except Exception:
        return None


def pick_workspace(assume_yes: bool = False) -> str:
    """Rich menu: 1) last used  2) current directory  3) custom path...

    Default = 1 when valid else 2. Custom paths are created on confirmation.
    """
    state = _load_tui_state()
    last = _valid_workspace(state.get("last_workspace"))
    current = str(Path.cwd())

    options: list[tuple[str, str]] = []
    if last:
        options.append(("1", f"Last used workspace: {last}"))
    options.append(("2", f"Current directory: {current}"))
    options.append(("3", "Custom path..."))

    default = "1" if last else "2"
    console.print(Panel(
        "\n".join(f"  [bright_cyan]{key})[/bright_cyan] {label}" for key, label in options),
        title="Choose workspace", border_style="bright_blue",
    ))

    choice = default
    if not assume_yes:
        valid = {k for k, _ in options}
        choice = Prompt.ask("Workspace", choices=sorted(valid), default=default)

    chosen: str | None = None
    if choice == "1" and last:
        chosen = last
    elif choice == "2":
        chosen = current
    else:
        while True:
            raw = Prompt.ask("Workspace path").strip().strip('"')
            candidate = Path(raw).expanduser()
            if candidate.is_dir():
                chosen = str(candidate.resolve())
                break
            if not candidate.exists():
                create = assume_yes or Prompt.ask(
                    f"[yellow]{candidate} does not exist. Create it?[/yellow]",
                    choices=["y", "n"], default="y",
                ) == "y"
                if create:
                    try:
                        candidate.mkdir(parents=True, exist_ok=True)
                        chosen = str(candidate.resolve())
                        break
                    except OSError as exc:
                        console.print(f"[red]Could not create: {exc}[/red]")
                        continue
            console.print("[red]Not a valid directory — try again.[/red]")

    _remember_workspace(chosen)
    return chosen


# ---------------------------------------------------------------------------
# Banner & UI helpers
# ---------------------------------------------------------------------------

def show_banner() -> None:
    console.print(
        Panel.fit(
            "[bold bright_cyan]OLLAMA MASTER CONTROLLER[/bold bright_cyan]\n"
            "[bright_magenta]Sequential VRAM Fencing | RTX 3050 4 GB ceiling[/bright_magenta]\n"
            "[dim]deepseek-r1:1.5b (Planner)  |  qwen2.5-coder:3b (Coder)[/dim]\n"
            "[dim]Reuses running Ollama · only own children are stopped on exit[/dim]\n"
            "[dim]CLI: --workspace PATH · --port N · --yes · --benchmark [bench args][/dim]",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
    )


def display_workspace(path: str) -> str:
    resolved = str(Path(path).expanduser().resolve())
    console.print(
        Panel(
            f"[bold bright_cyan]Active Workspace[/bold bright_cyan]\n"
            f"[white]{resolved}[/white]\n"
            "[dim]Generated code → .agent_staging/ · saved with /save[/dim]",
            border_style="bright_cyan",
        )
    )
    return resolved


def show_help() -> None:
    console.print(
        Panel(
            "/help        show commands\n"
            "/run         autonomous run through agent_core (plan → code → verify → save)\n"
            "/benchmark   benchmark role models (--quick, --roles, --compare, apply, last)\n"
            "/status      show service health + staged diff\n"
            "/workspace   change the active workspace path\n"
            "/save        save staged files into the workspace (/deploy alias)\n"
            "/cancel      cancel the in-progress run for this workspace\n"
            "/logs        tail planner · coder · sandbox logs\n"
            "/models      list installed Ollama models\n"
            "/clear       clear screen\n"
            "/quit        stop all services and exit",
            title="Commands",
            border_style="bright_cyan",
        )
    )


# ---------------------------------------------------------------------------
# Submit pipeline
# ---------------------------------------------------------------------------

async def submit(request_text: str, workspace_dir: str) -> None:
    lock = _get_gpu_lock()
    if lock.locked():
        console.print("[yellow]GPU busy (benchmark running?) — try again when it finishes.[/yellow]")
        return
    async with lock:
        await _submit_locked(request_text, workspace_dir)


EXPECTED_STAGES = ["ROUTE", "PLAN", "CODE", "CHECK", "VALIDATE", "SUMMARY"]


# ---------------------------------------------------------------------------
# Header: logo + system vitals (redrawn before each prompt; cached ~2s)
# ---------------------------------------------------------------------------

# ASCII-only logo: the Windows console codepage (cp1252) cannot encode
# box-drawing characters.
LOGO = (
    "[bold bright_cyan]+-------------------+[/bold bright_cyan]\n"
    "[bold bright_cyan]|[/bold bright_cyan]  [bold bright_magenta](*) OLLAMA AGENT[/bold bright_magenta] [bold bright_cyan]|[/bold bright_cyan]\n"
    "[bold bright_cyan]|[/bold bright_cyan]  [dim]local - 4GB GPU[/dim]  [bold bright_cyan]|[/bold bright_cyan]\n"
    "[bold bright_cyan]+-------------------+[/bold bright_cyan]"
)

_vitals_cache: tuple[float, str] = (0.0, "")
_last_trace: list[dict] = []  # stage timings from the latest pipeline run


def _vital_gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        )
        used, total, pct = [p.strip() for p in out.splitlines()[0].split(",")]
        return f"GPU: {used}/{total} MiB - {pct}%"
    except Exception:
        return "GPU: n/a"


def _vital_loaded_model() -> str:
    try:
        data = _http_get(OLLAMA_BASE + "/api/ps", timeout=2)
        models = data.get("models", [])
        if not models:
            return "Loaded: none"
        m = models[0]
        size_gb = round(int(m.get("size", 0)) / 1e9, 1)
        return f"Loaded: {m.get('name', '?')} ({size_gb} GB)"
    except Exception:
        return "Loaded: n/a"


def _vital_services() -> str:
    ollama = "READY" if _check_ollama_running().ok else "DOWN"
    try:
        _http_get(HEALTH_URL, timeout=2)
        fastapi = "READY"
    except Exception:
        fastapi = "DOWN"
    return f"Ollama: {ollama} - FastAPI: {fastapi}"


def _vital_last_run() -> str:
    if not _last_trace:
        return "Last run: —"
    parts = [f"{t.get('stage', '?')} {t.get('seconds', 0):.1f}s" for t in _last_trace]
    return "Last run: " + " ".join(parts)


def _collect_vitals(workspace: str) -> str:
    """Each vital in its own try/except — one failing probe never crashes the header."""
    lines = []
    for probe in (_vital_gpu, _vital_loaded_model, _vital_services):
        try:
            lines.append(probe())
        except Exception:
            lines.append("n/a")
    try:
        short = workspace
        if len(short) > 38:
            short = "..." + short[-35:]
        lines.append(f"Workspace: {short}")
    except Exception:
        pass
    try:
        lines.append(_vital_last_run())
    except Exception:
        pass
    return "\n".join(lines)


def print_header(workspace: str) -> None:
    """Compact two-column header: logo left, vitals right. Vitals cached ~2s."""
    global _vitals_cache
    now = time.monotonic()
    if now - _vitals_cache[0] > 2.0 or not _vitals_cache[1]:
        _vitals_cache = (now, _collect_vitals(workspace))
    console.print(Panel(
        Text.from_markup(LOGO) + Text("\n\n" + _vitals_cache[1]),
        border_style="bright_blue", box=box.ROUNDED, expand=False,
    ))


def pluralize(n: int, word: str) -> str:
    """'1 file' / '2 files'."""
    return f"{n} {word}" + ("" if n == 1 else "s")


async def _save_staged(workspace_dir: str) -> None:
    """List every staged file with (new)/(overwrites existing file), ask
    'Save? [y/N]' (default N), then save via the deploy endpoint."""
    diff_status = await post_json(STATUS_URL, {"workspace_dir": workspace_dir})
    if not diff_status.get("has_changes"):
        console.print("[dim]No staged changes.[/dim]")
        return
    dest_root = str(Path(workspace_dir).resolve())
    lines = []
    for diff_obj in diff_status.get("diffs", []):
        tag = "(new)" if diff_obj["action"] == "create" else "(overwrites existing file)"
        lines.append(f"  {diff_obj['file']}  [dim]{tag}[/dim]")
    console.print(
        f"[yellow]Saving to {dest_root}:[/yellow]\n" + "\n".join(lines)
    )
    # --yes auto-approves: only CHECK-passed files are ever staged, so
    # auto-approving the staged set is exactly "approve what passed CHECK".
    if _cli_yes:
        console.print("[dim]--yes: auto-saving staged (CHECK-passed) files.[/dim]")
        answer = "y"
    else:
        answer = Prompt.ask("[bold bright_green]Save? [y/N][/bold bright_green]",
                            choices=["y", "N"], default="N")
    if answer != "y":
        console.print(
            "[dim]Not saved. Files remain in .agent_staging/. "
            "Type /save to save them later.[/dim]"
        )
        return
    res = await post_json(DEPLOY_URL, {"workspace_dir": workspace_dir})
    saved = res.get("deployed_files", [])
    console.print(
        f"[bold bright_green]Saved {pluralize(len(saved), 'file')} to {dest_root}[/bold bright_green]"
    )
    for f in saved:
        console.print(f"  [dim]↳ {f}[/dim]")


def _print_trace(trace: list[dict]) -> None:
    """One line: 'ROUTE ok 0.0s | PLAN ok 12.3s | ...'.
    A stage missing from the trace prints as SKIPPED in red — a missing
    stage can never go unnoticed."""
    parts: list[str] = []
    seen = {t.get("stage") for t in trace}
    for entry in trace:
        stage = entry.get("stage", "?")
        status = entry.get("status", "ok")
        seconds = entry.get("seconds", 0.0)
        color = "bright_green" if status == "ok" else ("yellow" if status == "skipped" else "red")
        parts.append(f"[{color}]{stage} {status} {seconds:.1f}s[/{color}]")
    for stage in EXPECTED_STAGES:
        if stage not in seen:
            parts.append(f"[red]{stage} SKIPPED[/red]")
    console.print(" | ".join(parts))


async def _submit_locked(request_text: str, workspace_dir: str) -> None:
    try:
        # ── Initialise workspace context ──────────────────────────────────
        await post_json(INIT_URL, {"root_path": workspace_dir})

        # ── ONE call: ROUTE -> PLAN -> CODE -> CHECK -> VALIDATE -> SUMMARY ──
        console.print("[dim]⏳ Running pipeline (ROUTE → PLAN → CODE → CHECK → VALIDATE → SUMMARY)...[/dim]")
        with Progress(SpinnerColumn(), TextColumn("[dim]Pipeline running..."), transient=True, console=console) as p:
            p.add_task("", total=None)
            result = await post_json(RUN_URL, {"prompt": request_text, "workspace_dir": workspace_dir})

        # 1. Trace (also feeds the header's "Last run" vitals)
        global _last_trace
        _last_trace = result.get("trace", [])
        _print_trace(_last_trace)

        # 2a. Question path: print the answer and stop.
        if "answer" in result:
            console.print(Panel(result["answer"], title="Answer", border_style="bright_green"))
            return

        # 2b. Summary
        console.print(Panel(result.get("summary", "(no summary)"),
                            title="Summary", border_style="bright_blue"))
        for failed in result.get("failed_tasks", []):
            console.print(f"[red]NOT saved (failed): {failed['path']} — {failed['error']}[/red]")

        # 3. Staged diffs (only files that passed CHECK are staged)
        diff_status = await post_json(STATUS_URL, {"workspace_dir": workspace_dir})
        if diff_status.get("has_changes"):  # noqa: E501
            n = diff_status["staged_count"]
            a = diff_status["total_additions"]
            d = diff_status["total_deletions"]
            console.print(
                f"\n[yellow]STAGED: {pluralize(n, 'file')}  "
                f"[bright_green]+{a}[/bright_green] / [red]-{d}[/red][/yellow]"
            )
            for diff_obj in diff_status.get("diffs", []):
                console.print(
                    Panel(
                        diff_obj["diff"][:2000] or "(binary / empty diff)",
                        title=f"[{'bright_green' if diff_obj['action']=='create' else 'yellow'}]"
                              f"{diff_obj['action'].upper()}  {diff_obj['file']}[/]",
                        border_style="yellow",
                    )
                )
            # 4. Save prompt — default N
            await _save_staged(workspace_dir)
        else:
            console.print("[dim]No staged changes.[/dim]")

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        console.print(f"[bold red]HTTP {exc.code}:[/bold red] {body[:400]}")
    except Exception as exc:
        console.print(f"[bold red]Pipeline error:[/bold red] {exc}")


# ---------------------------------------------------------------------------
# Autonomous run with live task table (agent_core engine)
# ---------------------------------------------------------------------------

def _render_run_table(tasks: dict[str, dict], status: str, llm_calls: int) -> Table:
    table = Table(title=f"agent_core run — {status} (llm calls: {llm_calls})",
                  box=box.ROUNDED, border_style="bright_blue")
    table.add_column("Task", width=6)
    table.add_column("Name", style="bright_cyan", max_width=40)
    table.add_column("Status", width=14)
    table.add_column("Attempts", width=9, justify="right")
    for number, info in sorted(tasks.items(), key=lambda kv: int(kv[0])):
        status_str = info.get("status", "pending")
        color = {
            "complete": "bright_green", "running": "yellow",
            "failed": "red", "needs_human": "magenta",
        }.get(status_str, "dim")
        table.add_row(
            str(number), info.get("name", ""),
            f"[{color}]{status_str}[/{color}]", str(info.get("attempts", 0)),
        )
    return table


async def run_autonomous(request_text: str, workspace_dir: str) -> None:
    """/run — same single /agent/run pipeline as a typed request."""
    await submit(request_text, workspace_dir)


def _read_latest_run_state(workspace_dir: str) -> dict | None:
    """Read the newest .agent_runs/<id>/state.json for the live table."""
    runs_dir = Path(workspace_dir) / ".agent_runs"
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        reverse=True,
    )
    for d in candidates:
        state_path = d / "state.json"
        if state_path.exists():
            try:
                return json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# /benchmark slash command
# ---------------------------------------------------------------------------

_last_bench_report: dict | None = None  # most recent in-session report


def _bench_reports_dir(workspace: str) -> Path:
    return Path(workspace) / ".agent_runs" / "bench"


def _show_last_bench_report(workspace: str) -> None:
    bench_dir = _bench_reports_dir(workspace)
    reports = sorted(bench_dir.glob("bench-*.md"), reverse=True) if bench_dir.is_dir() else []
    if not reports:
        console.print("[yellow]No benchmark reports yet — run /benchmark first.[/yellow]")
        return
    newest = reports[0]
    console.print(Panel(
        newest.read_text(encoding="utf-8", errors="replace")[:4000],
        title=f"Latest report: {newest.name}", border_style="bright_blue",
    ))


def _print_bench_table(report: dict) -> None:
    table = Table(title="Benchmark results", box=box.ROUNDED, border_style="bright_blue")
    table.add_column("Role", style="bright_cyan")
    table.add_column("Model")
    table.add_column("OK", width=6)
    table.add_column("Secs", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Error", style="dim", max_width=40)
    for r in report.get("results", []):
        ok = "[bright_green]yes[/bright_green]" if r.get("ok") else "[red]no[/red]"
        table.add_row(
            r.get("role", ""), r.get("model", ""), ok,
            f"{r.get('seconds', 0):.2f}", f"{r.get('score', 0):.1f}",
            r.get("error", ""),
        )
    console.print(table)
    best = report.get("best_per_role", {})
    if best:
        console.print("[bold]Best per role:[/bold]")
        for role, model in best.items():
            console.print(f"  [bright_cyan]{role}[/bright_cyan] → [white]{model}[/white]")
    verdict = report.get("coexist_verdict")
    if verdict:
        console.print(f"[dim]Coexist: {verdict}[/dim]")


async def _apply_bench_roles(workspace: str) -> None:
    """Write best-per-role models into .agent under 'roles' after y/N."""
    global _last_bench_report
    if not _last_bench_report or not _last_bench_report.get("best_per_role"):
        console.print("[yellow]No benchmark results to apply — run /benchmark first.[/yellow]")
        return
    best = _last_bench_report["best_per_role"]
    console.print("[bold]Recommended models:[/bold]")
    for role, model in best.items():
        console.print(f"  {role} → [white]{model}[/white]")
    confirm = Prompt.ask("Write these into .agent under 'roles'?", choices=["y", "n"], default="n")
    if confirm != "y":
        console.print("[dim]Not applied.[/dim]")
        return

    import json as _json

    agent_path = Path(workspace) / ".agent"
    data: dict = {}
    if agent_path.exists():
        try:
            data = _json.loads(agent_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    roles = data.get("roles", {})
    for role, model in best.items():
        existing = roles.get(role, {}) if isinstance(roles.get(role), dict) else {}
        roles[role] = {**existing, "model": model}
    data["roles"] = roles
    agent_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    console.print("[bright_green]✓ roles written to .agent[/bright_green]")


async def run_benchmark_slash(arg_string: str, workspace: str) -> None:
    """/benchmark [args] — GPU-locked, asyncio-responsive, Ctrl-C safe."""
    global _last_bench_report
    import bench
    import shlex

    args = shlex.split(arg_string) if arg_string.strip() else []
    if args and args[0] == "last":
        _show_last_bench_report(workspace)
        return
    if args and args[0] == "apply":
        await _apply_bench_roles(workspace)
        return

    lock = _get_gpu_lock()
    if lock.locked():
        console.print("[yellow]GPU busy with another operation — try again when it finishes.[/yellow]")
        return

    bench_argv = ["--workspace", workspace, *args]
    console.print("[dim]Benchmark unloads models when done (keep_alive=0).[/dim]")
    async with lock:
        try:
            report, exit_code = await asyncio.to_thread(bench.run, bench_argv)
        except asyncio.CancelledError:
            console.print("[yellow]Benchmark cancelled — models unloaded, services untouched.[/yellow]")
            return
        except KeyboardInterrupt:
            console.print("[yellow]Benchmark cancelled.[/yellow]")
            return
    _last_bench_report = report
    if exit_code == 2:
        console.print("[bold red]Ollama unreachable — start it first.[/bold red]")
        return
    _print_bench_table(report)
    bench_dir = _bench_reports_dir(workspace)
    reports = sorted(bench_dir.glob("bench-*.*"), reverse=True) if bench_dir.is_dir() else []
    for path in reports[:2]:
        console.print(f"[dim]report: {path}[/dim]")


# ---------------------------------------------------------------------------
# Main TUI loop
# ---------------------------------------------------------------------------

async def main() -> None:
    show_banner()

    # ── Step 1: preflight ─────────────────────────────────────────────────
    console.print("\n[bold bright_cyan]── Preflight checks ──[/bold bright_cyan]")
    preflight_ok = _run_preflight_table()
    if not preflight_ok:
        fatal = [c for c in [
            _check_ollama_binary(),
            _check_required_models(),
        ] if not c.ok]
        if fatal:
            console.print(
                "[bold red]Fatal preflight failure — fix the issues above and retry.[/bold red]"
            )
            return
        console.print("[yellow]Non-fatal failures — continuing with service startup...[/yellow]")

    # ── Step 2: start services ────────────────────────────────────────────
    try:
        start_all_services()
    except RuntimeError as exc:
        console.print(f"[bold red]Service startup failed:[/bold red] {exc}")
        return

    # ── Step 3: post-startup health table ─────────────────────────────────
    console.print()
    table = Table(title="Service health", box=box.ROUNDED, border_style="bright_blue")
    table.add_column("Service",  style="bright_cyan", width=24)
    table.add_column("PID",      width=8)
    table.add_column("Status",   width=10)
    table.add_column("Endpoint", style="dim")

    if _ollama_reused:
        ollama_proc, fastapi_proc = None, (_owned_procs[0] if _owned_procs else None)
    else:
        ollama_proc = _owned_procs[0] if len(_owned_procs) > 0 else None
        fastapi_proc = _owned_procs[1] if len(_owned_procs) > 1 else None

    for label, proc, url, port in [
        ("Ollama serve",    ollama_proc,  OLLAMA_TAGS_URL, OLLAMA_PORT),
        ("FastAPI ctrl",    fastapi_proc, HEALTH_URL,      FASTAPI_PORT),
    ]:
        pid_str = str(proc.pid) if proc else "—"
        try:
            _http_get(url, timeout=3)
            status = "[bright_green]READY[/bright_green]"
        except Exception:
            status = "[red]OFFLINE[/red]"
        table.add_row(label, pid_str, status, url)

    console.print(table)
    console.print(
        Panel(
            "[bold bright_green]All services running under this process tree.[/bold bright_green]\n"
            "[dim]Ctrl-C or /quit will terminate Ollama + FastAPI cleanly.[/dim]",
            border_style="bright_green",
        )
    )

    # ── Step 4: check required models are actually present ─────────────────
    try:
        tags_data = _http_get(OLLAMA_TAGS_URL)
        installed = [m.get("name", "") for m in tags_data.get("models", [])]
        missing   = [
            r for r in REQUIRED_MODELS
            if not any(r.split(":")[0].lower() in m.lower() for m in installed)
        ]
        if missing:
            console.print(
                Panel(
                    "[yellow]The following models are not pulled yet:[/yellow]\n"
                    + "\n".join(f"  ollama pull {m}" for m in missing),
                    border_style="yellow",
                    title="⚠  Missing models",
                )
            )
    except Exception:
        pass

    # ── Step 5: interactive loop ───────────────────────────────────────────
    workspace = display_workspace(_cli_workspace or pick_workspace(assume_yes=_cli_yes))
    console.print("[dim]Type /help for commands. Enter a coding request to begin.[/dim]")

    try:
        while True:
            print_header(workspace)
            text    = Prompt.ask("[bold bright_green]you[/bold bright_green]")
            command = text.strip().lower()

            if command in {"/quit", "/exit", "q"}:
                break

            elif command == "/help":
                show_help()

            elif command == "/clear":
                console.clear()
                show_banner()

            elif command == "/workspace":
                workspace = display_workspace(pick_workspace(assume_yes=False))

            elif command == "/models":
                try:
                    data   = await get_json(OLLAMA_TAGS_URL)
                    models = [m.get("name", "") for m in data.get("models", [])]
                    console.print(Panel("\n".join(models) or "(none)", title="Installed Ollama models", border_style="bright_blue"))
                except Exception as exc:
                    console.print(f"[red]Could not list models: {exc}[/red]")

            elif command == "/status":
                try:
                    health = await get_json(HEALTH_URL)
                    console.print(JSON(json.dumps(health)))
                    diff   = await post_json(STATUS_URL, {"workspace_dir": workspace})
                    console.print(JSON(json.dumps({k: v for k, v in diff.items() if k != "diffs"})))
                except Exception as exc:
                    console.print(f"[red]{exc}[/red]")

            elif command in {"/save", "/deploy"}:  # /deploy kept as alias
                try:
                    await _save_staged(workspace)
                except Exception as exc:
                    console.print(f"[red]{exc}[/red]")

            elif command == "/run":
                request_text = Prompt.ask("Describe the change").strip()
                if request_text:
                    await run_autonomous(request_text, workspace)

            elif command == "/cancel":
                try:
                    res = await post_json(CANCEL_URL, {"workspace_dir": workspace})
                    console.print(JSON(json.dumps(res)))
                except Exception as exc:
                    console.print(f"[red]{exc}[/red]")

            elif command.startswith("/benchmark"):
                arg_string = text.strip()[len("/benchmark"):].strip()
                await run_benchmark_slash(arg_string, workspace)

            elif command == "/logs":
                for log_file in ("planner.log", "coder.log", "sandbox.log",
                                 "ollama_service.log", "fastapi_service.log"):
                    p = LOG_DIR / log_file
                    if p.exists():
                        tail = p.read_text(encoding="utf-8", errors="replace")[-1500:]
                        console.print(Panel(tail, title=log_file, border_style="bright_blue"))

            elif text.strip():
                await submit(text, workspace)

    except KeyboardInterrupt:
        console.print("\n[dim]Keyboard interrupt received.[/dim]")
    finally:
        _shutdown("TUI closed")


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def _parse_cli(argv: list[str]) -> "argparse.Namespace":
    """parse_known_args: unknown args are forwarded to bench."""
    import argparse

    parser = argparse.ArgumentParser(prog="tui.py", description="Ollama agent TUI launcher")
    parser.add_argument("--workspace", default=None, help="skip the workspace picker")
    parser.add_argument("--port", type=int, default=None, help="FastAPI port override")
    parser.add_argument("--yes", action="store_true", help="auto-answer prompts with defaults")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="headless: start/reuse Ollama only, run benchmark, print, exit",
    )
    known, unknown = parser.parse_known_args(argv)
    known.bench_args = unknown  # forwarded to bench.run()
    return known


async def _benchmark_headless(args: "argparse.Namespace") -> int:
    """--benchmark mode: Ollama only (NO FastAPI), run bench, exit."""
    import bench

    workspace = args.workspace or str(PROJECT_ROOT)
    # Ollama only — reuse if up, else start our own.
    global _ollama_reused
    _ollama_reused = _check_ollama_running().ok
    if not _ollama_reused:
        _start_ollama()
        if not _wait_for_http(OLLAMA_TAGS_URL, timeout=20):
            console.print("[bold red]Ollama unreachable — benchmark aborted[/bold red]")
            return 2
    bench_argv = ["--workspace", workspace, *args.bench_args]
    report, exit_code = await asyncio.to_thread(bench.run, bench_argv)
    return exit_code


def cli() -> None:
    args = _parse_cli(sys.argv[1:])
    global _cli_workspace, _cli_yes
    _cli_workspace = args.workspace
    _cli_yes = args.yes
    if args.port is not None:
        _set_fastapi_port(args.port)
    if args.benchmark:
        exit_code = asyncio.run(_benchmark_headless(args))
        _shutdown("benchmark done")
        raise SystemExit(exit_code)

    # Prefer the Textual UI; fall back to the Rich TUI when unavailable.
    try:
        import textual  # noqa: F401

        _textual_available = True
    except ImportError:
        _textual_available = False

    if not _textual_available:
        console.print("[yellow]Textual not installed — falling back to the Rich TUI "
                      "(pip install textual for the new layout).[/yellow]")
        asyncio.run(main())
        return

    # Textual path: start services, pick workspace, launch the app.
    show_banner()
    console.print("\n[bold bright_cyan]── Preflight checks ──[/bold bright_cyan]")
    _run_preflight_table()
    try:
        start_all_services(assume_yes=args.yes)
    except RuntimeError as exc:
        console.print(f"[bold red]Service startup failed:[/bold red] {exc}")
        return
    workspace = args.workspace or pick_workspace(assume_yes=args.yes)
    try:
        from tui_textual import run_textual

        run_textual(workspace)
    finally:
        _shutdown("TUI closed")


if __name__ == "__main__":
    cli()
