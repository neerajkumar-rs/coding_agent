"""agent_core: the single agentic pipeline engine.

All entry points (FastAPI endpoints, TUI, CLI) must go through
``engine.run(request, workspace)``. There is exactly one pipeline.
"""

from agent_core.engine import Engine, RunRequest, RunReport

__all__ = ["Engine", "RunRequest", "RunReport"]
