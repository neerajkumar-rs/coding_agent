"""Application entry point exporting the FastAPI Master Controller."""

from __future__ import annotations

import logging
from typing import Any
from main import app, state

logger = logging.getLogger("app")

# Ensure legacy compatibility routes if queried
@app.get("/api/v1/health")
async def legacy_health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/api/v1/status")
async def legacy_status() -> dict[str, Any]:
    installed = await state.ollama.list_installed_models()
    return {
        "status": "online",
        "supervisor_vram": "0MB",
        "workspace": str(state.workspace_root),
        "models": installed,
        "config": state.config.to_dict(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
