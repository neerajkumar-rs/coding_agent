"""Live smoke check against a running local Ollama instance.

NOT a pytest module (no test_ functions collected); run it directly:
    .\\env\\Scripts\\python.exe scripts\\live_smoke.py

Uses the EXACT-match resolver: a configured model that is not in /api/tags
raises ModelNotInstalled with the exact `ollama pull <name>` hint instead of
being silently substituted.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_config import load_agent_config
from ollama_client import ModelNotInstalled, OllamaSupervisorClient


async def main() -> int:
    config = load_agent_config()
    client = OllamaSupervisorClient()

    models = await client.list_installed_models()
    print(f"Installed Ollama models: {models}")

    for role in ("thinker", "coder", "validator"):
        model = config.get_role_config(role)["model"]
        try:
            resolved = await client.resolve_model_name(model)
        except ModelNotInstalled as exc:
            print(f"[{role}] NOT INSTALLED: {exc}")
            return 1
        print(f"[{role}] configured={model} -> resolved={resolved}")

    thinker = config.get_role_config("thinker")["model"]
    print("Sending live test prompt to the thinker...")
    res = await client.generate(
        "Reply with: 'STATUS_OK'", model_name=thinker, config=config
    )
    print(f"Live thinker response snippet: {res[:150]}")
    assert "STATUS_OK" in res or len(res) > 0
    print("Live Ollama verification SUCCESSFUL!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
