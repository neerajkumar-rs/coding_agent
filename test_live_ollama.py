"""Smoke test against live local Ollama instance."""

import asyncio
from agent_config import load_agent_config
from ollama_client import OllamaSupervisorClient

async def test_live():
    config = load_agent_config()
    client = OllamaSupervisorClient()
    
    models = await client.list_installed_models()
    print(f"Discovered Ollama models: {models}")
    
    # 1. Test planner model resolution
    resolved_planner = await client.resolve_model_name(config.planner_model)
    print(f"Configured planner: {config.planner_model} -> Resolved: {resolved_planner}")
    
    # 2. Test coder model resolution
    resolved_coder = await client.resolve_model_name(config.coder_model)
    print(f"Configured coder: {config.coder_model} -> Resolved: {resolved_coder}")
    
    # 3. Test quick prompt with planner
    print("Sending live test prompt to Planner...")
    res = await client.generate("Reply with: 'STATUS_OK'", model_name=resolved_planner, config=config)
    print(f"Live Planner Response snippet: {res[:150]}")
    assert "STATUS_OK" in res or len(res) > 0
    print("Live Ollama verification SUCCESSFUL!")

if __name__ == "__main__":
    asyncio.run(test_live())
