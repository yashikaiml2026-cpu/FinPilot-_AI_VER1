from __future__ import annotations

from typing import Any

from src.agent.agent import run_finpilot_agent


def run_agent_workflow(message: str, identified_customer_id: str | None = None, **agent_state: Any) -> dict[str, Any]:
    """Compatibility entry point routed through the single dynamic FinPilot agent."""
    return run_finpilot_agent(message, identified_customer_id=identified_customer_id, **agent_state)
