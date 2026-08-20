"""MCP-aware tool routing using LLM function calling.

When MCP_ENABLED=true, uses the tool registry to discover available tools
and lets the LLM decide which tool(s) to call via function calling.
Falls back to regex-based routing from the original tool_node.py when disabled.
"""
from __future__ import annotations

import logging

from config import settings
from src.llm_pool import get_llm
from src.mcp.tool_registry import get_tool_registry
from src.prompts import register
from src.resilience.circuit_breaker import CircuitBreakerOpen, get_breaker

logger = logging.getLogger(__name__)


def mcp_route_and_invoke(question: str) -> list[str]:
    """Route a question to tools using the MCP registry.

    Uses LLM function calling to select appropriate tools from the registry,
    then invokes them and returns results.

    Returns:
        List of tool result strings.
    """
    registry = get_tool_registry()
    tools = registry.list_tools()

    if not tools:
        logger.debug("No tools registered in MCP registry")
        return []

    # Build tool descriptions for the LLM
    tool_descriptions = "\n".join(
        f"- **{t.name}**: {t.description}" for t in tools
    )

    # Ask the LLM which tools to use
    try:
        from pydantic import BaseModel, Field

        class ToolSelection(BaseModel):
            """LLM selection of tools to invoke."""
            tool_name: str = Field(description="Name of the tool to use, or 'none' if no tool is needed")
            arguments: dict = Field(default_factory=dict, description="Arguments to pass to the tool")

        prompt = register(
            "mcp_tool_selection",
            "v1",
            [
            ("system",
             "You are a tool selection assistant. Given a user question and "
             "available tools, decide if any tool should be used.\n\n"
             f"Available tools:\n{tool_descriptions}\n\n"
             "If no tool is relevant, set tool_name to 'none'.\n"
             "If a tool is relevant, provide the tool_name and arguments."),
            ("human", "{question}"),
        ])

        cb = get_breaker(
            "llm",
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout,
        )
        chain = prompt | get_llm().with_structured_output(ToolSelection)
        selection: ToolSelection = cb.call(chain.invoke, {"question": question})

        if selection.tool_name == "none":
            return []

        # Invoke the selected tool
        try:
            result = registry.invoke(selection.tool_name, **selection.arguments)
            logger.info("MCP tool invoked: %s -> %d chars", selection.tool_name, len(str(result)))
            return [f"{selection.tool_name}: {result}"]
        except ValueError as e:
            logger.warning("MCP tool invocation failed: %s", e)
            return []

    except CircuitBreakerOpen as exc:
        logger.warning("LLM circuit open during MCP routing: %s", exc)
        return []
    except Exception:
        logger.debug("MCP tool routing failed", exc_info=True)
        return []
