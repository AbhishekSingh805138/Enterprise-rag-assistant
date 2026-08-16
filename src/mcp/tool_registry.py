"""MCP tool registry for extensible tool management.

Provides a central registry where tools can be registered with metadata
(name, description, parameters). The MCP tool router uses this registry
to discover available tools and route queries to them.

When MCP_ENABLED=true, tools are discovered from MCP servers.
When MCP_ENABLED=false, falls back to the built-in regex-based tool routing.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a registered tool."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    invoke_fn: Callable | None = None
    source: str = "builtin"  # "builtin", "mcp"


class ToolRegistry:
    """Central registry for tool discovery and invocation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._lock = threading.Lock()

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        with self._lock:
            self._tools[tool.name] = tool
            logger.debug("Registered tool: %s (%s)", tool.name, tool.source)

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        with self._lock:
            self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        with self._lock:
            return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        with self._lock:
            return list(self._tools.values())

    def invoke(self, name: str, **kwargs) -> str:
        """Invoke a tool by name with the given arguments."""
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        if tool.invoke_fn is None:
            raise ValueError(f"Tool {name} has no invoke function")
        return tool.invoke_fn(**kwargs)

    def clear(self) -> None:
        """Clear all registered tools."""
        with self._lock:
            self._tools.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ToolRegistry | None = None
_registry_lock = threading.Lock()


def get_tool_registry() -> ToolRegistry:
    """Return the singleton ToolRegistry. Thread-safe."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ToolRegistry()
            _register_builtin_tools(_registry)
            logger.info("ToolRegistry initialized with %d tools", len(_registry.list_tools()))
        return _registry


def reset_tool_registry() -> None:
    """Discard the singleton (for testing)."""
    global _registry
    with _registry_lock:
        _registry = None


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Register the built-in tools (calculator, data_lookup)."""
    try:
        from src.tools.calculator import calculator

        def calc_invoke(**kwargs):
            expr = kwargs.get("expression", "")
            return str(calculator.invoke(expr))

        registry.register(ToolDefinition(
            name="calculator",
            description="Evaluate mathematical expressions (e.g., '2 + 3 * 4')",
            parameters={"expression": {"type": "string", "description": "Math expression to evaluate"}},
            invoke_fn=calc_invoke,
            source="builtin",
        ))
    except ImportError:
        logger.debug("Calculator tool not available")

    try:
        from src.tools.data_lookup import data_lookup

        def lookup_invoke(**kwargs):
            query = kwargs.get("query", "")
            department = kwargs.get("department", "")
            return str(data_lookup.invoke({"query": query, "department": department}))

        registry.register(ToolDefinition(
            name="data_lookup",
            description="Look up data from a specific department in the knowledge base",
            parameters={
                "query": {"type": "string", "description": "The search query"},
                "department": {"type": "string", "description": "Target department"},
            },
            invoke_fn=lookup_invoke,
            source="builtin",
        ))
    except ImportError:
        logger.debug("Data lookup tool not available")
