"""Phase 18: MCP Integration tests.

Tests for:
- Tool registry CRUD
- Tool invocation
- MCP tool routing
- MCP-aware tool_router node
- /tools API endpoint
- Config defaults
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import settings


def _set_setting(name: str, value):
    object.__setattr__(settings, name, value)


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_list(self):
        from src.mcp.tool_registry import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            invoke_fn=lambda **kw: "result",
        )
        registry.register(tool)
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_get_tool(self):
        from src.mcp.tool_registry import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        tool = ToolDefinition(name="calc", description="Calculator")
        registry.register(tool)
        assert registry.get("calc") is not None
        assert registry.get("nonexistent") is None

    def test_unregister(self):
        from src.mcp.tool_registry import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        tool = ToolDefinition(name="temp", description="Temporary")
        registry.register(tool)
        registry.unregister("temp")
        assert registry.get("temp") is None

    def test_invoke_tool(self):
        from src.mcp.tool_registry import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        tool = ToolDefinition(
            name="echo",
            description="Echo input",
            invoke_fn=lambda **kw: f"echo: {kw.get('input', '')}",
        )
        registry.register(tool)
        result = registry.invoke("echo", input="hello")
        assert result == "echo: hello"

    def test_invoke_unknown_raises(self):
        from src.mcp.tool_registry import ToolRegistry

        registry = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            registry.invoke("nonexistent")

    def test_invoke_no_fn_raises(self):
        from src.mcp.tool_registry import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        tool = ToolDefinition(name="nofn", description="No function")
        registry.register(tool)
        with pytest.raises(ValueError, match="no invoke function"):
            registry.invoke("nofn")

    def test_clear(self):
        from src.mcp.tool_registry import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        registry.register(ToolDefinition(name="a", description="A"))
        registry.register(ToolDefinition(name="b", description="B"))
        registry.clear()
        assert registry.list_tools() == []


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestToolRegistrySingleton:
    def test_singleton_returns_same_instance(self):
        from src.mcp.tool_registry import get_tool_registry, reset_tool_registry

        reset_tool_registry()
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2
        reset_tool_registry()

    def test_reset_creates_new(self):
        from src.mcp.tool_registry import get_tool_registry, reset_tool_registry

        reset_tool_registry()
        r1 = get_tool_registry()
        reset_tool_registry()
        r2 = get_tool_registry()
        assert r1 is not r2
        reset_tool_registry()


# ---------------------------------------------------------------------------
# MCP tool routing tests
# ---------------------------------------------------------------------------


class TestMCPRouting:
    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig_mcp = settings.mcp_enabled
        orig_tools = settings.enable_tools
        yield
        _set_setting("mcp_enabled", orig_mcp)
        _set_setting("enable_tools", orig_tools)

    @patch("src.mcp.tool_router.get_breaker")
    @patch("src.mcp.tool_router.get_tool_registry")
    def test_mcp_selects_and_invokes_tool(self, mock_registry, mock_breaker):
        """MCP router should select and invoke a tool via LLM."""
        from src.mcp.tool_registry import ToolDefinition
        from src.mcp.tool_router import mcp_route_and_invoke

        mock_reg = MagicMock()
        mock_reg.list_tools.return_value = [
            ToolDefinition(name="calculator", description="Evaluate math")
        ]
        mock_reg.invoke.return_value = "42"
        mock_registry.return_value = mock_reg

        # Mock LLM to select the calculator tool
        from pydantic import BaseModel

        class FakeSelection(BaseModel):
            tool_name: str = "calculator"
            arguments: dict = {"expression": "6 * 7"}

        mock_cb = MagicMock()
        mock_breaker.return_value = mock_cb
        mock_cb.call.return_value = FakeSelection()

        results = mcp_route_and_invoke("What is 6 times 7?")
        assert len(results) == 1
        assert "calculator" in results[0]

    @patch("src.mcp.tool_router.get_tool_registry")
    def test_mcp_no_tools_returns_empty(self, mock_registry):
        """MCP router should return empty when no tools registered."""
        mock_reg = MagicMock()
        mock_reg.list_tools.return_value = []
        mock_registry.return_value = mock_reg

        from src.mcp.tool_router import mcp_route_and_invoke

        results = mcp_route_and_invoke("Test query")
        assert results == []


# ---------------------------------------------------------------------------
# Tool router node integration tests
# ---------------------------------------------------------------------------


class TestToolRouterMCPIntegration:
    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig_mcp = settings.mcp_enabled
        orig_tools = settings.enable_tools
        yield
        _set_setting("mcp_enabled", orig_mcp)
        _set_setting("enable_tools", orig_tools)

    def test_mcp_disabled_uses_regex(self):
        """When MCP is disabled, should use regex routing."""
        _set_setting("enable_tools", True)
        _set_setting("mcp_enabled", False)

        from src.graph.tool_node import tool_router

        state = {"question": "What is 2 + 3?"}
        result = tool_router(state)
        # Regex should detect the math expression
        assert isinstance(result["tool_results"], list)

    def test_tools_disabled_returns_empty(self):
        """When tools disabled, should return empty results."""
        _set_setting("enable_tools", False)

        from src.graph.tool_node import tool_router

        state = {"question": "Calculate 2 + 3"}
        result = tool_router(state)
        assert result["tool_results"] == []

    @patch("src.mcp.tool_router.mcp_route_and_invoke")
    def test_mcp_enabled_delegates(self, mock_mcp):
        """When MCP is enabled, should delegate to MCP router."""
        _set_setting("enable_tools", True)
        _set_setting("mcp_enabled", True)

        mock_mcp.return_value = ["calculator: 42"]

        from src.graph.tool_node import tool_router

        state = {"question": "What is 6 * 7?"}
        result = tool_router(state)
        assert "calculator: 42" in result["tool_results"]
        mock_mcp.assert_called_once()


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestToolsEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from api.app import app
        return TestClient(app, raise_server_exceptions=False)

    @patch("src.mcp.tool_registry.get_tool_registry")
    def test_tools_endpoint(self, mock_registry, client):
        from src.mcp.tool_registry import ToolDefinition, ToolRegistry

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="calculator",
            description="Math eval",
            parameters={"expression": {"type": "string"}},
            source="builtin",
        ))
        mock_registry.return_value = registry

        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["tools"][0]["name"] == "calculator"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_config_field_exists(self):
        assert hasattr(settings, "mcp_enabled")

    def test_defaults_disabled(self):
        assert settings.mcp_enabled is False
