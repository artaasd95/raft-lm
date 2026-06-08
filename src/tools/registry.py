"""ToolRegistry — list and dispatch risk tools."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from src.tools import risk_tools
from src.tools.schemas import TOOL_SCHEMAS, json_schema


ToolFn = Callable[..., Dict[str, Any]]

TOOLS: Dict[str, ToolFn] = {
    "compute_cvar": risk_tools.compute_cvar_tool,
    "compute_drawdown": risk_tools.compute_drawdown_tool,
    "compute_volatility": risk_tools.compute_volatility_tool,
    "compute_position_size": risk_tools.compute_position_size_tool,
}


class ToolRegistry:
    def list_tools(self) -> List[str]:
        return sorted(TOOLS.keys())

    def json_schema(self, name: str) -> Dict[str, Any]:
        return json_schema(name)

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in TOOLS:
            raise KeyError(f"Unknown tool: {name}")
        return TOOLS[name](**args)


_default_registry = ToolRegistry()


def list_tools() -> List[str]:
    return _default_registry.list_tools()


def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return _default_registry.call_tool(name, args)
