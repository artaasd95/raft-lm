"""ToolRegistry — list and dispatch risk tools."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from src.tools import risk_tools
from src.tools.schemas import json_schema

ToolFn = Callable[..., Dict[str, Any]]

TOOLS: Dict[str, ToolFn] = {
    "compute_cvar": risk_tools.compute_cvar_tool,
    "compute_drawdown": risk_tools.compute_drawdown_tool,
    "compute_volatility": risk_tools.compute_volatility_tool,
    "compute_position_size": risk_tools.compute_position_size_tool,
}


def _apply_defaults(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    schema = json_schema(name)
    params = schema.get("function", {}).get("parameters", {})
    properties = params.get("properties", {})
    merged = dict(args)
    for key, spec in properties.items():
        if key not in merged and "default" in spec:
            merged[key] = spec["default"]
    return merged


def _validate_tool_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    schema = json_schema(name)
    params = schema.get("function", {}).get("parameters", {})
    required = params.get("required", [])
    properties = params.get("properties", {})
    merged = _apply_defaults(name, args)
    missing = [key for key in required if key not in merged]
    if missing:
        raise ValueError(f"Missing required tool arguments for {name}: {missing}")
    unknown = [key for key in merged if key not in properties]
    if unknown:
        raise ValueError(f"Unknown tool arguments for {name}: {unknown}")
    return merged


class ToolRegistry:
    def list_tools(self) -> List[str]:
        return sorted(TOOLS.keys())

    def json_schema(self, name: str) -> Dict[str, Any]:
        return json_schema(name)

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in TOOLS:
            raise KeyError(f"Unknown tool: {name}")
        validated = _validate_tool_args(name, args)
        return TOOLS[name](**validated)


_default_registry = ToolRegistry()


def list_tools() -> List[str]:
    return _default_registry.list_tools()


def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return _default_registry.call_tool(name, args)
