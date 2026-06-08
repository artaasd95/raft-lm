"""OpenAI function-calling JSON schemas for risk tools."""

from __future__ import annotations

from typing import Any, Callable, Dict


def _object_schema(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: list[str],
) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def cvar_schema() -> Dict[str, Any]:
    return _object_schema(
        name="compute_cvar",
        description="Compute CVaR (expected shortfall) on losses from returns.",
        properties={
            "returns": {"type": "array", "items": {"type": "number"}},
            "alpha": {"type": "number", "default": 0.95},
        },
        required=["returns"],
    )


def drawdown_schema() -> Dict[str, Any]:
    return _object_schema(
        name="compute_drawdown",
        description="Maximum drawdown from a return series.",
        properties={
            "returns": {"type": "array", "items": {"type": "number"}},
        },
        required=["returns"],
    )


def volatility_schema() -> Dict[str, Any]:
    return _object_schema(
        name="compute_volatility",
        description="Realized volatility of returns.",
        properties={
            "returns": {"type": "array", "items": {"type": "number"}},
            "annualize": {"type": "boolean", "default": False},
        },
        required=["returns"],
    )


def position_size_schema() -> Dict[str, Any]:
    return _object_schema(
        name="compute_position_size",
        description="Vol-target position size given risk budget.",
        properties={
            "returns": {"type": "array", "items": {"type": "number"}},
            "risk_budget": {"type": "number", "default": 0.02},
            "max_leverage": {"type": "number", "default": 1.0},
        },
        required=["returns"],
    )


TOOL_SCHEMAS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "compute_cvar": cvar_schema,
    "compute_drawdown": drawdown_schema,
    "compute_volatility": volatility_schema,
    "compute_position_size": position_size_schema,
}


def json_schema(tool_name: str) -> Dict[str, Any]:
    if tool_name not in TOOL_SCHEMAS:
        raise KeyError(f"Unknown tool: {tool_name}")
    return TOOL_SCHEMAS[tool_name]()
