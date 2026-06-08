"""Unit tests for OpenAI function-calling schemas."""

from src.tools.registry import ToolRegistry
from src.tools.schemas import TOOL_SCHEMAS, json_schema


def test_all_tools_have_schema():
    registry = ToolRegistry()
    for name in registry.list_tools():
        assert name in TOOL_SCHEMAS


def test_json_schema_structure():
    schema = json_schema("compute_cvar")
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "compute_cvar"
    assert "returns" in fn["parameters"]["properties"]


def test_schema_registerable_with_mock_client():
    registry = ToolRegistry()
    tools = [registry.json_schema(n) for n in registry.list_tools()]
    assert len(tools) == 4
    names = {t["function"]["name"] for t in tools}
    assert "compute_cvar" in names
    assert "compute_drawdown" in names
