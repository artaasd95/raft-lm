"""Known-answer tests for risk tools."""

import math

import pytest

from src.tools.registry import call_tool, list_tools


@pytest.fixture
def sample_returns():
    return [0.01, -0.02, 0.015, -0.03, 0.005]


class TestToolRegistry:
    def test_list_tools(self):
        tools = list_tools()
        assert len(tools) == 4
        assert "compute_cvar" in tools

    def test_dispatch_all_tools(self, sample_returns):
        for name in list_tools():
            result = call_tool(name, {"returns": sample_returns})
            assert "value" in result
            assert "provenance" in result


class TestCVaRTool:
    def test_normal(self, sample_returns):
        r = call_tool("compute_cvar", {"returns": sample_returns, "alpha": 0.8})
        assert r["value"] >= 0

    def test_empty(self):
        r = call_tool("compute_cvar", {"returns": []})
        assert r["value"] == 0.0

    def test_nan_filtered(self):
        r = call_tool("compute_cvar", {"returns": [float("nan"), -0.1, 0.1]})
        assert math.isfinite(r["value"])

    def test_zero_losses(self):
        r = call_tool("compute_cvar", {"returns": [0.1, 0.2, 0.3]})
        assert r["value"] == 0.0

    def test_all_negative(self):
        r = call_tool("compute_cvar", {"returns": [-0.1, -0.2, -0.3]})
        assert r["value"] > 0


class TestDrawdownTool:
    def test_normal(self, sample_returns):
        r = call_tool("compute_drawdown", {"returns": sample_returns})
        assert r["value"] >= 0

    def test_empty(self):
        assert call_tool("compute_drawdown", {"returns": []})["value"] == 0.0

    def test_flat(self):
        assert call_tool("compute_drawdown", {"returns": [0.0, 0.0]})["value"] == 0.0

    def test_nan(self):
        r = call_tool("compute_drawdown", {"returns": [float("nan"), -0.5]})
        assert math.isfinite(r["value"])

    def test_crash(self):
        r = call_tool("compute_drawdown", {"returns": [0.1, -0.5]})
        assert r["value"] > 0


class TestVolatilityTool:
    def test_normal(self, sample_returns):
        r = call_tool("compute_volatility", {"returns": sample_returns})
        assert r["value"] > 0

    def test_empty(self):
        assert call_tool("compute_volatility", {"returns": []})["value"] == 0.0

    def test_zero_vol(self):
        assert call_tool("compute_volatility", {"returns": [0.01, 0.01]})["value"] == 0.0

    def test_annualize(self, sample_returns):
        r = call_tool("compute_volatility", {"returns": sample_returns, "annualize": True})
        assert r["units"] == "annualized"

    def test_nan(self):
        r = call_tool("compute_volatility", {"returns": [float("nan"), 0.01, -0.01]})
        assert math.isfinite(r["value"])


class TestPositionSizeTool:
    def test_normal(self, sample_returns):
        r = call_tool("compute_position_size", {"returns": sample_returns, "risk_budget": 0.02})
        assert 0 <= r["value"] <= 1.0

    def test_empty(self):
        assert call_tool("compute_position_size", {"returns": []})["value"] == 0.0

    def test_zero_vol(self):
        assert call_tool("compute_position_size", {"returns": [0.01, 0.01]})["value"] == 0.0

    def test_zero_budget(self, sample_returns):
        assert call_tool("compute_position_size", {"returns": sample_returns, "risk_budget": 0})["value"] == 0.0

    def test_max_leverage_cap(self, sample_returns):
        r = call_tool(
            "compute_position_size",
            {"returns": sample_returns, "risk_budget": 1.0, "max_leverage": 0.5},
        )
        assert r["value"] <= 0.5
