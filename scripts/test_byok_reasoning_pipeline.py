#!/usr/bin/env python3
"""End-to-end BYOK reasoning pipeline test (inference only, no training).

Reads API credentials from freellmapi.txt (KEY=VALUE lines) and exercises:
  1. Direct LLM provider completion (custom BYOK adapter)
  2. Context budget assembly for gpt-oss-20b
  3. Quantitative / probabilistic reasoning prompts
  4. Tool registry (risk tools)
  5. Standard RAG pipeline generation
  6. RAFT-LM pipeline generation
  7. Benchmark smoke (single question)

Usage:
  python scripts/test_byok_reasoning_pipeline.py
  python scripts/test_byok_reasoning_pipeline.py --config freellmapi.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_API_FILE = REPO_ROOT / "freellmapi.txt"
DEFAULT_MODEL = "gpt-oss-20b"
CORPUS_DIR = REPO_ROOT / "data" / "benchmark_corpus" / "financial_policy"


@dataclass
class ApiConfig:
    base_url: str
    api_key: str
    model: str = DEFAULT_MODEL

    def apply_env(self) -> None:
        os.environ["FREELLM_API_KEY"] = self.api_key
        os.environ["RAFT_LLM_CONFIG_PATH"] = str(REPO_ROOT / "configs" / "llm_freellm.yaml")


@dataclass
class StepResult:
    name: str
    passed: bool
    detail: str = ""
    latency_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestReport:
    steps: list[StepResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for s in self.steps if s.passed)

    @property
    def failed(self) -> int:
        return sum(1 for s in self.steps if not s.passed)

    def add(self, step: StepResult) -> None:
        self.steps.append(step)
        status = "PASS" if step.passed else "FAIL"
        print(f"  [{status}] {step.name}: {step.detail}")


def load_api_config(path: Path) -> ApiConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"API config not found: {path}\n"
            f"Copy freellmapi.example.txt to freellmapi.txt and set BASE_URL, API_KEY, MODEL."
        )
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip()
    base_url = values.get("BASE_URL", "").rstrip("/")
    # OpenAI SDK docs often include /v1; our adapter appends /v1/chat/completions.
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    api_key = values.get("API_KEY", "")
    model = values.get("MODEL", DEFAULT_MODEL)
    missing = [k for k, v in [("BASE_URL", base_url), ("API_KEY", api_key)] if not v]
    if missing:
        raise ValueError(f"freellmapi.txt missing required keys: {', '.join(missing)}")
    return ApiConfig(base_url=base_url, api_key=api_key, model=model)


def write_runtime_config(api: ApiConfig) -> Path:
    """Patch llm_freellm.yaml base_url for this run."""
    import yaml

    cfg_path = REPO_ROOT / "configs" / "llm_freellm.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw["base_url"] = api.base_url
    raw["model_id"] = api.model
    cfg_path.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return cfg_path


async def step_endpoint_health(api: ApiConfig) -> StepResult:
    import httpx

    started = time.perf_counter()
    url = f"{api.base_url}/v1/models"
    headers = {"Authorization": f"Bearer {api.api_key}"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        models = [m.get("id", m) if isinstance(m, dict) else str(m) for m in data.get("data", [])]
        latency = (time.perf_counter() - started) * 1000
        model_ok = api.model in models or any(api.model in str(m) for m in models)
        detail = f"{len(models)} models listed; target={api.model!r} {'found' if model_ok else 'not in list'}"
        return StepResult("endpoint_health", True, detail, latency, {"models": models[:10]})
    except Exception as exc:
        latency = (time.perf_counter() - started) * 1000
        return StepResult("endpoint_health", False, str(exc), latency)


async def step_direct_completion(api: ApiConfig) -> StepResult:
    from src.llm_integration.factory import create_llm_provider_for_name

    started = time.perf_counter()
    provider = create_llm_provider_for_name("freellm")
    prompt = (
        "Reasoning: medium\n"
        "You are a risk analyst. In one sentence, explain why CVaR matters for tail risk."
    )
    try:
        completion = await provider.complete(prompt, api.model, max_tokens=256)
        latency = (time.perf_counter() - started) * 1000
        text = (completion.text or "").strip()
        ok = len(text) > 20
        detail = f"backend={completion.backend_id} tokens={completion.token_usage} chars={len(text)}"
        return StepResult("direct_byok_completion", ok, detail, latency, {"preview": text[:200]})
    except Exception as exc:
        latency = (time.perf_counter() - started) * 1000
        return StepResult("direct_byok_completion", False, str(exc), latency)


def step_context_assembly(api: ApiConfig) -> StepResult:
    from src.llm_integration.context import ContextBudget, ContextSegment, PRIORITY_RETRIEVED, PRIORITY_SYSTEM

    started = time.perf_counter()
    segments = [
        ContextSegment(name="system", content="Reasoning: high\nYou answer risk policy questions.", priority=PRIORITY_SYSTEM, protected=True),
        ContextSegment(name="ctx1", content="[chunk_a] Minimum CET1 ratio is 4.5% under Basel III.", priority=PRIORITY_RETRIEVED),
        ContextSegment(name="ctx2", content="[chunk_b] Liquidity coverage ratio must exceed 100%.", priority=PRIORITY_RETRIEVED),
    ]
    budget = ContextBudget(api.model)
    assembled = budget.assemble(segments)
    latency = (time.perf_counter() - started) * 1000
    ok = assembled.estimated_tokens > 0 and "CET1" in assembled.text
    detail = f"tokens={assembled.estimated_tokens}/{assembled.max_input_tokens} kept={assembled.segments_kept}"
    return StepResult("context_assembly", ok, detail, latency)


async def step_reasoning_prompts(api: ApiConfig) -> StepResult:
    from src.llm_integration.factory import create_llm_provider_for_name

    started = time.perf_counter()
    provider = create_llm_provider_for_name("freellm")
    prompts = {
        "probabilistic": (
            "Reasoning: medium\n"
            "A portfolio has 60% chance of +2% return and 40% chance of -5%. "
            "Which outcome is more likely? Answer in one sentence with probability reasoning."
        ),
        "quantitative": (
            "Reasoning: medium\n"
            "Given returns [-0.02, 0.01, -0.04, 0.03], which single metric best captures tail loss: VaR or CVaR? "
            "Answer in one sentence."
        ),
    }
    results: dict[str, str] = {}
    try:
        for kind, prompt in prompts.items():
            completion = await provider.complete(prompt, api.model, max_tokens=200)
            results[kind] = (completion.text or "").strip()[:150]
        latency = (time.perf_counter() - started) * 1000
        ok = all(len(v) > 10 for v in results.values())
        detail = f"answered {len(results)}/2 reasoning prompts"
        return StepResult("reasoning_prompts", ok, detail, latency, results)
    except Exception as exc:
        latency = (time.perf_counter() - started) * 1000
        return StepResult("reasoning_prompts", False, str(exc), latency)


def step_tool_registry() -> StepResult:
    from src.tools.registry import call_tool, list_tools

    started = time.perf_counter()
    tools = list_tools()
    cvar = call_tool("compute_cvar", {"returns": [-0.02, 0.01, -0.05, 0.03, -0.08], "alpha": 0.95})
    vol = call_tool("compute_volatility", {"returns": [-0.02, 0.01, -0.05, 0.03]})
    latency = (time.perf_counter() - started) * 1000
    ok = len(tools) >= 4 and cvar.get("value", 0) >= 0 and vol.get("value", 0) >= 0
    detail = f"tools={len(tools)} cvar={cvar.get('value')} vol={vol.get('value')}"
    return StepResult("tool_registry", ok, detail, latency)


def _run_rag_pipeline(pipeline_name: str, api: ApiConfig) -> StepResult:
    started = time.perf_counter()
    try:
        pytest = __import__("pytest")
        pytest.importorskip("langgraph")
    except Exception as exc:
        return StepResult(f"rag_{pipeline_name}", False, f"langgraph unavailable: {exc}")

    from src.rag.pipelines import RaftLMPipeline, StandardRAGPipeline
    from src.rag.retrievers import BenchmarkBudget

    budget = BenchmarkBudget(
        max_retrieval_depth=2,
        max_context_tokens=2048,
        model_provider=str(REPO_ROOT / "configs" / "llm_freellm.yaml"),
        generation_model=api.model,
        embedding_model="deterministic-stub",
        vector_store="in_memory",
    )
    query = "What is the minimum CET1 ratio?"
    try:
        if pipeline_name == "standard_rag":
            pipeline = StandardRAGPipeline(CORPUS_DIR, budget=budget)
        else:
            pipeline = RaftLMPipeline(CORPUS_DIR, budget=budget)
        result = pipeline.run(query)
        latency = (time.perf_counter() - started) * 1000
        answer = (result.answer or "").strip()
        ok = len(answer) > 10 and bool(result.citations)
        detail = f"answer_chars={len(answer)} citations={len(result.citations)}"
        return StepResult(
            f"rag_{pipeline_name}",
            ok,
            detail,
            latency,
            {"answer_preview": answer[:250], "pipeline": result.pipeline_name},
        )
    except Exception as exc:
        latency = (time.perf_counter() - started) * 1000
        return StepResult(f"rag_{pipeline_name}", False, str(exc), latency)


def step_benchmark_smoke(api: ApiConfig) -> StepResult:
    started = time.perf_counter()
    os.environ["MODEL_PROVIDER"] = str(REPO_ROOT / "configs" / "llm_freellm.yaml")
    os.environ["GENERATION_MODEL"] = api.model
    os.environ["EMBEDDING_MODE"] = "mock"
    os.environ["EMBEDDING_MODEL"] = "deterministic-stub"
    os.environ["VECTOR_STORE"] = "in_memory"
    os.environ["BENCHMARK_MODE"] = "live"
    try:
        from src.evals.benchmark_runner import run_standard_rag_benchmark

        report = run_standard_rag_benchmark(
            corpus_dir=CORPUS_DIR,
            out_dir=REPO_ROOT / "docs" / "benchmarks" / "results" / "byok-smoke",
            questions_limit=1,
        )
        latency = (time.perf_counter() - started) * 1000
        run_id = report.run_id
        ok = bool(report.standard.artifact_path)
        detail = f"run_id={run_id} artifact={report.standard.artifact_path}"
        return StepResult("benchmark_smoke", ok, detail, latency)
    except Exception as exc:
        latency = (time.perf_counter() - started) * 1000
        return StepResult("benchmark_smoke", False, str(exc), latency)


async def run_all(api: ApiConfig) -> TestReport:
    report = TestReport()
    api.apply_env()
    write_runtime_config(api)

    print(f"\n=== BYOK Reasoning Pipeline Test ===")
    print(f"Endpoint: {api.base_url}")
    print(f"Model:    {api.model}\n")

    report.add(await step_endpoint_health(api))
    report.add(await step_direct_completion(api))
    report.add(step_context_assembly(api))
    report.add(await step_reasoning_prompts(api))
    report.add(step_tool_registry())
    report.add(_run_rag_pipeline("standard_rag", api))
    report.add(_run_rag_pipeline("raft_lm", api))
    report.add(step_benchmark_smoke(api))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BYOK reasoning pipeline integration test")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_API_FILE,
        help="Path to freellmapi.txt (default: repo root freellmapi.txt)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run local-only checks (context, tools) without live API",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "docs" / "benchmarks" / "results" / "byok-pipeline-test.json",
        help="Write structured results JSON",
    )
    return parser.parse_args()


def run_offline(api: ApiConfig | None = None) -> TestReport:
    """Validate local pipeline wiring without calling a live LLM endpoint."""
    report = TestReport()
    model = api.model if api else DEFAULT_MODEL
    print("\n=== BYOK Reasoning Pipeline Test (offline) ===\n")
    report.add(step_context_assembly(ApiConfig(base_url="http://offline", api_key="offline", model=model)))
    report.add(step_tool_registry())
    try:
        pytest = __import__("pytest")
        pytest.importorskip("langgraph")
        from src.rag.pipelines import RaftLMPipeline, StandardRAGPipeline
        from src.rag.retrievers import BenchmarkBudget

        budget = BenchmarkBudget(
            max_retrieval_depth=2,
            model_provider="mock",
            generation_model="mock",
            embedding_model="deterministic-stub",
            vector_store="in_memory",
        )
        for name, cls in [("standard_rag", StandardRAGPipeline), ("raft_lm", RaftLMPipeline)]:
            started = time.perf_counter()
            try:
                result = cls(CORPUS_DIR, budget=budget).run("What is the minimum CET1 ratio?")
                latency = (time.perf_counter() - started) * 1000
                ok = bool(result.answer) and bool(result.citations)
                report.add(StepResult(f"rag_{name}_mock", ok, f"answer_chars={len(result.answer)}", latency))
            except Exception as exc:
                latency = (time.perf_counter() - started) * 1000
                report.add(StepResult(f"rag_{name}_mock", False, str(exc), latency))
    except Exception as exc:
        report.add(StepResult("rag_mock_import", False, str(exc)))
    return report


def main() -> int:
    args = _parse_args()
    if args.offline:
        report = run_offline()
        api = ApiConfig(base_url="offline", api_key="offline", model=DEFAULT_MODEL)
    else:
        try:
            api = load_api_config(args.config)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print("\nRunning offline checks only...\n", file=sys.stderr)
            report = run_offline()
            api = ApiConfig(base_url="missing", api_key="missing", model=DEFAULT_MODEL)
            args.json_out = args.json_out.with_name("byok-pipeline-test-offline.json")
            print(f"\n=== Offline Summary: {report.passed}/{len(report.steps)} passed ===")
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "mode": "offline",
                "model": api.model,
                "passed": report.passed,
                "failed": report.failed,
                "steps": [
                    {
                        "name": s.name,
                        "passed": s.passed,
                        "detail": s.detail,
                        "latency_ms": round(s.latency_ms, 1),
                    }
                    for s in report.steps
                ],
            }
            args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Report written: {args.json_out}")
            print("\nAdd freellmapi.txt with BASE_URL, API_KEY, MODEL=gpt-oss-20b then re-run without --offline.")
            return 2

        report = asyncio.run(run_all(api))

    print(f"\n=== Summary: {report.passed}/{len(report.steps)} passed ===")
    if report.failed:
        print("Failed steps:")
        for step in report.steps:
            if not step.passed:
                print(f"  - {step.name}: {step.detail}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "endpoint": api.base_url,
        "model": api.model,
        "passed": report.passed,
        "failed": report.failed,
        "steps": [
            {
                "name": s.name,
                "passed": s.passed,
                "detail": s.detail,
                "latency_ms": round(s.latency_ms, 1),
                **({"extra": s.extra} if s.extra else {}),
            }
            for s in report.steps
        ],
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nReport written: {args.json_out}")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
