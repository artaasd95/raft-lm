# Benchmark Summary

**Corpus:** financial_policy_v1  
**Run ID:** 5f46a57c  
**Created:** 2026-05-30T15:17:34.294128+00:00

## Environment

| Field | Value |
|-------|-------|
| Benchmark mode | stub |
| Embedding model | deterministic-stub |
| Generation model | deterministic-stub |
| Vector store | in_memory |
| Model provider | stub |

## Run config

| Field | Value |
|-------|-------|
| Corpus path | G:\repositories\raft-lm\data\benchmark_corpus\financial_policy |
| top-k | 4 |
| Max context chars | 4096 |
| Run count | 1 |
| Seed | None |
| Pipeline | both |
| Policy version | 1.0.0 |

## Standard RAG vs RAFT-LM

| Metric | Standard RAG | RAFT-LM |
|--------|--------------|---------|
| Context Precision | 0.6 | 0.7 |
| Faithfulness | 0.5 | 0.6 |
| Severity events | 0 | 0 |
| Severity (legal / financial / compliance / operational) | 0/0/0/0 | 0/0/0/0 |
| Max severity | none | none |

_Ragas headline metrics are stubbed until S4; slots may be null in live JSON._

Artifacts: `report.json`, `metrics.csv`, `comparison_chart.json`
