# Benchmark Summary

**Corpus:** financial_policy_v1  
**Run ID:** 89af48ba  
**Created:** 2026-06-25T14:31:22.978074+00:00

## Environment

| Field | Value |
|-------|-------|
| Benchmark mode | live |
| Embedding model | deterministic-stub |
| Generation model | gpt-oss-20b |
| Vector store | in_memory |
| Model provider | G:\repositories\raft-lm\configs\llm_freellm.yaml |

## Run config

| Field | Value |
|-------|-------|
| Corpus path | G:\repositories\raft-lm\data\benchmark_corpus\financial_policy |
| top-k | 4 |
| Max context chars | 4096 |
| Run count | 1 |
| Seed | None |
| Pipeline | standard_rag |
| Policy version | n/a |

## Standard RAG vs RAFT-LM

| Metric | Standard RAG | RAFT-LM |
|--------|--------------|---------|
| Context Precision | 0.6 | 0.0 |
| Faithfulness | 0.5 | 0.0 |
| Severity events | 0 | 0 |
| Severity (legal / financial / compliance / operational) | 0/0/0/0 | 0/0/0/0 |
| Max severity | none | none |



Artifacts: `report.json`, `metrics.csv`, `comparison_chart.json`
