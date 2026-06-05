# Project Hub — RAFT-LM (Risk Aware Framework for Training LMs)

**Record:** RF-2026-28  
**Code repo:** [raft-lm](https://github.com/artaasd95/raft-lm)  
**Status:** Active — risk-training path (S7/S8); RAG benchmark is evaluation slice, not primary identity

## North star

Train and evaluate language models that **internalize risk** — engine-derived labels, risk-aware losses, and policy-driven train/eval — not merely retrieve and cite policy documents.

## What changed (S0-01)

| Prior narrative | Current narrative |
|-----------------|-------------------|
| RAFT-LM = enterprise RAG + distractor-aware retrieval | **Risk Aware Framework for Training LMs** — training infrastructure first |
| Benchmark corpus as product centerpiece | **Engine labels → optimizers → benchmark** as the vertical slice |
| Dashboard / multi-pipeline RAG expansion (S6) | **Superseded** — frozen `financial_policy_v1` RAG contract retained for regression only |

## Active workstreams

| Stream | Sprint | Code / docs |
|--------|--------|-------------|
| Risk data platform | S7 (DP-01…DP-08) | `docs/data-platform.md`, `src/data_platform/`, `configs/data/` |
| Policy + logging + loaders | S8 (01…03) | `src/training/policies/`, `src/logging/`, `src/models/loaders/` |
| Risk-training benchmark | S7-03+ | `docs/benchmarks/BENCHMARK.md` § Risk-training contract |
| RAG regression | Maintained | `src/rag/`, `docs/benchmarks/BENCHMARK.md` § Enterprise RAG |

## Adapter projects (portfolio)

| Project | Role | Doc |
|---------|------|-----|
| **Meridian** | Market/scenario feature adapter → engine label inputs | [Projects/Meridian/Architecture.md](../../Projects/Meridian/Architecture.md) |
| **RADA** | Risk analytics & decision annotations adapter | [Projects/RADA/Architecture.md](../../Projects/RADA/Architecture.md) |

## Key artifacts

- [Decision-Log.md](Decision-Log.md) — ADR-style decisions including RF-2026-28
- [Portfolio/Agent-Stack-Definitions.md](Portfolio/Agent-Stack-Definitions.md) — agent roles on the training path
- [issues/sprint-s7.yaml](../../issues/sprint-s7.yaml) — S7 execution cards
- [Roadmap.md](../../Roadmap.md) — phased delivery

## Out of scope (this phase)

- S6 RAG dashboard expansion (tasks 31–35 superseded)
- SP-TRAIN-* / SP-BENCH-* multi-seed live runs (start at task 50)
- Production serving / model zoo
