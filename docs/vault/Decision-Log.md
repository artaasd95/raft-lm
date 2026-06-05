# Decision Log — RAFT-LM Vault

Decisions are numbered `RF-YYYY-NN`. Superseded entries remain for traceability.

---

## RF-2026-28 — Vault identity: Risk Aware Framework for Training LMs

**Date:** 2026-06  
**Status:** Accepted  
**Context:** Vault and code README emphasized enterprise RAG benchmarking. The research program’s durable goal is **risk-aware LM training** (engine labels, losses, policies), with RAG as one evaluation harness.

**Decision**

1. Canonical name: **Risk Aware Framework for Training LMs (RAFT-LM)**.
2. Primary vertical slice: **engine labels → optimizers → benchmark** (S7 data platform + S8 policy/logging/loaders).
3. Enterprise RAG benchmark (`financial_policy_v1`) stays **frozen** for Standard vs RAFT-LM regression; it does not define product identity.
4. Supersede S6 dashboard/RAG-primary expansion (tasks 31–35); do not schedule net-new RAG features ahead of S7/S8.

**Consequences**

- Update [Project-Hub.md](Project-Hub.md), code [README.md](../../README.md), and adapter docs (Meridian, RADA).
- Extend [BENCHMARK.md](../benchmarks/BENCHMARK.md) with a **risk-training** contract separate from the RAG showcase.
- Seed [issues/sprint-s7.yaml](../../issues/sprint-s7.yaml) and [Roadmap.md](../../Roadmap.md).

---

## RF-2026-02 — LangGraph for RAG orchestration

**Status:** Accepted (unchanged)  
**Doc:** [docs/adr/0002-rag-orchestration-framework.md](../adr/0002-rag-orchestration-framework.md)

RAG pipelines remain LangGraph-based; scope is regression and citation/evidence policy testing only.

---

## RF-2026-15 — Offline-first benchmark CI

**Status:** Accepted (unchanged)

Stub/mock embedding and generation modes are the CI default. Non-stub headline claims require a documented live run artifact path.
