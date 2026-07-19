# ADR 0003: Hybrid RL Architecture

## Status

Accepted — 2026-07-19

## Context

RAFT-LM started as a risk-aware supervised training framework (MLP + CVaR/tail losses) with auxiliary RAG inference. The project needs to signal RL engineering capability: preference optimization, on-policy LM RL, classical env RL, and extensible reward design — while keeping risk training and RAG/BYOK inference as first-class pillars.

## Decision

1. **Hybrid RL identity** — Add LLM alignment (SFT → DPO/KTO → PPO-LM/GRPO) and a Gymnasium risk-allocation env (PPO + DQN from scratch), unified by a pluggable reward layer.
2. **LoRA default path** — `transformers` + `peft` for all LM training/alignment; Unsloth optional for SFT acceleration only.
3. **Clean architecture layers** — `domain` (specs), `rewards`, `rl`, `alignment`, `application` (orchestrators), existing `training`/`rag`/`llm_integration`.
4. **RAG = inference plane** — Training via `method:` YAML; usage via RAG + BYOK/local models (`scripts/infer.py`).
5. **Documentation** — Keep Sphinx (Furo + MyST); expand toctrees; do not introduce MkDocs.
6. **Distributed honesty** — `ddp`/`fsdp` backends remain MLP process-group stubs; not marketed as production LLM multi-GPU RL.

## Consequences

- New packages and backends; config `method` discriminator and expanded `SUPPORTED_*` lists.
- PreferencePair and feedback pipelines wired to DPO/KTO trainers.
- README and Sphinx docs lead with RL + risk + inference narrative.
