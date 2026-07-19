# RAFT-LM Roadmap

**Identity:** Risk-Aware RL Framework for Training & Aligning LMs (see [ADR 0003](docs/adr/0003-hybrid-rl-architecture.md))

## Now — Hybrid RL redesign (shipped in tree)

| ID | Deliverable | Status |
|----|-------------|--------|
| RL-01 | Reward framework (`src/rewards/`) | Done |
| RL-02 | Classical env PPO + DQN | Done |
| RL-03 | DPO/KTO + PPO-LM/GRPO backends | Done |
| RL-04 | PEFT LoRA loader + `peft` backend | Done |
| RL-05 | `scripts/infer.py` inference plane | Done |
| RL-06 | Sphinx docs reorder + RL narrative | Done |
| RL-07 | Method YAML configs | Done |

## Next

| ID | Deliverable |
|----|-------------|
| RL-08 | Multi-seed RL benchmark table in README |
| RL-09 | GPU-marked PEFT/DPO end-to-end on tiny HF model |
| RL-10 | Preference JSONL from feedback → DPO train loop with optimizer |
| RL-11 | Real multi-GPU LLM RL (honest scope — not started) |

## Shipped (prior)

| Area | Path |
|------|------|
| Risk MLP + CVaR losses | `src/training/backends/mlp_backend.py` |
| Data platform | `src/data_platform/` |
| RAG benchmark harness | `scripts/run_benchmark.py` |
| Unsloth SFT | `src/training/backends/unsloth_trainer.py` |

## Explicit non-goals (near term)

- SAC/TD3/CQL classical suite
- Multi-agent / Nash
- C++/CUDA kernels
- MkDocs (Sphinx is canonical)
