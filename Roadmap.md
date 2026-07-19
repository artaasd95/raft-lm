# RAFT-LM Roadmap

**Identity:** Risk-Aware Fine-Tuning for training LLMs on financial risk-aware decision making.

## Shipped (v0.2 restructure)

| ID | Deliverable | Status |
|----|-------------|--------|
| RL-01 | RL-first package layout (`algorithms/`, `trainers/`, `search/`) | Done |
| RL-02 | Remove inference/RAG/serving plane | Done |
| RL-03 | GiGPO advantage + smoke backend | Done |
| RL-04 | ReST-MCTS* search scaffold + CLI | Done |
| RL-05 | Custom rewards directory + registry | Done |
| RL-06 | Config tree rewrite + generation mock | Done |
| RL-07 | Training-only Docker + manual CI | Done |
| RL-08 | wandb/comet/ray optional extras | Done |

## Next

| ID | Deliverable |
|----|-------------|
| RL-09 | Full GPU PEFT/DPO optimizer loops on tiny HF model |
| RL-10 | Real multi-GPU LM RL (DDP/FSDP on causal LM) |
| RL-11 | Multi-turn GiGPO agent env with tool calls |
| RL-12 | Preference JSONL from search → DPO train at scale |

## Explicit non-goals

- Inference serving, RAG product, BYOK pipelines
- Production deployment containers (training-only Docker only)
