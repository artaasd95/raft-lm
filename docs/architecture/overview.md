# Architecture overview

RAFT-LM has two complementary planes:

## Training plane

- **Supervised risk** — MLP + CVaR/tail losses
- **Preference** — DPO/KTO on `PreferencePair`
- **On-policy LM RL** — PPO-LM, GRPO with composite rewards
- **Classical RL** — Gymnasium env + PPO/DQN from scratch
- **LoRA** — `transformers`+`peft` default; Unsloth optional for SFT

Packages: `src/training/`, `src/alignment/`, `src/rl/`, `src/rewards/`

## Inference plane

- **RAG** — retrieval-augmented generation
- **BYOK / local** — Ollama, vLLM, LiteLLM adapters
- **Optional LoRA** — adapter load at serve time

Packages: `src/rag/`, `src/llm_integration/`, `scripts/infer.py`

See ADR [0003-hybrid-rl-architecture.md](../adr/0003-hybrid-rl-architecture.md).
