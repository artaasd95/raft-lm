Overview
========

**RAFT-LM** is a **Risk-Aware RL Framework for Training and Aligning Language Models**.

Three pillars:

1. **Hybrid RL training** — supervised risk, DPO/KTO, PPO-LM/GRPO, classical env PPO/DQN
2. **Extensible rewards** — composable YAML reward recipes wired to risk metrics
3. **RAG + BYOK inference** — train with ``method:`` configs; serve with RAG and local/cloud LLMs

Quick links: :doc:`../getting-started`, :doc:`../training/lora-peft`, :doc:`../rewards/design`.

Build full docs: ``cd docs && make html``.
