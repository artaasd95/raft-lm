# RAFT-LM Project Structure

RL-first layout for risk-aware LLM fine-tuning.

```
raft-lm/
├── configs/
│   ├── methods/          # sft, dpo, grpo, gigpo, actor_critic, ...
│   ├── lora/             # default_lora, qlora
│   ├── rewards/
│   ├── data/
│   ├── distributed/
│   ├── logging/
│   ├── eval/
│   ├── search/
│   └── generation/       # mock rollout generator
├── src/
│   ├── algorithms/       # preference, on_policy, actor_critic, value
│   ├── trainers/         # factory + backends
│   ├── rewards/          # builtin + custom/
│   ├── search/           # PGTS, ReST-MCTS*, orchestrator
│   ├── generation/       # mock generator for rollouts
│   ├── data/pipeline/    # ingest → label → split
│   ├── models/           # MLP + causal_peft
│   ├── tools/            # risk tool schemas
│   ├── metrics/          # task + risk metrics
│   ├── evaluation/       # training eval only
│   ├── logging/          # local, wandb, comet
│   ├── domain/           # MethodSpec, LoRASpec
│   ├── application/      # train orchestrator
│   ├── envs/             # RiskAllocationEnv
│   ├── buffers/          # rollout, replay
│   └── training/         # loss_factory, callbacks (shared)
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── build_dataset.py
│   └── run_search.py
├── tests/
│   ├── unit/
│   └── integration/
└── docs/
```

## Entry points

| Goal | Command |
|------|---------|
| Train | `python scripts/train.py --config configs/methods/grpo.yaml` |
| Search | `python scripts/run_search.py --config configs/search/pgts.yaml --output out.jsonl` |
| Test | `pytest` |
