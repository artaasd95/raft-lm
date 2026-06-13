# Specialized Reasoning Overview

The specialized package introduces three task classes:
- probabilistic reasoning
- quantitative reasoning
- tool-aware reasoning

## Components
- src/data/probabilistic_dataset.py
- src/data/quantitative_dataset.py
- src/data/tool_call_dataset.py
- src/losses/probabilistic_losses.py
- src/losses/quantitative_losses.py
- src/losses/tool_aware_losses.py
- src/training/specialized_trainer.py

## Distributed Backends
- experiments/configs/distributed_ddp.yaml
- experiments/configs/distributed_fsdp.yaml

## LLM Integration Demo
python scripts/demo_llm_integration.py --adapter mock --model-id test
