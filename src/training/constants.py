"""Training backend and method constants (no heavy imports)."""

from __future__ import annotations

SUPPORTED_BACKENDS = frozenset(
    {
        "mlp",
        "unsloth",
        "peft",
        "ddp",
        "fsdp",
        "dpo",
        "kto",
        "ppo_lm",
        "grpo",
        "ppo_env",
        "dqn_env",
    }
)

METHOD_TO_BACKEND = {
    "supervised": "mlp",
    "dpo": "dpo",
    "kto": "kto",
    "ppo_lm": "ppo_lm",
    "grpo": "grpo",
    "ppo_env": "ppo_env",
    "dqn_env": "dqn_env",
}
