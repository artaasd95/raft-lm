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
        "gigpo",
        "ppo_env",
        "dqn_env",
        "ray",
    }
)

METHOD_TO_BACKEND = {
    "supervised": "mlp",
    "sft": "peft",
    "dpo": "dpo",
    "kto": "kto",
    "ppo_lm": "ppo_lm",
    "grpo": "grpo",
    "gigpo": "gigpo",
    "ppo_env": "ppo_env",
    "actor_critic": "ppo_env",
    "dqn_env": "dqn_env",
}

UNSLOTH_ALLOWED_METHODS = frozenset({"supervised", "sft"})
