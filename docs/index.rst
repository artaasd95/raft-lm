raft-lm
=======

**Risk-Aware Fine-Tuning for training LLMs** on financial risk-aware decision making.

Build docs: ``cd docs && make html``

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting-started
   narrative/overview

.. toctree::
   :maxdepth: 2
   :caption: Training

   training/supervised-risk
   training/lora-peft
   training/preference-dpo-kto
   training/rl-ppo-grpo
   training/classical-env-rl

.. toctree::
   :maxdepth: 2
   :caption: Rewards & search

   rewards/design
   rewards/custom-rewards
   unlabeled-guidance

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   config/yaml-reference
   artifacts-schema

.. toctree::
   :maxdepth: 1
   :caption: Architecture & ADRs

   architecture/overview
   adr/0003-hybrid-rl-architecture

.. toctree::
   :maxdepth: 2
   :caption: API

   api/index
