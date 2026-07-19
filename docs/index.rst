raft-lm
=======

**Risk-Aware RL Framework for Training and Aligning Language Models** — hybrid RL (DPO, PPO-LM, GRPO, env PPO/DQN), extensible rewards, supervised risk training, and RAG/BYOK inference.

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
   :caption: Rewards

   rewards/design
   rewards/custom-rewards

.. toctree::
   :maxdepth: 2
   :caption: Inference

   inference/rag
   inference/byok-and-local
   inference/serving-adapters

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   config/yaml-reference
   artifacts-schema

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   benchmarks/BENCHMARK
   benchmarks/reproduce

.. toctree::
   :maxdepth: 1
   :caption: Architecture & ADRs

   architecture/overview
   adr/0003-hybrid-rl-architecture

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
