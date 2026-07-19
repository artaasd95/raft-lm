# RAFT-LM v0.2 CodeQA notes

## Intent alignment

Product: train LLMs for financial risk-aware decision making via RL/preference/search; no inference plane.

## Findings addressed

| Area | Fix |
|------|-----|
| Layering | Packages split into `algorithms/`, `trainers/`, `search/`, `generation/` |
| Dead code | Removed `rag/`, `llm_integration/`, `evals/`, `demo/`, `deploy/` |
| Factory | Single dispatch in `src/trainers/factory.py` |
| Rollouts | `src/generation/mock.py` replaces serving adapters |
| Rewards | Registry + `rewards/custom/` with `compute()` API |
| Config | Central validation; `generation`, `distributed`, `training.env` fields |
| Stubs | LM RL backends label `status: smoke_complete` |
| Callbacks | Opt-in via `logging.callbacks: true` |

## Remaining Roadmap gaps (honest)

- Full GPU DPO/PEFT optimizer loops
- Real DDP wrap on causal LM (DDP today delegates MLP path)
- Multi-turn GiGPO agent environment
