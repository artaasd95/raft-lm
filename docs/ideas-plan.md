# Project 1: Risk‑Aware LLM Training (DPO, PPO, Nash, and related)

## Goal
- Train models that understand payoff logic, tail risk, and constraints; then make risk‑aware choices (or produce risk‑aware outputs). Cover supervised (SFT), preference learning (DPO), constrained RL (PPO/TRPO variants), and multi‑agent/Nash learning for trading or allocation policies.

## Where and how to start
- Choose a concrete initial task:
  - Classification: label synthetic or curated scenarios as low/medium/high risk.
  - Regression: predict 1‑day CVaR of a specific position (e.g., long call spread).
  - Decision: choose between two actions with different expected return and tail risk (e.g., enter trade A vs B under a CVaR constraint).
- Adopt an experiment harness from your Raft‑LM docs:
  - Produce complete artifacts per run (config, metrics, evaluation, logs, checkpoints, plots).
  - Minimum 3 seeds for comparative claims; statistical tests (t‑tests, effect sizes).
- Use synthetic data first for tight ground truth, then add real data.
  - Generate paths with heavy tails (Student‑t, α‑stable) via TailWarp or CPU EVT.
  - Bake in payoff functions, constraints, and stress scenarios.
- Define core metrics early:
  - Task metrics: accuracy/F1 (classification) or MSE/MAE (regression).
  - Risk metrics: CVaR (primary), tail event precision/recall, calibration error.
  - Decision metrics: constraint satisfaction rate, risk‑adjusted return (Sharpe/Sortino), maximum drawdown.

## Reading list (start here, then expand)
- From Statistical Consequences of Fat Tails:
  - Ch 3.10: X vs F(X) (focus training/evaluation on payoff functions, convexity/fragility).
  - Ch 2.2.19: VaR vs CVaR (adopt CVaR for tail risk; use VaR cautiously).
  - Ch 11: Probability Calibration Under Fat Tails (calibration pitfalls).
  - Ch 12: Single‑Point Forecasts (why to avoid naïve point estimates in tails).
  - Ch 13.3: Error Propagation (why VaR fails; design robust evaluation).
  - Ch 30: Tail Risk Constraints & Maximum Entropy (encode constraints).
  - Ch 26/27/25: Option pricing heuristics under power laws, unique measure (risk logic grounded in tradable forward/anchors and tail index).
- ML/Alignment/RL references:
  - PPO/TRPO (Schulman et al.): trust regions and implicit natural gradient; useful for constrained training.
  - DPO/preference learning papers: encode risk preferences over pairs (high return & high risk vs medium return & low risk).
  - Safe RL and constrained MDPs (Achiam et al., Constrained Policy Optimization; Lagrangian methods).
  - Risk‑sensitive RL (CVaR RL, distributional RL): Tamar et al., Chow et al., Bellemare et al.
  - Multi‑agent/Nash learning (PSRO, independent PPO, fictitious play; robust MARL min‑max).

## Specific trade/position risk metrics: how to pursue in LLM context
- Treat risk as a function of payoff, not just probability of X:
  - Define the position's payoff function g(x) explicitly (e.g., for options use static arbitrage and put‑call parity).
  - Anchor prices and tail index α (Ch 27) for robust tail option heuristics.
- For labels/targets:
  - Generate loss distribution of the specific position under heavy tails (Student‑t, α‑stable). Compute CVaR at chosen levels (e.g., 95%, 99%) for targets.
  - For classification tasks, bucket scenarios by CVaR thresholds (low/medium/high risk).
  - For decision tasks, label pairs with a preference that respects CVaR constraints or a utility U = expected return − λ·CVaR.
- Validate sample reliability:
  - Use κ metric (Ch 8) to set Monte Carlo scenario counts per asset/position.
  - Apply EVT (Ch 9) and shadow mean to ensure tails are not underestimated.
- Constraints:
  - Encode tail risk constraints (Ch 30) into training/evaluation (e.g., penalty when predicted policy violates CVaR limit).
- Produce per‑trade outputs:
  - CVaR and expected shortfall by horizon, drawdown profiles, fragility score (sensitivity to uncertainty), convexity indicator (benefits from uncertainty).
  - Confidence flags tied to κ and EVT diagnostics.

## Training plan: sequencing methods
- Phase 0–1 (Baselines):
  - Implement supervised baselines: classification/regression of risk metrics using SFT with standard losses. Track calibration and simple tail metrics.
  - Build risk metric library (CVaR, drawdown, Sharpe/Sortino, constraint violation rate).
- Phase 2 (Risk‑aware losses):
  - Add CVaR‑based penalty to base task loss; multi‑objective loss Total_Loss = α*Task + β*CVaR_penalty + γ*Constraint_violation.
  - Evaluate improvement in tail metrics and constraint satisfaction.
  - Consider distributional losses that emphasize extremes (quantile losses, pinball loss at high quantiles).
- Phase 3 (Preference learning, DPO):
  - Construct pair datasets where A vs B tradeoffs codify risk preferences.
  - Train DPO to prefer lower risk under similar return; or higher risk only if return compensates utility.
  - Evaluate pairwise accuracy and aggregate policy metrics (CVaR compliance).
- Phase 4 (Constrained PPO/TRPO):
  - Build a simple environment (episode = day; action = position size or trade selection; reward = realized P&L; constraints = CVaR max).
  - Train PPO with Lagrangian or TRPO‑style trust region plus constraint enforcement.
  - Measure risk‑adjusted returns and violation rates.
- Phase 5 (Multi‑agent/Nash learning):
  - Simulate adversary or market "opponent" (stress or liquidity shocks).
  - Apply PSRO/self‑play to converge toward policies robust under adversarial tails.
  - Evaluate worst‑case CVaR and robust returns.

## R&D roadmap and gates
- Gate A (Vertical slice done):
  - One baseline task and metric suite working; synthetic scenarios; CVaR computed; artifacts per run.
- Gate B (Risk‑aware loss effectiveness):
  - Show ≥10–15% improvement in tail metrics or constraint compliance vs baseline across multiple seeds.
- Gate C (Preference learning adds value):
  - DPO improves decision quality under tail constraints; pairwise risk preference accuracy > baseline by statistically significant margin.
- Gate D (Constrained RL viable):
  - PPO/TRPO maintains constraint compliance >95% while achieving competitive risk‑adjusted returns; training stable across seeds.
- Gate E (Robust/Nash behavior):
  - Under adversarial tails, policy performance degrades gracefully; worst‑case CVaR bounded per constraints.

## Pitfalls to avoid
- Over‑reliance on VaR; use CVaR/expected shortfall for evaluation and training signals.
- Single‑seed anecdotes; use multiple seeds and statistical tests.
- Ignoring sample reliability; use κ to decide scenario counts.
- Forecasting point values in tails; prefer intervals and tail quantiles.

## Outputs
- Per‑scenario risk predictions (JSON schema): risk_score, CVaR_xx, constraints_violated, confidence (κ/EVT), explanation (linking to payoff logic g(x)).
- Policy models and ablations showing tradeoffs between return and tail risk.