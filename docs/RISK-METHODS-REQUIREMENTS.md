# Risk Methods: Base Requirements & Calculation Spec

This document defines **what** Raft-LM’s quantitative layer must support, **how** inputs/outputs are interpreted, and **how** implementations in `src/metrics/` map to these requirements. It aligns with `docs/research_notes/risk-categories-init.md` (category lenses) and `docs/IDEA-AND-ACTION-PLAN.md` (engine as source of truth for labels and inference).

---

## 1. Conventions (must be consistent everywhere)

| Topic | Specification |
|--------|----------------|
| **Simple returns** | \(r_t = \frac{P_t - P_{t-1}}{P_{t-1}}\). Default input to path metrics unless stated. |
| **Log returns** | \(\ell_t = \ln(P_t/P_{t-1})\). Use for additive aggregation over time; translate to simple when compounding wealth. |
| **Wealth / equity curve** | \(W_0 > 0\), \(W_t = W_{t-1}(1+r_t)\) for simple returns. Drawdown is defined on \(W_t\), not on \(\sum r_t\). |
| **Loss samples** | Vector \(L\) where **larger = worse** (e.g., absolute errors, dollar losses). Used by legacy `compute_var` / `compute_cvar`. |
| **VaR sign (returns-based)** | **Report VaR as a positive number = loss magnitude** at level \(\alpha\): e.g. 95% VaR = \(-Q_{0.05}(r)\) when quantiles refer to the return distribution (loss is left tail). |
| **CVaR / ES** | Expected shortfall: mean of returns **at or below** the VaR quantile (tail mean). Matches regulatory “ES” for continuous distributions; discrete samples use empirical tail mean. |
| **Annualization** | Document `periods_per_year` (e.g. 252). \(\sigma_{\text{ann}} = \sigma_{\text{step}} \sqrt{\text{periods\_per\_year}}\). |
| **Position / portfolio** | Weights \(w\) sum to gross/net definitions as documented; covariance \(\Sigma\) for linear (variance–covariance) methods. |

All new functions in code use explicit names (`*_returns`, `*_wealth`, `*_losses`) to avoid silent mixing.

---

## 2. Tier A — Core path & tail (P0)

These are **required** for baseline training labels, evaluation, and CVaR-style losses.

| Method | Definition / requirement | API (implementation) |
|--------|--------------------------|------------------------|
| **Historical VaR (returns)** | Empirical quantile of \(r\); report as positive loss magnitude | `var_historical_returns` |
| **Historical CVaR / ES (returns)** | Mean of \(r\) for \(r \le -\)VaR | `cvar_historical_returns` |
| **VaR / CVaR (loss samples)** | Quantile / tail mean on loss vector \(L\) | `compute_var`, `compute_cvar` |
| **Wealth from simple returns** | Compound to \(W_t\) | `wealth_from_simple_returns` |
| **Maximum drawdown** | \(\max_t \frac{\max_{s\le t} W_s - W_t}{\max_{s\le t} W_s}\) on \(W_t > 0\) | `max_drawdown_wealth`, `max_drawdown_from_returns` |
| **Batch CVaR (tensor)** | Empirical tail mean of per-example losses (differentiable via `topk`) | `batch_cvar_from_losses` |

**Consistency rule:** `var_historical_returns` uses the full return sample (left tail). `compute_var` expects **nonnegative loss magnitudes**; `losses_from_simple_returns` uses \(\max(0,-r)\) (down days only). The two VaRs answer different questions—do not mix without intent.

---

## 3. Tier B — Volatility, downside, asymmetry (P1)

| Method | Definition / requirement | API |
|--------|--------------------------|-----|
| **Realized volatility** | Std dev of returns; optional annualize | `realized_volatility` |
| **Downside deviation** | Std dev of \((r - \text{MAR})_-\) | `downside_deviation` |
| **Semi-variance** | Mean of squared negative deviations from MAR | `semi_variance` |
| **Sharpe ratio** | \(\frac{\mathbb{E}[r-r_f]}{\sigma(r)}\) | `sharpe_ratio` |
| **Sortino ratio** | \(\frac{\mathbb{E}[r-\text{MAR}]}{\text{downside deviation}}\) | `sortino_ratio` |
| **Portfolio variance / vol** | \(w^\top \Sigma w\), \(\sqrt{\cdot}\) | `portfolio_variance`, `portfolio_volatility` |
| **Parametric (linear) portfolio VaR** | Gaussian: \(\sqrt{w^\top\Sigma w}\cdot z_{1-\alpha}\) as **loss magnitude** | `portfolio_var_gaussian` |
| **Beta (vs benchmark)** | \(\text{Cov}(r,r_b)/\text{Var}(r_b)\) | `beta_vs_benchmark` |
| **Omega ratio** | \(\sum \max(r-L,0) / \sum \max(L-r,0)\) | `omega_ratio` |
| **Calmar / Sterling** | CAGR vs max / mean drawdown | `calmar_ratio`, `sterling_ratio` |
| **Ulcer index** | \(\sqrt{\mathbb{E}[\text{DD}_t^2]}\) on wealth | `ulcer_index_wealth`, `ulcer_index_from_returns` |
| **Information ratio** | \(\mu(r-r_b) / \sigma(r-r_b)\) | `information_ratio` |
| **Historical portfolio VaR/CVaR** | \(R w\) then tail metrics | `portfolio_var_historical`, `portfolio_cvar_historical` |
| **Marginal / component VaR** | Linear Gaussian decomposition | `marginal_var_gaussian`, `component_var_gaussian` |
| **Tail ratio** | CVaR / VaR (returns) | `tail_ratio_returns` |

---

## 4. Tier C — Position & exposure (P1)

| Method | Definition / requirement | API |
|--------|--------------------------|-----|
| **Gross / net exposure** | \(\sum |w_i|\), \(\sum w_i\) | `gross_exposure`, `net_exposure` |
| **Concentration (Herfindahl)** | \(\sum w_i^2\) (normalized weights) | `concentration_herfindahl` |
| **Constraint violation rate** | Fraction of samples above threshold | `constraint_violation_rate` |

*Note:* “Position risk” in linear models is often **marginal contribution to risk** or scenario PV01; those are **Phase 2 extensions**—documented here as future entries (`marginal_var`, `component_var`) once the covariance/scenario engine exists.

---

## 5. Tier D — Risk of ruin & survival (P0/P1)

| Method | Definition / requirement | API |
|--------|--------------------------|-----|
| **Consecutive-loss bound (i.i.d.)** | If each period has loss prob \(p\), \(P(\ge k\) consecutive losses\() \approx p^k\) (upper bound / stylized) | `probability_consecutive_losses` |
| **Gambler’s ruin (symmetric random walk)** | Classic formula ruin prob vs opponent; used for **illustration** only | `gambler_ruin_symmetric` |
| **GBM log-barrier sketch** | Optional rough survival intuition from drift/vol (document assumptions in code docstring) | `risk_of_ruin_gbm_log_barrier_approx` |

Full **Monte Carlo ruin** with path-dependent constraints is out of scope for Tier D but should use the same return/wealth conventions (Section 1).

---

## 6. Tier E — Volatility surface & smile (P2)

These support **options-aware** risk: smile/skew/term structure are inputs to stress and scenario engines, not a replacement for historical time-series VaR.

| Method | Definition / requirement | API |
|--------|--------------------------|-----|
| **Black–Scholes call (scalar)** | Standard BSM with carry \(q\) | `black_scholes_call_price` |
| **Implied vol (bisection)** | Invert BSM price → \(\sigma\) | `implied_volatility_bisection` |
| **1-D smile interpolation** | Linear IV in strike (or log-strike) | `interpolate_iv_1d` |
| **ATM index** | Index of strike nearest forward | `atm_strike_index` |
| **Smile skew proxy** | Central difference \(\partial \sigma / \partial K\) at ATM | `iv_skew_finite_difference` |
| **Total variance** | \(\sigma^2 T\) for a tenor slice | `total_implied_variance` |

**Extensions (documented, not all implemented day one):**

- SVI / SSVI parameterization of a slice  
- Arbitrage checks (butterfly, calendar)  
- Local vol from Dupire (research track)

---

## 7. Mapping to risk categories (from `risk-categories-init.md`)

| Category | Primary methods in this doc |
|----------|------------------------------|
| 1 Structural / ruin | §5, drawdown + constraints |
| 2 Volatility & noise | §3 realized vol, §6 IV surface |
| 3 Downside & tail | §2 VaR/CVaR |
| 4 Drawdown & pain | §2 max drawdown |
| 5 Asymmetry & convexity | §3 Sortino, semi-variance, smile skew §6 |
| 6 Exposure & leverage | §4 |
| 7 Liquidity | Future: spreads, impact (not in base Tier A–E) |
| 8–10 | Behavioral, narrative, systemic — mostly data/features, not core `metrics` |

---

## 8. Loss functions (compatibility)

| Loss | Uses metric layer |
|------|-------------------|
| `CVaRLoss` | `batch_cvar_from_losses` (same tail definition as `compute_cvar` on batch) |
| `TailAwareLoss` | Quantile-based weights aligned with §2 |

---

## 9. Versioning & testing

- Each non-trivial function has **unit tests** with closed-form or small hand-computed examples.  
- **Sign / convention tests:** VaR on synthetic Gaussian vs `scipy.stats.norm` optional; repo stays scipy-optional with fixed seeds.  
- Breaking changes to conventions require bumping this doc and `src/metrics/conventions.py` docstrings.

---

## 10. Implementation status

| Tier | Status |
|------|--------|
| A–E base APIs | Implemented in `src/metrics/risk_metrics.py`, `vol_surface.py`, `conventions.py` |
| Marginal/component VaR, liquidity | Spec only (future) |
| Full SVI / local vol | Spec only (future) |

For actionable project sequencing, see `docs/IDEA-AND-ACTION-PLAN.md`.

---

## 11. Breaking note: `compute_var` on loss samples

Earlier drafts used `np.quantile(losses, 1 - alpha)` for VaR on nonnegative losses. **That is incorrect** for the usual definition “α-quantile of the loss distribution” when larger loss is worse (e.g. α = 0.95 → 95th percentile). The implementation now uses `np.quantile(losses, alpha)`. If you had code assuming the old behavior, update it or use `var_historical_returns` for return series instead.
