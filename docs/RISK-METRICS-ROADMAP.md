# Risk Metrics: Future Backlog & Actionable Plan

Companion to `docs/RISK-METHODS-REQUIREMENTS.md` (definitions) and `docs/IDEA-AND-ACTION-PLAN.md` (project phases). This file lists **what is not yet implemented** and **how to schedule** work.

---

## 1. Already in code (reference)

| Area | Functions (non-exhaustive) |
|------|----------------------------|
| Tail / path | `var_historical_returns`, `cvar_historical_returns`, `compute_var`, `compute_cvar`, `batch_cvar_from_losses` |
| Drawdown / wealth | `wealth_from_simple_returns`, `max_drawdown_wealth`, `max_drawdown_from_returns` |
| Vol / downside | `realized_volatility`, `downside_deviation`, `semi_variance`, `sharpe_ratio`, `sortino_ratio` |
| Linear portfolio | `portfolio_variance`, `portfolio_volatility`, `portfolio_var_gaussian`, `beta_vs_benchmark`, exposures |
| Ruin (stylized) | `probability_consecutive_losses`, `gambler_ruin_symmetric`, `risk_of_ruin_gbm_log_barrier_approx` |
| Options slice | `black_scholes_call_price`, `implied_volatility_bisection`, smile interp, skew proxy (`src/metrics/vol_surface.py`) |

---

## 2. Phased backlog

### Phase F1 — Pain, asymmetry, multi-asset historical (**implemented**)

**Goal:** Metrics that appear in benchmarks and “convexity / pain” categories without new data infra.

| Metric | Purpose | API |
|--------|---------|-----|
| Omega ratio | Gain/loss mass above threshold | `omega_ratio` |
| Calmar ratio | CAGR / max DD | `calmar_ratio` |
| Ulcer index | RMS of drawdowns | `ulcer_index_wealth`, `ulcer_index_from_returns` |
| Drawdown diagnostics | Mean DD, series | `drawdown_series_wealth`, `average_drawdown_wealth` |
| Sterling ratio | CAGR / average DD | `sterling_ratio` |
| Moments | Skew / excess kurtosis | `skewness`, `excess_kurtosis` |
| Information ratio | Active return / tracking error | `information_ratio` |
| Historical portfolio tail | VaR/CVaR from return matrix × weights | `portfolio_var_historical`, `portfolio_cvar_historical` |
| Gaussian decomposition | Marginal / component VaR | `marginal_var_gaussian`, `component_var_gaussian` |
| Tail ratio | CVaR / VaR (returns) | `tail_ratio_returns` |

**Status:** Implemented in `src/metrics/risk_metrics.py` with tests in `tests/unit/test_metrics.py` (`TestRoadmapF1Metrics`). Gaussian quantiles use `statistics.NormalDist` (no SciPy).

### Phase F2 — Liquidity & microstructure (**in progress**)

| Metric | Purpose | API |
|--------|---------|-----|
| Amihud illiquidity | \|return\| / dollar volume | `amihud_illiquidity` |
| Roll’s spread estimator | From price autocovariance | `roll_spread_estimator` |
| Volume z-score | Regime flag for thin markets | `volume_zscore` |

**Status:** Implemented in `src/metrics/risk_metrics.py` with tests in `tests/unit/test_metrics.py` (`TestRoadmapF2Metrics`). Dataset adapters shipped in `src/data/adapters.py` (`build_aligned_panel`, `compute_f2_liquidity_features`).

**Needs:** Hook into production dataset builders when raw feeds include stable volume columns.

### Phase F3 — Dependence & systemic

| Metric | Purpose | API |
|--------|---------|-----|
| Correlation / rolling β to factors | Category 10 | `rolling_correlation`, `rolling_beta` |
| Sample copula tail dependence | Upper/lower tail λ | `sample_copula_tail_dependence` |
| Diversification ratio | (w'σ) / √(w'Σw) | `diversification_ratio` |

**Status:** Implemented in `src/metrics/risk_metrics.py` with tests in `tests/unit/test_metrics.py` (`TestRoadmapF3Metrics`). Multi-asset panel alignment + feature aggregation added in `src/data/adapters.py` (`build_aligned_panel`, `compute_f3_dependence_features`).

**Needs:** Wire factor definitions from research config for consistent rolling-β semantics.

### Phase F4 — Options surface (advanced, **implemented primitives**)

| Item | Purpose |
|------|---------|
| SVI / SSVI fit per slice | Smooth, arbitrage-aware smile |
| Butterfly / calendar no-arb checks | Validation |
| Local vol (Dupire) | Scenario engine |

**Status:** Implemented in `src/metrics/vol_surface.py` with tests in `tests/unit/test_vol_surface.py`: `fit_svi_slice`, `fit_ssvi_slice`, `butterfly_no_arb_check`, `calendar_no_arb_check`, `dupire_local_vol`.

**Needs:** Curated option chain snapshots for robust calibration; optional upgrade to optimizer-backed calibration (`scipy.optimize`) for tighter production fits.

### Phase F5 — Simulation engines

| Item | Purpose |
|------|---------|
| Monte Carlo portfolio VaR | Non-Gaussian, user copula |
| Path-dependent ruin | Barriers, margin calls |
| RL / scenario env hooks | Reward + constraint signals |

**Needs:** `src/data` scenario generators + performance budget (`06-PERFORMANCE-PROTOCOL.md`).

---

## 3. Suggested sequencing (sprints)

| Sprint | Focus | Exit criteria |
|--------|--------|----------------|
| **S1** | F1 implementation + tests | All F1 APIs merged; CI green on numpy-only path |
| **S2** | Wire F1 metrics into `scripts/evaluate.py` / eval JSON | At least 3 new fields in `evaluation.json` on sample run |
| **S3** | F2 spike on one CSV with volume | One Amihud + volume feature in dataset builder |
| **S4** | F3 on 3+ asset synthetic panel | Rolling corr + diversification ratio in report |

---

## 4. How to add a metric (checklist)

1. Add definition row to `RISK-METHODS-REQUIREMENTS.md` (Tier table) if public API.  
2. Implement in `src/metrics/risk_metrics.py` (or `vol_surface.py` if options-only).  
3. Export via `src/metrics/__init__.py` (`_LAZY_RISK` set).  
4. Unit test in `tests/unit/test_metrics.py` or focused file.  
5. If training labels use it, document schema in dataset README or research note.

---

## 5. Revision history

- **2026-04-26:** Initial roadmap; Phase F1 metrics shipped (`omega_ratio`, `calmar_ratio`, ulcer/sterling, moments, IR, historical portfolio VaR/CVaR, marginal/component VaR, `tail_ratio_returns`). PyTorch limited to `batch_cvar_from_losses` only; `portfolio_var_gaussian` uses `NormalDist.inv_cdf`.
