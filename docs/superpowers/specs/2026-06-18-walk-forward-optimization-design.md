# Design: Walk-Forward Strategy Optimization

**Date**: 2026-06-18
**Branch**: `002-walk-forward-optimization`
**Spec**: `specs/002-walk-forward-optimization/spec.md`
**Status**: Approved

## Summary

Replace total-return-only strategy parameter selection with a research-only robust optimization flow. Extends existing `src/backtest` and `backtest_agent` surfaces with deterministic walk-forward folds, risk-adjusted candidate scoring, transaction-cost assumptions, regime performance summaries, optimization evidence generation, API/UI result exposure, and safety framing.

## Scope

Full implementation: T001–T056 across Phase 1–6 + Final (all user stories).

**Execution strategy**: TDD-first with intra-phase parallelism (Approach B). Tests written before implementation within each phase. Phase dependencies respected: Phase 1 → Phase 2 → US1 → US2 → US3 → US4.

**Out of scope (deferred)**: PostgreSQL, Pinecone, Neo4j, OpenSearch, Redis, user accounts, portfolios, multi-store indexing.

---

## Architecture

### Module Map

```
src/backtest/
├── optimizer.py      ← existing, untouched (optimize_backtest compatible)
├── strategy.py       ← existing, reused (run_backtest, BacktestParams)
├── data_loader.py    ← existing, monkeypatched in tests
├── robust.py         ← NEW: metrics, scoring, fold generation, guardrails
└── regime.py         ← NEW: regime classification, per-regime summaries

src/evidence/evidence_builder.py  ← extended
src/graph/state.py                ← GraphState extended
src/storage/report_store.py       ← extended
src/agents/backtest_agent.py      ← extended
src/agents/coordinator_agent.py   ← extended
src/agents/evaluator_agent.py     ← extended
src/agents/rewrite_agent.py       ← extended
main.py                           ← POST /backtest/optimize added
app.py                            ← Streamlit robust optimization UI added
```

**Constraint**: `optimize_backtest` in `optimizer.py` is not modified. Existing tests and callers remain unaffected.

---

## Core Data Flow

```
POST /backtest/optimize
  → Pydantic request validation
  → data_loader (monkeypatchable)
  → WalkForwardConfig → fold list generation
  → pre-check: can we produce ≥3 folds? (fail fast)

  → per-fold execution:
      1. train window: Optuna n_trials
         - train_trial_filter: reject candidates where in-sample MDD > 25% OR trades < 30
         - train_trial_score = weighted(Sharpe, Sortino, −MDD, −turnover)  [in-sample only]
         - best_params selected by train_trial_score
      2. test window: OOS evaluation of best_params
         - NO train data used in test evaluation (leakage guard)
      3. fold-level EvidenceItem generation

  → valid OOS fold count check: < 3 → insufficient_data + limitation_only_output

  → fold aggregation:
      - median OOS return
      - worst-fold return
      - fold return stddev
      - max OOS drawdown
      - total completed OOS trades

  → final_robust_score (5-component weighted, from OOS data):
      30% OOS return + 25% risk-adjusted + 20% drawdown control
      + 15% worst-fold resilience + 10% stability/turnover penalty

  → OOS robust_label_allowed guardrail:
      - False if aggregated OOS completed_trades < 30
      - False if aggregated OOS max_drawdown > 25%
      - When False: strong positive expressions PROHIBITED in all output text

  → regime classification (post-hoc, explanation only, NOT used in trial selection):
      - bull: rolling return ≥ +5%
      - bear: rolling return ≤ −5%
      - sideways: rolling return in (−5%, +5%)
      - high-volatility: realized vol ≥ 70th percentile
      - low-volatility: realized vol ≤ 30th percentile
      - low_confidence when: completed_trades < 10 OR trading_days < 60

  → manual_baseline + passive_baseline comparison (same periods, same costs)
      - baseline EvidenceItem generated for each comparison claim

  → aggregate EvidenceItem generation (robust score, MDD, trade count, worst fold, baselines, regimes)

  → local JSON/MD report storage

  → Coordinator (report generation with limitations + disclaimer)
  → Evaluator:
      - unsupported claim check (numeric claim without EvidenceItem)
      - limitation missing check ("과거 시뮬레이션" framing required)
      - evidence missing check (key metrics must map to EvidenceItem)
      - robust_label_allowed=False → strong expression check
  → Rewrite (max 2 attempts):
      - INVARIANT: evidence_id references must not change
      - INVARIANT: numeric values must not change
      - removes unsafe wording, preserves evidence-grounded reasoning
  → if rewrite limit exceeded: status = completed_degraded + evaluator_errors list
  → final response
```

---

## Key Data Contracts

### train_trial_score (in-sample, per Optuna trial)
```
score = 0.35 * z(Sharpe)
      + 0.35 * z(Sortino)
      + 0.20 * z(−MDD_pct)
      + 0.10 * z(−turnover)

z(x) = (x − mean(x)) / std(x)  [across all completed trials in same study]
       = x itself if std == 0 (single trial or all identical)
```
Weights sum to 1.0. Used only within Optuna to rank in-sample candidates. Never exposed as the final score.

### final_robust_score (OOS, after fold aggregation)
```
score = 0.30 * norm(median_oos_return)
      + 0.25 * norm(sharpe_oos)
      + 0.20 * norm(−max_drawdown_oos)
      + 0.15 * norm(−worst_fold_return_loss)
      + 0.10 * norm(−fold_return_stddev)
```
Weights sum to 1.0. All components exposed in response for explainability.

### Guardrail Separation
- **train_trial_filter**: applied during Optuna trials on in-sample data → prune obviously bad candidates early
- **OOS robust_label_guardrail**: applied after fold aggregation on OOS data → final eligibility check

### Status Values
| Status | Meaning |
|---|---|
| `success` | ≥3 valid folds, robust candidate found, all checks passed |
| `degraded` | data load issue, partial results available |
| `insufficient_data` | < 3 valid OOS folds, no robust candidate |
| `failed` | unrecoverable error |
| `completed_degraded` | evaluator checks failed after max rewrites; `evaluator_errors` list included |

---

## Safety Invariants

1. Selected parameters are always labeled **"historical simulation candidates"** — never recommendations, signals, or trading instructions.
2. `robust_label_allowed=False` → output text must not use strong positive expressions about the candidate.
3. Every numeric claim in a report must reference an `EvidenceItem` (including baseline comparisons).
4. Final Korean reports must include the exact education-only disclaimer from the constitution.
5. Rewrite agent must not alter `evidence_id`, numeric values, or core evidence-grounded reasoning.
6. Regime classification results must not be used in trial selection (post-hoc only).

---

## Testing Strategy

**Invariant**: all tests use deterministic synthetic price fixtures. No live yfinance, OpenAI, DB, cache, vector, graph, or search services.

### Critical Test Scenarios (additions beyond spec)
- `test window leakage`: assert no test-period data is present in train feature calculations
- `train_trial_filter`: assert candidates with in-sample MDD > 25% or trades < 30 are pruned before OOS
- `robust_label_allowed=False language check`: assert output text has no strong positive expressions
- `baseline evidence`: assert manual and passive baseline comparison claims have EvidenceItem records
- `completed_degraded response`: assert evaluator_errors list is non-empty when rewrite limit exceeded

### Phase Execution Order (TDD-first, intra-phase parallel where marked)
```
Phase 1: T001 → [T002, T003, T004] parallel
Phase 2: T005–T010 sequential (blocking all user stories)
Phase 3 US1: [T011–T014] tests → [T015–T020] impl
Phase 4 US2: [T021–T025] tests → [T026–T033] impl
Phase 5 US3: [T034–T036] tests → [T037–T041] impl
Phase 6 US4: [T042–T046] tests → [T047–T051] impl
Final:   T052–T056
```

---

## Cost Assumptions

Default (user-adjustable):
- `fee_pct_one_way`: 0.05%
- `slippage_pct_one_way`: 0.05%

All return metrics (train and OOS) apply costs before presentation. Cost assumptions are always visible in the response.

---

## Deferred

- PostgreSQL, Pinecone, Neo4j, OpenSearch, Redis
- User accounts, portfolios, settings, notifications
- Regime-specific parameter application (post-MVP)
- Multi-objective Pareto candidate exploration
