# Data Model: Walk-Forward Strategy Optimization

## OptimizationRun

Represents one robust optimization request.

**Fields**

- `run_id`: string, unique workflow or optimization run identifier.
- `ticker`: string, validated ticker symbol.
- `start`: ISO date string.
- `end`: ISO date string.
- `initial_balance`: number greater than 0.
- `cost_assumptions`: `CostAssumptions`.
- `scoring_policy`: `RobustScoringPolicy`.
- `fold_setup`: `WalkForwardConfig`.
- `status`: `success`, `degraded`, `insufficient_data`, or `failed`.
- `manual_baseline`: `ParameterCandidateResult | null`.
- `passive_baseline`: `BaselineResult | null`.
- `robust_candidate`: `ParameterCandidateResult | null`.
- `folds`: list of `WalkForwardFold`.
- `regime_summary`: list of `MarketRegimeSegment`.
- `warnings`: list of strings.
- `evidence`: list of `OptimizationEvidenceItem`.

**Validation Rules**

- `start` must be before `end`.
- A robust candidate must not be produced unless at least 3 valid test folds exist.
- `status` must be `insufficient_data` when valid fold count is below 3.
- Output must state that results are historical simulations and not trading advice.
- Canonical persisted records live in PostgreSQL `analysis_requests`, `analysis_results`, and `robust_optimization_runs`.
- Derived search/graph/vector projections must reference the PostgreSQL run ID.

## Persistent Storage Entities

These entities define the source-of-truth PostgreSQL model used by this feature and the broader platform direction.

### User

- `id`: UUID.
- `email`: unique text or null for local/demo mode.
- `display_name`: text.
- `created_at`, `updated_at`: timestamps.

### Ticker

- `id`: UUID.
- `symbol`: unique ticker string.
- `market`: text or null.
- `name`: text or null.
- `created_at`, `updated_at`: timestamps.

### AnalysisRequest

- `id`: UUID.
- `user_id`: nullable `User` reference.
- `ticker_id`: `Ticker` reference.
- `request_type`: `research`, `backtest`, or `robust_optimization`.
- `parameters`: JSON object.
- `status`: `pending`, `running`, `success`, `degraded`, `insufficient_data`, or `failed`.
- `created_at`, `started_at`, `completed_at`: timestamps.

### AnalysisResult

- `id`: UUID.
- `request_id`: `AnalysisRequest` reference.
- `result_type`: `market`, `fundamental`, `news`, `backtest`, `optimization`, or `evaluation`.
- `summary`: text.
- `metrics`: JSON object.
- `warnings`: JSON array.
- `created_at`: timestamp.

### Report

- `id`: UUID.
- `request_id`: `AnalysisRequest` reference.
- `ticker_id`: `Ticker` reference.
- `title`: text.
- `language`: text.
- `report_json`: JSON object.
- `report_markdown`: text.
- `evaluation_score`: number or null.
- `safety_status`: `pass`, `fail`, or `not_evaluated`.
- `created_at`: timestamp.

### Setting

- `id`: UUID.
- `user_id`: nullable `User` reference.
- `key`: text.
- `value`: JSON object.
- `updated_at`: timestamp.

### Notification

- `id`: UUID.
- `user_id`: nullable `User` reference.
- `notification_type`: text.
- `payload`: JSON object.
- `status`: `pending`, `sent`, `read`, or `failed`.
- `created_at`, `updated_at`: timestamps.

### Portfolio

- `id`: UUID.
- `user_id`: `User` reference.
- `name`: text.
- `holdings`: JSON object.
- `created_at`, `updated_at`: timestamps.

## Derived Index Entities

### PineconeDocumentChunk

- `chunk_id`: stable chunk identifier.
- `source_document_id`: PostgreSQL/provider source document identifier.
- `source_type`: `news`, `financial_statement_explanation`, or `wave_theory_material`.
- `ticker`: optional ticker symbol.
- `text`: chunk text used for embedding.
- `metadata`: JSON-compatible source metadata.

### Neo4jEvidencePath

- Nodes: `WaveRule`, `Scenario`, `InvalidationCondition`, `EvidenceItem`, `Ticker`, `Report`, `OptimizationRun`.
- Relationships: `SCENARIO_USES_RULE`, `RULE_HAS_INVALIDATION`, `EVIDENCE_SUPPORTS_SCENARIO`, `EVIDENCE_CONTRADICTS_SCENARIO`, `REPORT_CITES_EVIDENCE`, `OPTIMIZATION_PRODUCED_EVIDENCE`.

### OpenSearchDocument

- Indexes: `news_documents`, `reports`, `logs`, `keywords`.
- Documents include PostgreSQL IDs where applicable so search hits can resolve back to canonical records.

### RedisRuntimeKey

- Namespaces: `cache:*`, `queue:*`, `rate_limit:*`, `session:*`, `run_status:*`.
- Redis values are ephemeral and must not be the only durable copy of reports, analysis results, or optimization outputs.

## CostAssumptions

Transaction cost inputs applied before return metrics are presented.

**Fields**

- `fee_pct_one_way`: number, default `0.05`.
- `slippage_pct_one_way`: number, default `0.05`.
- `user_adjustable`: boolean, true for API/UI supplied assumptions.

**Validation Rules**

- Fee and slippage must be greater than or equal to 0.
- All optimized return metrics must use cost-adjusted results.

## RobustScoringPolicy

Transparent score component weights.

**Fields**

- `out_of_sample_return_weight`: number, default `0.30`.
- `risk_adjusted_return_weight`: number, default `0.25`.
- `drawdown_control_weight`: number, default `0.20`.
- `worst_fold_resilience_weight`: number, default `0.15`.
- `stability_turnover_penalty_weight`: number, default `0.10`.

**Validation Rules**

- Weights must sum to 1.0 within normal floating-point tolerance.
- Score components must be exposed in output so the ranking is explainable.

## WalkForwardConfig

Defines train/test split behavior.

**Fields**

- `train_window_days`: integer greater than 0.
- `test_window_days`: integer greater than 0.
- `step_days`: integer greater than 0.
- `minimum_valid_test_folds`: integer, fixed at 3 for robust candidate output.

**Validation Rules**

- Every test period must start after its train period.
- Fold order must follow the original time order.
- If windows cannot be created, the result must be limitation-only and not robust.

## WalkForwardFold

One selection period and subsequent validation period.

**Fields**

- `fold_index`: integer starting at 1.
- `train_start`: ISO date string.
- `train_end`: ISO date string.
- `test_start`: ISO date string.
- `test_end`: ISO date string.
- `selected_params`: `BacktestParams`.
- `candidate_metrics`: `CandidateMetrics`.
- `warnings`: list of strings.
- `status`: `valid`, `no_trades`, `missing_data`, or `invalid`.

**Validation Rules**

- `test_start` must be after `train_end`.
- Invalid folds are reported but excluded from robust candidate eligibility.

## ParameterCandidateResult

Metrics for a strategy parameter set.

**Fields**

- `params`: `BacktestParams`.
- `score`: number.
- `score_components`: object containing each robust score component.
- `metrics`: `CandidateMetrics`.
- `fold_metrics`: list of fold-level `CandidateMetrics`.
- `robust_label_allowed`: boolean.
- `warnings`: list of strings.

**Validation Rules**

- `robust_label_allowed` is false if completed trades are fewer than 30.
- `robust_label_allowed` is false if maximum drawdown is greater than 25%.
- High total return alone must not determine candidate selection.

## CandidateMetrics

Comparable performance and risk metrics.

**Fields**

- `total_return_pct`: number.
- `cost_adjusted_return_pct`: number.
- `max_drawdown_pct`: number.
- `sharpe`: number or null.
- `sortino`: number or null.
- `win_rate_pct`: number or null.
- `profit_factor`: number or null.
- `completed_trades`: integer.
- `average_holding_days`: number or null.
- `turnover`: number or null.
- `median_oos_return_pct`: number or null.
- `worst_fold_return_pct`: number or null.
- `fold_return_stddev`: number or null.

**Validation Rules**

- Important numeric report claims must map to evidence items.
- Missing or undefined values must be shown as unavailable, not invented.

## MarketRegimeSegment

Performance grouped by market condition.

**Fields**

- `regime`: `bull`, `bear`, `sideways`, `high_volatility`, or `low_volatility`.
- `start`: ISO date string.
- `end`: ISO date string.
- `trading_days`: integer.
- `completed_trades`: integer.
- `metrics`: `CandidateMetrics`.
- `confidence`: `normal` or `low`.
- `low_confidence_reason`: string or null.

**Validation Rules**

- `confidence` is `low` when completed trades are fewer than 10 or trading days are fewer than 60.
- Low-confidence regime results must not be overinterpreted in report text.

## BaselineResult

Comparable manual-parameter or passive benchmark result.

**Fields**

- `baseline_type`: `manual_parameters` or `passive_buy_and_hold`.
- `metrics`: `CandidateMetrics`.
- `warnings`: list of strings.

**Validation Rules**

- Baselines use the same evaluation periods and transaction-cost assumptions where applicable.

## OptimizationEvidenceItem

Evidence record for optimization-derived numeric claims.

**Fields**

- `evidence_id`: string.
- `source_type`: `backtest`.
- `source_name`: string, for example `FinSight robust optimization (historical simulation)`.
- `source_url`: string or null.
- `collected_at`: datetime.
- `ticker`: string.
- `metric_name`: string.
- `metric_value`: string, number, boolean, or null.
- `description`: string.

**Validation Rules**

- Every important numeric claim in a report or agent output must reference a corresponding evidence item.
- Descriptions must include historical-simulation framing and avoid future-return guarantees.

## State Transitions

```text
requested
  -> data_loaded
  -> folds_created
  -> candidates_scored
  -> regime_summarized
  -> evidence_built
  -> reported
```

Failure and degraded paths:

```text
requested -> insufficient_data -> limitation_only_output
requested -> data_load_failed -> degraded_output
candidates_scored -> no_robust_candidate -> limitation_only_output
reported -> evaluator_failed -> rewrite_requested
```
