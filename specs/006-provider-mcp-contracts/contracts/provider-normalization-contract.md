# Contract: Provider Normalization And Graph Mapping

## Purpose

This contract defines the boundary between future MCP/provider adapters and the
FinSight Guard agents, persistence layer, and GraphRAG pipeline.

Agents must consume normalized contract objects only. They must not read raw
provider payload fields directly.

## Provider Interfaces

### NewsProvider

Input:
- `ticker`
- `company_hint`
- `as_of_date`
- `max_results`

Output:
- `raw_response_ref`
- `normalization_status`
- `news_events`
- `warnings`
- `errors`

Rules:
- Must preserve raw response lineage.
- Must return NewsEvent objects or an explicit degraded/failed status.
- Must not expose provider-specific raw fields to NewsAgent.

### FinancialProvider

Input:
- `ticker`
- `company_hint`
- `as_of_date`
- `requested_metrics`

Output:
- `raw_response_ref`
- `normalization_status`
- `company_profile`
- `financial_metrics`
- `warnings`
- `errors`

Rules:
- Must preserve raw response lineage.
- Must return CompanyProfile and FinancialMetric objects when available.
- Must not expose provider-specific raw fields to FinancialAgent.

### MarketDataProvider

Input:
- `ticker`
- `period`
- `interval`
- `as_of_date`

Output:
- `raw_response_ref`
- `normalization_status`
- `normalized_market_data_ref`
- `warnings`
- `errors`

Rules:
- Must preserve raw response lineage.
- Must return normalized market data references, not technical analysis results.
- TechnicalAnalysisResult and WaveAnalysisResult are produced by internal
  analysis after normalized market data is available.

## Normalized Objects

Required provider-normalized contracts:
- CompanyProfile
- NewsEvent
- FinancialMetric

Required internal derived contracts:
- TechnicalAnalysisResult
- WaveAnalysisResult

Required scenario input:
- ScenarioReportInput

## Lineage Rules

- RawProviderResponse -> CompanyProfile
- RawProviderResponse -> NewsEvent
- RawProviderResponse -> FinancialMetric
- Normalized market data -> TechnicalAnalysisResult
- Normalized market data + NEoWaveRule + Evidence -> WaveAnalysisResult
- Normalized and derived records -> ScenarioReportInput

TechnicalAnalysisResult and WaveAnalysisResult must not claim direct raw-provider
lineage unless the raw provider actually supplied the final analysis result.

## Persistence Contract

PostgreSQL owns canonical persistence. This contract requires the following
logical records; the plan must decide whether each is already covered by 004 or
requires an explicit extension/migration:

- raw_provider_responses
- company_profiles
- news_events
- financial_metrics
- technical_analysis_results
- wave_analysis_results

Each canonical record must carry:
- request reference
- ticker/company reference
- source or input references
- status
- warnings or error metadata
- evidence references when used in report claims

## Graph Mapping Contract

006 defines mapping eligibility only. 005 owns graph model and retrieval
semantics.

Eligible graph nodes:
- Company
- Ticker
- NewsEvent
- FinancialMetric
- TechnicalAnalysisResult
- WaveAnalysisResult
- Risk
- Evidence

Eligible relationships:
- HAS_TICKER
- MENTIONED_IN
- HAS_METRIC
- HAS_TECHNICAL_ANALYSIS
- HAS_WAVE_ANALYSIS
- CHECKED_BY
- SUPPORTS
- AFFECTS
- HAS_RISK

Projection rules:
- RawProviderResponse is not projected directly.
- Graph nodes must resolve to canonical PostgreSQL references.
- Graph edges must resolve to Evidence or canonical source references.
- Do not create one node per raw candle, news sentence, financial statement row,
  or temporary calculation.

## ScenarioReportInput Contract

ScenarioReportInput must include:
- request id
- ticker
- company profile
- news events
- financial metrics
- technical analysis results
- wave analysis results
- graph context
- evidence ids
- vector references (news originals, NEoWave rule explanations, report chunks)
- missing-data notes
- degradation status
- warnings

Vector references are `VectorReference` objects: `source_kind`,
`canonical_ref_id`, optional `source_uri`, optional `chunk_id`. They reference
semantically retrievable material while preserving canonical source references
(FR-018). They MUST NOT embed raw payloads, vector-store internals, retrieval
scores, or embedding-model details in this feature.

ScenarioReportInput must not include:
- raw provider payload bodies
- provider-specific field names
- buy/sell/hold instructions
- guaranteed target or return fields
- live order execution fields

## Safety Field Contract

Goal: provider and scenario contracts in this feature MUST NOT carry
trading-instruction fields (no buy/sell/hold, no order execution, no guaranteed
return/target). This is enforced structurally, not just by inspection.

Token-based matching (NOT substring):
- Field names are split into snake_case tokens before matching, so incidental
  substrings never false-positive (e.g. `threshold`/`household` do not match
  `hold`).
- `FORBIDDEN_TOKENS` (single tokens): `buy`, `sell`, `hold`, `order`, `execute`,
  `guaranteed`, `recommend`, `recommendation`.
- `FORBIDDEN_TOKEN_PHRASES` (contiguous token tuples): `(target, price)`,
  `(position, size)`, `(stop, loss)`, `(take, profit)`, `(guaranteed, return)`.
- A field violates the contract if any token is in `FORBIDDEN_TOKENS` or any
  contiguous token subsequence is in `FORBIDDEN_TOKEN_PHRASES`. Examples that
  MUST be rejected: `buy_signal`, `order_action`, `target_price`.
- Matching applies to field NAMES only, not to field values (e.g. a
  `metric_name` whose value is `"price"` is unaffected).

Enforcement:
- All contract models set `model_config = ConfigDict(extra="forbid")` so unknown
  trading fields cannot be silently attached.
- A shared `assert_no_trading_fields(model)` helper and the token/phrase
  constants live in `src/providers/safety.py`.

Scope (`SAFETY_CHECKED_CONTRACTS`):
- The safety check applies ONLY to this feature's MVP contracts: `CompanyProfile`,
  `NewsEvent`, `FinancialMetric`, `TechnicalAnalysisResult`, `WaveAnalysisResult`,
  `ScenarioReportInput`, and provider-interface outputs. It does NOT blanket-scan
  the whole codebase.
- Future Phase 2/3 simulation contracts (`SignalCandidate`, `StrategyRule`,
  `PaperTradingExecution`) live in a separate namespace (e.g. `src/simulation/`)
  with their own allowlist; this safety contract MUST NOT block them, so the
  long-term simulated paper-trading goal stays open. Those contracts remain
  simulated-only (SER-007) and still MUST NOT carry live order-execution fields.

## Observability Contract

Degradation must be observable, not silent:
- `degradation_status` and `warnings` are structured fields on normalized
  records, persistence metadata, and `ScenarioReportInput` (already required).
- A structured log event MUST be emitted at (1) the persistence boundary when
  normalization is partial or failed, and (2) the graph-context boundary when
  graph projection is missing or stale.
- Each event carries `request_id`, `ticker`, `status`, and `warning_count`.
- Tests assert the structured warning/degradation FIELDS are populated; the log
  emission itself is an implementation detail layered on top of those fields.

## Degradation Rules

Provider failure:
- Store raw failure metadata when available.
- Return degraded status and missing-data notes.
- Continue only if enough normalized evidence exists.

Partial provider success:
- Preserve successful normalized records.
- Attach warnings for missing fields or unsupported fields.
- ScenarioReportInput must expose missing-data notes.

Normalization failure:
- Preserve raw response and failure reason.
- Do not fabricate normalized records.
- Return insufficient-data when required categories are unavailable.

Graph mapping failure:
- Preserve canonical normalized records.
- Mark graph context as degraded or stale.
- Continue from canonical evidence when safe.
