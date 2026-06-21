# Data Model: Provider-Agnostic MCP Contracts

## Ownership Map

| Object | Contract Owner | Canonical Storage Owner | Graph Behavior Owner |
|---|---|---|---|
| RawProviderResponse | 006 | 004 extension/migration if missing | Not projected directly |
| CompanyProfile | 006 | 004 extension/migration if missing | 005 mapping target |
| NewsEvent | 006 | 004 extension/migration if missing | 005 mapping target |
| FinancialMetric | 006 | 004 extension/migration if missing | 005 mapping target |
| TechnicalAnalysisResult | 006 contract for derived output | Existing `analysis_results` or 004 extension | 005 mapping target |
| WaveAnalysisResult | 006 contract for derived output | Existing `analysis_results`/wave tables or 004 extension | 005 mapping target |
| ScenarioReportInput | 006 | Usually not persisted as canonical record; may be reproducible from request/result IDs | Feeds 005 scenario generation |
| GraphMappingRule | 006 eligibility contract | Projection status owned by 004 | Graph semantics owned by 005 |
| VectorReference | 006 | Resolves to 004 canonical source/evidence references; vector store itself is a future feature | Not projected directly |

## RawProviderResponse

Represents the original response from an external MCP or provider adapter.

Fields:
- `raw_response_id`
- `request_id`
- `ticker_id`
- `provider_name`
- `provider_kind`: `news`, `financial`, `market_data`
- `provider_request`
- `status`: `success`, `partial_success`, `degraded`, `failed`
- `collected_at`
- `payload_ref`
- `payload_body`
- `payload_hash`
- `error_message`
- `metadata`

Validation:
- Exactly one of `payload_ref` or `payload_body` should be present for successful
  responses.
- Failed responses may omit payload but must include status and error metadata.
- RawProviderResponse is not projected to graph directly.

## CompanyProfile

Provider-normalized company identity and business profile.

Fields:
- `company_profile_id`
- `request_id`
- `ticker_id`
- `raw_response_id`
- `company_name`
- `legal_name`
- `sector`
- `industry`
- `country`
- `exchange`
- `currency`
- `description`
- `normalization_status`
- `warnings`
- `evidence_id`

Relationships:
- Traces to RawProviderResponse.
- May produce/update the Company graph node through 005 mapping.

## NewsEvent

Provider-normalized event-level news record.

Fields:
- `news_event_id`
- `request_id`
- `ticker_id`
- `raw_response_id`
- `title`
- `summary`
- `source_name`
- `source_url`
- `published_at`
- `collected_at`
- `event_type`
- `sentiment_label`
- `risk_tags`
- `normalization_status`
- `warnings`
- `evidence_id`

Relationships:
- Traces to RawProviderResponse.
- May map to NewsEvent graph node and MENTIONED_IN/AFFECTS/SUPPORTS edges.

## FinancialMetric

Provider-normalized financial metric.

Fields:
- `financial_metric_id`
- `request_id`
- `ticker_id`
- `raw_response_id`
- `metric_name`
- `metric_value`
- `period`
- `currency`
- `unit`
- `source_name`
- `source_url`
- `collected_at`
- `normalization_status`
- `warnings`
- `evidence_id`

Relationships:
- Traces to RawProviderResponse.
- May map to FinancialMetric graph node and HAS_METRIC/SUPPORTS edges.

## TechnicalAnalysisResult

Internal derived technical analysis output.

Fields:
- `technical_analysis_result_id`
- `request_id`
- `ticker_id`
- `source_market_data_refs`
- `indicator_values`
- `trend_state`
- `momentum_state`
- `volatility_state`
- `normalization_or_derivation_status`
- `warnings`
- `evidence_ids`

Relationships:
- Traces to normalized market data and Evidence records, not directly to raw
  provider response.
- May map to TechnicalAnalysisResult graph node and HAS_TECHNICAL_ANALYSIS edges.

## WaveAnalysisResult

Internal derived NEoWave candidate analysis output.

Fields:
- `wave_analysis_result_id`
- `request_id`
- `ticker_id`
- `source_market_data_refs`
- `rule_refs`
- `candidate_summary`
- `rule_statuses`: `passed`, `failed`, `unknown`, `needs_human_review`
- `confirmation_conditions`
- `invalidation_conditions`
- `uncertainty_notes`
- `warnings`
- `evidence_ids`

Relationships:
- Traces to normalized market data, NEoWave rule references, and Evidence
  records.
- May map to WaveAnalysisResult graph node and HAS_WAVE_ANALYSIS/CHECKED_BY
  edges.

## ScenarioReportInput

Stable ScenarioAgent input.

Fields:
- `request_id`
- `ticker`
- `company_profile`
- `news_events`
- `financial_metrics`
- `technical_analysis_results`
- `wave_analysis_results`
- `graph_context`
- `evidence_ids`
- `vector_references`
- `missing_data_notes`
- `degradation_status`
- `warnings`

Validation:
- Must not include raw provider payload fields.
- Must include missing-data notes when required evidence categories are absent.
- Must include enough evidence references for Evaluator grounding checks.
- Must not expose any trading-instruction field (see Safety Field Contract in
  `contracts/provider-normalization-contract.md`).

## VectorReference

Lightweight reference from `ScenarioReportInput` to semantically retrievable
material. It carries only canonical references; the vector store, scoring, and
embedding model are intentionally out of scope and deferred to a future
retrieval feature.

Fields:
- `source_kind`: `news_original`, `neowave_rule_explanation`, `report_chunk`
- `canonical_ref_id`: canonical PostgreSQL reference (e.g. source_document or
  evidence record) the chunk derives from
- `source_uri`: optional original locator (URL or document path)
- `chunk_id`: optional future vector-store chunk identifier

Validation:
- Every VectorReference MUST resolve to a canonical PostgreSQL source or evidence
  reference; a dangling reference with no `canonical_ref_id` is invalid.
- VectorReference is never graph-projected directly.

## GraphMappingRule

Defines how contract objects are eligible for 005 graph projection.

Fields:
- `source_contract_type`
- `source_canonical_ref`
- `graph_node_type`
- `relationship_type`
- `required_fields`
- `evidence_id_required`
- `projection_scope`

Validation:
- RawProviderResponse is never projected directly.
- One node per raw candle/news sentence/financial statement row is forbidden.
- Every projected relationship must resolve to canonical source references.

## State Transitions

Normalization status:

```text
received -> normalized
received -> partial_success
received -> failed
normalized -> graph_mapping_ready
partial_success -> graph_mapping_ready_with_warnings
graph_mapping_ready -> projected
graph_mapping_ready -> projection_failed
```

Degradation status:

```text
complete
partial_provider_failure
partial_normalization_failure
graph_mapping_degraded
insufficient_data
```
