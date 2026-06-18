# Data Model: PostgreSQL Source-of-Truth Table Schema

## Status Vocabulary

- Request status: `pending`, `running`, `success`, `degraded`, `insufficient_data`, `failed`, `cancelled`.
- Result status: `success`, `degraded`, `insufficient_data`, `failed`.
- Report status: `draft`, `final`, `failed_review`, `archived`.
- Safety status: `pass`, `fail`, `not_evaluated`.
- Projection status: `pending`, `success`, `failed`, `stale`.
- User status: `active`, `disabled`, `anonymized`, `deleted`.
- In-app notification status: `unread`, `read`, `archived`, `deleted`.

## MVP Core Tables

### users

Represents local/demo ownership and future user profile context.

**Fields**

- `id`: UUID primary key.
- `email`: text, nullable, unique when present.
- `display_name`: text.
- `role`: text, default `demo`.
- `status`: user status.
- `anonymized_at`: timestamp nullable.
- `created_at`, `updated_at`, `deleted_at`: timestamps.

**Relationships**

- One user may own analysis requests, settings, notifications, portfolios, and portfolio items through portfolios.

**Validation Rules**

- Email may be absent.
- Login credentials, password hashes, SSO identifiers, and durable sessions are not MVP fields.
- Deleted users may be anonymized while non-PII audit records remain.

### tickers

Canonical research subject.

**Fields**

- `id`: UUID primary key.
- `symbol`: text.
- `market`: text nullable.
- `exchange`: text nullable.
- `currency`: text nullable.
- `name`: text nullable.
- `sector`: text nullable.
- `industry`: text nullable.
- `provider_metadata`: JSON object.
- `is_active`: boolean.
- `created_at`, `updated_at`: timestamps.

**Relationships**

- Referenced by requests, results, evidence, source documents, reports, notifications, portfolios, and graph scenarios.

**Validation Rules**

- `(symbol, market)` is unique.
- Symbol is normalized to uppercase for storage.

### analysis_requests

One workflow request.

**Fields**

- `id`: UUID primary key.
- `user_id`: nullable foreign key to `users`.
- `ticker_id`: foreign key to `tickers`.
- `request_type`: `research`, `backtest`, `robust_optimization`, or `graph_context`.
- `horizon`: text nullable.
- `risk_profile`: text nullable.
- `parameters`: JSON object.
- `status`: request status.
- `degraded_reason`: text nullable.
- `warning_summary`: text nullable.
- `error_summary`: text nullable.
- `created_at`, `started_at`, `completed_at`: timestamps.

**Relationships**

- Owns node runs, analysis results, reports, evidence items, structured log events, and evidence paths.

**Validation Rules**

- A completed request has `completed_at`.
- Failed/degraded requests preserve warning or error summary without fabricated data.

### workflow_node_runs

One observable workflow node attempt.

**Fields**

- `id`: UUID primary key.
- `request_id`: foreign key to `analysis_requests`.
- `run_id`: text workflow run identifier.
- `node_name`: text.
- `attempt_number`: integer.
- `status`: text.
- `started_at`, `ended_at`: timestamps.
- `duration_ms`: integer nullable.
- `error_type`: text nullable.
- `error_message`: text nullable.
- `evaluation_score`: numeric nullable.
- `metadata`: JSON object.

**Relationships**

- Belongs to an analysis request; may be associated with an analysis result through result metadata.

**Validation Rules**

- `(request_id, node_name, attempt_number)` is unique.
- Failed attempts preserve error details.

### analysis_results

Structured output from a workflow component.

**Fields**

- `id`: UUID primary key.
- `request_id`: foreign key to `analysis_requests`.
- `ticker_id`: nullable foreign key to `tickers`.
- `result_type`: `market`, `fundamental`, `news`, `graph_context`, `backtest`, `optimization`, `coordinator_draft`, `evaluation`, or `rewrite`.
- `summary`: text.
- `metrics`: JSON object.
- `warnings`: JSON array.
- `missing_data_notes`: JSON array.
- `status`: result status.
- `created_at`: timestamp.

**Relationships**

- May be cited by evidence items and feed reports.

**Validation Rules**

- Missing values are represented as unavailable notes, not invented metrics.

### reports

Report container and current status.

**Fields**

- `id`: UUID primary key.
- `request_id`: foreign key to `analysis_requests`.
- `ticker_id`: foreign key to `tickers`.
- `current_version_id`: nullable foreign key to `report_versions`.
- `title`: text.
- `language`: text, default `ko`.
- `status`: report status.
- `safety_status`: safety status.
- `evaluation_score`: numeric nullable.
- `disclaimer_present`: boolean.
- `created_at`, `updated_at`: timestamps.

**Relationships**

- Owns immutable report versions and citations.

**Validation Rules**

- Final Korean reports must preserve the required disclaimer.
- Final reports must have a current version.

### report_versions

Immutable report content snapshot.

**Fields**

- `id`: UUID primary key.
- `report_id`: foreign key to `reports`.
- `version_number`: integer.
- `stage`: `draft`, `rewrite`, `final`, or `failed`.
- `report_json`: JSON object.
- `report_markdown`: text.
- `created_by_node`: text.
- `created_at`: timestamp.

**Relationships**

- Cites evidence through `report_evidence_citations`.

**Validation Rules**

- `(report_id, version_number)` is unique.
- Report versions are append-only.

### evidence_items

EvidenceItem-compatible canonical evidence.

**Fields**

- `id`: UUID primary key.
- `evidence_id`: unique text.
- `request_id`: nullable foreign key to `analysis_requests`.
- `ticker_id`: nullable foreign key to `tickers`.
- `analysis_result_id`: nullable foreign key to `analysis_results`.
- `source_document_id`: nullable foreign key to `source_documents`.
- `source_type`: text.
- `source_name`: text.
- `source_url`: text nullable.
- `collected_at`: timestamp.
- `metric_name`: text.
- `metric_value`: JSON scalar nullable.
- `description`: text.
- `created_at`: timestamp.

**Relationships**

- Cited by report versions and evidence paths.

**Validation Rules**

- `evidence_id` is unique and externally reportable.
- Important factual/numeric report claims must resolve to evidence.

### report_evidence_citations

Report-to-evidence grounding join.

**Fields**

- `id`: UUID primary key.
- `report_version_id`: foreign key to `report_versions`.
- `evidence_item_id`: foreign key to `evidence_items`.
- `section_name`: text.
- `claim_text`: text.
- `created_at`: timestamp.

**Validation Rules**

- Duplicate citation for the same report version, section, claim, and evidence item is prevented.

## Source Documents And Projections

### source_documents

Canonical provider or generated document metadata.

**Fields**

- `id`: UUID primary key.
- `ticker_id`: nullable foreign key to `tickers`.
- `document_type`: `news`, `financial_statement_explanation`, `wave_theory_material`, `report`, `log`, or `provider_payload`.
- `source_name`: text.
- `source_url`: text nullable.
- `title`: text nullable.
- `language`: text nullable.
- `published_at`: timestamp nullable.
- `collected_at`: timestamp.
- `raw_content_ref`: text nullable.
- `content_hash`: text.
- `revision_group_id`: UUID grouping recrawled or corrected versions of the same logical document.
- `supersedes_document_id`: nullable self-referential foreign key to the prior source document version.
- `metadata`: JSON object.
- `status`: text.

**Validation Rules**

- Source URL plus content hash identifies duplicate crawls when URL exists.
- Raw large content may be referenced instead of stored inline.
- Recrawled or corrected documents preserve lineage by sharing `revision_group_id`.
- `supersedes_document_id`, when present, references an earlier source document in the same revision group.

### document_chunks

Chunk provenance for semantic and keyword search.

**Fields**

- `id`: UUID primary key.
- `source_document_id`: foreign key to `source_documents`.
- `chunk_index`: integer.
- `chunk_text`: text.
- `chunk_hash`: text.
- `token_count`: integer nullable.
- `metadata`: JSON object.
- `created_at`: timestamp.

**Validation Rules**

- `(source_document_id, chunk_index)` is unique.

### index_projection_status

Projection ledger for Pinecone, Neo4j, and OpenSearch.

**Fields**

- `id`: UUID primary key.
- `source_table`: text.
- `source_id`: UUID.
- `target_system`: `pinecone`, `neo4j`, or `opensearch`.
- `projection_type`: text.
- `projection_key`: text.
- `status`: projection status.
- `attempt_count`: integer.
- `last_attempt_at`: timestamp nullable.
- `last_success_at`: timestamp nullable.
- `error_message`: text nullable.
- `idempotency_key`: text.

**Validation Rules**

- `(target_system, projection_type, idempotency_key)` is unique.
- Projection failures do not change canonical source records.

### keyword_terms

Optional normalized keyword catalog.

**Fields**

- `id`: UUID primary key.
- `term`: text.
- `normalized_term`: text.
- `language`: text nullable.
- `created_at`: timestamp.

**Validation Rules**

- `(normalized_term, language)` is unique.

### structured_log_events

Durable operational event summary.

**Fields**

- `id`: UUID primary key.
- `request_id`: nullable foreign key to `analysis_requests`.
- `run_id`: text nullable.
- `ticker_id`: nullable foreign key to `tickers`.
- `node_name`: text nullable.
- `event_name`: text.
- `status`: text.
- `message`: text nullable.
- `error_message`: text nullable.
- `evaluation_score`: numeric nullable.
- `occurred_at`: timestamp.
- `metadata`: JSON object.

**Validation Rules**

- This table stores audit summaries, not high-volume raw tracing.

## MVP Graph Knowledge Tables

### wave_rules

Canonical wave-theory rule.

**Fields**

- `id`: UUID primary key.
- `rule_code`: unique text.
- `name`: text.
- `description`: text.
- `rule_type`: text.
- `status`: text.
- `source_document_id`: nullable foreign key to `source_documents`.
- `created_at`, `updated_at`: timestamps.

### wave_scenarios

Canonical wave scenario for research context.

**Fields**

- `id`: UUID primary key.
- `ticker_id`: nullable foreign key to `tickers`.
- `name`: text.
- `description`: text.
- `timeframe`: text nullable.
- `status`: text.
- `confidence_label`: text nullable.
- `created_at`, `updated_at`: timestamps.

**Validation Rules**

- Scenario text must be research framing, not trading advice.

### wave_invalidation_conditions

Condition that invalidates or requires review of a scenario.

**Fields**

- `id`: UUID primary key.
- `scenario_id`: foreign key to `wave_scenarios`.
- `condition_text`: text.
- `metric_name`: text nullable.
- `threshold_value`: JSON scalar nullable.
- `direction`: text nullable.
- `source_document_id`: nullable foreign key to `source_documents`.
- `created_at`: timestamp.

**Validation Rules**

- Thresholds are review conditions, not guaranteed targets.

### wave_scenario_rules

Join table connecting scenarios and rules.

**Fields**

- `id`: UUID primary key.
- `scenario_id`: foreign key to `wave_scenarios`.
- `rule_id`: foreign key to `wave_rules`.
- `role`: text.
- `created_at`: timestamp.

**Validation Rules**

- Duplicate scenario-rule-role rows are prevented.

### evidence_paths

Canonical explainability path.

**Fields**

- `id`: UUID primary key.
- `request_id`: nullable foreign key to `analysis_requests`.
- `ticker_id`: nullable foreign key to `tickers`.
- `path_type`: text.
- `path_summary`: text.
- `source_node_ref`: text.
- `target_node_ref`: text.
- `confidence_label`: text nullable.
- `created_at`: timestamp.

**Relationships**

- Owns ordered path steps and may cite evidence through step references.

### evidence_path_steps

Ordered evidence path step.

**Fields**

- `id`: UUID primary key.
- `evidence_path_id`: foreign key to `evidence_paths`.
- `step_index`: integer.
- `node_table`: text.
- `node_id`: UUID.
- `relationship_type`: text.
- `description`: text.
- `created_at`: timestamp.

**Validation Rules**

- `(evidence_path_id, step_index)` is unique.

## MVP User Experience Tables

### user_settings

User or system preference.

**Fields**

- `id`: UUID primary key.
- `user_id`: nullable foreign key to `users`.
- `setting_key`: text.
- `setting_value`: JSON object.
- `scope`: text.
- `updated_at`: timestamp.
- `deleted_at`: timestamp nullable.

**Validation Rules**

- `(user_id, setting_key, scope)` is unique.
- User-owned settings are deleted or anonymized when the user is deleted.

### notifications

In-app notification state.

**Fields**

- `id`: UUID primary key.
- `user_id`: nullable foreign key to `users`.
- `ticker_id`: nullable foreign key to `tickers`.
- `notification_type`: text.
- `title`: text.
- `body`: text.
- `payload`: JSON object.
- `status`: notification status.
- `created_at`, `read_at`, `updated_at`, `deleted_at`: timestamps nullable as applicable.

**Validation Rules**

- Notifications are informational only and must not instruct trading.
- External delivery channel state is deferred.

### portfolios

Research-only portfolio context.

**Fields**

- `id`: UUID primary key.
- `user_id`: nullable foreign key to `users`.
- `name`: text.
- `description`: text nullable.
- `base_currency`: text nullable.
- `status`: text.
- `created_at`, `updated_at`, `deleted_at`: timestamps.

**Validation Rules**

- Portfolios are context/watchlist records only.
- No brokerage or order execution fields are allowed.

### portfolio_items

Ticker association inside a portfolio.

**Fields**

- `id`: UUID primary key.
- `portfolio_id`: foreign key to `portfolios`.
- `ticker_id`: foreign key to `tickers`.
- `label`: text nullable.
- `quantity_note`: text nullable.
- `cost_basis_note`: text nullable.
- `metadata`: JSON object.
- `created_at`, `updated_at`, `deleted_at`: timestamps.

**Validation Rules**

- Numeric-like values are optional research notes and must not trigger orders.

## Deferred Expansion Tables

- `provider_sync_runs`
- `provider_payloads`
- `analysis_request_links`
- `notification_preferences`
- `notification_deliveries`
- `auth_identities`
- `user_sessions`

These entities remain documented for later planning and must not block the first PostgreSQL MVP.

## State Transitions

### Analysis Request

```text
pending -> running -> success
pending -> running -> degraded
pending -> running -> insufficient_data
pending -> running -> failed
pending -> cancelled
```

### Report

```text
draft -> final
draft -> failed_review
draft -> rewrite -> final
draft -> rewrite -> failed_review
final -> archived
```

### Projection

```text
pending -> success
pending -> failed
success -> stale -> pending
failed -> pending
```

### User Deletion

```text
active -> anonymized
active -> disabled
anonymized -> deleted
```
