# Feature Specification: PostgreSQL Source-of-Truth Table Schema

**Feature Branch**: `[004-postgresql-table-schema]`

**Created**: 2026-06-18

**Status**: Draft

**Input**: User description: "PostgreSQL is the real source of truth for users, tickers, analysis requests, reports, analysis results, settings, notifications, and portfolios. Pinecone is a semantic chunk index for news, financial statement explanations, and wave-theory materials. Neo4j is a relationship index/knowledge graph for wave rules, scenarios, invalidation conditions, and evidence paths. OpenSearch is a keyword/full-text index for news originals, reports, logs, and keyword search. Redis is ephemeral cache, job queue, rate limit, and session storage. Start data modeling with the PostgreSQL table specification."

## Clarifications

### Session 2026-06-18

- Q: Which PostgreSQL table scope should be included in the first MVP implementation? → A: Full platform MVP: include users, settings, notifications, and portfolios in the first PostgreSQL implementation.
- Q: What authentication scope should the MVP user model support? → A: Local/demo user first: users are optional owners, email is nullable, and login, password, SSO, and durable session credential storage are deferred.
- Q: What deletion and retention policy should apply to user-owned data? → A: Anonymize user data: delete or anonymize PII and user-owned UX records while preserving reports, evidence, and audit records under an anonymous owner.
- Q: What notification delivery scope should the MVP support? → A: In-app only: MVP notifications are stored app-internal records; email, webhook, push delivery, and channel retry handling are deferred.
- Q: Should graph knowledge and evidence path tables be included in the first PostgreSQL MVP? → A: Include graph tables now: wave rules, wave scenarios, invalidation conditions, and evidence path tables are part of the first MVP implementation.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define Canonical Research Records (Priority: P1)

A product owner and developer need a complete source-of-truth table catalog for research workflow records so that every analysis request, agent result, report, evidence item, and safety evaluation can be stored durably and audited later.

**Why this priority**: The existing MVP can collect data and generate reports, but durable canonical records are required before derived indexes, user history, notifications, and portfolio views can be implemented safely.

**Independent Test**: Review the table catalog against one complete research workflow run and verify that the request, ticker, agent outputs, evidence, evaluator result, final report, warnings, and run lifecycle events all have an assigned canonical table.

**Acceptance Scenarios**:

1. **Given** a completed research workflow, **When** reviewers map each workflow artifact to the schema, **Then** every canonical artifact is assigned to exactly one primary owner table.
2. **Given** a degraded workflow with missing market, fundamental, or news data, **When** reviewers inspect the schema, **Then** warnings, failure reasons, retry attempts, and degraded status can be stored without fabricating missing facts.

---

### User Story 2 - Rebuild Derived Search And Graph Indexes (Priority: P2)

A developer needs canonical records that allow Pinecone, Neo4j, and OpenSearch projections to be rebuilt without treating those systems as the original source of truth.

**Why this priority**: The platform architecture depends on clear role separation: PostgreSQL is the ledger, while semantic, graph, keyword, and runtime stores are rebuildable projections or ephemeral state.

**Independent Test**: Select a news item, report, evidence item, and wave-theory scenario and verify that each derived index record can resolve back to a canonical table record and stable identifier.

**Acceptance Scenarios**:

1. **Given** a deleted or stale semantic index, **When** a re-index job reads canonical source documents and chunks, **Then** the semantic index can be repopulated with stable source references.
2. **Given** a graph evidence path displayed to a user, **When** auditors trace the path, **Then** every cited rule, scenario, condition, evidence item, ticker, request, and report resolves to a canonical record.

---

### User Story 3 - Support User History, Settings, In-App Notifications, And Portfolios (Priority: P3)

A demo user needs persistent history, preferences, notifications, and portfolio watch context so the assistant can provide a richer research experience without adding trading or order execution.

**Why this priority**: User-facing persistence is part of the first platform MVP, but it must remain research-only and must not imply brokerage, execution, or investment recommendation behavior.

**Independent Test**: Create a user profile, settings record, notification, portfolio, and watchlist-style holding reference, then verify that no table requires order execution, brokerage credentials, guaranteed targets, or buy/sell instructions.

**Acceptance Scenarios**:

1. **Given** a user changes a preference, **When** the preference is saved, **Then** the setting is traceable to that user and can be updated without changing historical reports.
2. **Given** a user maintains portfolio context, **When** a ticker is associated with the portfolio, **Then** the association supports research and monitoring only, not order execution.

### Edge Cases

- Duplicate ticker symbols from different markets must remain distinguishable by market/exchange metadata.
- A local/demo run may have no authenticated user; user references must allow a system or anonymous owner where appropriate.
- MVP user records may be local/demo owners without login credentials; authentication, password, SSO, and durable credential/session storage must remain deferred.
- MVP notifications may exist without external delivery addresses; email, webhook, push delivery, and channel retry failures are out of scope for first implementation.
- A provider may return partial data, missing URLs, delayed filings, or no news; canonical records must preserve missing-data notes and source status.
- A generated report may fail evaluator review; failed drafts, rewrite attempts, final failed status, and safety reasons must remain auditable.
- A source document may be re-crawled or corrected; the schema must preserve provider identity, source URL, collection time, and version relationship.
- Derived index writes may fail after canonical data is saved; projection status must record pending, success, or failure without losing the source records.
- Redis runtime state may expire; durable run status, report path, and final outcomes must remain recoverable from canonical records.
- User deletion must not break research auditability; PII and user-owned UX records are deleted or anonymized, while reports, evidence, source documents, workflow node runs, and audit records remain under an anonymous owner.
- Graph projection failures must not remove canonical wave-rule, scenario, invalidation-condition, or evidence-path records from PostgreSQL.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST define PostgreSQL as the canonical owner for durable business records: users, tickers, analysis requests, analysis results, reports, evidence, settings, notifications, portfolios, source documents, workflow events, and index projection status.
- **FR-002**: System MUST assign every durable research workflow artifact to a single canonical table and document the table's purpose, primary key, important columns, relationships, and retention role.
- **FR-003**: System MUST preserve stable identifiers for requests, reports, evidence items, source documents, document chunks, graph rules, graph scenarios, and index projection records so derived systems can reference canonical records.
- **FR-004**: System MUST support unauthenticated local/demo runs and local/demo user ownership without requiring login credentials; `users.email` may be absent and authentication, password, SSO, and durable session credential storage are deferred.
- **FR-005**: System MUST represent ticker identity with symbol, market/exchange context, display name, provider metadata, active status, and timestamps.
- **FR-006**: System MUST record analysis request inputs, request type, lifecycle status, timestamps, retry/degraded status, request parameters, and user-facing error or warning summaries.
- **FR-007**: System MUST record agent and workflow node execution details, including node name, start/end time, status, retry count, degraded reason, error message, and evaluation score when available.
- **FR-008**: System MUST record agent analysis results separately from generated reports so market, fundamental, news, backtest, optimization, graph context, evaluation, and rewrite results can be inspected independently.
- **FR-009**: System MUST store final and intermediate report records with language, title, structured content, rendered text, safety status, disclaimer presence, evaluation score, and report version.
- **FR-010**: System MUST store EvidenceItem-compatible records with evidence ID, source type, source name, source URL, collection time, ticker, metric name, metric value, description, and relationship to request/result/report where applicable.
- **FR-011**: System MUST store source document metadata and chunk provenance for news, financial-statement explanations, report text, logs, and wave-theory materials so semantic and keyword indexes are rebuildable.
- **FR-012**: System MUST store canonical wave-theory rules, scenarios, invalidation conditions, and evidence path records when graph reasoning is used, while keeping graph databases as derived projections.
- **FR-013**: System MUST store index projection status for Pinecone, Neo4j, and OpenSearch targets, including source record, projection type, status, last attempt time, error message, and idempotency key.
- **FR-014**: System MUST store user settings, notifications, portfolios, portfolio items, and watchlist context without any field that represents brokerage connection, order placement, or guaranteed target execution.
- **FR-015**: System MUST define uniqueness and integrity rules that prevent duplicate canonical evidence IDs, duplicate report versions for the same request, duplicate user setting keys per user, and duplicate ticker records for the same symbol/market pair.
- **FR-016**: System MUST support soft deletion or archival status for user-facing records where deletion would otherwise break report, evidence, or audit history.
- **FR-017**: System MUST include users, settings, notifications, portfolios, portfolio items, wave rules, wave scenarios, invalidation conditions, evidence paths, and evidence path steps in the first PostgreSQL MVP table set, alongside research workflow, source document, and projection status tables.
- **FR-018**: System MUST keep provider batch sync, immutable raw provider payload retention, request-link history, and detailed notification routing preferences as deferred expansion tables unless later planning explicitly promotes them.
- **FR-019**: System MUST support user deletion by deleting or anonymizing PII and user-owned UX records while preserving reports, evidence, source documents, workflow node runs, and audit records under an anonymous owner.
- **FR-020**: System MUST limit MVP notifications to app-internal notification records and MUST defer email, webhook, push delivery, channel preferences, and delivery retry tracking.

### Safety, Evidence & Reliability Requirements *(mandatory for research workflow changes)*

- **SER-001**: System MUST avoid direct buy/sell/hold recommendations, trading instructions, guaranteed return claims, guaranteed target claims, and order execution behavior.
- **SER-002**: System MUST back important numeric or factual report claims with `EvidenceItem` records when the feature affects research output.
- **SER-003**: Final Korean reports MUST include the required education-only, no-recommendation disclaimer exactly as defined in the constitution.
- **SER-004**: System MUST disclose unavailable or degraded market, fundamental, or news data instead of fabricating missing facts.
- **SER-005**: Workflow-affecting features MUST define deterministic behavior for validation failure, provider failure, evaluator failure, and rewrite limits.
- **SER-006**: Runtime-affecting features MUST preserve structured logs, report storage, health checks, and metrics required by the constitution.
- **SER-007**: Redis MUST NOT be the only durable copy of run status, sessions that affect saved research state, generated reports, notifications, or user-visible analysis results.
- **SER-008**: Pinecone, Neo4j, and OpenSearch projection failures MUST be recorded as projection warnings or failed projection statuses, not as missing canonical data.

### Key Entities *(include if feature involves data)*

#### MVP Core Tables

- **users**: Optional owner profile for local/demo and future authenticated use. Key columns: `id`, `email`, `display_name`, `role`, `status`, `anonymized_at`, `created_at`, `updated_at`, `deleted_at`. Relationships: owns analysis requests, settings, notifications, portfolios. Rules: email is unique when present; local/demo system runs may use no user; login credentials, password hashes, SSO identifiers, and durable session credentials are not MVP fields; deleted users may be anonymized while audit records remain.
- **tickers**: Canonical security identifier for research. Key columns: `id`, `symbol`, `market`, `exchange`, `currency`, `name`, `sector`, `industry`, `provider_metadata`, `is_active`, `created_at`, `updated_at`. Relationships: referenced by requests, evidence, source documents, reports, portfolios. Rules: `(symbol, market)` is unique.
- **analysis_requests**: One user research/backtest/optimization request. Key columns: `id`, `user_id`, `ticker_id`, `request_type`, `horizon`, `risk_profile`, `parameters`, `status`, `degraded_reason`, `warning_summary`, `error_summary`, `created_at`, `started_at`, `completed_at`. Relationships: owns workflow events, results, reports, evidence. Rules: status values include pending, running, success, degraded, insufficient_data, failed, cancelled.
- **workflow_node_runs**: Observable node-level execution record. Key columns: `id`, `request_id`, `run_id`, `node_name`, `attempt_number`, `status`, `started_at`, `ended_at`, `duration_ms`, `error_type`, `error_message`, `evaluation_score`, `metadata`. Relationships: belongs to an analysis request and may link to an analysis result. Rules: one record per node attempt.
- **analysis_results**: Structured result from a workflow component. Key columns: `id`, `request_id`, `ticker_id`, `result_type`, `summary`, `metrics`, `warnings`, `missing_data_notes`, `status`, `created_at`. Relationships: may cite evidence and feed reports. Rules: result types include market, fundamental, news, graph_context, backtest, optimization, coordinator_draft, evaluation, rewrite.
- **reports**: Generated Korean research report record. Key columns: `id`, `request_id`, `ticker_id`, `current_version_id`, `title`, `language`, `status`, `safety_status`, `evaluation_score`, `disclaimer_present`, `created_at`, `updated_at`. Relationships: owns report versions and report-evidence citations. Rules: status values include draft, final, failed_review, archived.
- **report_versions**: Immutable report content snapshot. Key columns: `id`, `report_id`, `version_number`, `stage`, `report_json`, `report_markdown`, `created_by_node`, `created_at`. Relationships: cited by evaluator and search projections. Rules: `(report_id, version_number)` is unique.
- **evidence_items**: EvidenceItem-compatible canonical evidence. Key columns: `id`, `evidence_id`, `request_id`, `ticker_id`, `analysis_result_id`, `source_document_id`, `source_type`, `source_name`, `source_url`, `collected_at`, `metric_name`, `metric_value`, `description`, `created_at`. Relationships: cited by reports and evidence paths. Rules: `evidence_id` is unique and reportable.
- **report_evidence_citations**: Join table between report versions and evidence items. Key columns: `id`, `report_version_id`, `evidence_item_id`, `section_name`, `claim_text`, `created_at`. Relationships: supports evaluator grounding checks. Rules: duplicate citation for the same report section and evidence item is prevented.

#### Source Document And Projection Tables

- **source_documents**: Canonical provider or generated document metadata. Key columns: `id`, `ticker_id`, `document_type`, `source_name`, `source_url`, `title`, `language`, `published_at`, `collected_at`, `raw_content_ref`, `content_hash`, `revision_group_id`, `supersedes_document_id`, `metadata`, `status`. Relationships: owns chunks, may produce evidence, and may supersede an earlier document version in the same revision group. Rules: source URL plus content hash identifies duplicate crawls when URL exists; recrawled or corrected documents preserve version lineage through `revision_group_id` and `supersedes_document_id`.
- **document_chunks**: Canonical chunk provenance for semantic and keyword indexing. Key columns: `id`, `source_document_id`, `chunk_index`, `chunk_text`, `chunk_hash`, `token_count`, `metadata`, `created_at`. Relationships: projected to semantic and keyword indexes. Rules: `(source_document_id, chunk_index)` is unique.
- **index_projection_status**: Projection ledger for derived systems. Key columns: `id`, `source_table`, `source_id`, `target_system`, `projection_type`, `projection_key`, `status`, `attempt_count`, `last_attempt_at`, `last_success_at`, `error_message`, `idempotency_key`. Relationships: points to source records by table and ID. Rules: `(target_system, projection_type, idempotency_key)` is unique.
- **keyword_terms**: Optional normalized keyword catalog for search support. Key columns: `id`, `term`, `normalized_term`, `language`, `created_at`. Relationships: may connect to source documents and reports. Rules: normalized term and language are unique.
- **structured_log_events**: Durable operational event summary when logs need canonical retention beyond files. Key columns: `id`, `request_id`, `run_id`, `ticker_id`, `node_name`, `event_name`, `status`, `message`, `error_message`, `evaluation_score`, `occurred_at`, `metadata`. Relationships: projected to full-text log search. Rules: used for audit summaries, not high-volume raw tracing.

#### MVP Graph Knowledge Tables

- **wave_rules**: Canonical wave-theory rule record. Key columns: `id`, `rule_code`, `name`, `description`, `rule_type`, `status`, `source_document_id`, `created_at`, `updated_at`. Relationships: used by scenarios and invalidation conditions. Rules: `rule_code` is unique.
- **wave_scenarios**: Canonical scenario record. Key columns: `id`, `ticker_id`, `name`, `description`, `timeframe`, `status`, `confidence_label`, `created_at`, `updated_at`. Relationships: uses rules, has invalidation conditions, cites evidence. Rules: scenario text must be research framing, not trading advice.
- **wave_invalidation_conditions**: Scenario invalidation condition. Key columns: `id`, `scenario_id`, `condition_text`, `metric_name`, `threshold_value`, `direction`, `source_document_id`, `created_at`. Relationships: belongs to a scenario and may cite evidence. Rules: thresholds are conditions for scenario review, not guaranteed targets.
- **wave_scenario_rules**: Join table connecting scenarios and rules. Key columns: `id`, `scenario_id`, `rule_id`, `role`, `created_at`. Relationships: projected to graph index. Rules: duplicate scenario-rule-role rows are prevented.
- **evidence_paths**: Canonical explainability path. Key columns: `id`, `request_id`, `ticker_id`, `path_type`, `path_summary`, `source_node_ref`, `target_node_ref`, `confidence_label`, `created_at`. Relationships: owns path steps and may cite evidence. Rules: path summary must resolve to canonical evidence/rule/scenario records.
- **evidence_path_steps**: Ordered path step. Key columns: `id`, `evidence_path_id`, `step_index`, `node_table`, `node_id`, `relationship_type`, `description`, `created_at`. Relationships: projected to graph index. Rules: `(evidence_path_id, step_index)` is unique.

#### MVP User Experience Tables

- **user_settings**: User or system preference. Key columns: `id`, `user_id`, `setting_key`, `setting_value`, `scope`, `updated_at`, `deleted_at`. Relationships: optionally belongs to user. Rules: `(user_id, setting_key, scope)` is unique; system defaults may use no user; user-owned settings are deleted or anonymized when the user is deleted.
- **notifications**: User-facing in-app message or alert state. Key columns: `id`, `user_id`, `ticker_id`, `notification_type`, `title`, `body`, `payload`, `status`, `created_at`, `read_at`, `updated_at`, `deleted_at`. Relationships: may reference request/report/ticker in payload. Rules: notifications are informational only and must not instruct trading; MVP notifications do not store external delivery channel state; user-owned notifications are deleted or anonymized when the user is deleted.
- **portfolios**: Research-only portfolio context. Key columns: `id`, `user_id`, `name`, `description`, `base_currency`, `status`, `created_at`, `updated_at`, `deleted_at`. Relationships: owns portfolio items. Rules: portfolio records are for context and monitoring, not brokerage execution; user-owned portfolio records are deleted or anonymized when the user is deleted.
- **portfolio_items**: Ticker association inside a portfolio. Key columns: `id`, `portfolio_id`, `ticker_id`, `label`, `quantity_note`, `cost_basis_note`, `metadata`, `created_at`, `updated_at`, `deleted_at`. Relationships: belongs to portfolio and ticker. Rules: numeric values are optional research notes and must not trigger orders; user-owned portfolio items are deleted or anonymized when the user is deleted.

#### Deferred Expansion Tables

- **provider_sync_runs**: Provider collection batch metadata for yfinance, news search, filings, and document ingestion. Key columns: `id`, `provider_name`, `sync_type`, `status`, `started_at`, `completed_at`, `error_message`, `metadata`.
- **provider_payloads**: Immutable raw or normalized provider payload references. Key columns: `id`, `sync_run_id`, `ticker_id`, `source_name`, `source_url`, `payload_ref`, `payload_hash`, `collected_at`, `metadata`.
- **analysis_request_links**: Relationship between requests, such as rewrite, rerun, comparison, or optimization child run. Key columns: `id`, `from_request_id`, `to_request_id`, `link_type`, `created_at`.
- **notification_preferences**: Future notification routing preferences for email, webhook, push, or other external channels. Key columns: `id`, `user_id`, `channel`, `is_enabled`, `rules`, `updated_at`.
- **notification_deliveries**: Future external notification delivery attempts. Key columns: `id`, `notification_id`, `channel`, `delivery_target_ref`, `status`, `attempt_count`, `last_attempt_at`, `error_message`.
- **auth_identities**: Future login identity mapping for passwordless, OAuth, or SSO providers. Key columns: `id`, `user_id`, `provider`, `provider_subject`, `created_at`, `updated_at`.
- **user_sessions**: Future durable authenticated session records when real login exists. Key columns: `id`, `user_id`, `session_ref`, `created_at`, `expires_at`, `revoked_at`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Reviewers can map 100% of MVP research workflow artifacts to a canonical table without adding an unplanned table.
- **SC-002**: A complete sample platform flow can be represented with user, settings, portfolio context, request, node event, market result, fundamental result, news result, evidence, draft report, evaluator result, final report, notification, and projection status records.
- **SC-003**: 100% of report evidence citations in a sample report resolve to canonical evidence records with source, collection time, ticker, metric name, metric value, and description.
- **SC-004**: 100% of semantic, graph, and keyword projection records in a sample indexing run resolve back to canonical source records.
- **SC-005**: The schema review identifies zero fields that enable brokerage connection, order execution, guaranteed return, guaranteed target, or direct buy/sell/hold instruction behavior.
- **SC-006**: A degraded provider scenario can be stored with warnings and missing-data notes while preserving a user-facing final report and audit trail.
- **SC-007**: The MVP table set can be implemented independently of deferred provider-sync and preference-routing tables while still supporting later projections through stable IDs.
- **SC-008**: A user deletion scenario removes or anonymizes user-identifying data while preserving 100% of report, evidence, source document, workflow node, and audit references needed to review historical research output.
- **SC-009**: A notification scenario can create, list, mark read, and delete or anonymize in-app notifications without requiring any external delivery channel.
- **SC-010**: A graph-context sample can store wave rules, scenarios, invalidation conditions, evidence paths, and ordered path steps canonically before any graph index projection succeeds.

## Assumptions

- PostgreSQL is the durable source of truth; Pinecone, Neo4j, and OpenSearch are rebuildable projections, and Redis is ephemeral runtime support.
- MVP implementation starts with core research tables, source document/chunk provenance, projection status, users, settings, notifications, portfolios, portfolio items, wave rules, wave scenarios, invalidation conditions, evidence paths, and evidence path steps; detailed provider sync, raw payload retention, request-link history, and notification preference tables may be deferred.
- Local/demo mode may run without real user authentication, and the first MVP user table stores ownership/profile context only while leaving room for future authenticated identity tables.
- Reports remain research-only and Korean final reports keep the constitution-required disclaimer.
- Portfolio records are research context and watchlist-like organization only; real trading, brokerage integration, and order execution remain out of scope.
- MVP notifications are in-app records only; external delivery channels and retry queues are future expansion.
- Raw large content can be referenced through a durable content reference when storing the full body inline would be impractical.
- User deletion prioritizes privacy for user-identifying fields and user-owned UX records while preserving non-PII research audit records required for evidence-grounded report review.
