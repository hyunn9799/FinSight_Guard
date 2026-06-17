# Storage Architecture: Portfolio Research Platform

## Role Separation

PostgreSQL is the real source of truth. All durable business entities must be recoverable from PostgreSQL records and immutable/raw source payloads where applicable.

Pinecone, Neo4j, OpenSearch, and Redis are specialized systems:

- PostgreSQL: users, tickers, analysis requests, reports, analysis results, settings, notifications, portfolios, canonical evidence metadata, optimization runs.
- Pinecone: semantic search over document chunks such as news, financial-statement explanations, and wave-theory materials.
- Neo4j: wave-theory rules, scenarios, invalidation conditions, evidence paths, and explainable relationship traversal.
- OpenSearch: news full-text search, report full-text search, log search, keyword search, and operational search views.
- Redis: cache, task queue, rate limits, session state, and short-lived workflow coordination.

## Source-of-Truth Rules

- PostgreSQL owns canonical IDs, lifecycle status, user ownership, request parameters, generated report records, analysis result summaries, settings, notifications, and portfolio records.
- Pinecone vectors are projections of source documents/chunks and may be rebuilt.
- Neo4j graph nodes/edges are projections of canonical rule/evidence/scenario records and may be rebuilt.
- OpenSearch documents are projections of raw news, generated reports, and logs and may be rebuilt.
- Redis data is ephemeral and must never be the only durable copy of an analysis result or report.

## PostgreSQL Core Tables

### users

- `id`: UUID primary key.
- `email`: unique text, nullable for local/demo users if auth is deferred.
- `display_name`: text.
- `created_at`, `updated_at`: timestamps.

### tickers

- `id`: UUID primary key.
- `symbol`: unique text.
- `market`: text or null.
- `name`: text or null.
- `created_at`, `updated_at`: timestamps.

### analysis_requests

- `id`: UUID primary key.
- `user_id`: UUID nullable foreign key to `users`.
- `ticker_id`: UUID foreign key to `tickers`.
- `request_type`: `research`, `backtest`, `robust_optimization`.
- `parameters`: JSON object containing horizon, risk profile, date range, cost assumptions, fold setup, and strategy parameters.
- `status`: `pending`, `running`, `success`, `degraded`, `insufficient_data`, or `failed`.
- `created_at`, `started_at`, `completed_at`: timestamps.

### analysis_results

- `id`: UUID primary key.
- `request_id`: UUID foreign key to `analysis_requests`.
- `result_type`: `market`, `fundamental`, `news`, `backtest`, `optimization`, `evaluation`.
- `summary`: text.
- `metrics`: JSON object.
- `warnings`: JSON array.
- `created_at`: timestamp.

### reports

- `id`: UUID primary key.
- `request_id`: UUID foreign key to `analysis_requests`.
- `ticker_id`: UUID foreign key to `tickers`.
- `title`: text.
- `language`: text, default `ko`.
- `report_json`: JSON object.
- `report_markdown`: text.
- `evaluation_score`: numeric or null.
- `safety_status`: `pass`, `fail`, or `not_evaluated`.
- `created_at`: timestamp.

### evidence_items

- `id`: UUID primary key.
- `evidence_id`: unique text.
- `request_id`: UUID nullable foreign key to `analysis_requests`.
- `ticker_id`: UUID nullable foreign key to `tickers`.
- `source_type`: text.
- `source_name`: text.
- `source_url`: text or null.
- `collected_at`: timestamp.
- `metric_name`: text.
- `metric_value`: JSON scalar or null.
- `description`: text.

### robust_optimization_runs

- `id`: UUID primary key.
- `request_id`: UUID foreign key to `analysis_requests`.
- `ticker_id`: UUID foreign key to `tickers`.
- `cost_assumptions`: JSON object.
- `scoring_policy`: JSON object.
- `fold_setup`: JSON object.
- `status`: `success`, `degraded`, `insufficient_data`, or `failed`.
- `robust_candidate`: JSON object or null.
- `manual_baseline`: JSON object or null.
- `passive_baseline`: JSON object or null.
- `created_at`: timestamp.

### walk_forward_folds

- `id`: UUID primary key.
- `optimization_run_id`: UUID foreign key to `robust_optimization_runs`.
- `fold_index`: integer.
- `train_start`, `train_end`, `test_start`, `test_end`: dates.
- `selected_params`: JSON object.
- `metrics`: JSON object.
- `status`: text.
- `warnings`: JSON array.

### market_regime_results

- `id`: UUID primary key.
- `optimization_run_id`: UUID foreign key to `robust_optimization_runs`.
- `regime`: text.
- `start_date`, `end_date`: dates.
- `trading_days`: integer.
- `completed_trades`: integer.
- `metrics`: JSON object.
- `confidence`: `normal` or `low`.
- `low_confidence_reason`: text or null.

### settings

- `id`: UUID primary key.
- `user_id`: UUID nullable foreign key to `users`.
- `key`: text.
- `value`: JSON object.
- `updated_at`: timestamp.

### notifications

- `id`: UUID primary key.
- `user_id`: UUID nullable foreign key to `users`.
- `notification_type`: text.
- `payload`: JSON object.
- `status`: `pending`, `sent`, `read`, or `failed`.
- `created_at`, `updated_at`: timestamps.

### portfolios

- `id`: UUID primary key.
- `user_id`: UUID foreign key to `users`.
- `name`: text.
- `holdings`: JSON object.
- `created_at`, `updated_at`: timestamps.

## Pinecone Projection

Semantic chunk records should include:

- `chunk_id`: stable ID derived from source document ID and chunk index.
- `source_document_id`: PostgreSQL or provider document ID.
- `source_type`: `news`, `financial_statement_explanation`, `wave_theory_material`, or similar.
- `ticker`: optional symbol.
- `published_at` or `collected_at`: timestamp.
- `text`: source chunk text for embedding generation.
- `metadata`: source URL, title, language, ticker, document type, and provenance.

PostgreSQL stores canonical document metadata and chunk provenance; Pinecone stores the embedding index.

## Neo4j Projection

Graph nodes:

- `WaveRule`
- `Scenario`
- `InvalidationCondition`
- `EvidenceItem`
- `Ticker`
- `Report`
- `OptimizationRun`

Graph relationships:

- `SCENARIO_USES_RULE`
- `RULE_HAS_INVALIDATION`
- `EVIDENCE_SUPPORTS_SCENARIO`
- `EVIDENCE_CONTRADICTS_SCENARIO`
- `REPORT_CITES_EVIDENCE`
- `OPTIMIZATION_PRODUCED_EVIDENCE`

Neo4j supports explainable evidence paths; PostgreSQL remains the canonical record store.

## OpenSearch Projection

Indexes:

- `news_documents`: original/news article full text and metadata.
- `reports`: report markdown/text, ticker, run ID, safety status, evaluation score.
- `logs`: structured logs with run ID, ticker, node, timestamps, status, error message, and evaluation score.
- `keywords`: optional normalized keyword/search helper index.

OpenSearch is optimized for retrieval and operational search, not canonical writes.

## Redis Usage

Redis keys should be short-lived and scoped:

- `cache:*`: market/news/fundamental provider cache.
- `queue:*`: background analysis or indexing jobs.
- `rate_limit:*`: API and provider throttling.
- `session:*`: UI/API session state.
- `run_status:*`: transient workflow status for polling.

Durable run status must be persisted back to PostgreSQL.

## Consistency Rules

- PostgreSQL commit happens before derived index writes are considered successful.
- Derived index failures must produce warnings/degraded index status, not fabricate missing retrieval results.
- Re-index jobs must be idempotent.
- Tests must monkeypatch all external store clients or use local fakes.
