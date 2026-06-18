# Implementation Plan: PostgreSQL Source-of-Truth Table Schema

**Branch**: `[004-postgresql-table-schema]` | **Date**: 2026-06-18 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/004-postgresql-table-schema/spec.md`

## Summary

Introduce PostgreSQL as the canonical persistence layer for the financial research platform while preserving the existing research-only safety posture. The implementation will replace local-only/in-memory persistence paths with a source-of-truth schema covering research workflow records, evidence, reports, source documents, projection status, local/demo users, in-app notifications, research-only portfolios, and canonical wave-theory graph records. Pinecone, Neo4j, OpenSearch, and Redis remain derived or ephemeral systems whose records must resolve back to PostgreSQL IDs.

## Technical Context

**Language/Version**: Python 3.12, matching the active project virtual environment.

**Primary Dependencies**: Existing FastAPI, Streamlit, Pydantic v2, pytest, LangGraph workflow modules, yfinance/news providers, plus SQLAlchemy 2.x, Alembic, and psycopg for PostgreSQL persistence.

**Storage**: PostgreSQL becomes the durable source of truth. Existing local JSON/Markdown reports remain transitional export/demo artifacts. Pinecone, Neo4j, and OpenSearch are rebuildable projections. Redis is ephemeral cache/queue/rate-limit/session support only.

**Testing**: pytest with deterministic repository tests using isolated transactions or a local test database fixture; external providers and derived index clients must be faked or monkeypatched.

**Target Platform**: Local portfolio/demo runtime through FastAPI, Streamlit, Docker Compose, and pytest.

**Project Type**: Single Python web-service plus Streamlit UI and LangGraph research workflow.

**Performance Goals**: MVP repository operations for one deterministic local-demo workflow run should complete within 5 seconds on a developer laptop; schema tests should run deterministically without live provider calls; projection status queries should support resolving records by source table/source ID and idempotency key.

**Constraints**: No brokerage integration, order execution, buy/sell/hold recommendations, guaranteed returns, guaranteed targets, paid provider requirement, real login implementation, external notification delivery, or treating derived indexes as canonical stores. User deletion must anonymize/delete user-identifying UX data while preserving non-PII research audit records.

**Scale/Scope**: First PostgreSQL MVP includes research workflow tables, source document/chunk tables, projection status, local/demo users, settings, in-app notifications, portfolios, portfolio items, wave rules, wave scenarios, invalidation conditions, evidence paths, and evidence path steps. Provider sync batches, immutable raw payload retention, request-link history, auth identities, durable sessions, notification preferences, and external delivery attempts are deferred.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Evidence grounding: PASS. The plan keeps `evidence_items` and `report_evidence_citations` as canonical tables and requires all important report claims to resolve to EvidenceItem-compatible records.
- Financial safety: PASS. Persistence tables explicitly exclude brokerage connections, orders, guaranteed targets, and trading instructions. Portfolio tables are research-only context.
- LangGraph workflow: PASS. The schema includes `analysis_requests`, `workflow_node_runs`, `analysis_results`, `reports`, `report_versions`, and evaluator/rewrite result types so conditional routing and retry outcomes are auditable.
- Deterministic quality: PASS. Tests will use deterministic DB fixtures/fakes and will not call live yfinance, Tavily, Firecrawl, OpenAI, Pinecone, Neo4j, OpenSearch, or Redis.
- Observability: PASS. `workflow_node_runs`, `structured_log_events`, projection statuses, report persistence, health checks, and metrics behavior remain part of the design.

**Constitution Result**: PASS with one documented complexity exception. The constitution's original domain constraints excluded production databases, but this feature explicitly promotes PostgreSQL as the source of truth. The exception is justified by the user-approved platform storage direction and bounded to a local/demo PostgreSQL MVP with derived stores remaining optional/rebuildable.

## Project Structure

### Documentation (this feature)

```text
specs/004-postgresql-table-schema/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── postgresql-storage-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── db/
│   ├── postgres.py
│   ├── models.py
│   ├── repositories/
│   │   ├── analysis_repository.py
│   │   ├── evidence_repository.py
│   │   ├── report_repository.py
│   │   ├── source_document_repository.py
│   │   ├── projection_repository.py
│   │   ├── user_repository.py
│   │   ├── notification_repository.py
│   │   ├── portfolio_repository.py
│   │   └── graph_repository.py
│   └── migrations/
├── storage/
│   ├── report_store.py
│   └── run_store.py
├── graph/
├── agents/
└── observability/

tests/
├── test_storage_postgres_schema.py
├── test_storage_postgres_repositories.py
├── test_storage_postgres_research_flow.py
├── test_storage_postgres_privacy.py
└── test_storage_postgres_projection_status.py
```

**Structure Decision**: Use the existing single-project layout. Add `src/db` as a narrow persistence boundary rather than spreading SQL concerns through agents, graph nodes, API handlers, or UI code. Keep local file stores as compatibility/export helpers until repository-backed persistence replaces the relevant paths.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design & Contracts

- Data model: [data-model.md](./data-model.md)
- Storage contract: [contracts/postgresql-storage-contract.md](./contracts/postgresql-storage-contract.md)
- Validation guide: [quickstart.md](./quickstart.md)

## Post-Design Constitution Check

- Evidence grounding: PASS. Data model includes canonical evidence, citation, source document, and graph evidence path records.
- Financial safety: PASS. User and portfolio models remain research-only; auth and brokerage/order concepts are deferred or excluded.
- LangGraph workflow: PASS. Request, node run, result, report, version, evaluator, rewrite, and degraded/failure states are represented.
- Deterministic quality: PASS. Quickstart and contract require deterministic schema/repository tests with faked external services.
- Observability: PASS. Node runs, structured log event summaries, projection status, health/metrics preservation, and report persistence are covered.

**Post-Design Constitution Result**: PASS with the same PostgreSQL persistence exception documented below.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Production-style PostgreSQL persistence beyond original MVP local files | User explicitly selected PostgreSQL as source of truth for users, tickers, requests, reports, results, settings, notifications, portfolios, evidence, source documents, and graph records | Local JSON/Markdown cannot support user history, anonymization, portfolios, notification state, rebuildable derived indexes, or auditable source-of-truth relationships |
| First MVP includes user UX and graph knowledge tables | Clarification selected full platform MVP and graph tables now | A smaller research-ledger-only schema would require immediate rework once portfolios, in-app notifications, and wave evidence paths are added |
