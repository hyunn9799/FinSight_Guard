# Design: PostgreSQL Research Ledger (004 — US1 Slice)

**Date**: 2026-06-18
**Source spec**: `specs/004-postgresql-table-schema/` (spec.md, plan.md, data-model.md, research.md, tasks.md)
**Slice**: User Story 1 — Define Canonical Research Records (P1)
**Status**: Design approved, pending written-spec review

## Context

`specs/004` already defines the full PostgreSQL source-of-truth schema (24 MVP tables + 7 deferred,
74 tasks across US1/US2/US3). This document does not redefine that schema — it scopes the **first
delivery slice (US1)** and records the implementation decisions made during brainstorming so the
implementation plan can be written against a settled design.

Today the project has **no DB layer**: `src/db` is absent, `requirements.txt` has no
SQLAlchemy/Alembic/psycopg, `docker-compose.yml` has no postgres service. Persistence is local files
(`src/storage/report_store.py` JSON/MD, `src/storage/run_store.py`). The closed-but-deferred storage
work (issue #71) was split into `specs/003` (umbrella platform-storage vision) and `specs/004` (this
PostgreSQL schema). US2 of this feature is the canonical foundation for the index layer #71 described.

## Full-feature sequencing (confirmed)

004 is complete only after **all three** user stories ship. This slice is US1; US2 and US3 follow,
each as its own design/plan/implementation cycle. The slice order is chosen for incremental, testable
delivery — not to drop scope.

1. **US1 — Research ledger** (this slice): core 12 tables + DB infrastructure (Phase 1·2).
   Establishes the `src/db` boundary, Alembic, test fixtures, and the `report_store` integration seam
   that US2/US3 reuse.
2. **US2 — Projection & graph canonical records**: `index_projection_status`, `keyword_terms`,
   `wave_*`, `evidence_paths`, `evidence_path_steps`. This is the canonical foundation for the index
   layer described in issue #71.
3. **US3 — User UX**: `user_settings`, `notifications`, `portfolios`, `portfolio_items` plus the
   anonymizing user-deletion flow.
4. **Polish**: docs, regression tests, safety review.

## Decisions (brainstorming outcomes)

| Decision | Choice | Rationale |
|---|---|---|
| Delivery scope of this slice | US1 research ledger only | Independently testable MVP; matches tasks.md "First Delivery Slice" |
| Test database engine | Always real PostgreSQL | Validate JSONB, server-side UUID, and real constraints directly |
| Runtime requirement | PostgreSQL **required**; JSON/MD files kept as **export** | Matches source-of-truth principle; files remain demo/export artifacts |
| Test PG provisioning | `docker-compose` postgres service + env DSN | Same path for dev, team, and CI service container |
| Test schema creation | Run **Alembic migrations** in test setup | Prevents schema/migration drift; migration is exercised by every run |

## US1 scope — tables

Created in this slice (canonical for one end-to-end research run):

`users` · `tickers` · `analysis_requests` · `workflow_node_runs` · `analysis_results` · `reports` ·
`report_versions` · `evidence_items` · `report_evidence_citations` · `source_documents` ·
`document_chunks` · `structured_log_events`

Field-level definitions are authoritative in `specs/004-postgresql-table-schema/data-model.md`.

`users` and `tickers` are created in their minimal US1 form (ownership/identity only); US3 extends the
user-facing surface. No US2/US3/deferred tables are created in this slice.

## Module structure (`src/db` boundary)

```
src/db/
  __init__.py
  postgres.py          # SQLAlchemy 2.x engine/session factory; DSN from env
  models.py            # US1 ORM models (declarative; JSONB, server-side UUID, timestamps)
  constants.py         # status vocabularies (request/result/report/safety)
  migrations/
    env.py             # Alembic; target_metadata = models.metadata
    versions/0001_postgresql_source_of_truth.py   # US1 tables, single initial revision
  repositories/
    base.py                          # session/transaction helper
    analysis_repository.py           # ticker, request, node_run, result, structured_log_event
    evidence_repository.py           # evidence_items, report_evidence_citations
    report_repository.py             # report, report_version, current-version, safety status
    source_document_repository.py    # source_documents (+ revision lineage), document_chunks
```

Agents, LangGraph nodes, API handlers, and UI access PostgreSQL **only through repository
interfaces**. Raw SQL does not leak outside `src/db`. The initial Alembic revision covers the US1
table set; US2/US3 add later revisions.

## Test infrastructure

- Add a `db` service (postgres:16) with a healthcheck to `docker-compose.yml`.
- `tests/fixtures/postgres.py` + `tests/conftest.py`: at session start, **create schema by running
  Alembic migrations**; wrap each test in a transaction and **roll back** for isolation (fast,
  deterministic).
- Tests connect via an env DSN. When the DSN is unset, PostgreSQL tests fail/skip explicitly. No live
  external provider APIs (yfinance, Tavily, Firecrawl, OpenAI, Pinecone, Neo4j, OpenSearch, Redis) are
  called.

## Workflow / API integration seam (highest-risk area)

- Keep the **existing `report_store.py` / `run_store.py` function signatures**. Internally they write
  to PostgreSQL first (canonical), then continue to emit the existing JSON/MD as export. Callers in
  `src/graph` and `main.py` change minimally.
- `src/graph/workflow.py`: persist request · node_run · result · report · report_version · evidence on
  **both success and degraded paths**. Missing data is stored as `missing_data_notes` — never
  fabricated.
- `main.py` `/analyze`, `/backtest`, `/backtest/optimize`: responses stay **backward compatible** while
  using the persisted request/report IDs.
- **PG required**: on a PostgreSQL connection/persistence failure, return a degraded warning or an
  explicit persistence error (deterministic behavior per SER-005). The app does not silently fall back
  to files-only.

## Safety preservation (unchanged guarantees)

- **Zero** brokerage/order/guaranteed-return/buy-sell-hold fields — verified by a schema review test
  (SC-005, FR-014).
- Korean final-report disclaimer preserved; important numeric/factual claims resolve to
  `evidence_items` (SER-002, SER-003).
- Redis is not part of this slice, so SER-007 is not engaged here.

## Task mapping (from `specs/004/tasks.md`)

In scope for this slice:
- **Phase 1 Setup**: T001–T007
- **Phase 2 Foundational**: T008–T019
- **Phase 3 US1**: T020–T038
- **Relevant polish**: T066 (README), T069–T074 (health/metrics check, compile, US1 storage tests,
  regression tests, no-live-service check, safety field review)

Out of scope for this slice (later slices): T039–T065 (US2/US3) and US2/US3-specific polish.

## Acceptance for this slice

- A deterministic sample research run persists and re-fetches ticker, request, node attempts,
  market/fundamental/news results, evidence, draft/final report versions, evaluator result, citations,
  source documents, chunks, and degraded warnings — with **no live provider calls** (SC-002, SC-006).
- 100% of sample report evidence citations resolve to canonical `evidence_items` (SC-003).
- Schema review finds zero trading/brokerage/guaranteed-outcome fields (SC-005).
- Existing local JSON/MD export still works as a compatibility artifact.

## Out of scope (this feature, all slices)

Login/password/SSO, durable sessions, brokerage integration, order execution, guaranteed targets,
external notification delivery, and treating any derived index (Pinecone/Neo4j/OpenSearch) or Redis as
a canonical store.
