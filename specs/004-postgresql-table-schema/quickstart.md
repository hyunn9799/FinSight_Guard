# Quickstart: PostgreSQL Source-of-Truth Table Schema

## Prerequisites

- Python dependencies installed from `requirements.txt`.
- PostgreSQL available locally for implementation validation once dependencies and migrations are added.
- No live yfinance, Tavily, Firecrawl, OpenAI, Pinecone, Neo4j, OpenSearch, or Redis calls in tests.

## Validation Goal

Prove that PostgreSQL can store a complete research workflow ledger while preserving financial safety, evidence traceability, user privacy behavior, in-app notifications, research-only portfolios, graph canonical records, and rebuildable projection status.

## Phase 1: Static Validation

Expected commands after implementation:

```bash
python -m compileall src
pytest tests/test_storage_postgres_schema.py
```

Expected outcome:

- All schema metadata loads.
- Required tables and uniqueness rules are present.
- Forbidden trading/order/brokerage fields are absent from user and portfolio tables.

## Phase 2: Repository Validation

Expected command:

```bash
pytest tests/test_storage_postgres_repositories.py
```

Expected outcome:

- Ticker upsert is idempotent.
- Analysis request lifecycle transitions are stored.
- Node attempts preserve retry/failure information.
- Evidence IDs are unique.
- Report versions are append-only.
- Projection status records resolve to canonical source records.

## Phase 3: End-To-End Research Ledger

Expected command:

```bash
pytest tests/test_storage_postgres_research_flow.py
```

Expected outcome:

- A sample research request stores user, ticker, request, node runs, market/fundamental/news results, evidence, draft report, evaluator result, final report, citations, source document chunks, and projection statuses.
- A degraded provider sample stores warnings and missing-data notes without fabricated facts.
- Final report persistence preserves the required Korean disclaimer flag and safety status.

## Phase 4: Privacy And UX Scope

Expected command:

```bash
pytest tests/test_storage_postgres_privacy.py
```

Expected outcome:

- Local/demo user records can exist without email or credentials.
- User deletion anonymizes or deletes PII and user-owned UX records.
- Historical reports, evidence, source documents, node runs, and audit references remain resolvable under an anonymous owner.
- In-app notifications can be created, listed, marked read, and deleted without external delivery fields.
- Portfolio records remain research-only and contain no order execution fields.

## Phase 5: Graph And Projection Validation

Expected command:

```bash
pytest tests/test_storage_postgres_projection_status.py
```

Expected outcome:

- Wave rules, scenarios, invalidation conditions, evidence paths, and ordered path steps are stored canonically.
- Neo4j projection failures do not delete canonical graph records.
- Pinecone/OpenSearch projection statuses can be marked pending, success, failed, and stale.

## Manual Demo Check

After repository integration with the existing workflow:

```bash
uvicorn main:app --reload
```

Then run a deterministic or monkeypatched analyze request and verify:

- The API response still returns the existing response shape.
- A canonical `analysis_requests` record exists for the run.
- Evidence and report citations resolve from PostgreSQL.
- `/health` and `/metrics` still respond.

## Notes

- Existing local JSON/Markdown report export may remain during migration, but PostgreSQL is the durable source of truth.
- Derived stores are not required for this quickstart; their projection status records are enough for deterministic validation.
