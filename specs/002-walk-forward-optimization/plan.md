# Implementation Plan: Walk-Forward Strategy Optimization

**Branch**: `[002-walk-forward-optimization]` | **Date**: 2026-06-17 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/002-walk-forward-optimization/spec.md`

## Summary

Replace total-return-only strategy parameter selection with a research-only robust optimization flow. The implementation will extend the existing `src/backtest` and `backtest_agent` surfaces with deterministic walk-forward folds, risk-adjusted candidate scoring, transaction-cost assumptions, regime performance summaries, evidence generation, API/UI result exposure, and safety framing that prevents optimized parameters from being presented as trading advice. The broader persistence direction is now explicit: PostgreSQL is the source of truth, with Pinecone, Neo4j, OpenSearch, and Redis serving specialized index/cache/queue roles.

## Technical Context

**Language/Version**: Python 3.x, matching the existing project runtime.

**Primary Dependencies**: pandas, numpy, scipy/statsmodels, optuna, FastAPI, Streamlit, Pydantic, pytest, existing LangGraph workflow modules, PostgreSQL client/ORM layer, Pinecone client, Neo4j driver, OpenSearch client, Redis client/queue layer.

**Storage**: PostgreSQL is the real source of truth for users, tickers, analysis requests, reports, analysis results, settings, notifications, and portfolios. Pinecone stores semantic document chunks for news, financial-statement explanations, and wave-theory materials. Neo4j stores wave-theory rules, scenarios, invalidation conditions, and evidence paths. OpenSearch stores news full text, report full text, logs, and keyword-search indexes. Redis supports cache, job queue, rate limiting, and sessions. Local JSON/Markdown remains acceptable only as a transitional export/demo artifact.

**Testing**: pytest with deterministic synthetic price fixtures and monkeypatched providers; no live yfinance, Tavily, Firecrawl, or OpenAI calls in tests.

**Target Platform**: Local portfolio/demo runtime via Streamlit, FastAPI/uvicorn, and Docker.

**Project Type**: Python web-service plus Streamlit UI and LangGraph research workflow.

**Performance Goals**: Robust optimization should be bounded by API/UI request limits and remain suitable for portfolio demo usage; default validation scenarios should complete in a small deterministic pytest suite without external network calls.

**Constraints**: No real trading, brokerage integration, order execution, buy/sell/hold recommendations, guaranteed return claims, paid data provider requirement, or live external calls in tests. PostgreSQL is authoritative; Pinecone, Neo4j, and OpenSearch are derived indexes that must be rebuildable from PostgreSQL source records and raw/provider documents.

**Scale/Scope**: MVP supports one robust all-regime parameter candidate first, with regime-specific parameter evaluation deferred until enough out-of-sample evidence exists.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Evidence grounding: PASS. Optimization metrics that appear in reports will be represented by `Optimization Evidence Item` records using the existing `EvidenceItem` schema; PostgreSQL stores canonical evidence metadata while Neo4j can store relationship paths and Pinecone/OpenSearch can index supporting documents.
- Financial safety: PASS. Outputs must call selected parameters historical simulation candidates, not recommendations or trading settings. The required Korean disclaimer remains mandatory for final reports.
- LangGraph workflow: PASS. The feature extends the existing Backtest Agent path and report/evaluator flow without bypassing Coordinator/Evaluator safety checks.
- Deterministic quality: PASS. New tests will use synthetic price series and monkeypatched loaders; no live API dependency is allowed.
- Observability: PASS. Optimization runs will preserve structured logs, report persistence, API health/metrics behavior, degraded-data notes, Redis job/cache status, and OpenSearch log/report search indexing.

**Constitution Result**: PASS with explicit infrastructure expansion. The constitution allows persistent storage beyond local project storage when a later plan explicitly expands infrastructure; this plan records PostgreSQL as source of truth and the other stores as rebuildable specialized indexes.

## Project Structure

### Documentation (this feature)

```text
specs/002-walk-forward-optimization/
├── plan.md
├── research.md
├── data-model.md
├── storage-architecture.md
├── quickstart.md
├── contracts/
│   └── robust-optimization-api.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── agents/
│   └── backtest_agent.py          # surface robust optimization evidence in workflow
├── backtest/
│   ├── optimizer.py               # keep legacy optimize_backtest; add robust/walk-forward orchestration
│   ├── strategy.py                # reuse run_backtest and BacktestParams
│   └── data_loader.py             # existing price-history loader, monkeypatched in tests
├── db/
│   ├── postgres.py                # canonical persistence connection/session
│   ├── redis.py                   # cache, queue, rate limit, session integration
│   └── repositories/              # user, ticker, request, report, result, settings, notification, portfolio repos
├── evidence/
│   └── evidence_builder.py        # add/build optimization evidence items
├── graph/
│   └── state.py                   # add typed robust optimization analysis/result contracts if needed
├── indexes/
│   ├── pinecone_index.py          # semantic chunk index writer/query facade
│   ├── neo4j_graph.py             # evidence path and wave-rule graph facade
│   └── opensearch_index.py        # full-text news/report/log index facade
└── observability/
    ├── logger.py                  # structured node/run logging remains active
    └── metrics.py                 # existing metrics endpoint remains active

main.py                           # add request/response contract for robust optimization
app.py                            # add Streamlit controls/result view for robust optimization

tests/
├── test_backtest_charts_optimizer.py
├── test_backtest_agent.py
├── test_api.py
└── test_workflow_routing.py
```

**Structure Decision**: Use the existing single-project Python layout while adding storage/index adapter boundaries. PostgreSQL is the canonical write path; Pinecone, Neo4j, and OpenSearch are not treated as source-of-truth databases.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design & Contracts

- Data model: [data-model.md](./data-model.md)
- Storage architecture: [storage-architecture.md](./storage-architecture.md)
- API contract: [contracts/robust-optimization-api.md](./contracts/robust-optimization-api.md)
- Validation guide: [quickstart.md](./quickstart.md)

## Post-Design Constitution Check

- Evidence grounding: PASS. Data model includes `OptimizationEvidenceItem` mapping, PostgreSQL canonical tables, and index projections for semantic, graph, and full-text retrieval.
- Financial safety: PASS. Contracts require historical-simulation wording, limitations, and no advice-like parameter labels.
- LangGraph workflow: PASS. Backtest Agent integration routes optimization output through existing Coordinator/Evaluator paths.
- Deterministic quality: PASS. Quickstart and design require deterministic synthetic fixtures and pytest coverage.
- Observability: PASS. Optimization status, warnings, degraded notes, metrics preservation, Redis queue/cache signals, and OpenSearch log indexing are part of the planned contract.

**Post-Design Constitution Result**: PASS with documented complexity exceptions for the broader storage architecture.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Persistent storage beyond local files | User has explicitly selected PostgreSQL as the source of truth for users, tickers, requests, reports, results, settings, notifications, and portfolios | Local JSON/Markdown cannot support user/account workflows, portfolios, notification state, or reliable report/result history |
| Multiple specialized indexes | Pinecone, Neo4j, OpenSearch, and Redis each serve a distinct retrieval/runtime role: semantic chunks, graph evidence paths, keyword/full-text/log search, cache/queue/rate-limit/session | A single PostgreSQL-only design would simplify MVP but would not support semantic search, graph reasoning paths, full-text search, and queue/cache concerns at the intended architecture level |
