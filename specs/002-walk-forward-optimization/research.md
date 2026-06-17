# Research: Walk-Forward Strategy Optimization

## Decision: PostgreSQL Is the Source of Truth

**Rationale**: Users, tickers, analysis requests, reports, analysis results, settings, notifications, portfolios, evidence metadata, and robust optimization outputs need durable canonical records. PostgreSQL provides transactional consistency, relational constraints, and a clear recovery point for derived indexes.

**Alternatives considered**: Local JSON/Markdown only, OpenSearch as primary storage, or Redis as primary storage. Rejected because local files do not support user/portfolio workflows, OpenSearch is a search index rather than a transactional source of truth, and Redis is ephemeral.

## Decision: Pinecone Stores Semantic Document Chunk Indexes

**Rationale**: News, financial-statement explanations, and wave-theory materials need semantic retrieval beyond keyword matching. Pinecone should store chunk embeddings and metadata, while PostgreSQL keeps canonical document/chunk provenance.

**Alternatives considered**: PostgreSQL-only text search or OpenSearch-only keyword retrieval. Rejected because the user explicitly wants semantic search, and keyword retrieval does not capture conceptual similarity as well.

## Decision: Neo4j Stores Wave-Theory Rules and Evidence Paths

**Rationale**: Wave-theory rules, scenarios, invalidation conditions, and evidence paths are graph-shaped. Neo4j provides relationship traversal and explainable evidence-path queries without forcing deeply recursive joins into PostgreSQL.

**Alternatives considered**: PostgreSQL adjacency tables only. Rejected because the intended graph reasoning and evidence-path inspection are clearer in a graph index.

## Decision: OpenSearch Stores Full-Text and Operational Search Indexes

**Rationale**: News original text, generated reports, structured logs, and keyword search are search workloads. OpenSearch supports full-text queries, filtering, and operational log search while keeping PostgreSQL free of search-specific indexing concerns.

**Alternatives considered**: PostgreSQL full-text search only. Rejected because the user wants dedicated keyword/news/report/log search and future scale beyond simple SQL filtering.

## Decision: Redis Supports Cache, Queue, Rate Limit, and Sessions

**Rationale**: Provider calls, optimization jobs, UI/API sessions, and rate limits need short-lived fast state. Redis should not own durable results; PostgreSQL persists canonical statuses and outputs.

**Alternatives considered**: In-process cache and synchronous-only execution. Rejected because they do not handle multi-worker API/Streamlit operation or background indexing/optimization cleanly.

## Decision: Extend Existing Backtest Modules Instead of Adding a New Optimization Service

**Rationale**: The repository already has `src/backtest/strategy.py`, `src/backtest/optimizer.py`, `src/agents/backtest_agent.py`, FastAPI `/backtest`, Streamlit backtest UI, and deterministic backtest tests. Extending these modules keeps the feature compatible with the current app while the new storage adapters provide persistence/index boundaries.

**Alternatives considered**: A separate optimization microservice. Rejected because this portfolio project should remain a single deployable application until task planning proves a split is necessary.

## Decision: Keep Legacy `optimize_backtest` Compatible and Add Robust Optimization Separately

**Rationale**: Existing tests and UI may rely on the current Optuna return shape. A new robust orchestration surface can compute walk-forward folds, candidate metrics, and robust score without breaking current callers.

**Alternatives considered**: Rewriting `optimize_backtest` in place. Rejected because it increases regression risk and mixes a legacy total-return optimizer with the new research-safe robust ranking behavior.

## Decision: Use Time-Ordered Walk-Forward Folds With At Least 3 Valid Test Folds

**Rationale**: The spec requires validation periods to occur after selection periods and clarified that robust candidates need at least 3 valid test folds. This is enough to compute median, worst-fold, and stability behavior without making MVP test fixtures too large.

**Alternatives considered**: Two folds or five folds. Two folds is too weak for stability claims; five folds is stronger but increases fixture and runtime burden for the MVP.

## Decision: Use Risk-Balanced Robust Score as the MVP Default

**Rationale**: The clarified default weights are 30% out-of-sample return, 25% risk-adjusted return, 20% drawdown control, 15% worst-fold resilience, and 10% stability or turnover penalty. This prevents total return from dominating while remaining explainable in reports.

**Alternatives considered**: Performance-led weighting and conservative resilience-led weighting. Performance-led undercuts the safety goal; conservative resilience-led may hide useful out-of-sample performance too aggressively for an exploratory research tool.

## Decision: Apply User-Adjustable One-Way Fee and Slippage Defaults

**Rationale**: The spec clarified 0.05% one-way fee and 0.05% one-way slippage defaults. Exposing both assumptions keeps result interpretation transparent and makes deterministic tests straightforward.

**Alternatives considered**: Hard-coded costs or no default. Hard-coded costs reduce scenario comparison value; no default creates ambiguous API/UI behavior.

## Decision: Mark Regime Results Low-Confidence Below Sample Thresholds

**Rationale**: The spec clarified low-confidence when a regime has fewer than 10 completed trades or fewer than 60 trading days. This avoids overinterpreting thin bull, bear, sideways, high-volatility, or low-volatility segments.

**Alternatives considered**: Only trading-day threshold or only trade-count threshold. A combined rule better captures both market coverage and strategy activity.

## Decision: Use Synthetic Data and Monkeypatched Loaders for Tests

**Rationale**: The constitution requires deterministic tests with no live external APIs. Synthetic dated price series can force drawdown breaches, low trade counts, unstable folds, insufficient folds, and regime low-confidence cases.

**Alternatives considered**: Cached yfinance fixtures. Rejected for Phase 1 planning because static synthetic data is easier to reason about and keeps tests fully deterministic.

## Decision: Preserve Research-Only Framing Through Coordinator and Evaluator

**Rationale**: Robust optimization output affects final reports and must not become trading advice. The Backtest Agent should label outputs as historical simulation candidates, attach evidence, and rely on Coordinator/Evaluator checks for disclaimer, limitations, and forbidden wording.

**Alternatives considered**: Returning optimization results only through a standalone endpoint. Rejected because the project goal is an evidence-grounded multi-agent workflow, and report integration is required.
