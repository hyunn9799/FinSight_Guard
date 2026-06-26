# PROJECT_PLAN.md

## 1. Project Name

GraphRAG-Based Financial Scenario Research Assistant

## 2. Goal

Build a LangGraph and Neo4j GraphRAG-based financial research assistant that
collects market, fundamental, news, technical, and NEoWave analysis evidence,
retrieves a Company-centered evidence subgraph, generates Bull/Base/Bear
scenario reports, and validates the result through an Evaluator Agent.

This project is designed as a portfolio-grade practical AI Agent workflow.

## 3. Product Positioning

This system is not a stock recommendation engine.
It is a responsible AI research assistant that supports decision-making by
organizing evidence, graph relationships, risks, confirmation conditions,
invalidation conditions, and scenarios.

The current MVP is report-first and does not execute real trades. The long-term
roadmap may add signal candidate extraction and paper-trading or mock investment
API validation, but live trading remains outside the MVP and requires separate
approval, regulatory review, risk controls, user consent, and operational
stability review.

The product identity is:

```text
Ticker input
→ Company-centered graph retrieval
→ News + financial + technical + NEoWave evidence context
→ Bull/Base/Bear scenario generation
→ Evaluator safety review
→ Report and evidence path persistence
```

PostgreSQL is the canonical durable store and source of truth. Pinecone is the
semantic vector index. Neo4j is the GraphRAG relationship index and knowledge
graph. OpenSearch is the keyword, full-text, and log search index. Redis is
ephemeral infrastructure for cache, work queues, sessions, and rate limits. All
non-PostgreSQL stores must resolve back to canonical PostgreSQL IDs.

## 4. Main Workflow

```text
User Input
  ↓
Input Validator
  ↓
Market Agent
  ↓
Fundamental Agent
  ↓
News Agent
  ↓
Wave Agent
  ↓
Evidence Builder
  ↓
PostgreSQL Canonical Persistence
  ↓
Neo4j Graph Projection
  ↓
Graph Context Builder
  ↓
Scenario / Coordinator Agent
  ↓
Bull/Base/Bear Draft Report
  ↓
Evaluator Agent
  ↓
Conditional Edge
    ├─ PASS → Save Report → Response
    └─ FAIL → Rewrite Agent → Evaluator Agent → Save or Fail
```

# Project Specification: MVP Plus

## 5. MVP Plus Scope

### ✅ Included
*   **Data Sources & Integration**
    *   `yfinance` market data
    *   `yfinance` financial data
    *   `Tavily` / `Firecrawl` optional news search
    *   Mock news fallback
    *   NEoWave rule/source material ingestion as research references
*   **Analysis Logic**
    *   Technical indicators
    *   WaveAnalysis candidates based on OHLCV-derived swing or monowave candidates
    *   NEoWave rule checks with passed/failed/unknown/needs_human_review labels
    *   Bull/Base/Bear scenario generation
    *   Confirmation and invalidation condition extraction
    *   EvidenceItem tracking
    *   Graph evidence path tracking
*   **LangGraph Workflow**
    *   LangGraph workflow design
    *   Conditional branching
    *   Retry logic
    *   Wave Agent
    *   Graph Context Builder
    *   Evaluator Agent
    *   Rewrite Agent
*   **GraphRAG**
    *   Company-centered graph retrieval
    *   Neo4j projection for NewsEvent, FinancialMetric, TechnicalAnalysis, WaveAnalysis, NEoWaveRule, Scenario, Risk, Evidence, and Report relationships
    *   Subgraph context injection into scenario generation
    *   Scenario-to-evidence relationship persistence after generation
*   **Roadmap Documentation**
    *   Signal Candidate Layer defined as future work
    *   Paper Trading / Mock Investment API validation defined as future work
    *   Live trading explicitly outside MVP
*   **Interface & Delivery**
    *   Streamlit UI
    *   FastAPI API
*   **Infrastructure & DevOps**
    *   PostgreSQL canonical persistence
    *   Neo4j relationship projection
    *   Report storage
    *   Structured logs
    *   Metrics endpoint
    *   Dockerfile
    *   Tests (`pytest`)

### ❌ Excluded
*   Real trading
*   Brokerage API
*   Order execution
*   Portfolio optimization
*   Paid data providers
*   User login
*   Production database
*   Financial advice claims
*   Guaranteed price targets or guaranteed returns
*   Definitive automatic NEoWave counts
*   Graphing every raw candle, news sentence, financial statement row, or temporary calculation
*   Treating Neo4j as the canonical source of truth

---

## 6. Product Roadmap

### Phase 1: Research Report MVP

*   Bull/Base/Bear scenario report generation
*   News, financial, technical, NEoWave, risk, and GraphRAG evidence integration
*   Scenario-to-evidence path persistence
*   No real trading and no order execution

### Phase 2: Signal Candidate Layer

*   Convert confirmation, invalidation, and risk conditions into structured signal candidates
*   Still no order execution

### Phase 3: Paper Trading / Mock Investment API

*   Validate signal candidates in a simulated environment only
*   Store performance, drawdown, win rate, profit factor, and risk metrics

### Phase 4: Live Trading Review

*   Outside the MVP
*   Requires separate approval, regulatory review, risk controls, user consent, and operational stability review

---

## 7. Target GraphRAG Model

### Core Nodes

*   Company
*   Ticker
*   NewsEvent
*   FinancialMetric
*   TechnicalAnalysis
*   WaveAnalysis
*   NEoWaveRule
*   Scenario
*   Risk
*   Evidence
*   Report

### Core Relationships

*   HAS_TICKER
*   MENTIONED_IN
*   HAS_METRIC
*   HAS_TECHNICAL_ANALYSIS
*   HAS_WAVE_ANALYSIS
*   CHECKED_BY
*   SUPPORTS
*   AFFECTS
*   HAS_RISK
*   SUPPORTED_BY
*   CONTAINS

### Storage Responsibilities

*   PostgreSQL owns canonical records, stable IDs, audit history, report versions, evidence items, source documents, and projection status.
*   Pinecone owns rebuildable semantic vector retrieval over news, financial narrative, NEoWave/wave-theory, and report chunks.
*   Neo4j owns rebuildable relationship retrieval and evidence-path traversal for GraphRAG context.
*   OpenSearch owns rebuildable keyword, full-text, and log search over news originals, report text, logs, tickers, and event keywords.
*   Redis owns ephemeral cache, queue, session, rate-limit, deduplication, and short-lived workflow state.
*   Pinecone, Neo4j, OpenSearch, and Redis records must be rebuildable or disposable from PostgreSQL canonical records and source documents.
*   Projection/index failures must degrade retrieval, not erase canonical data or block safe report storage.

---

## 8. Definition of Done (DoD)

The project is considered complete when all the following conditions are met:

*   **Operational Success**
    *   [ ] `streamlit run app.py` works as expected.
    *   [ ] `uvicorn main:app --reload` runs the API server successfully.
    *   [ ] `pytest` passes all test cases.
*   **Storage & Logging**
    *   [ ] Reports are successfully saved through canonical persistence and optional export files.
    *   [ ] Scenario-to-evidence relationships are stored.
    *   [ ] Neo4j projection status is recorded for graph records.
    *   [ ] Logs are structured and saved under the `/logs` directory.
*   **Agent Quality & Reliability**
    *   [ ] **Wave Agent** produces candidate wave analysis with confirmation/invalidation conditions and explicit uncertainty.
    *   [ ] **Graph Context Builder** retrieves bounded Company-centered graph context for at least one ticker.
    *   [ ] **Scenario / Coordinator Agent** generates Bull/Base/Bear reports with evidence, risks, limitations, and disclaimer.
    *   [ ] **Evaluator Agent** correctly identifies and fails unsafe or low-quality reports.
    *   [ ] **Workflow/Rewrite Agent** successfully rewrites failed reports.
*   **Deployment & Documentation**
    *   [ ] Docker container builds and starts successfully.
    *   [ ] `README.md` clearly explains the LangGraph, PostgreSQL, Neo4j GraphRAG, safety design, and limitations.
    *   [ ] Demo scenario works for at least one ticker symbol.


---

## 7. 006 Provider Contract Boundary

Feature branch 006 establishes the provider-agnostic MCP contract layer. Ownership is
split across three feature units:

*   **006 — contracts and mapping:** Defines normalized contract types
    (CompanyProfile, FinancialMetric, NewsEvent, MarketData, TechnicalAnalysisResult,
    WaveAnalysisResult, RiskAssessment, EvidenceItem), the normalization boundary
    (raw provider responses must not cross into agent logic), and the GraphRAG mapping
    spec that feeds the 005 graph model.
*   **004 — canonical tables:** Owns the underlying database schema. The provider
    contract records introduced by 006 are persisted via Alembic migration
    `8f1a2b3c4d5e` (down_revision `56464a69bd55`), which extends the 004 schema
    without replacing it.
*   **005 — graph semantics:** Owns the graph node/edge model. The 006 mapping
    contract produces graph-eligible specs for 005 to consume; raw provider payloads
    are never projected directly into the graph.

---

<!-- # 8. Codex 프롬프트 1 — 전체 설계 업그레이드

첫 프롬프트는 이걸 써.

```text
Read AGENTS.md and PROJECT_PLAN.md first.

I want to build this project as a portfolio-grade practical AI Agent workflow, not a simple RAG chatbot.

Task:
Analyze the current repository and create a detailed implementation plan for the following target:

"Evidence-Grounded Financial Research Multi-Agent Workflow"

The final system must include:
1. LangGraph role-based agents
2. EvidenceItem-based grounding
3. Evaluator Agent
4. Rewrite Agent
5. Conditional branching
6. Retry/fallback logic
7. Streamlit UI
8. FastAPI API
9. report storage
10. structured logging
11. tests
12. Docker support

Do not implement code yet.

Output:
1. Current repository assessment
2. Missing modules
3. Recommended folder structure
4. Step-by-step implementation phases
5. Risks and how to reduce scope if needed
6. Definition of Done

Be strict and practical. Avoid overengineering. -->
