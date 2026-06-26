# AGENTS.md

## Project Identity

This project is a portfolio-grade LangGraph and Neo4j GraphRAG-based financial
research multi-agent workflow.

It is NOT a stock recommendation system.
The current MVP is NOT a live automated trading system.
It is NOT allowed to guarantee profit, predict returns with certainty, or tell users to buy/sell a stock.

The system is an evidence-based financial research assistant that helps users
compare Bull/Base/Bear scenarios using market data, financial data, news events,
technical analysis, NEoWave rule candidates, and graph evidence paths.

Longer-term roadmap work may add signal candidates and paper-trading or mock
investment API validation. Those stages must remain simulated-only until a
separate live-trading review approves user consent, regulatory, risk-management,
and operational-stability requirements.

## MVP Definition

The MVP is a GraphRAG-based AI Research Assistant that generates ticker-level
Bull/Base/Bear scenario reports by combining:

1. news events
2. financial metrics
3. technical analysis results
4. NEoWave-based wave analysis candidates
5. risks and confirmation/invalidation conditions
6. evidence paths retrieved from a Company-centered graph

The system should use a Company node as the center of graph retrieval, find the
related NewsEvent, FinancialMetric, TechnicalAnalysis, WaveAnalysis, NEoWaveRule,
Risk, Scenario, Evidence, and Report nodes, and use that subgraph as LLM context
for scenario generation.

After report generation, the system must persist which Scenario was supported
by which NewsEvent, FinancialMetric, TechnicalAnalysis, WaveAnalysis, Evidence,
and Risk relationships.

## Roadmap Boundaries

Phase 1 is the current Research Report MVP:
- Bull/Base/Bear scenario report generation
- news, financial, technical, NEoWave, risk, and GraphRAG evidence integration
- no real trading and no order execution

Phase 2 may introduce Signal Candidate records:
- convert confirmation, invalidation, and risk conditions into structured candidates
- no order execution

Phase 3 may introduce Paper Trading or Mock Investment API validation:
- simulated orders only
- record performance, drawdown, win rate, profit factor, and risk metrics

Phase 4 live trading remains outside the MVP:
- requires separate approval, regulatory review, risk controls, user consent, and operational stability review

## Core Portfolio Goal

The final project must demonstrate:

1. LangGraph workflow design
2. Role-based agents
3. Tool/API integration
4. Evidence-grounded report generation
5. Neo4j-backed GraphRAG retrieval over research evidence relationships
6. Evaluator Agent for responsible AI review
7. Conditional branching and retry logic
8. Failure handling and fallback behavior
9. Safety filters for financial advice
10. Tests, logs, report storage, Streamlit UI
11. FastAPI, Docker, health check, and basic monitoring endpoint

## Required Agents

### 1. Market Agent
- Uses yfinance price history.
- Calculates MA20, MA60, MA120, RSI, MACD, ATR.
- Produces structured market analysis.
- Must attach EvidenceItem objects for numeric facts.

### 2. Fundamental Agent
- Uses yfinance company info and financial metrics.
- Handles missing financial fields gracefully.
- Produces valuation, profitability, and stability summary.
- Must attach EvidenceItem objects.

### 3. News Agent
- Uses Tavily or Firecrawl if API key exists.
- Falls back to mock news provider when API key is missing.
- Extracts positive factors, negative factors, event risks, and source URLs.
- Must attach EvidenceItem objects.

### 4. Wave Agent
- Uses OHLCV-derived swing or monowave candidates and NEoWave rule materials.
- Produces WaveAnalysis candidates, not definitive wave counts.
- Must label rules as passed, failed, unknown, or needs_human_review.
- Must generate confirmation and invalidation conditions when evidence supports them.
- Must attach EvidenceItem objects and NEoWaveRule references.
- Must not claim that a wave count is certain or that a price target is guaranteed.

### 5. Graph Context Builder
- Builds or queries the Company-centered GraphRAG context.
- Uses Neo4j as a relationship index/knowledge graph for retrieval.
- Uses PostgreSQL IDs as canonical references for graph nodes and relationships.
- Retrieves related NewsEvent, FinancialMetric, TechnicalAnalysis, WaveAnalysis, NEoWaveRule, Risk, Scenario, Evidence, and Report paths.
- Must keep graph output bounded to the context needed for scenario generation.
- Must tolerate missing or stale graph projection records and fall back to canonical PostgreSQL evidence when possible.

### 6. Coordinator / Scenario Agent
- Combines Market/Fundamental/News/Wave analysis and Graph Context Builder output.
- Generates a Korean research report.
- Must use Bull/Base/Bear scenario framing:
  - Bull / 상승 시나리오
  - Base / 기준 시나리오
  - Bear / 하락 시나리오
- Each scenario must include supporting evidence, confirmation conditions, invalidation conditions, key risks, limitations, and data date.
- Must not directly recommend buy/sell/hold.
- Must include evidence summary, graph evidence path summary, risks, limitations, data date, and disclaimer.

### 7. Evaluator Agent
- Reviews the draft report.
- Checks:
  - source grounding
  - graph evidence path grounding
  - numeric consistency
  - excessive investment recommendation language
  - risk disclosure
  - limitations
  - data freshness
  - disclaimer
- Returns pass/fail and scores.
- If evaluation fails, workflow must route to Rewrite Agent.

### 8. Rewrite Agent
- Revises unsafe or weak reports based on Evaluator feedback.
- Must remove direct investment advice.
- Must add missing risk, limitation, or disclaimer sections.
- Must preserve evidence-based reasoning.

## Required Safety Rules

Forbidden phrases include:

- 무조건 매수
- 강력 매수
- 반드시 매수
- 지금 사야 합니다
- 매도해야 합니다
- 수익 보장
- 손실 없음
- 확실한 수익
- 원금 보장
- 목표가 보장

Every final report must include this disclaimer:

"본 보고서는 교육 및 정보 제공 목적의 AI 리서치 결과이며, 특정 종목의 매수·매도·보유를 권유하지 않습니다. 최종 투자 판단과 책임은 투자자 본인에게 있습니다."

## Required Conditional Branching

The LangGraph workflow must include conditional routing:

1. If ticker validation fails:
   - stop workflow and return user-friendly error.

2. If market data fails:
   - retry once.
   - if still failing, return degraded report with missing data notes.

3. If news search fails:
   - fallback to mock news provider or continue with no-news warning.

4. If graph projection or GraphRAG retrieval fails:
   - continue from canonical PostgreSQL evidence when possible.
   - include graph-context warning or missing-data note.

5. If Wave Agent cannot validate a NEoWave candidate:
   - mark the rule status as unknown or needs_human_review.
   - do not fabricate a wave count.

6. If Evaluator passes:
   - save final report.

7. If Evaluator fails:
   - route to Rewrite Agent.
   - re-run Evaluator.
   - stop after max 2 rewrite attempts.

## Evidence Rules

All important numeric or factual claims must be backed by EvidenceItem.

EvidenceItem fields:
- evidence_id
- source_type
- source_name
- source_url
- collected_at
- ticker
- metric_name
- metric_value
- description

The final report should include an evidence summary section.

## GraphRAG Rules

PostgreSQL is the canonical durable store for requests, reports, evidence,
analysis results, source documents, projection status, users, tickers, settings,
notifications, and portfolios.

Neo4j is the relationship index and GraphRAG retrieval layer. It should store
only graph nodes and edges that improve scenario retrieval, explanation, and
evidence-path tracing.

Pinecone is the semantic vector index for news chunks, financial narrative
chunks, NEoWave and wave-theory materials, and report chunks.

OpenSearch is the keyword, full-text, and log search index for news originals,
report text, logs, tickers, event keywords, and operational search.

Redis is ephemeral infrastructure for cache, work queues, rate limits, sessions,
deduplication, and short-lived workflow state.

PostgreSQL records are authoritative. Pinecone, Neo4j, OpenSearch, and Redis
must be rebuildable or disposable from canonical records and source documents.

MVP graph nodes:
- Company
- Ticker
- NewsEvent
- FinancialMetric
- TechnicalAnalysis
- WaveAnalysis
- NEoWaveRule
- Scenario
- Risk
- Evidence
- Report

MVP graph relationships:
- HAS_TICKER
- MENTIONED_IN
- HAS_METRIC
- HAS_TECHNICAL_ANALYSIS
- HAS_WAVE_ANALYSIS
- CHECKED_BY
- SUPPORTS
- AFFECTS
- HAS_RISK
- SUPPORTED_BY
- CONTAINS

Do not graph every raw candle, every news sentence, every financial statement
row, or every temporary calculation. Raw and canonical records belong in
PostgreSQL; Neo4j stores retrieval-ready relationships and evidence paths.

## Observability Rules

Add structured logs for:
- run_id
- ticker
- node name
- start/end time
- success/failure
- error message
- evaluation score

Add a simple metrics endpoint in FastAPI:
- total_runs
- successful_runs
- failed_runs
- average_evaluation_score

## Testing Rules

Tests must be deterministic.
Tests must not depend on live external APIs.

Required tests:
- technical indicator tests
- wave analysis candidate tests
- Graph Context Builder tests with mocked Neo4j/repository clients
- safety checker tests
- evaluator pass/fail tests
- workflow routing tests
- report storage tests
- FastAPI health/analyze tests

## Development Rules for Codex

Before editing:
1. Inspect the repository.
2. Read AGENTS.md and PROJECT_PLAN.md.
3. Propose a short implementation plan.

While editing:
1. Modify only relevant files.
2. Keep changes small and testable.
3. Do not add unnecessary infrastructure.
4. Do not implement real trading or order execution.

After editing:
1. Run `python -m compileall src`.
2. Run relevant tests.
3. Summarize changed files and remaining TODOs.

## GitHub Issue Rules

All GitHub issues for this repository must be written in natural Korean.

When converting Spec Kit tasks into GitHub issues, do not create one issue per
T### task by default. Group tasks by phase, user story, or implementation
milestone, and include the detailed T### task IDs inside the issue body as an
included-work section or checklist.

For large features, create or maintain a high-level MVP tracking issue that
summarizes the first practical delivery slice. If too many detailed issues
already exist, close them with a comment linking to the grouped issue instead of
deleting them.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
at specs/006-provider-mcp-contracts/plan.md
<!-- SPECKIT END -->
