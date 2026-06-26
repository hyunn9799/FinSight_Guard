<!--
Sync Impact Report
Version change: 1.1.0 -> 1.2.0
Modified principles:
- Domain Constraints -> Domain Constraints
Added sections:
- Storage and Index Roles
Removed sections:
- None
Templates requiring updates:
- ✅ updated .specify/templates/plan-template.md
- ✅ updated .specify/templates/spec-template.md
- ✅ updated .specify/templates/tasks-template.md
- ✅ checked .specify/templates/commands/*.md (directory absent)
- ✅ updated README.md
- ✅ updated AGENTS.md
- ✅ updated PROJECT_PLAN.md
Follow-up TODOs:
- None
-->
# FinSight Guard Constitution

## Core Principles

### I. Graph-Grounded Financial Research
FinSight Guard MUST produce financial research from traceable market,
fundamental, news, technical, NEoWave, risk, and graph-context evidence.
Important numeric, factual, or relationship claims MUST be backed by
`EvidenceItem` records or graph evidence paths that resolve to canonical source
records. Reports MUST include an evidence summary and MUST NOT invent missing
metrics, URLs, financial facts, news events, wave-rule outcomes, graph
relationships, or evidence IDs.

GraphRAG is a core product structure, not a decorative storage layer. The system
MUST use Company/Ticker-centered graph context to connect news events, financial
metrics, technical analysis, NEoWave rules, WaveAnalysis candidates, risks, and
evidence when generating scenario reports. Generated Scenario and Evidence paths
MUST be persisted so reviewers can trace which evidence supported each scenario.

Rationale: The project exists to demonstrate evidence-grounded agentic research,
relationship-aware retrieval, and auditable scenario reasoning, not unsupported
financial commentary.

### II. Bull/Base/Bear Scenario Safety
FinSight Guard MUST NOT provide stock recommendations, trading instructions,
real brokerage integration, live order execution, guaranteed returns, guaranteed
targets, or principal protection claims in the current research MVP. Final
Korean reports MUST use Bull/Base/Bear scenario framing instead of direct buy,
sell, or hold recommendations:

- Bull Scenario: positive conditions improve or upside evidence strengthens.
- Base Scenario: current evidence supports a neutral or reference-case view.
- Bear Scenario: negative conditions strengthen or invalidation risk increases.

Legacy wording such as 관망, 분할 접근, and 리스크 회피 MAY appear only as
scenario interpretation or risk guidance, never as investment action
instructions. For example, "관망 추천" MUST be reframed as "Base Scenario에서는
추가 확인 조건 대기", "분할 매수" MUST be reframed as "Bull Scenario 확인 조건
충족 시 signal candidate로 검토 가능", and "리스크 회피 전략 실행" MUST be
reframed as "Bear Scenario에서는 invalidation condition 발생 가능성 증가".

Every final report MUST include this disclaimer exactly:

```text
본 보고서는 교육 및 정보 제공 목적의 AI 리서치 결과이며, 특정 종목의 매수·매도·보유를 권유하지 않습니다. 최종 투자 판단과 책임은 투자자 본인에게 있습니다.
```

Rationale: The current MVP is a responsible research assistant. It may organize
decision-support evidence, but it must not act as an investment advisor or live
trading product.

### III. Role-Based LangGraph and GraphRAG Workflow
The implementation MUST preserve explicit agent responsibilities and LangGraph
conditional routing. Required workflow roles are Input Validator, Supervisor or
Coordinator routing logic, Market Agent, Fundamental Agent, News Agent, Graph
Context Builder when evidence relationships are used, Wave Agent when NEoWave
evidence is used, Scenario or Coordinator Agent, Evaluator Agent, Rewrite Agent,
and report persistence.

The Graph Context Builder MUST retrieve bounded Company/Ticker-centered context
from graph relationships when available and MUST fall back or degrade safely when
graph projection is missing or stale. The Wave Agent MUST produce candidate,
confirmation, invalidation, and review-status outputs; it MUST NOT claim a
definitive automated wave count.

Ticker validation failure MUST stop with a user-friendly error. Market data
failure MUST retry once before degraded reporting. News failure MUST fall back to
mock news or continue with a no-news warning. Graph retrieval failure MUST
degrade or fall back to canonical evidence without fabricating relationships.
Evaluator failure MUST route to Rewrite and stop after at most two rewrite
attempts.

Rationale: The portfolio value of this project depends on visible multi-agent
workflow design, GraphRAG retrieval, conditional branching, retries, and failure
handling.

### IV. Deterministic Quality Gates
Tests MUST be deterministic and MUST NOT depend on live external APIs. Required
coverage includes technical indicators, WaveAnalysis candidates, Graph Context
Builder retrieval and fallback, safety checking, evaluator pass/fail behavior,
workflow routing, scenario-to-evidence persistence, report storage, and FastAPI
health/analyze behavior. External providers such as yfinance, Tavily, Firecrawl,
OpenAI, Neo4j, Pinecone, OpenSearch, broker APIs, and paper-trading APIs MUST be
mocked, stubbed, or replaced with deterministic fallbacks in tests. Changes
touching runtime code MUST pass `python -m compileall src` and relevant pytest
suites before completion.

Rationale: Reproducible tests are mandatory for a financial safety workflow and
for portfolio-grade review.

### V. Observable, Recoverable Operation
Runtime workflow execution MUST emit structured logs with run ID, ticker, node
name, start/end time, success/failure, error message, and evaluation score when
available. FastAPI MUST expose health and metrics endpoints, including
`total_runs`, `successful_runs`, `failed_runs`, and
`average_evaluation_score`. Reports, scenario outputs, evidence paths, and graph
projection status MUST remain recoverable through canonical persistence or
explicit export artifacts. Degraded-data notes MUST be clear when an upstream
source, graph retrieval, or projection path fails.

Rationale: Observability and recoverable degraded behavior make agent failures
auditable instead of silent.

## Domain Constraints

The approved MVP scope is a LangGraph, PostgreSQL, and Neo4j GraphRAG-based
financial scenario research assistant with Streamlit UI, FastAPI API, canonical
report/evidence persistence, structured logs, metrics, Docker support, and
pytest coverage. The current MVP MUST NOT add real trading, live brokerage
execution, live order placement, guaranteed targets, guaranteed returns, or
financial advice claims.

PostgreSQL is the canonical durable store for requests, reports, evidence,
analysis results, source documents, and projection status. Neo4j is the
GraphRAG relationship projection used to retrieve Company/Ticker-centered
subgraphs and evidence paths. Neo4j MUST NOT become the canonical source of
truth for reports, evidence, requests, or source documents.

Market analysis MUST use yfinance price history and calculate MA20, MA60, MA120,
RSI, MACD, and ATR when data is available. Fundamental analysis MUST use yfinance
company and financial metrics and handle missing fields gracefully. News analysis
MUST use Tavily or Firecrawl only when an API key is configured, otherwise it
MUST use deterministic mock news fallback or disclose missing news coverage.

NEoWave analysis MUST be presented as candidate, evidence, confirmation, and
invalidation analysis. It MUST NOT be presented as a certain wave count,
guaranteed target, or guaranteed return.

## Storage and Index Roles

PostgreSQL is the authoritative ledger and source of truth for users, tickers,
analysis requests, raw provider responses, normalized records, evidence, reports,
analysis results, settings, notifications, portfolios, source documents, and
projection status.

Pinecone is the semantic vector index for rebuildable meaning-based retrieval
over news chunks, financial narrative chunks, NEoWave and wave-theory materials,
and report chunks. Pinecone records MUST resolve back to canonical PostgreSQL
records or source documents.

Neo4j is the GraphRAG relationship index and knowledge graph for Company/Ticker
centered relationships, NEoWave rules, scenarios, confirmation conditions,
invalidation conditions, risks, and evidence paths. Neo4j graph records MUST be
rebuildable from canonical PostgreSQL records and MUST NOT become the canonical
ledger.

OpenSearch is the keyword, full-text, and log search index for news originals,
report text, logs, tickers, event keywords, and operational search. OpenSearch
documents MUST resolve back to canonical PostgreSQL records, source documents,
or structured log records.

Redis is ephemeral infrastructure for cache, work queues, rate limits, sessions,
deduplication, and short-lived workflow state. Redis MUST NOT hold canonical
research, evidence, report, portfolio, or trading records.

PostgreSQL records are authoritative. Pinecone, Neo4j, OpenSearch, and Redis are
rebuildable or disposable indexes/runtime stores and MUST degrade safely without
mutating or erasing canonical records.

## Product Roadmap Boundaries

Phase 1 is the Research Report MVP. It MUST generate Bull/Base/Bear scenario
reports from news, financial, technical, NEoWave, risk, and GraphRAG evidence.
It MUST NOT execute trades.

Phase 2 is the Signal Candidate Layer. It MAY structure report conditions into
signal candidates, confirmation conditions, invalidation conditions, and risk
conditions. It MUST NOT execute orders.

Phase 3 is Paper Trading or Mock Investment API validation. It MAY execute
simulated orders only in a mock or paper-trading environment and MUST record
performance, drawdown, win rate, profit factor, and risk metrics as historical
simulation evidence. It MUST NOT connect report output directly to live orders.

Phase 4 is Live Trading Review. It is outside the MVP and MUST require separate
approval, regulatory review, risk management design, user consent design, and
operational stability review before any real brokerage integration or live order
execution is considered.

## Development Workflow

Before editing code or project governance, contributors MUST inspect the
repository, read `AGENTS.md` and `PROJECT_PLAN.md`, and state a short
implementation plan. Changes MUST be limited to relevant files and kept
testable. New infrastructure MUST be justified by a constitution principle or a
documented project requirement.

Feature specs MUST include safety, evidence, failure-mode, and observability
requirements when the feature affects report generation, workflow routing,
provider integration, persistence, GraphRAG retrieval, paper-trading validation,
API behavior, or UI delivery. Implementation plans MUST document the
constitution check before design work and re-check it after design. Task lists
MUST include deterministic tests for any changed safety, evidence, routing,
storage, GraphRAG, paper-trading, API, indicator, or WaveAnalysis behavior.

## Governance

This constitution supersedes conflicting repository guidance. `AGENTS.md`,
`PROJECT_PLAN.md`, Spec Kit templates, README documentation, and feature plans
MUST remain consistent with these principles.

Amendments MUST document the reason for change, the semantic version bump, and
the dependent templates or runtime guidance that were checked or updated. Version
bumping follows semantic versioning:

- MAJOR for removing or redefining core principles in a backward-incompatible way.
- MINOR for adding principles, required sections, quality gates, or materially
  expanded governance.
- PATCH for wording clarifications, typo fixes, and non-semantic refinements.

Compliance review MUST occur during `/speckit-plan`, `/speckit-tasks`, code
review, and final implementation summary. Any approved exception MUST be listed
in the plan's Complexity Tracking section with the simpler alternative that was
rejected.

**Version**: 1.2.0 | **Ratified**: 2026-06-17 | **Last Amended**: 2026-06-21
