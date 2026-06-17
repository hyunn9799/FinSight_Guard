<!--
Sync Impact Report
Version change: template -> 1.0.0
Modified principles:
- Template Principle 1 -> I. Evidence-Grounded Financial Research
- Template Principle 2 -> II. Financial Safety and Scenario Framing
- Template Principle 3 -> III. Role-Based LangGraph Workflow
- Template Principle 4 -> IV. Deterministic Quality Gates
- Template Principle 5 -> V. Observable, Recoverable Operation
Added sections:
- Domain Constraints
- Development Workflow
Removed sections:
- None
Templates requiring updates:
- ✅ updated .specify/templates/plan-template.md
- ✅ updated .specify/templates/spec-template.md
- ✅ updated .specify/templates/tasks-template.md
- ✅ checked .specify/templates/commands/*.md (directory absent)
- ✅ checked README.md, AGENTS.md, PROJECT_PLAN.md, CLAUDE.md
Follow-up TODOs:
- None
-->
# FinSight Guard Constitution

## Core Principles

### I. Evidence-Grounded Financial Research
FinSight Guard MUST produce financial research from traceable market, fundamental,
news, and graph-context evidence. Important numeric or factual claims MUST be
backed by `EvidenceItem` records containing source, collection time, ticker,
metric name, metric value, and description. Reports MUST include an evidence
summary and MUST NOT invent missing metrics, URLs, financial facts, news events,
or evidence IDs.

Rationale: The project exists to demonstrate evidence-grounded agentic research,
not unsupported financial commentary.

### II. Financial Safety and Scenario Framing
FinSight Guard MUST NOT provide stock recommendations, trading instructions,
brokerage integration, order execution, guaranteed returns, guaranteed targets,
or principal protection claims. Final Korean reports MUST use scenario-based
language, including 관망 시나리오, 분할 접근 시나리오, and 리스크 회피 시나리오,
instead of direct buy, sell, or hold recommendations. Every final report MUST
include this disclaimer exactly:

```text
본 보고서는 교육 및 정보 제공 목적의 AI 리서치 결과이며, 특정 종목의 매수·매도·보유를 권유하지 않습니다. 최종 투자 판단과 책임은 투자자 본인에게 있습니다.
```

Rationale: The system is a responsible research assistant, not an investment
advisor or automated trading product.

### III. Role-Based LangGraph Workflow
The implementation MUST preserve explicit agent responsibilities and LangGraph
conditional routing. Required workflow roles are Input Validator, Supervisor or
Coordinator routing logic, Market Agent, Fundamental Agent, News Agent, Graph
Context Builder when evidence relationships are used, Coordinator Agent,
Evaluator Agent, Rewrite Agent, and report persistence. Ticker validation failure
MUST stop with a user-friendly error. Market data failure MUST retry once before
degraded reporting. News failure MUST fall back to mock news or continue with a
no-news warning. Evaluator failure MUST route to Rewrite and stop after at most
two rewrite attempts.

Rationale: The portfolio value of this project depends on visible multi-agent
workflow design, conditional branching, retries, and failure handling.

### IV. Deterministic Quality Gates
Tests MUST be deterministic and MUST NOT depend on live external APIs. Required
coverage includes technical indicators, safety checking, evaluator pass/fail
behavior, workflow routing, report storage, and FastAPI health/analyze behavior.
External providers such as yfinance, Tavily, Firecrawl, and OpenAI MUST be mocked,
stubbed, or replaced with deterministic fallbacks in tests. Changes touching
runtime code MUST pass `python -m compileall src` and relevant pytest suites
before completion.

Rationale: Reproducible tests are mandatory for a financial safety workflow and
for portfolio-grade review.

### V. Observable, Recoverable Operation
Runtime workflow execution MUST emit structured logs with run ID, ticker, node
name, start/end time, success/failure, error message, and evaluation score when
available. FastAPI MUST expose health and metrics endpoints, including
`total_runs`, `successful_runs`, `failed_runs`, and
`average_evaluation_score`. Reports MUST be stored locally in JSON and/or
Markdown with clear degraded-data notes when an upstream source fails.

Rationale: Observability and recoverable degraded behavior make agent failures
auditable instead of silent.

## Domain Constraints

The approved scope is a LangGraph-based, evidence-centered financial research
assistant with Streamlit UI, FastAPI API, local report storage, structured logs,
metrics, Docker support, and pytest coverage. The implementation MUST remain
small enough for portfolio demonstration and MUST NOT add real trading, brokerage
APIs, order execution, portfolio optimization, paid data provider requirements,
user login, production databases, or financial advice claims.

Market analysis MUST use yfinance price history and calculate MA20, MA60, MA120,
RSI, MACD, and ATR when data is available. Fundamental analysis MUST use yfinance
company and financial metrics and handle missing fields gracefully. News analysis
MUST use Tavily or Firecrawl only when an API key is configured, otherwise it
MUST use deterministic mock news fallback or disclose missing news coverage.

## Development Workflow

Before editing code or project governance, contributors MUST inspect the
repository, read `AGENTS.md` and `PROJECT_PLAN.md`, and state a short
implementation plan. Changes MUST be limited to relevant files and kept
testable. New infrastructure MUST be justified by a constitution principle or a
documented project requirement.

Feature specs MUST include safety, evidence, failure-mode, and observability
requirements when the feature affects report generation, workflow routing,
provider integration, persistence, API behavior, or UI delivery. Implementation
plans MUST document the constitution check before design work and re-check it
after design. Task lists MUST include deterministic tests for any changed safety,
evidence, routing, storage, API, or indicator behavior.

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

**Version**: 1.0.0 | **Ratified**: 2026-06-17 | **Last Amended**: 2026-06-17
