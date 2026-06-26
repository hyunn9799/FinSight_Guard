<!--
Sync Impact Report
Version change: 1.0.0 -> 1.1.0
Modified principles:
- I. Evidence-Grounded Financial Research -> unchanged
- II. Financial Safety and Scenario Framing -> unchanged
- III. Role-Based LangGraph Workflow -> unchanged
- IV. Deterministic Quality Gates -> unchanged
- V. Observable, Recoverable Operation -> unchanged
- VI. Learning-Oriented Spec Kit Governance -> VI. 학습 중심 Spec Kit 거버넌스
Added sections:
- VI. 학습 중심 Spec Kit 거버넌스
- Spec Kit Stage Review requirements in Development Workflow
- Assumptions requirement for ambiguous requirements
Removed sections:
- None
Templates requiring updates:
- ✅ updated .specify/templates/plan-template.md
- ✅ updated .specify/templates/spec-template.md
- ✅ updated .specify/templates/tasks-template.md
- ✅ checked .specify/templates/commands/*.md (directory absent)
- ✅ updated AGENTS.md
- ✅ checked README.md, PROJECT_PLAN.md, CLAUDE.md
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

### VI. 학습 중심 Spec Kit 거버넌스
이 프로젝트의 Spec Kit 작업은 빠른 구현보다 설계 결정을 이해하는 것을
우선해야 한다. Spec Kit 산출물을 만들거나 수정하기 전에는 범위, 절충안,
가정, 예상 변경 파일을 포함해 어떤 판단을 할 것인지 먼저 설명해야 한다.
산출물을 만들거나 수정한 뒤에는 단계 목적, 새로 생긴 파일 또는 변경된
파일, 핵심 설계 결정, 가능한 대안, 현재 설계를 선택한 이유, 학습자가
반드시 읽어야 할 부분을 요약해야 한다.

애매한 요구사항은 조용히 확정하면 안 된다. 막히지 않는 불확실성은
Assumptions 섹션에 기록해야 하며, 막히는 불확실성은 산출물 생성 전에
질문으로 확인해야 한다. 구현 단계에서는 사용자가 명시한 task 범위만
실행해야 하며, 각 task 전에는 변경 대상 파일과 변경 이유를 설명하고,
각 task 후에는 실제 변경 내용과 테스트 결과를 요약해야 한다.

Rationale: 이 프로젝트는 Spec Kit이 요구사항을 결정, 계획, 작업 목록,
구현으로 바꾸는 과정을 학습하기 위한 실습 프로젝트다.

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

모든 Spec Kit 단계에서는 산출물을 작성하기 전에 짧은 판단 브리핑을 먼저
제시해야 하며, 작성 후에는 사후 리뷰를 제공해야 한다. 사후 리뷰에는
단계 목적, 새로 생긴 파일 또는 변경된 파일, 핵심 설계 결정, 검토한 대안,
현재 설계를 선택한 이유, 학습자가 반드시 읽어야 할 구체적인 섹션을
포함해야 한다. 해석 여지가 있는 요구사항은 Assumptions 섹션에 명시해야
한다.

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
review, and final implementation summary. `/speckit-implement` 작업은 사용자가
선택한 task 범위 안에서만 수행해야 하며, 기본값으로 전체 tasks.md를
실행하면 안 된다. 승인된 예외는 plan의 Complexity Tracking 섹션에 기록하고,
거절한 더 단순한 대안도 함께 적어야 한다.

**Version**: 1.1.0 | **Ratified**: 2026-06-17 | **Last Amended**: 2026-06-26
