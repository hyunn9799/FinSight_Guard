# AGENTS.md

## Project Identity

This project is a portfolio-grade LangGraph-based financial research multi-agent workflow.

It is NOT a stock recommendation system.
It is NOT an automated trading system.
It is NOT allowed to guarantee profit, predict returns with certainty, or tell users to buy/sell a stock.

The system is an evidence-based financial research assistant that helps users compare scenarios using market data, financial data, and news evidence.

## Core Portfolio Goal

The final project must demonstrate:

1. LangGraph workflow design
2. Role-based agents
3. Tool/API integration
4. Evidence-grounded report generation
5. Evaluator Agent for responsible AI review
6. Conditional branching and retry logic
7. Failure handling and fallback behavior
8. Safety filters for financial advice
9. Tests, logs, report storage, Streamlit UI
10. FastAPI, Docker, health check, and basic monitoring endpoint

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

### 4. Coordinator Agent
- Combines Market/Fundamental/News analysis.
- Generates a Korean research report.
- Must use scenario-based wording:
  - 관망 시나리오
  - 분할 접근 시나리오
  - 리스크 회피 시나리오
- Must not directly recommend buy/sell/hold.
- Must include evidence summary, risks, limitations, data date, and disclaimer.

### 5. Evaluator Agent
- Reviews the draft report.
- Checks:
  - source grounding
  - numeric consistency
  - excessive investment recommendation language
  - risk disclosure
  - limitations
  - data freshness
  - disclaimer
- Returns pass/fail and scores.
- If evaluation fails, workflow must route to Rewrite Agent.

### 6. Rewrite Agent
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

4. If Evaluator passes:
   - save final report.

5. If Evaluator fails:
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

## Spec Kit 학습 규칙

이 프로젝트는 구현 속도를 높이는 것보다 Spec Kit의 설계 결정을 공부하는
것을 우선한다.

모든 Spec Kit 단계에서 다음 규칙을 지켜야 한다.
1. 요청된 산출물 단계가 허용하기 전에는 구현 코드를 수정하지 않는다.
2. 산출물을 만들거나 수정하기 전에 어떤 판단을 할지 먼저 설명한다.
3. 산출물을 만들거나 수정한 뒤에는 다음 항목을 요약한다.
   - 이번 단계의 목적
   - 새로 생긴 파일 또는 변경된 파일
   - 핵심 설계 결정
   - 검토한 대안
   - 현재 설계를 선택한 이유
   - 사용자가 반드시 읽어야 할 부분
4. 애매한 요구사항은 임의로 확정하지 말고 Assumptions 섹션에 기록한다.
5. 구현 단계에서는 tasks.md의 모든 작업을 기본값으로 실행하지 않는다.
   사용자가 명시한 task 범위만 실행한다.
6. 각 구현 task 전에는 변경 대상 파일과 변경 이유를 설명한다.
7. 각 구현 task 후에는 실제 변경 내용과 테스트 결과를 요약한다.

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
at specs/004-postgresql-table-schema/plan.md
<!-- SPECKIT END -->
