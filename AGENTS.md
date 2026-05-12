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