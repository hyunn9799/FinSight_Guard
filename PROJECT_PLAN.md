# PROJECT_PLAN.md

## 1. Project Name

Evidence-Grounded Financial Research Multi-Agent Workflow

## 2. Goal

Build a LangGraph-based financial research assistant that collects market, fundamental, and news data, converts them into structured evidence, generates a scenario-based research report, and validates the result through an Evaluator Agent.

This project is designed as a portfolio-grade practical AI Agent workflow.

## 3. Product Positioning

This system is not a stock recommendation engine.
It is a responsible AI research assistant that supports decision-making by organizing evidence, risks, and scenarios.

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
Evidence Builder
  ↓
Coordinator Agent
  ↓
Draft Report
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
*   **Analysis Logic**
    *   Technical indicators
    *   EvidenceItem tracking
*   **LangGraph Workflow**
    *   LangGraph workflow design
    *   Conditional branching
    *   Retry logic
    *   Evaluator Agent
    *   Rewrite Agent
*   **Interface & Delivery**
    *   Streamlit UI
    *   FastAPI API
*   **Infrastructure & DevOps**
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

---

## 6. Definition of Done (DoD)

The project is considered complete when all the following conditions are met:

*   **Operational Success**
    *   [ ] `streamlit run app.py` works as expected.
    *   [ ] `uvicorn main:app --reload` runs the API server successfully.
    *   [ ] `pytest` passes all test cases.
*   **Storage & Logging**
    *   [ ] Reports are successfully saved under the `/reports` directory.
    *   [ ] Logs are structured and saved under the `/logs` directory.
*   **Agent Quality & Reliability**
    *   [ ] **Evaluator Agent** correctly identifies and fails unsafe or low-quality reports.
    *   [ ] **Workflow/Rewrite Agent** successfully rewrites failed reports.
*   **Deployment & Documentation**
    *   [ ] Docker container builds and starts successfully.
    *   [ ] `README.md` clearly explains the architecture, safety design, and limitations.
    *   [ ] Demo scenario works for at least one ticker symbol.


---

<!-- # 7. Codex 프롬프트 1 — 전체 설계 업그레이드

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