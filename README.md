# Evidence-Grounded Financial Research Multi-Agent Workflow

Portfolio-grade LangGraph workflow for evidence-grounded financial research.
The system compares market, fundamental, and news evidence through role-based
agents, then validates the Korean research report with an Evaluator Agent.

This project is not a stock recommendation system and does not provide buy,
sell, hold, profit guarantee, or trading advice.

## Architecture

```text
User Input
  -> Input Validator
  -> Market Agent
      - yfinance price history
      - MA20, MA60, MA120, RSI, MACD, ATR
      - EvidenceItem objects
  -> Fundamental Agent
      - yfinance company info and financial metrics
      - missing-field handling
      - EvidenceItem objects
  -> News Agent
      - Tavily or Firecrawl when configured
      - mock fallback when no API key exists
      - EvidenceItem objects
  -> Coordinator Agent
      - Korean scenario-based report
      - evidence, risks, limitations, disclaimer
  -> Evaluator Agent
      - source grounding, numeric consistency, safety, risk, freshness
      - PASS -> Save Report
      - FAIL -> Rewrite Agent -> Evaluator Agent
```

## Local Setup

Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional news search providers can be configured in `.env`. If no provider key
is set, the workflow continues with the mock news fallback.

## Streamlit Run

```bash
streamlit run app.py
```

The Streamlit UI runs on `http://localhost:8501` by default.

## FastAPI Run

```bash
uvicorn main:app --reload
```

Useful endpoints:

```text
GET  /health
GET  /metrics
POST /analyze
GET  /reports/{run_id}
```

Example API request:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","investment_horizon":"중기","risk_profile":"중립형"}'
```

## Docker Run

Build and run the FastAPI service with Docker Compose:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`.

To run the Streamlit UI from the same image:

```bash
docker build -t finsight-guard .
docker run --rm -p 8501:8501 --env-file .env finsight-guard \
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Tests

```bash
python -m compileall src
pytest
```

Tests are deterministic and should not depend on live external APIs.

## Safety Disclaimer

본 보고서는 교육 및 정보 제공 목적의 AI 리서치 결과이며, 특정 종목의 매수·매도·보유를 권유하지 않습니다. 최종 투자 판단과 책임은 투자자 본인에게 있습니다.
