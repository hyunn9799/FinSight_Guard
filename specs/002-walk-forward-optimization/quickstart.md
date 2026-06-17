# Quickstart: Walk-Forward Strategy Optimization Validation

## Prerequisites

- Python dependencies installed from `requirements.txt`.
- No live API keys are required for validation.
- Tests use deterministic synthetic data or monkeypatched loaders.
- For full infrastructure validation, provide local/dev services for PostgreSQL, Pinecone-compatible vector indexing or a fake client, Neo4j, OpenSearch, and Redis. Unit tests should still use fakes or monkeypatches by default.

## Static Validation

```bash
python3 -m compileall src
```

Expected outcome: all modules under `src` compile successfully.

## Focused Test Runs

```bash
pytest tests/test_backtest_charts_optimizer.py tests/test_backtest_agent.py tests/test_api.py
```

Expected outcome: existing backtest, agent, and API tests pass after compatibility-preserving changes.

## New Test Scenarios To Add During Implementation

1. Candidate with highest return but fewer than 30 completed trades is not labeled robust.
2. Candidate with maximum drawdown above 25% is not labeled robust.
3. Walk-forward output with fewer than 3 valid test folds returns insufficient-data status.
4. Every valid test fold starts after its train fold.
5. Robust score uses the risk-balanced weights from the spec.
6. Cost-adjusted returns reflect 0.05% one-way fee and 0.05% one-way slippage by default.
7. Regime performance with fewer than 10 trades or fewer than 60 trading days is marked low-confidence.
8. Optimization evidence items are generated for numeric report claims.
9. Final report wording keeps historical-simulation framing and avoids direct trading advice.
10. API request validation rejects malformed dates, out-of-range parameters, and invalid cost assumptions.
11. PostgreSQL persistence is the canonical write path for optimization runs and reports.
12. Pinecone, Neo4j, and OpenSearch indexing failures produce degraded/index-pending warnings without losing PostgreSQL records.
13. Redis cache/queue/session/rate-limit behavior is treated as ephemeral and can be rebuilt from PostgreSQL state.

## Manual API Smoke Test

Start the API server:

```bash
uvicorn main:app --reload
```

Submit a robust optimization request:

```bash
curl -X POST http://127.0.0.1:8000/backtest/optimize \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker": "AAPL",
    "investment_horizon": "중기",
    "risk_profile": "중립형",
    "start": "2022-01-01",
    "end": "2025-12-31",
    "initial_balance": 10000,
    "n_trials": 10,
    "walk_forward": {
      "train_window_days": 360,
      "test_window_days": 90,
      "step_days": 90
    },
    "costs": {
      "fee_pct_one_way": 0.05,
      "slippage_pct_one_way": 0.05
    }
  }'
```

Expected outcome:

- Response includes `status`, `optimization`, `warnings`, and `evidence`.
- If fewer than 3 valid test folds exist, response is limitation-only and does not expose a robust candidate.
- If a robust candidate exists, score components and fold summaries are visible.
- Text frames the result as historical simulation research, not a trading recommendation.

## Streamlit Smoke Test

```bash
streamlit run app.py
```

Expected outcome:

- UI exposes robust optimization controls only as historical simulation research.
- Result view shows score components, fold breakdown, regime summary, manual baseline, passive baseline, warnings, and evidence where available.
- Required no-recommendation framing remains visible in report output.

## Infrastructure Smoke Tests

These are planning targets for implementation tasks, not required before code exists.

```bash
pytest tests/test_storage_postgres.py tests/test_indexes.py tests/test_redis_runtime.py
```

Expected outcome:

- PostgreSQL repository tests persist users, tickers, analysis requests, reports, analysis results, settings, notifications, portfolios, evidence, and robust optimization records.
- Pinecone, Neo4j, and OpenSearch adapters are tested with local fakes or monkeypatched clients.
- Redis queue/cache/rate-limit/session tests verify no durable report/result data depends only on Redis.
