# Quickstart: Walk-Forward Strategy Optimization Validation

## Prerequisites

- Python dependencies installed from `requirements.txt`.
- No live API keys are required for validation.
- Tests use deterministic synthetic data or monkeypatched loaders.
- No database, cache, vector index, graph database, or search engine is required for this MVP validation.

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

## Validated Test Scenarios

All of the following now have deterministic pytest coverage:

1. ✅ Candidate with highest return but fewer than 30 completed trades is not labeled robust.
2. ✅ Candidate with maximum drawdown above 25% is not labeled robust.
3. ✅ Walk-forward output with fewer than 3 valid test folds returns `insufficient_data`.
4. ✅ Every valid test fold starts after its train fold (leakage prevention).
5. ✅ Robust score uses the 5-component risk-balanced weights from the spec.
6. ✅ Cost-adjusted returns reflect 0.05% one-way fee + 0.05% slippage.
7. ✅ Regime results with < 10 trades or < 60 trading days marked `low_confidence`.
8. ✅ Optimization evidence items generated for all numeric report claims.
9. ✅ Safety checker rejects forbidden phrases and strong expressions on guardrail failure.
10. ✅ API validates dates, n_trials ≤ 50, positive cost assumptions.
11. ✅ Manual and passive baselines use same periods and cost assumptions.
12. ✅ train_trial_filter rejects candidates before OOS evaluation (separate from OOS guardrail).
13. ✅ completed_degraded status + evaluator_errors returned when rewrite limit exceeded.

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

## MVP Storage and Workflow Smoke Tests

These are planning targets for implementation tasks, not required before code exists.

```bash
pytest tests/test_report_store.py tests/test_workflow_routing.py tests/test_optimization_evidence.py
```

Expected outcome:

- Local report-store tests persist optimization-containing reports and evaluation results.
- Workflow routing tests cover validation failure, data-load failure, evaluator failure, and max rewrite attempts.
- Evidence tests prove numeric optimization claims map to `EvidenceItem` records.
