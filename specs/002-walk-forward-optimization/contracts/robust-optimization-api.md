# Contract: Robust Optimization API

## Endpoint

`POST /backtest/optimize`

Runs robust walk-forward optimization for historical simulation research. The endpoint must not place orders, connect to brokerage APIs, or describe returned parameters as trading recommendations.

## Request

```json
{
  "ticker": "AAPL",
  "investment_horizon": "중기",
  "risk_profile": "중립형",
  "start": "2022-01-01",
  "end": "2025-12-31",
  "initial_balance": 10000,
  "n_trials": 30,
  "walk_forward": {
    "train_window_days": 360,
    "test_window_days": 90,
    "step_days": 90
  },
  "costs": {
    "fee_pct_one_way": 0.05,
    "slippage_pct_one_way": 0.05
  },
  "manual_params": {
    "rsi_period": 14,
    "kr_window": 30,
    "kr_bandwidth": 5.0,
    "bb_k": 2.0,
    "extrema_order": 5,
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0
  }
}
```

## Request Validation

- `ticker` follows the existing ticker pattern and length bounds.
- `start` and `end` are valid ISO dates; `start` is before `end`.
- Date range remains bounded by existing server limits or a similarly explicit optimization-specific cap.
- `initial_balance` is greater than 0.
- `n_trials` is bounded for synchronous demo use.
- Fee and slippage are greater than or equal to 0.
- RSI oversold must be below RSI overbought.
- Train/test/step windows are positive integers.

## Success Response

```json
{
  "run_id": "wf123",
  "status": "success",
  "optimization": {
    "optimization_run_id": "2d6e3d3f-5b2f-4ef5-a5b5-0b9fd0d5f7d2",
    "analysis_request_id": "9cfda05d-72c5-40c5-a764-66b3df195781",
    "ticker": "AAPL",
    "robust_candidate": {
      "params": {
        "rsi_period": 14,
        "kr_window": 30,
        "kr_bandwidth": 5.0,
        "bb_k": 2.0,
        "extrema_order": 5,
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0
      },
      "score": 0.73,
      "score_components": {
        "out_of_sample_return": 0.30,
        "risk_adjusted_return": 0.25,
        "drawdown_control": 0.20,
        "worst_fold_resilience": 0.15,
        "stability_turnover_penalty": 0.10
      },
      "metrics": {
        "cost_adjusted_return_pct": 8.4,
        "max_drawdown_pct": 18.2,
        "sharpe": 0.91,
        "sortino": 1.15,
        "win_rate_pct": 54.0,
        "profit_factor": 1.31,
        "completed_trades": 42,
        "average_holding_days": 8.5,
        "median_oos_return_pct": 2.1,
        "worst_fold_return_pct": -3.4,
        "fold_return_stddev": 2.8
      },
      "robust_label_allowed": true,
      "warnings": []
    },
    "folds": [],
    "regime_summary": [],
    "manual_baseline": {},
    "passive_baseline": {},
    "warnings": [
      "결과는 과거 시뮬레이션이며 매수·매도·보유 권유가 아닙니다."
    ],
    "evidence": [],
    "storage": {
      "source_of_truth": "postgresql",
      "semantic_index": "pinecone",
      "graph_index": "neo4j",
      "keyword_index": "opensearch",
      "runtime_cache_queue": "redis"
    }
  },
  "final_report": {},
  "evaluation_result": {},
  "report_path": "reports/wf123_AAPL_20260617T000000Z.json",
  "errors": []
}
```

## Insufficient Data Response

```json
{
  "run_id": "wf124",
  "status": "insufficient_data",
  "optimization": {
    "ticker": "AAPL",
    "robust_candidate": null,
    "folds": [],
    "warnings": [
      "Walk-forward validation produced fewer than 3 valid test folds.",
      "결과를 robust parameter로 표시하지 않습니다."
    ],
    "evidence": []
  },
  "final_report": null,
  "evaluation_result": null,
  "report_path": null,
  "errors": []
}
```

## Safety Contract

- Returned parameter sets are `historical simulation candidates`.
- Response text must not include buy/sell/hold instructions, guaranteed targets, guaranteed returns, or automatic execution wording.
- Any final Korean report containing optimization metrics must include the constitution disclaimer exactly.
- Numeric report claims must be backed by evidence items.

## Persistence Contract

- PostgreSQL write succeeds before the response is considered durable.
- Pinecone, Neo4j, and OpenSearch indexing may be asynchronous and must report degraded/index-pending status instead of blocking core report persistence indefinitely.
- Redis may hold transient run status for polling, but PostgreSQL stores the durable final status.
- Search/index IDs must resolve back to PostgreSQL `analysis_requests`, `reports`, `analysis_results`, `evidence_items`, or `robust_optimization_runs`.
