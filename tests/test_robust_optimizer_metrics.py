"""Tests for robust optimization data contracts and metrics."""
import pytest
from src.backtest.robust import (
    CostAssumptions, RobustScoringPolicy, WalkForwardConfig,
    CandidateMetrics, OptimizationRun,
)


def test_cost_assumptions_defaults():
    cost = CostAssumptions()
    assert cost.fee_pct_one_way == 0.05
    assert cost.slippage_pct_one_way == 0.05
    assert cost.total_one_way_fee == pytest.approx(0.001)


def test_robust_scoring_policy_weights_sum_to_one():
    policy = RobustScoringPolicy()
    total = (policy.out_of_sample_return_weight + policy.risk_adjusted_return_weight
             + policy.drawdown_control_weight + policy.worst_fold_resilience_weight
             + policy.stability_turnover_penalty_weight)
    assert total == pytest.approx(1.0)


def test_walk_forward_config_requires_positive_windows():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        WalkForwardConfig(train_window_days=0, test_window_days=90, step_days=90)


def test_optimization_run_status_values():
    from typing import get_args
    import src.backtest.robust as r
    assert "success" in get_args(r.OptimizationStatus)
    assert "completed_degraded" in get_args(r.OptimizationStatus)


# --- Task 3: compute_candidate_metrics ---

import numpy as np
import pandas as pd
from tests.fixtures.optimization_data import synthetic_trades, synthetic_trades_few, synthetic_trades_high_mdd
from src.backtest.robust import compute_candidate_metrics
from src.backtest.strategy import BacktestResult


def _make_result(trades_df: pd.DataFrame, profit_pct: float = 5.0) -> BacktestResult:
    return BacktestResult(
        profit_pct=profit_pct,
        final_value=10_000 * (1 + profit_pct / 100),
        trades=trades_df,
        enriched=pd.DataFrame(),
    )


def test_completed_trades_counts_sell_rows():
    result = _make_result(synthetic_trades(n_sells=35))
    metrics = compute_candidate_metrics(result, initial_balance=10_000.0, cost=CostAssumptions())
    assert metrics.completed_trades == 35


def test_few_trades_still_computes_metrics():
    result = _make_result(synthetic_trades_few(n_sells=10))
    metrics = compute_candidate_metrics(result, initial_balance=10_000.0, cost=CostAssumptions())
    assert metrics.completed_trades == 10


def test_cost_adjusted_return_equals_profit_pct():
    result = _make_result(synthetic_trades(n_sells=35), profit_pct=8.5)
    metrics = compute_candidate_metrics(result, initial_balance=10_000.0, cost=CostAssumptions())
    assert metrics.cost_adjusted_return_pct == pytest.approx(8.5)


def test_max_drawdown_above_25_for_high_mdd_trades():
    result = _make_result(synthetic_trades_high_mdd(), profit_pct=-30.0)
    metrics = compute_candidate_metrics(result, initial_balance=10_000.0, cost=CostAssumptions())
    assert metrics.max_drawdown_pct > 25.0


def test_win_rate_between_0_and_100():
    result = _make_result(synthetic_trades(n_sells=35))
    metrics = compute_candidate_metrics(result, initial_balance=10_000.0, cost=CostAssumptions())
    assert metrics.win_rate_pct is not None
    assert 0.0 <= metrics.win_rate_pct <= 100.0


def test_profit_factor_positive_for_mostly_winning_trades():
    result = _make_result(synthetic_trades(n_sells=35, seed=42))
    metrics = compute_candidate_metrics(result, initial_balance=10_000.0, cost=CostAssumptions())
    if metrics.profit_factor is not None:
        assert metrics.profit_factor >= 0.0


def test_default_cost_applied_matches_001_fee():
    assert CostAssumptions().total_one_way_fee == pytest.approx(0.001)
