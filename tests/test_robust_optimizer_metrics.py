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
