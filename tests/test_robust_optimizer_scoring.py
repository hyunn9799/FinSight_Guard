"""Tests for robust scoring: train filter, train score, final score, guardrails."""
import pytest
from src.backtest.robust import (
    CandidateMetrics, RobustScoringPolicy,
    passes_train_trial_filter, compute_train_trial_score,
    compute_final_robust_score, compute_robust_label_allowed,
)


def _metrics(**kwargs) -> CandidateMetrics:
    defaults = dict(
        total_return_pct=10.0, cost_adjusted_return_pct=10.0,
        max_drawdown_pct=15.0, sharpe=1.0, sortino=1.2,
        win_rate_pct=55.0, profit_factor=1.4, completed_trades=35,
        average_holding_days=8.0, turnover=1.5,
    )
    defaults.update(kwargs)
    return CandidateMetrics(**defaults)


# --- train_trial_filter ---

def test_passes_filter_with_acceptable_metrics():
    assert passes_train_trial_filter(_metrics()) is True


def test_fails_filter_when_mdd_exceeds_25():
    assert passes_train_trial_filter(_metrics(max_drawdown_pct=26.0)) is False


def test_fails_filter_when_trades_below_30():
    assert passes_train_trial_filter(_metrics(completed_trades=29)) is False


def test_fails_filter_when_both_mdd_and_trades_bad():
    assert passes_train_trial_filter(_metrics(max_drawdown_pct=30.0, completed_trades=5)) is False


# --- train_trial_score ---

def test_train_trial_score_returns_float():
    score = compute_train_trial_score(_metrics())
    assert isinstance(score, float)


def test_train_trial_score_higher_sharpe_gives_better_score():
    low_sharpe = compute_train_trial_score(_metrics(sharpe=0.2))
    high_sharpe = compute_train_trial_score(_metrics(sharpe=2.0))
    assert high_sharpe > low_sharpe


def test_train_trial_score_lower_mdd_gives_better_score():
    high_mdd = compute_train_trial_score(_metrics(max_drawdown_pct=24.0))
    low_mdd = compute_train_trial_score(_metrics(max_drawdown_pct=5.0))
    assert low_mdd > high_mdd


# --- final_robust_score ---

def test_final_robust_score_uses_policy_weights():
    fold_metrics = [_metrics(cost_adjusted_return_pct=5.0)] * 4
    score, components = compute_final_robust_score(fold_metrics, RobustScoringPolicy())
    assert isinstance(score, float)
    assert set(components.keys()) == {
        "out_of_sample_return", "risk_adjusted_return",
        "drawdown_control", "worst_fold_resilience", "stability_turnover_penalty",
    }


def test_final_robust_score_higher_for_better_oos():
    weak = [_metrics(cost_adjusted_return_pct=-2.0, sharpe=0.1)] * 4
    strong = [_metrics(cost_adjusted_return_pct=8.0, sharpe=1.5)] * 4
    score_weak, _ = compute_final_robust_score(weak, RobustScoringPolicy())
    score_strong, _ = compute_final_robust_score(strong, RobustScoringPolicy())
    assert score_strong > score_weak


def test_final_score_penalizes_high_stddev():
    stable = [_metrics(cost_adjusted_return_pct=3.0)] * 5
    volatile_folds = [_metrics(cost_adjusted_return_pct=v) for v in [10.0, -5.0, 8.0, -4.0, 9.0]]
    score_stable, _ = compute_final_robust_score(stable, RobustScoringPolicy())
    score_volatile, _ = compute_final_robust_score(volatile_folds, RobustScoringPolicy())
    assert score_stable > score_volatile


# --- robust_label_allowed ---

def test_robust_label_allowed_when_guardrails_pass():
    assert compute_robust_label_allowed(_metrics()) is True


def test_robust_label_not_allowed_when_few_trades():
    assert compute_robust_label_allowed(_metrics(completed_trades=29)) is False


def test_robust_label_not_allowed_when_mdd_exceeds_25():
    assert compute_robust_label_allowed(_metrics(max_drawdown_pct=25.1)) is False


def test_high_return_alone_does_not_grant_robust_label():
    m = _metrics(cost_adjusted_return_pct=200.0, max_drawdown_pct=30.0, completed_trades=35)
    assert compute_robust_label_allowed(m) is False
