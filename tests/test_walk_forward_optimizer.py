"""Tests for walk-forward fold generation and orchestration."""
import pytest
import pandas as pd
from src.backtest.robust import WalkForwardConfig, generate_fold_windows


def test_fold_windows_test_starts_after_train_end():
    config = WalkForwardConfig(train_window_days=180, test_window_days=60, step_days=60)
    folds = generate_fold_windows("2022-01-01", "2024-12-31", config)
    for fold in folds:
        assert pd.Timestamp(fold["test_start"]) > pd.Timestamp(fold["train_end"])


def test_fold_windows_are_time_ordered():
    config = WalkForwardConfig(train_window_days=180, test_window_days=60, step_days=60)
    folds = generate_fold_windows("2022-01-01", "2024-12-31", config)
    for i in range(1, len(folds)):
        assert pd.Timestamp(folds[i]["train_start"]) >= pd.Timestamp(folds[i-1]["train_start"])


def test_fold_windows_produces_at_least_3_folds_with_enough_data():
    config = WalkForwardConfig(train_window_days=180, test_window_days=60, step_days=60)
    folds = generate_fold_windows("2020-01-01", "2024-12-31", config)
    assert len(folds) >= 3


def test_fold_windows_empty_when_data_too_short():
    config = WalkForwardConfig(train_window_days=360, test_window_days=90, step_days=90)
    folds = generate_fold_windows("2022-01-01", "2022-06-01", config)
    assert len(folds) == 0


def test_fold_windows_no_test_data_leakage():
    """No train window may overlap with its own test window."""
    config = WalkForwardConfig(train_window_days=180, test_window_days=60, step_days=60)
    folds = generate_fold_windows("2021-01-01", "2024-12-31", config)
    for fold in folds:
        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        assert test_start > train_end, (
            f"Fold {fold['fold_index']}: test_start={test_start} must be after train_end={train_end}"
        )


def test_fold_index_starts_at_1():
    config = WalkForwardConfig(train_window_days=180, test_window_days=60, step_days=60)
    folds = generate_fold_windows("2021-01-01", "2024-12-31", config)
    assert folds[0]["fold_index"] == 1
    assert folds[-1]["fold_index"] == len(folds)


# --- Task 7: orchestration ---

from tests.fixtures.optimization_data import synthetic_prices
from src.backtest.robust import (
    CostAssumptions, RobustScoringPolicy,
    run_walk_forward_optimization,
)


def _fast_config() -> WalkForwardConfig:
    return WalkForwardConfig(train_window_days=120, test_window_days=40, step_days=40)


def test_insufficient_folds_returns_insufficient_data_status():
    df = synthetic_prices(n=100)  # too short for 3 folds
    result = run_walk_forward_optimization(
        df=df, ticker="TEST", run_id="r1",
        initial_balance=10_000, config=_fast_config(),
        cost=CostAssumptions(), policy=RobustScoringPolicy(), n_trials=2,
    )
    assert result.status == "insufficient_data"
    assert result.robust_candidate is None


def test_valid_run_produces_at_least_3_folds():
    df = synthetic_prices(n=800)
    result = run_walk_forward_optimization(
        df=df, ticker="TEST", run_id="r2",
        initial_balance=10_000, config=_fast_config(),
        cost=CostAssumptions(), policy=RobustScoringPolicy(), n_trials=3,
    )
    # The orchestration must process every generated fold window and append a
    # WalkForwardFold for each. With weak synthetic data + few optuna trials the
    # in-sample filter may prune all candidates (status "invalid"), so we assert
    # on the number of folds processed rather than their data-dependent status.
    assert len(result.folds) >= 3


def test_fold_aggregate_includes_median_worst_stddev():
    df = synthetic_prices(n=800)
    result = run_walk_forward_optimization(
        df=df, ticker="TEST", run_id="r3",
        initial_balance=10_000, config=_fast_config(),
        cost=CostAssumptions(), policy=RobustScoringPolicy(), n_trials=3,
    )
    if result.robust_candidate:
        m = result.robust_candidate.metrics
        assert m.median_oos_return_pct is not None
        assert m.worst_fold_return_pct is not None
        assert m.fold_return_stddev is not None


def test_warnings_list_non_empty_on_insufficient_data():
    df = synthetic_prices(n=100)
    result = run_walk_forward_optimization(
        df=df, ticker="TEST", run_id="r4",
        initial_balance=10_000, config=_fast_config(),
        cost=CostAssumptions(), policy=RobustScoringPolicy(), n_trials=2,
    )
    assert any("fold" in w.lower() for w in result.warnings)


def test_empty_dataframe_fails_gracefully():
    df = synthetic_prices(n=800).iloc[0:0]  # empty frame, keeps columns
    result = run_walk_forward_optimization(
        df=df, ticker="TEST", run_id="r5",
        initial_balance=10_000, config=_fast_config(),
        cost=CostAssumptions(), policy=RobustScoringPolicy(), n_trials=2,
    )
    assert result.status == "failed"
    assert result.robust_candidate is None
    assert any("empty" in w.lower() for w in result.warnings)


def test_pruned_trials_do_not_fall_back_to_default_params():
    """When robust_optimize_window returns no params, the fold is invalid,
    not silently backtested with default parameters."""
    import src.backtest.optimizer as optimizer_mod

    df = synthetic_prices(n=800)
    original = optimizer_mod.robust_optimize_window
    optimizer_mod.robust_optimize_window = lambda *a, **k: ({}, None)
    try:
        result = run_walk_forward_optimization(
            df=df, ticker="TEST", run_id="r6",
            initial_balance=10_000, config=_fast_config(),
            cost=CostAssumptions(), policy=RobustScoringPolicy(), n_trials=2,
        )
    finally:
        optimizer_mod.robust_optimize_window = original

    # All folds had no surviving in-sample candidate → none valid, no robust label
    assert result.robust_candidate is None
    assert all(f.status != "valid" for f in result.folds)
    invalid = [f for f in result.folds if f.status == "invalid"]
    assert invalid and any("in-sample filter" in w for f in invalid for w in f.warnings)
