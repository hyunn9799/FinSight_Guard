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
