"""Optuna Bayesian optimization for backtest strategy parameters.

Streamlit-independent port of the standalone project's inline Optuna loop. The
objective maximizes the historical-simulation return; this is parameter fitting
on past data and carries no forward-looking guarantee or investment advice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import optuna
import pandas as pd

from src.backtest.strategy import BacktestParams, run_backtest
from src.backtest.robust import (
    CandidateMetrics, CostAssumptions,
    compute_candidate_metrics, compute_train_trial_score,
    passes_train_trial_filter,
)

# Quiet Optuna's per-trial logging; callers drive their own progress reporting.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class SearchSpace:
    """Inclusive ranges for the Optuna parameter search."""

    kr_window: tuple[int, int] = (20, 100)
    kr_bandwidth: tuple[float, float] = (0.5, 10.0)
    bb_k: tuple[float, float] = (0.1, 2.0)
    rsi_period: tuple[int, int] = (7, 21)
    extrema_order: tuple[int, int] = (3, 10)
    rsi_oversold: tuple[int, int] = (20, 40)
    rsi_overbought: tuple[int, int] = (60, 80)


@dataclass
class OptimizationResult:
    """Outcome of an Optuna optimization run."""

    best_params: dict
    best_profit_pct: float
    n_trials: int
    completed_trials: int = 0
    notes: list[str] = field(default_factory=list)


def _suggest_params(trial: optuna.Trial, space: SearchSpace) -> dict:
    return {
        "kr_window": trial.suggest_int("kr_window", *space.kr_window),
        "kr_bandwidth": trial.suggest_float("kr_bandwidth", *space.kr_bandwidth),
        "bb_k": trial.suggest_float("bb_k", *space.bb_k),
        "rsi_period": trial.suggest_int("rsi_period", *space.rsi_period),
        "extrema_order": trial.suggest_int("extrema_order", *space.extrema_order),
        "rsi_oversold": trial.suggest_int("rsi_oversold", *space.rsi_oversold),
        "rsi_overbought": trial.suggest_int("rsi_overbought", *space.rsi_overbought),
    }


def optimize_backtest(
    df: pd.DataFrame,
    *,
    initial_balance: float,
    fee: float = 0.001,
    n_trials: int = 100,
    search_space: SearchSpace | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> OptimizationResult:
    """Search for the parameter set with the highest historical-simulation return.

    Args:
        df: OHLC price history with a ``Close`` column and DatetimeIndex.
        initial_balance: Starting capital for each simulated run.
        fee: Per-trade proportional fee.
        n_trials: Number of Optuna trials.
        search_space: Parameter ranges (defaults to ``SearchSpace()``).
        progress_callback: Optional ``(completed, total)`` callback per trial.

    Returns:
        An ``OptimizationResult`` with the best parameters and simulated return.
    """
    space = search_space or SearchSpace()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, space)
        result = run_backtest(df, BacktestParams.from_dict(params), initial_balance, fee)
        return result.profit_pct

    study = optuna.create_study(direction="maximize")
    completed = 0
    for _ in range(int(n_trials)):
        study.optimize(objective, n_trials=1)
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, int(n_trials))

    return OptimizationResult(
        best_params=dict(study.best_params),
        best_profit_pct=float(study.best_value),
        n_trials=int(n_trials),
        completed_trials=completed,
    )


def robust_optimize_window(
    df: pd.DataFrame,
    *,
    initial_balance: float,
    cost: CostAssumptions,
    n_trials: int = 30,
    search_space: SearchSpace | None = None,
) -> tuple[dict, CandidateMetrics]:
    """Run Optuna on `df` using train_trial_score (not total return). Return best params + metrics."""
    space = search_space or SearchSpace()

    best_params: dict = {}
    best_score = float("-inf")
    best_metrics: CandidateMetrics = CandidateMetrics()

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_params, best_score, best_metrics
        params = _suggest_params(trial, space)
        result = run_backtest(
            df, BacktestParams.from_dict(params), initial_balance, cost.total_one_way_fee
        )
        metrics = compute_candidate_metrics(result, initial_balance, cost)
        if not passes_train_trial_filter(metrics):
            return float("-inf")
        score = compute_train_trial_score(metrics)
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials))
    return best_params, best_metrics
