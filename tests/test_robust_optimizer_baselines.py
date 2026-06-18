"""Tests: baseline comparisons use same evaluation periods and cost assumptions."""
import pytest
from tests.fixtures.optimization_data import synthetic_prices
from src.backtest.robust import CostAssumptions, compute_baselines


def test_baselines_return_both_types():
    df = synthetic_prices(n=300)
    manual, passive = compute_baselines(
        df=df, start="2022-01-01", end="2022-12-31",
        initial_balance=10_000, cost=CostAssumptions(),
        manual_params={"rsi_period": 14, "kr_window": 30, "kr_bandwidth": 5.0,
                       "bb_k": 2.0, "extrema_order": 5, "rsi_oversold": 30.0, "rsi_overbought": 70.0},
    )
    assert manual.baseline_type == "manual_parameters"
    assert passive.baseline_type == "passive_buy_and_hold"


def test_passive_baseline_cost_adjusted_return_equals_buy_hold():
    df = synthetic_prices(n=300)
    _, passive = compute_baselines(
        df=df, start="2022-01-01", end="2022-12-31",
        initial_balance=10_000, cost=CostAssumptions(), manual_params=None,
    )
    window_df = df.loc[(df.index >= "2022-01-01") & (df.index <= "2022-12-31")]
    first_close = float(window_df["Close"].iloc[0])
    last_close = float(window_df["Close"].iloc[-1])
    expected_pct = (last_close - first_close) / first_close * 100.0
    assert abs(passive.metrics.cost_adjusted_return_pct - expected_pct) < 1.0


def test_baselines_have_evidence_fields_available():
    df = synthetic_prices(n=300)
    manual, passive = compute_baselines(
        df=df, start="2022-01-01", end="2022-12-31",
        initial_balance=10_000, cost=CostAssumptions(), manual_params=None,
    )
    assert isinstance(manual.metrics.cost_adjusted_return_pct, float)
    assert isinstance(passive.metrics.cost_adjusted_return_pct, float)
