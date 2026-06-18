"""Robust walk-forward optimization — models, metrics, scoring, orchestration."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional
from pydantic import BaseModel, Field, model_validator

import numpy as np
import pandas as pd

OptimizationStatus = Literal[
    "success", "degraded", "insufficient_data", "failed", "completed_degraded"
]
FoldStatus = Literal["valid", "no_trades", "missing_data", "invalid"]
RegimeType = Literal["bull", "bear", "sideways", "high_volatility", "low_volatility"]
ConfidenceLevel = Literal["normal", "low"]


class CostAssumptions(BaseModel):
    fee_pct_one_way: float = Field(default=0.05, ge=0.0)
    slippage_pct_one_way: float = Field(default=0.05, ge=0.0)
    user_adjustable: bool = True

    @property
    def total_one_way_fee(self) -> float:
        return (self.fee_pct_one_way + self.slippage_pct_one_way) / 100.0


class RobustScoringPolicy(BaseModel):
    out_of_sample_return_weight: float = 0.30
    risk_adjusted_return_weight: float = 0.25
    drawdown_control_weight: float = 0.20
    worst_fold_resilience_weight: float = 0.15
    stability_turnover_penalty_weight: float = 0.10

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "RobustScoringPolicy":
        total = (self.out_of_sample_return_weight + self.risk_adjusted_return_weight
                 + self.drawdown_control_weight + self.worst_fold_resilience_weight
                 + self.stability_turnover_penalty_weight)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring policy weights must sum to 1.0, got {total}")
        return self


class WalkForwardConfig(BaseModel):
    train_window_days: int = Field(gt=0)
    test_window_days: int = Field(gt=0)
    step_days: int = Field(gt=0)
    minimum_valid_test_folds: int = 3


class CandidateMetrics(BaseModel):
    total_return_pct: float = 0.0
    cost_adjusted_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    win_rate_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    completed_trades: int = 0
    average_holding_days: Optional[float] = None
    turnover: Optional[float] = None
    median_oos_return_pct: Optional[float] = None
    worst_fold_return_pct: Optional[float] = None
    fold_return_stddev: Optional[float] = None


class WalkForwardFold(BaseModel):
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_params: dict
    candidate_metrics: CandidateMetrics = Field(default_factory=CandidateMetrics)
    warnings: list[str] = Field(default_factory=list)
    status: FoldStatus = "valid"


class ParameterCandidateResult(BaseModel):
    params: dict
    score: float = 0.0
    score_components: dict = Field(default_factory=dict)
    metrics: CandidateMetrics = Field(default_factory=CandidateMetrics)
    fold_metrics: list[CandidateMetrics] = Field(default_factory=list)
    robust_label_allowed: bool = False
    warnings: list[str] = Field(default_factory=list)


class BaselineResult(BaseModel):
    baseline_type: Literal["manual_parameters", "passive_buy_and_hold"]
    metrics: CandidateMetrics = Field(default_factory=CandidateMetrics)
    warnings: list[str] = Field(default_factory=list)


class OptimizationRun(BaseModel):
    run_id: str
    ticker: str
    start: str
    end: str
    initial_balance: float
    cost_assumptions: CostAssumptions = Field(default_factory=CostAssumptions)
    scoring_policy: RobustScoringPolicy = Field(default_factory=RobustScoringPolicy)
    fold_setup: WalkForwardConfig = Field(
        default_factory=lambda: WalkForwardConfig(
            train_window_days=360, test_window_days=90, step_days=90
        )
    )
    status: OptimizationStatus = "failed"
    manual_baseline: Optional[BaselineResult] = None
    passive_baseline: Optional[BaselineResult] = None
    robust_candidate: Optional[ParameterCandidateResult] = None
    folds: list[WalkForwardFold] = Field(default_factory=list)
    regime_summary: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    report_path: Optional[str] = None
    evaluator_errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _build_equity_curve(trades_df: pd.DataFrame, initial_balance: float) -> list[float]:
    if trades_df.empty:
        return [initial_balance]
    sell_rows = trades_df[trades_df["type"] == "Sell"]
    return [initial_balance] + sell_rows["balance"].tolist()


def _compute_max_drawdown(equity: list[float]) -> float:
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (peak - val) / peak * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


def _compute_sharpe_sortino(
    sell_trades: pd.DataFrame, initial_balance: float
) -> tuple[Optional[float], Optional[float]]:
    if sell_trades.empty or len(sell_trades) < 2:
        return None, None
    profits = sell_trades["profit"].to_numpy(dtype=float)
    returns = profits / initial_balance
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    if std_r == 0:
        return None, None
    sharpe = mean_r / std_r
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) >= 2 else std_r
    sortino = mean_r / downside_std if downside_std > 0 else None
    return float(sharpe), sortino


def _compute_average_holding_days(trades_df: pd.DataFrame) -> Optional[float]:
    if trades_df.empty:
        return None
    buys = trades_df[trades_df["type"] == "Buy"].reset_index(drop=True)
    sells = trades_df[trades_df["type"] == "Sell"].reset_index(drop=True)
    n = min(len(buys), len(sells))
    if n == 0:
        return None
    days = [
        (pd.Timestamp(sells.loc[i, "date"]) - pd.Timestamp(buys.loc[i, "date"])).days
        for i in range(n)
    ]
    return float(np.mean(days)) if days else None


def compute_candidate_metrics(
    result: "BacktestResult",
    initial_balance: float,
    cost: CostAssumptions,
) -> CandidateMetrics:
    """Compute all CandidateMetrics from a BacktestResult."""
    from src.backtest.strategy import BacktestResult  # noqa: F401 — type guard

    trades_df = result.trades if result.trades is not None else pd.DataFrame()
    sell_trades = trades_df[trades_df["type"] == "Sell"] if not trades_df.empty else pd.DataFrame()
    completed_trades = len(sell_trades)

    equity = _build_equity_curve(trades_df, initial_balance)
    max_dd = _compute_max_drawdown(equity)

    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    if completed_trades > 0:
        profits = sell_trades["profit"].to_numpy(dtype=float)
        win_rate = float((profits > 0).mean() * 100.0)
        gross_profit = float(profits[profits > 0].sum())
        gross_loss = float(abs(profits[profits < 0].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    sharpe, sortino = _compute_sharpe_sortino(sell_trades, initial_balance)
    avg_holding = _compute_average_holding_days(trades_df)

    total_days = max(1, (len(equity) - 1) * 10)
    turnover: Optional[float] = (completed_trades / total_days * 252.0) if completed_trades > 0 else None

    return CandidateMetrics(
        total_return_pct=result.profit_pct,
        cost_adjusted_return_pct=result.profit_pct,  # fee already applied in run_backtest
        max_drawdown_pct=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        completed_trades=completed_trades,
        average_holding_days=avg_holding,
        turnover=turnover,
    )
