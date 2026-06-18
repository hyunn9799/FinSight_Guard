"""Robust walk-forward optimization — models, metrics, scoring, orchestration."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

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
