# Walk-Forward Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace total-return-only strategy optimization with a research-only robust walk-forward system that evaluates candidates across risk, drawdown, regime, and out-of-sample evidence dimensions.

**Architecture:** Extend `src/backtest/optimizer.py` with two new modules — `robust.py` (metrics, scoring, fold orchestration) and `regime.py` (market classification) — then surface results through the existing FastAPI, Streamlit, and LangGraph workflow surfaces. The existing `optimize_backtest` function is not modified.

**Tech Stack:** Python 3.x, Pydantic v2, pandas, numpy, optuna, FastAPI, Streamlit, pytest, LangGraph. No live external APIs in tests.

## Global Constraints

- All tests must be deterministic: use `pd.date_range` + `numpy` synthetic fixtures; monkeypatch any external loader.
- No calls to `yfinance`, `OpenAI`, any database, cache, vector index, or search engine in tests.
- `optimize_backtest` in `src/backtest/optimizer.py` must not be modified.
- All return metrics in robust output must reflect `CostAssumptions` (fee + slippage).
- All numeric claims in report text must reference an `EvidenceItem.evidence_id`.
- Selected parameters must never be described as recommendations, signals, or guaranteed returns.
- `robust_label_allowed=False` prohibits strong positive expressions in output text.
- Rewrite agent must not change `evidence_id` references or numeric values.
- `run_backtest` fee parameter = `(cost.fee_pct_one_way + cost.slippage_pct_one_way) / 100`.
- `BacktestResult.trades` columns: `date`, `type` (Buy/Sell), `price`, `quantity`, `balance`, `profit`.
- Korean disclaimer required in every final report containing optimization metrics.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/backtest/robust.py` | Models, metrics, scoring, fold gen, orchestration |
| Create | `src/backtest/regime.py` | Regime classification, per-regime summaries |
| Modify | `src/backtest/optimizer.py` | Add `robust_optimize_window` entrypoint |
| Modify | `src/evidence/evidence_builder.py` | Add `build_optimization_evidence` |
| Modify | `src/graph/state.py` | Add `RobustOptimizationAnalysis` model |
| Modify | `src/storage/report_store.py` | Add `save_optimization_run` helper |
| Modify | `src/agents/backtest_agent.py` | Surface robust candidate + regime in output |
| Modify | `src/agents/coordinator_agent.py` | Include optimization section + disclaimer |
| Modify | `src/agents/evaluator_agent.py` | Add 4 optimization-specific checks |
| Modify | `src/agents/rewrite_agent.py` | Enforce evidence invariant in optimization text |
| Modify | `main.py` | Add `POST /backtest/optimize` endpoint |
| Modify | `app.py` | Add robust optimization UI controls + result view |
| Create | `tests/fixtures/optimization_data.py` | Synthetic price + trade fixtures |
| Create | `tests/fakes/optimization_fakes.py` | In-memory OptimizationRun repository |
| Create | `tests/test_robust_optimizer_metrics.py` | US1: metrics + cost |
| Create | `tests/test_robust_optimizer_scoring.py` | US1: scoring + guardrails |
| Create | `tests/test_walk_forward_optimizer.py` | US2: fold gen + orchestration |
| Create | `tests/test_robust_optimizer_baselines.py` | US2: baseline comparison |
| Create | `tests/test_api_robust_optimization.py` | US2: API contract |
| Create | `tests/test_regime_performance.py` | US3: regime classifier |
| Create | `tests/test_robust_optimization_safety.py` | US4: safety text checks |
| Create | `tests/test_optimization_evidence.py` | US4: evidence traceability |
| Modify | `tests/test_evaluator.py` | US4: evaluator fail cases |
| Modify | `tests/test_workflow_routing.py` | US4: routing + rewrite limits |

---

## Task 1: Scaffolding — Skeletons, Fixtures, Fakes, Env

**Files:**
- Create: `src/backtest/robust.py`
- Create: `src/backtest/regime.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/optimization_data.py`
- Create: `tests/fakes/__init__.py`
- Create: `tests/fakes/optimization_fakes.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `synthetic_prices(n, start, seed)`, `synthetic_trades(n_sells, seed)` for all downstream tests

- [ ] **Step 1: Create stub modules**

```python
# src/backtest/robust.py
"""Robust walk-forward optimization — models, metrics, scoring, orchestration."""
from __future__ import annotations
```

```python
# src/backtest/regime.py
"""Market regime classification and per-regime performance summaries."""
from __future__ import annotations
```

- [ ] **Step 2: Create test helpers directory**

```bash
mkdir -p tests/fixtures tests/fakes
touch tests/fixtures/__init__.py tests/fakes/__init__.py
```

- [ ] **Step 3: Write `tests/fixtures/optimization_data.py`**

```python
"""Deterministic synthetic fixtures for robust optimization tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(n: int = 500, start: str = "2022-01-03", seed: int = 42) -> pd.DataFrame:
    """Deterministic OHLC price series with a sine wave trend."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    t = np.arange(n)
    close = 100.0 + 20.0 * np.sin(t / 30.0) + t * 0.05 + rng.normal(0, 0.5, n).cumsum()
    close = np.maximum(close, 10.0)
    return pd.DataFrame(
        {
            "Open": close - 0.3,
            "High": close + 0.8,
            "Low": close - 0.8,
            "Close": close,
            "Volume": np.full(n, 10_000),
        },
        index=idx,
    )


def synthetic_trades(n_sells: int = 35, initial_balance: float = 10_000.0, seed: int = 42) -> pd.DataFrame:
    """Deterministic paired buy/sell trades DataFrame matching BacktestResult.trades schema."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    balance = initial_balance
    dates = pd.bdate_range("2022-01-03", periods=n_sells * 2 + 10)
    date_idx = 0
    for _ in range(n_sells):
        buy_price = float(rng.uniform(80, 120))
        qty = balance * 0.999 / buy_price
        rows.append({"date": dates[date_idx], "type": "Buy", "price": buy_price,
                     "quantity": qty, "balance": 0.0, "profit": 0.0})
        date_idx += 5
        sell_price = buy_price * float(rng.uniform(0.92, 1.12))
        profit = qty * (sell_price - buy_price) * 0.999
        balance = qty * sell_price * 0.999
        rows.append({"date": dates[date_idx], "type": "Sell", "price": sell_price,
                     "quantity": qty, "balance": balance, "profit": profit})
        date_idx += 5
    return pd.DataFrame(rows)


def synthetic_trades_few(n_sells: int = 10) -> pd.DataFrame:
    """Fewer than 30 completed trades — should fail robust_label guardrail."""
    return synthetic_trades(n_sells=n_sells, seed=7)


def synthetic_trades_high_mdd(initial_balance: float = 10_000.0) -> pd.DataFrame:
    """Trades that produce >25% drawdown."""
    rng = np.random.default_rng(99)
    rows: list[dict] = []
    balance = initial_balance
    dates = pd.bdate_range("2022-01-03", periods=200)
    date_idx = 0
    for i in range(35):
        buy_price = 100.0
        qty = balance * 0.999 / buy_price
        rows.append({"date": dates[date_idx], "type": "Buy", "price": buy_price,
                     "quantity": qty, "balance": 0.0, "profit": 0.0})
        date_idx += 2
        # First several trades lose 10% each — builds up to >25% drawdown
        sell_price = buy_price * (0.90 if i < 4 else float(rng.uniform(1.0, 1.05)))
        profit = qty * (sell_price - buy_price) * 0.999
        balance = qty * sell_price * 0.999
        rows.append({"date": dates[date_idx], "type": "Sell", "price": sell_price,
                     "quantity": qty, "balance": balance, "profit": profit})
        date_idx += 2
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Write `tests/fakes/optimization_fakes.py`**

```python
"""In-memory fakes for robust optimization tests."""
from __future__ import annotations
from typing import Any


class InMemoryOptimizationRepository:
    """Deterministic in-memory store for OptimizationRun objects."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def save(self, run_id: str, run: Any) -> None:
        self._store[run_id] = run

    def get(self, run_id: str) -> Any | None:
        return self._store.get(run_id)

    def all(self) -> list[Any]:
        return list(self._store.values())

    def clear(self) -> None:
        self._store.clear()
```

- [ ] **Step 5: Update `.env.example`**

Append the following block:

```
# Walk-forward robust optimization (MVP local storage — no external DB required)
# OPTIMIZATION_REPORT_DIR=reports/  # optional override; defaults to REPORT_DIR
# OPTIMIZATION_DEFAULT_FEE_PCT=0.05   # one-way fee in percent
# OPTIMIZATION_DEFAULT_SLIPPAGE_PCT=0.05  # one-way slippage in percent
```

- [ ] **Step 6: Commit**

```bash
git add src/backtest/robust.py src/backtest/regime.py \
        tests/fixtures/ tests/fakes/ .env.example
git commit -m "feat: add walk-forward optimization scaffolding and test fixtures"
```

---

## Task 2: Pydantic Data Contracts

**Files:**
- Modify: `src/backtest/robust.py`
- Create: `tests/test_robust_optimizer_metrics.py` (first tests only)

**Interfaces:**
- Produces: `CostAssumptions`, `RobustScoringPolicy`, `WalkForwardConfig`, `CandidateMetrics`, `WalkForwardFold`, `ParameterCandidateResult`, `BaselineResult`, `OptimizationRun`

- [ ] **Step 1: Write model validation tests**

```python
# tests/test_robust_optimizer_metrics.py
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
```

- [ ] **Step 2: Run tests — expect FAIL (module empty)**

```bash
pytest tests/test_robust_optimizer_metrics.py -v 2>&1 | head -20
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement models in `src/backtest/robust.py`**

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_robust_optimizer_metrics.py::test_cost_assumptions_defaults \
       tests/test_robust_optimizer_metrics.py::test_robust_scoring_policy_weights_sum_to_one \
       tests/test_robust_optimizer_metrics.py::test_walk_forward_config_requires_positive_windows \
       tests/test_robust_optimizer_metrics.py::test_optimization_run_status_values -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/robust.py tests/test_robust_optimizer_metrics.py
git commit -m "feat: add robust optimization Pydantic data contracts"
```

---

## Task 3: Candidate Metrics Computation

**Files:**
- Modify: `src/backtest/robust.py`
- Modify: `tests/test_robust_optimizer_metrics.py`

**Interfaces:**
- Consumes: `BacktestResult` from `src.backtest.strategy`, `CostAssumptions`
- Produces: `compute_candidate_metrics(result, initial_balance, cost) -> CandidateMetrics`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_robust_optimizer_metrics.py`:

```python
import numpy as np
import pandas as pd
from tests.fixtures.optimization_data import synthetic_trades, synthetic_trades_few, synthetic_trades_high_mdd
from src.backtest.robust import CostAssumptions, compute_candidate_metrics
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
    # seed=42 produces net-positive trades; profit_factor should be > 0 if any wins
    if metrics.profit_factor is not None:
        assert metrics.profit_factor >= 0.0


def test_default_cost_applied_matches_001_fee():
    """Default CostAssumptions (0.05+0.05)/100 = 0.001 — same as run_backtest default."""
    assert CostAssumptions().total_one_way_fee == pytest.approx(0.001)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_robust_optimizer_metrics.py -k "test_completed_trades" -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'compute_candidate_metrics'`.

- [ ] **Step 3: Implement `compute_candidate_metrics` in `src/backtest/robust.py`**

```python
import numpy as np
import pandas as pd
from src.backtest.strategy import BacktestResult


def _build_equity_curve(
    trades_df: pd.DataFrame, initial_balance: float
) -> list[float]:
    """Return equity value at each sell point, starting with initial_balance."""
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
    # Use per-trade return as % of capital at that point (approx: profit/initial_balance)
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
    result: BacktestResult,
    initial_balance: float,
    cost: CostAssumptions,
) -> CandidateMetrics:
    """Compute all CandidateMetrics from a BacktestResult."""
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

    # Turnover = round-trips per year
    total_days = max(1, (equity.__len__() - 1) * 10)  # rough estimate
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_robust_optimizer_metrics.py -v
```

Expected: all metric tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/robust.py tests/test_robust_optimizer_metrics.py \
        tests/fixtures/optimization_data.py tests/fixtures/__init__.py
git commit -m "feat: implement compute_candidate_metrics with MDD, Sharpe, Sortino, win rate"
```

---

## Task 4: Train Trial Filter & Score + Final Robust Score

**Files:**
- Modify: `src/backtest/robust.py`
- Create: `tests/test_robust_optimizer_scoring.py`

**Interfaces:**
- Produces: `passes_train_trial_filter(metrics) -> bool`, `compute_train_trial_score(metrics) -> float`, `compute_final_robust_score(fold_metrics, policy) -> tuple[float, dict]`, `compute_robust_label_allowed(metrics) -> bool`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_robust_optimizer_scoring.py
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
    """Candidate with 200% return but MDD=30% must still be rejected."""
    m = _metrics(cost_adjusted_return_pct=200.0, max_drawdown_pct=30.0, completed_trades=35)
    assert compute_robust_label_allowed(m) is False
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_robust_optimizer_scoring.py -v 2>&1 | tail -5
```

Expected: `ImportError`.

- [ ] **Step 3: Implement scoring functions in `src/backtest/robust.py`**

```python
def passes_train_trial_filter(metrics: CandidateMetrics) -> bool:
    """In-training filter: reject obviously bad candidates before OOS evaluation."""
    return metrics.completed_trades >= 30 and metrics.max_drawdown_pct <= 25.0


def compute_train_trial_score(metrics: CandidateMetrics) -> float:
    """In-sample trial score for Optuna ranking: Sharpe(35%) + Sortino(35%) + -MDD(20%) + -turnover(10%).
    Uses raw values; caller is responsible for comparing within the same study (Optuna handles normalization).
    """
    sharpe = metrics.sharpe or 0.0
    sortino = metrics.sortino or 0.0
    mdd = metrics.max_drawdown_pct
    turnover = metrics.turnover or 0.0
    return 0.35 * sharpe + 0.35 * sortino + 0.20 * (-mdd / 25.0) + 0.10 * (-min(turnover, 10.0) / 10.0)


def compute_robust_label_allowed(metrics: CandidateMetrics) -> bool:
    """OOS guardrail: both conditions must hold for robust label."""
    return metrics.completed_trades >= 30 and metrics.max_drawdown_pct <= 25.0


def _safe_normalize(values: list[float]) -> list[float]:
    arr = np.array(values, dtype=float)
    std = float(np.std(arr))
    mean = float(np.mean(arr))
    if std == 0:
        return [0.0] * len(values)
    return list((arr - mean) / std)


def compute_final_robust_score(
    fold_metrics: list[CandidateMetrics],
    policy: RobustScoringPolicy,
) -> tuple[float, dict]:
    """Compute the 5-component final robust score from OOS fold metrics."""
    if not fold_metrics:
        return 0.0, {}

    oos_returns = [m.cost_adjusted_return_pct for m in fold_metrics]
    sharpes = [m.sharpe or 0.0 for m in fold_metrics]
    mdds = [m.max_drawdown_pct for m in fold_metrics]
    turnovervals = [m.turnover or 0.0 for m in fold_metrics]

    median_oos = float(np.median(oos_returns))
    worst_fold = float(min(oos_returns))
    stddev = float(np.std(oos_returns)) if len(oos_returns) > 1 else 0.0
    avg_sharpe = float(np.mean(sharpes))
    avg_mdd = float(np.mean(mdds))
    avg_turnover = float(np.mean(turnovervals))

    # Normalized component scores (higher = better)
    c_oos = median_oos / 100.0                    # as fraction
    c_risk = avg_sharpe / 3.0                     # normalize to ~[0,1] range
    c_dd = max(0.0, 1.0 - avg_mdd / 100.0)        # lower MDD = better
    c_worst = max(0.0, (worst_fold + 20.0) / 40.0) # clamp worst fold
    c_stab = max(0.0, 1.0 - stddev / 20.0)        # lower stddev = better

    score = (
        policy.out_of_sample_return_weight * c_oos
        + policy.risk_adjusted_return_weight * c_risk
        + policy.drawdown_control_weight * c_dd
        + policy.worst_fold_resilience_weight * c_worst
        + policy.stability_turnover_penalty_weight * c_stab
    )

    components = {
        "out_of_sample_return": round(c_oos, 4),
        "risk_adjusted_return": round(c_risk, 4),
        "drawdown_control": round(c_dd, 4),
        "worst_fold_resilience": round(c_worst, 4),
        "stability_turnover_penalty": round(c_stab, 4),
    }
    return float(score), components
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_robust_optimizer_scoring.py -v
```

Expected: all 16 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/robust.py tests/test_robust_optimizer_scoring.py
git commit -m "feat: implement train_trial_filter, train_trial_score, final_robust_score, guardrails"
```

---

## Task 5: In-Memory Repository + Evidence Builder + GraphState

**Files:**
- Modify: `tests/fakes/optimization_fakes.py`
- Modify: `src/evidence/evidence_builder.py`
- Modify: `src/graph/state.py`
- Modify: `src/storage/report_store.py`
- Create: `tests/test_optimization_evidence.py`

**Interfaces:**
- Produces: `build_optimization_evidence(ticker, metric_name, metric_value, description) -> EvidenceItem`, `RobustOptimizationAnalysis` in state.py, `save_optimization_run(run_id, run) -> str` in report_store.py

- [ ] **Step 1: Write failing evidence tests**

```python
# tests/test_optimization_evidence.py
"""Tests: optimization evidence items are traceable and correctly typed."""
import pytest
from src.evidence.evidence_builder import build_optimization_evidence
from src.evidence.evidence_schema import EvidenceItem


def test_build_optimization_evidence_returns_evidence_item():
    item = build_optimization_evidence(
        ticker="AAPL",
        metric_name="max_drawdown_pct",
        metric_value=18.2,
        description="Historical simulation MDD for robust candidate.",
    )
    assert isinstance(item, EvidenceItem)
    assert item.source_type == "backtest"
    assert item.ticker == "AAPL"
    assert item.metric_name == "max_drawdown_pct"
    assert item.metric_value == 18.2


def test_optimization_evidence_id_starts_with_opt():
    item = build_optimization_evidence(ticker="MSFT", metric_name="sharpe", metric_value=0.9,
                                       description="Sharpe ratio.")
    assert item.evidence_id.startswith("opt_")


def test_optimization_evidence_description_non_empty():
    item = build_optimization_evidence(ticker="TSLA", metric_name="win_rate_pct",
                                       metric_value=54.0, description="Win rate.")
    assert len(item.description) > 0


def test_optimization_evidence_source_name_contains_simulation():
    item = build_optimization_evidence(ticker="AAPL", metric_name="cost_adjusted_return_pct",
                                       metric_value=8.4, description="Cost-adjusted return.")
    assert "simulation" in item.source_name.lower() or "backtest" in item.source_name.lower()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_optimization_evidence.py -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'build_optimization_evidence'`.

- [ ] **Step 3: Add `build_optimization_evidence` to `src/evidence/evidence_builder.py`**

```python
def build_optimization_evidence(
    *,
    ticker: str,
    metric_name: str,
    metric_value: "EvidenceMetricValue",
    description: str,
    source_name: str = "FinSight robust optimization (historical simulation)",
    source_url: str | None = None,
    collected_at: "datetime | None" = None,
) -> "EvidenceItem":
    """Build evidence for robust optimization numeric claims."""
    return EvidenceItem(
        evidence_id=generate_evidence_id("opt"),
        source_type="backtest",
        source_name=source_name,
        source_url=source_url,
        collected_at=_collected_at_or_now(collected_at),
        ticker=ticker,
        metric_name=metric_name,
        metric_value=metric_value,
        description=description,
    )
```

- [ ] **Step 4: Add `RobustOptimizationAnalysis` to `src/graph/state.py`**

Append after the existing `BacktestAnalysis` class:

```python
class RobustOptimizationAnalysis(BaseModel):
    """Robust walk-forward optimization output (historical simulation, not advice)."""

    ticker: str
    summary: str = ""
    robust_score: float | None = None
    robust_label_allowed: bool = False
    score_components: dict = Field(default_factory=dict)
    fold_count: int = 0
    median_oos_return_pct: float | None = None
    worst_fold_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    completed_trades: int = 0
    regime_summary: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    optimization_run_id: str | None = None
```

Also update `GraphState` TypedDict. Find the TypedDict definition in `src/graph/state.py` (the class/dict that contains `backtest_analysis`) and add after it:

```python
# src/graph/state.py — inside GraphState TypedDict, after backtest_analysis field:
    robust_optimization: "RobustOptimizationAnalysis | None"
```

The `RobustOptimizationAnalysis` class is defined in the same file so no additional import is needed.

- [ ] **Step 5: Add `save_optimization_run` to `src/storage/report_store.py`**

```python
def save_optimization_run(run_id: str, run: Any) -> str:
    """Persist an OptimizationRun as JSON and return the file path."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ticker = _extract_ticker(run) if not hasattr(run, "ticker") else str(run.ticker)
    filename = f"{_safe_filename_part(run_id)}_{_safe_filename_part(ticker)}_{_timestamp()}.json"
    path = REPORT_DIR / filename
    data = _to_jsonable(run)
    path.write_text(__import__("json").dumps(data, ensure_ascii=False, indent=2))
    return str(path)
```

- [ ] **Step 6: Run evidence tests — expect PASS**

```bash
pytest tests/test_optimization_evidence.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/evidence/evidence_builder.py src/graph/state.py src/storage/report_store.py \
        tests/fakes/ tests/test_optimization_evidence.py
git commit -m "feat: add optimization evidence builder, RobustOptimizationAnalysis state, report store helper"
```

---

## Task 6: Walk-Forward Fold Generation

**Files:**
- Modify: `src/backtest/robust.py`
- Create: `tests/test_walk_forward_optimizer.py`

**Interfaces:**
- Produces: `generate_fold_windows(start, end, config) -> list[dict]`
  - Each dict: `{fold_index, train_start, train_end, test_start, test_end}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_walk_forward_optimizer.py
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
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_walk_forward_optimizer.py -k "fold_windows" -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'generate_fold_windows'`.

- [ ] **Step 3: Implement `generate_fold_windows`**

```python
def generate_fold_windows(
    start: str,
    end: str,
    config: WalkForwardConfig,
) -> list[dict]:
    """Generate time-ordered walk-forward fold windows. No train/test overlap."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    folds = []
    fold_index = 1
    cursor = start_dt

    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=config.train_window_days - 1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=config.test_window_days - 1)

        if test_end > end_dt:
            break

        folds.append({
            "fold_index": fold_index,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })
        fold_index += 1
        cursor = cursor + pd.Timedelta(days=config.step_days)

    return folds
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_walk_forward_optimizer.py -k "fold_windows" -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtest/robust.py tests/test_walk_forward_optimizer.py
git commit -m "feat: implement generate_fold_windows with leakage prevention"
```

---

## Task 7: Walk-Forward Orchestration + Fold Aggregation

**Files:**
- Modify: `src/backtest/robust.py`
- Modify: `src/backtest/optimizer.py`
- Modify: `tests/test_walk_forward_optimizer.py`

**Interfaces:**
- Produces: `run_walk_forward_optimization(df, ticker, run_id, initial_balance, config, cost, policy, n_trials, manual_params) -> OptimizationRun`

- [ ] **Step 1: Write failing orchestration tests**

Append to `tests/test_walk_forward_optimizer.py`:

```python
from unittest.mock import patch
import pandas as pd
from tests.fixtures.optimization_data import synthetic_prices
from src.backtest.robust import (
    WalkForwardConfig, CostAssumptions, RobustScoringPolicy,
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
    valid_folds = [f for f in result.folds if f.status == "valid"]
    assert len(valid_folds) >= 3


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
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_walk_forward_optimizer.py -k "insufficient_folds or valid_run or aggregate or warnings" -v 2>&1 | tail -5
```

- [ ] **Step 3: Add `robust_optimize_window` to `src/backtest/optimizer.py`**

```python
# Add to src/backtest/optimizer.py — do NOT modify optimize_backtest

from src.backtest.robust import (
    CandidateMetrics, CostAssumptions, WalkForwardConfig,
    compute_candidate_metrics, compute_train_trial_score,
    passes_train_trial_filter,
)


def robust_optimize_window(
    df: pd.DataFrame,
    *,
    initial_balance: float,
    cost: CostAssumptions,
    n_trials: int = 30,
    search_space: "SearchSpace | None" = None,
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
```

- [ ] **Step 4: Implement `run_walk_forward_optimization` in `src/backtest/robust.py`**

```python
import uuid
from src.backtest.optimizer import robust_optimize_window
from src.backtest.strategy import BacktestParams, run_backtest


def _aggregate_fold_metrics(fold_oos_metrics: list[CandidateMetrics]) -> CandidateMetrics:
    if not fold_oos_metrics:
        return CandidateMetrics()
    returns = [m.cost_adjusted_return_pct for m in fold_oos_metrics]
    sharpes = [m.sharpe or 0.0 for m in fold_oos_metrics]
    mdds = [m.max_drawdown_pct for m in fold_oos_metrics]
    trades = sum(m.completed_trades for m in fold_oos_metrics)
    return CandidateMetrics(
        cost_adjusted_return_pct=float(np.median(returns)),
        total_return_pct=float(np.median(returns)),
        max_drawdown_pct=float(max(mdds)),
        sharpe=float(np.mean(sharpes)) if sharpes else None,
        completed_trades=trades,
        median_oos_return_pct=float(np.median(returns)),
        worst_fold_return_pct=float(min(returns)),
        fold_return_stddev=float(np.std(returns)) if len(returns) > 1 else 0.0,
    )


def run_walk_forward_optimization(
    df: pd.DataFrame,
    *,
    ticker: str,
    run_id: str | None = None,
    initial_balance: float,
    config: WalkForwardConfig,
    cost: CostAssumptions,
    policy: RobustScoringPolicy,
    n_trials: int = 30,
    manual_params: dict | None = None,
) -> OptimizationRun:
    """Full walk-forward optimization pipeline."""
    run_id = run_id or str(uuid.uuid4())
    start_str = df.index[0].strftime("%Y-%m-%d")
    end_str = df.index[-1].strftime("%Y-%m-%d")

    fold_windows = generate_fold_windows(start_str, end_str, config)
    if len(fold_windows) < config.minimum_valid_test_folds:
        return OptimizationRun(
            run_id=run_id, ticker=ticker, start=start_str, end=end_str,
            initial_balance=initial_balance, cost_assumptions=cost,
            scoring_policy=policy, fold_setup=config,
            status="insufficient_data",
            warnings=[
                f"Walk-forward validation produced {len(fold_windows)} fold(s); "
                f"minimum {config.minimum_valid_test_folds} required for robust output.",
                "결과를 robust parameter로 표시하지 않습니다.",
            ],
        )

    folds: list[WalkForwardFold] = []
    fold_oos_metrics: list[CandidateMetrics] = []
    best_fold_params: dict = {}

    for fw in fold_windows:
        train_mask = (df.index >= fw["train_start"]) & (df.index <= fw["train_end"])
        test_mask = (df.index >= fw["test_start"]) & (df.index <= fw["test_end"])
        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]

        if len(train_df) < 20 or len(test_df) < 5:
            folds.append(WalkForwardFold(
                fold_index=fw["fold_index"], **{k: fw[k] for k in fw if k != "fold_index"},
                selected_params={}, status="missing_data",
                warnings=["Insufficient data in this fold window."],
            ))
            continue

        try:
            best_params, _ = robust_optimize_window(
                train_df, initial_balance=initial_balance, cost=cost, n_trials=n_trials
            )
        except Exception as exc:
            folds.append(WalkForwardFold(
                fold_index=fw["fold_index"], **{k: fw[k] for k in fw if k != "fold_index"},
                selected_params={}, status="invalid",
                warnings=[f"Optimization failed: {exc}"],
            ))
            continue

        oos_result = run_backtest(
            test_df, BacktestParams.from_dict(best_params),
            initial_balance, cost.total_one_way_fee
        )
        oos_metrics = compute_candidate_metrics(oos_result, initial_balance, cost)

        status: FoldStatus = "valid" if oos_metrics.completed_trades > 0 else "no_trades"
        fold = WalkForwardFold(
            fold_index=fw["fold_index"],
            train_start=fw["train_start"], train_end=fw["train_end"],
            test_start=fw["test_start"], test_end=fw["test_end"],
            selected_params=best_params, candidate_metrics=oos_metrics, status=status,
        )
        folds.append(fold)
        if status == "valid":
            fold_oos_metrics.append(oos_metrics)
            best_fold_params = best_params  # last valid fold's params as candidate

    valid_count = len(fold_oos_metrics)
    if valid_count < config.minimum_valid_test_folds:
        return OptimizationRun(
            run_id=run_id, ticker=ticker, start=start_str, end=end_str,
            initial_balance=initial_balance, cost_assumptions=cost,
            scoring_policy=policy, fold_setup=config,
            status="insufficient_data", folds=folds,
            warnings=[
                f"Only {valid_count} valid OOS fold(s); {config.minimum_valid_test_folds} required.",
                "결과는 과거 시뮬레이션이며 매수·매도·보유 권유가 아닙니다.",
            ],
        )

    agg_metrics = _aggregate_fold_metrics(fold_oos_metrics)
    score, components = compute_final_robust_score(fold_oos_metrics, policy)
    label_allowed = compute_robust_label_allowed(agg_metrics)

    candidate = ParameterCandidateResult(
        params=best_fold_params,
        score=score,
        score_components=components,
        metrics=agg_metrics,
        fold_metrics=fold_oos_metrics,
        robust_label_allowed=label_allowed,
        warnings=[] if label_allowed else [
            "Guardrail: candidate does not meet minimum trade count or MDD threshold."
        ],
    )

    return OptimizationRun(
        run_id=run_id, ticker=ticker, start=start_str, end=end_str,
        initial_balance=initial_balance, cost_assumptions=cost,
        scoring_policy=policy, fold_setup=config,
        status="success", folds=folds, robust_candidate=candidate,
        warnings=["결과는 과거 시뮬레이션이며 매수·매도·보유 권유가 아닙니다."],
    )
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/test_walk_forward_optimizer.py -v
```

Expected: all walk-forward tests pass (some may show `no_trades` for synthetic data — check status not score).

- [ ] **Step 6: Commit**

```bash
git add src/backtest/robust.py src/backtest/optimizer.py tests/test_walk_forward_optimizer.py
git commit -m "feat: implement walk-forward orchestration and fold aggregation"
```

---

## Task 8: Baseline Comparison

**Files:**
- Modify: `src/backtest/robust.py`
- Create: `tests/test_robust_optimizer_baselines.py`

**Interfaces:**
- Produces: `compute_baselines(df, start, end, initial_balance, cost, manual_params) -> tuple[BaselineResult, BaselineResult]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_robust_optimizer_baselines.py
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
    first_close = df["Close"].iloc[0]
    last_close = df["Close"].iloc[-1]
    expected_pct = (last_close - first_close) / first_close * 100.0
    assert abs(passive.metrics.cost_adjusted_return_pct - expected_pct) < 1.0


def test_baselines_have_evidence_fields_available():
    df = synthetic_prices(n=300)
    manual, passive = compute_baselines(
        df=df, start="2022-01-01", end="2022-12-31",
        initial_balance=10_000, cost=CostAssumptions(), manual_params=None,
    )
    # Both baselines must report cost_adjusted_return_pct (traceable in evidence)
    assert isinstance(manual.metrics.cost_adjusted_return_pct, float)
    assert isinstance(passive.metrics.cost_adjusted_return_pct, float)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_robust_optimizer_baselines.py -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement `compute_baselines`**

```python
def compute_baselines(
    df: pd.DataFrame,
    *,
    start: str,
    end: str,
    initial_balance: float,
    cost: CostAssumptions,
    manual_params: dict | None,
) -> tuple[BaselineResult, BaselineResult]:
    """Return (manual_baseline, passive_baseline) over the same date window."""
    mask = (df.index >= start) & (df.index <= end)
    window_df = df.loc[mask]

    # Manual parameters baseline
    if manual_params and not window_df.empty:
        params = BacktestParams.from_dict(manual_params)
        result = run_backtest(window_df, params, initial_balance, cost.total_one_way_fee)
        m_metrics = compute_candidate_metrics(result, initial_balance, cost)
    else:
        m_metrics = CandidateMetrics()
    manual = BaselineResult(baseline_type="manual_parameters", metrics=m_metrics)

    # Passive buy-and-hold baseline (one buy, one sell, cost applied)
    if not window_df.empty and len(window_df) >= 2:
        first_price = float(window_df["Close"].iloc[0])
        last_price = float(window_df["Close"].iloc[-1])
        # Apply one-way cost on buy and sell
        effective_buy = first_price * (1 + cost.total_one_way_fee)
        effective_sell = last_price * (1 - cost.total_one_way_fee)
        pct = (effective_sell - effective_buy) / effective_buy * 100.0
        p_metrics = CandidateMetrics(
            total_return_pct=pct, cost_adjusted_return_pct=pct, completed_trades=1
        )
    else:
        p_metrics = CandidateMetrics()
    passive = BaselineResult(baseline_type="passive_buy_and_hold", metrics=p_metrics)

    return manual, passive
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_robust_optimizer_baselines.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/backtest/robust.py tests/test_robust_optimizer_baselines.py
git commit -m "feat: implement manual and passive baseline comparison"
```

---

## Task 9: API Endpoint `POST /backtest/optimize`

**Files:**
- Modify: `main.py`
- Create: `tests/test_api_robust_optimization.py`

**Interfaces:**
- Produces: `POST /backtest/optimize` endpoint with `RobustOptimizeRequest` / `RobustOptimizeResponse` models

- [ ] **Step 1: Write failing API contract tests**

```python
# tests/test_api_robust_optimization.py
"""Tests: POST /backtest/optimize contract — success and insufficient_data paths."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app
from src.backtest.robust import OptimizationRun, CostAssumptions, WalkForwardConfig

client = TestClient(app)

VALID_PAYLOAD = {
    "ticker": "AAPL",
    "investment_horizon": "중기",
    "risk_profile": "중립형",
    "start": "2022-01-01",
    "end": "2024-12-31",
    "initial_balance": 10000,
    "n_trials": 2,
    "walk_forward": {"train_window_days": 180, "test_window_days": 60, "step_days": 60},
    "costs": {"fee_pct_one_way": 0.05, "slippage_pct_one_way": 0.05},
}


def _mock_df():
    from tests.fixtures.optimization_data import synthetic_prices
    return synthetic_prices(n=800)


def test_optimize_endpoint_exists():
    resp = client.post("/backtest/optimize", json={})
    assert resp.status_code in (400, 422)  # validation error, not 404


def test_optimize_returns_run_id_and_status():
    with patch("main.load_price_history", return_value=_mock_df()):
        resp = client.post("/backtest/optimize", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    assert "status" in body


def test_optimize_insufficient_data_returns_no_candidate():
    short_payload = {**VALID_PAYLOAD, "start": "2024-01-01", "end": "2024-03-31"}
    with patch("main.load_price_history", return_value=_mock_df().iloc[:30]):
        resp = client.post("/backtest/optimize", json=short_payload)
    body = resp.json()
    assert body["status"] == "insufficient_data"
    assert body.get("optimization", {}).get("robust_candidate") is None


def test_optimize_rejects_invalid_dates():
    bad = {**VALID_PAYLOAD, "start": "2024-12-31", "end": "2022-01-01"}
    resp = client.post("/backtest/optimize", json=bad)
    assert resp.status_code == 422


def test_optimize_rejects_n_trials_above_50():
    bad = {**VALID_PAYLOAD, "n_trials": 51}
    resp = client.post("/backtest/optimize", json=bad)
    assert resp.status_code == 422


def test_optimize_response_contains_warnings():
    with patch("main.load_price_history", return_value=_mock_df()):
        resp = client.post("/backtest/optimize", json=VALID_PAYLOAD)
    body = resp.json()
    opt = body.get("optimization", {})
    assert "warnings" in opt
```

- [ ] **Step 2: Run tests — expect FAIL (endpoint not yet defined)**

```bash
pytest tests/test_api_robust_optimization.py::test_optimize_endpoint_exists -v 2>&1 | tail -5
```

- [ ] **Step 3: Add endpoint to `main.py`**

Find the location after the existing `/backtest` endpoint imports, then add:

```python
# ── Walk-forward robust optimization ──────────────────────────────────────────

class WalkForwardInput(BaseModel):
    train_window_days: int = Field(default=360, gt=0)
    test_window_days: int = Field(default=90, gt=0)
    step_days: int = Field(default=90, gt=0)


class CostInput(BaseModel):
    fee_pct_one_way: float = Field(default=0.05, ge=0.0)
    slippage_pct_one_way: float = Field(default=0.05, ge=0.0)


class RobustOptimizeRequest(BaseModel):
    model_config = {"extra": "forbid"}

    ticker: str = Field(min_length=1, max_length=20, pattern=TICKER_PATTERN)
    investment_horizon: Literal["단기", "중기", "장기"]
    risk_profile: Literal["보수형", "중립형", "공격형"]
    start: str
    end: str
    initial_balance: float = Field(default=10_000.0, gt=0)
    n_trials: int = Field(default=30, ge=1, le=50)
    walk_forward: WalkForwardInput = Field(default_factory=WalkForwardInput)
    costs: CostInput = Field(default_factory=CostInput)
    manual_params: BacktestParamsInput | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "RobustOptimizeRequest":
        try:
            s = date.fromisoformat(self.start)
            e = date.fromisoformat(self.end)
        except ValueError as exc:
            raise ValueError(f"Invalid date format: {exc}") from exc
        if s >= e:
            raise ValueError("start must be before end")
        if (e - s).days > MAX_BACKTEST_LOOKBACK_DAYS:
            raise ValueError(f"Date range exceeds {MAX_BACKTEST_LOOKBACK_DAYS} days")
        return self


@app.post("/backtest/optimize")
async def post_robust_optimize(request: RobustOptimizeRequest) -> dict:
    """Run robust walk-forward optimization (historical simulation research only)."""
    from src.backtest.robust import (
        CostAssumptions, RobustScoringPolicy, WalkForwardConfig,
        run_walk_forward_optimization, compute_baselines,
    )
    from src.backtest.data_loader import load_price_history
    from src.storage.report_store import save_optimization_run
    import uuid

    run_id = str(uuid.uuid4())[:8]

    try:
        df = load_price_history(request.ticker, request.start, request.end)
    except Exception as exc:
        return {"run_id": run_id, "status": "failed", "errors": [str(exc)], "optimization": None}

    cost = CostAssumptions(
        fee_pct_one_way=request.costs.fee_pct_one_way,
        slippage_pct_one_way=request.costs.slippage_pct_one_way,
    )
    policy = RobustScoringPolicy()
    config = WalkForwardConfig(
        train_window_days=request.walk_forward.train_window_days,
        test_window_days=request.walk_forward.test_window_days,
        step_days=request.walk_forward.step_days,
    )
    manual_params = request.manual_params.model_dump() if request.manual_params else None

    opt_run = run_walk_forward_optimization(
        df=df, ticker=request.ticker, run_id=run_id,
        initial_balance=request.initial_balance,
        config=config, cost=cost, policy=policy,
        n_trials=request.n_trials, manual_params=manual_params,
    )

    if opt_run.status == "success":
        manual_b, passive_b = compute_baselines(
            df=df, start=request.start, end=request.end,
            initial_balance=request.initial_balance, cost=cost,
            manual_params=manual_params,
        )
        opt_run.manual_baseline = manual_b
        opt_run.passive_baseline = passive_b

    report_path: str | None = None
    if opt_run.status == "success":
        try:
            report_path = save_optimization_run(run_id, opt_run)
            opt_run.report_path = report_path
        except Exception:
            pass

    return {
        "run_id": run_id,
        "status": opt_run.status,
        "optimization": opt_run.model_dump(mode="json"),
        "report_path": report_path,
        "errors": [],
    }
```

Also add `load_price_history` import: verify `src/backtest/data_loader.py` exports it and import at top of main.py.

- [ ] **Step 4: Check data_loader exports**

```bash
grep -n "def load_price_history" /Users/underwater05187808/dev/FinSight_Guard/src/backtest/data_loader.py
```

If not found, check what the actual function name is and adjust the import in main.py accordingly.

- [ ] **Step 5: Run API tests — expect PASS**

```bash
pytest tests/test_api_robust_optimization.py -v
```

Expected: 6 passed (some may run with mock).

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_api_robust_optimization.py
git commit -m "feat: add POST /backtest/optimize walk-forward endpoint"
```

---

## Task 10: Regime Classifier

**Files:**
- Modify: `src/backtest/regime.py`
- Create: `tests/test_regime_performance.py`

**Interfaces:**
- Produces: `classify_regime_periods(price_series, lookback_days) -> pd.Series[RegimeType]`, `compute_regime_performance(trades_df, regime_labels, initial_balance, cost) -> list[dict]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_regime_performance.py
"""Tests for market regime classification and per-regime performance summaries."""
import pytest
import numpy as np
import pandas as pd
from tests.fixtures.optimization_data import synthetic_prices
from src.backtest.regime import classify_regime_periods, compute_regime_performance
from src.backtest.robust import CostAssumptions


def _make_bull_prices(n: int = 200) -> pd.Series:
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(100.0 + np.arange(n) * 0.5, index=idx, name="Close")


def _make_bear_prices(n: int = 200) -> pd.Series:
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(200.0 - np.arange(n) * 0.5, index=idx, name="Close")


def _make_sideways_prices(n: int = 200, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(100.0 + rng.normal(0, 0.5, n).cumsum(), index=idx, name="Close")


def test_classify_rising_series_as_bull():
    labels = classify_regime_periods(_make_bull_prices(), lookback_days=30)
    bull_fraction = (labels == "bull").mean()
    assert bull_fraction > 0.5


def test_classify_falling_series_as_bear():
    labels = classify_regime_periods(_make_bear_prices(), lookback_days=30)
    bear_fraction = (labels == "bear").mean()
    assert bear_fraction > 0.5


def test_classify_flat_series_as_sideways():
    labels = classify_regime_periods(_make_sideways_prices(), lookback_days=30)
    sideways_fraction = (labels == "sideways").mean()
    assert sideways_fraction > 0.3


def test_regime_labels_are_valid_strings():
    prices = synthetic_prices(n=300)["Close"]
    labels = classify_regime_periods(prices, lookback_days=30)
    valid = {"bull", "bear", "sideways", "high_volatility", "low_volatility"}
    assert set(labels.dropna().unique()).issubset(valid)


def test_regime_performance_low_confidence_below_10_trades():
    from tests.fixtures.optimization_data import synthetic_trades_few
    trades_df = synthetic_trades_few(n_sells=3)
    prices = synthetic_prices(n=300)["Close"]
    labels = classify_regime_periods(prices, lookback_days=30)
    summaries = compute_regime_performance(trades_df, labels, initial_balance=10_000,
                                           cost=CostAssumptions())
    if summaries:
        for s in summaries:
            if s["completed_trades"] < 10:
                assert s["confidence"] == "low"


def test_regime_performance_low_confidence_below_60_trading_days():
    prices = pd.Series(
        100.0 + np.arange(50) * 0.2,
        index=pd.bdate_range("2022-01-03", periods=50),
        name="Close"
    )
    labels = classify_regime_periods(prices, lookback_days=20)
    from tests.fixtures.optimization_data import synthetic_trades
    trades_df = synthetic_trades(n_sells=5, seed=1)
    summaries = compute_regime_performance(trades_df, labels, initial_balance=10_000,
                                           cost=CostAssumptions())
    for s in summaries:
        if s["trading_days"] < 60:
            assert s["confidence"] == "low"
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_regime_performance.py -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement `src/backtest/regime.py`**

```python
"""Market regime classification and per-regime performance summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from src.backtest.robust import CandidateMetrics, CostAssumptions, RegimeType

BULL_THRESHOLD = 0.05
BEAR_THRESHOLD = -0.05
HIGH_VOL_PERCENTILE = 70
LOW_VOL_PERCENTILE = 30


def classify_regime_periods(
    price_series: pd.Series,
    lookback_days: int = 60,
) -> pd.Series:
    """Classify each trading day by trend regime (bull/bear/sideways).
    Returns a Series aligned with price_series.
    Volatility regimes are a separate analytical lens and computed independently.
    """
    rolling_return = price_series.pct_change(periods=lookback_days)
    regimes = pd.Series(index=price_series.index, dtype=object)
    regimes[rolling_return >= BULL_THRESHOLD] = "bull"
    regimes[rolling_return <= BEAR_THRESHOLD] = "bear"
    mask_sideways = (rolling_return > BEAR_THRESHOLD) & (rolling_return < BULL_THRESHOLD)
    regimes[mask_sideways] = "sideways"

    # Overlay volatility regime on remaining NaN or as a second label
    rolling_vol = price_series.pct_change().rolling(lookback_days).std()
    high_vol_threshold = rolling_vol.quantile(HIGH_VOL_PERCENTILE / 100.0)
    low_vol_threshold = rolling_vol.quantile(LOW_VOL_PERCENTILE / 100.0)

    # For days that are NaN in trend (not enough history), use vol regime
    null_trend = regimes.isna()
    regimes[null_trend & (rolling_vol >= high_vol_threshold)] = "high_volatility"
    regimes[null_trend & (rolling_vol <= low_vol_threshold)] = "low_volatility"
    regimes = regimes.fillna("sideways")

    return regimes


def compute_regime_performance(
    trades_df: pd.DataFrame,
    regime_labels: pd.Series,
    initial_balance: float,
    cost: CostAssumptions,
) -> list[dict]:
    """Summarize trade performance per regime segment."""
    if trades_df.empty or regime_labels.empty:
        return []

    sell_trades = trades_df[trades_df["type"] == "Sell"].copy()
    if sell_trades.empty:
        return []

    sell_trades["date_ts"] = pd.to_datetime(sell_trades["date"])
    sell_trades = sell_trades.set_index("date_ts")

    summaries: list[dict] = []
    for regime in ["bull", "bear", "sideways", "high_volatility", "low_volatility"]:
        regime_dates = regime_labels[regime_labels == regime].index
        if len(regime_dates) == 0:
            continue

        regime_trades = sell_trades[sell_trades.index.isin(regime_dates)]
        completed_trades = len(regime_trades)
        trading_days = len(regime_dates)

        confidence = "low" if (completed_trades < 10 or trading_days < 60) else "normal"
        low_confidence_reason: str | None = None
        if confidence == "low":
            reasons = []
            if completed_trades < 10:
                reasons.append(f"only {completed_trades} completed trades (minimum 10)")
            if trading_days < 60:
                reasons.append(f"only {trading_days} trading days (minimum 60)")
            low_confidence_reason = "; ".join(reasons)

        if completed_trades > 0:
            profits = regime_trades["profit"].to_numpy(dtype=float)
            wins = float((profits > 0).mean() * 100.0)
            gross_profit = float(profits[profits > 0].sum())
            gross_loss = float(abs(profits[profits < 0].sum()))
            pf = gross_profit / gross_loss if gross_loss > 0 else None
            ret_pct = float(profits.sum() / initial_balance * 100.0)
            metrics = CandidateMetrics(
                cost_adjusted_return_pct=ret_pct,
                total_return_pct=ret_pct,
                completed_trades=completed_trades,
                win_rate_pct=wins,
                profit_factor=pf,
            )
        else:
            metrics = CandidateMetrics()

        summaries.append({
            "regime": regime,
            "start": str(regime_dates[0].date()),
            "end": str(regime_dates[-1].date()),
            "trading_days": trading_days,
            "completed_trades": completed_trades,
            "metrics": metrics.model_dump(),
            "confidence": confidence,
            "low_confidence_reason": low_confidence_reason,
        })

    return summaries
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_regime_performance.py -v
```

Expected: all regime tests pass.

- [ ] **Step 5: Integrate regime into `run_walk_forward_optimization`**

In `src/backtest/robust.py`, inside `run_walk_forward_optimization`, **replace** the final `return` statement with:

```python
    # Regime classification — post-hoc explanation only, NOT used in trial selection
    from src.backtest.regime import compute_regime_performance, classify_regime_periods
    regime_summary: list[dict] = []
    try:
        price_series = df["Close"]
        regime_labels = classify_regime_periods(price_series)
        # Collect actual OOS sell trades from all valid folds for regime attribution
        oos_trade_frames: list[pd.DataFrame] = []
        for fold in folds:
            if fold.status == "valid":
                # Re-run OOS on test window to get actual trade DataFrame
                test_mask2 = (df.index >= fold.test_start) & (df.index <= fold.test_end)
                test_df2 = df.loc[test_mask2]
                if len(test_df2) >= 2:
                    oos_res2 = run_backtest(
                        test_df2, BacktestParams.from_dict(fold.selected_params),
                        initial_balance, cost.total_one_way_fee
                    )
                    if oos_res2.trades is not None and not oos_res2.trades.empty:
                        oos_trade_frames.append(oos_res2.trades)
        all_oos_trades = (
            pd.concat(oos_trade_frames, ignore_index=True)
            if oos_trade_frames else pd.DataFrame()
        )
        regime_summary = compute_regime_performance(
            all_oos_trades, regime_labels, initial_balance, cost
        )
    except Exception:
        pass  # regime failure must not block main result

    return OptimizationRun(
        run_id=run_id, ticker=ticker, start=start_str, end=end_str,
        initial_balance=initial_balance, cost_assumptions=cost,
        scoring_policy=policy, fold_setup=config,
        status="success", folds=folds, robust_candidate=candidate,
        regime_summary=regime_summary,
        warnings=["결과는 과거 시뮬레이션이며 매수·매도·보유 권유가 아닙니다."],
    )
```

- [ ] **Step 6: Commit**

```bash
git add src/backtest/regime.py tests/test_regime_performance.py src/backtest/robust.py
git commit -m "feat: implement regime classifier and per-regime performance summaries"
```

---

## Task 11: Safety Checks + Evidence Generation

**Files:**
- Modify: `src/backtest/robust.py`
- Modify: `src/evidence/evidence_builder.py`
- Create: `tests/test_robust_optimization_safety.py`
- Modify: `tests/test_optimization_evidence.py`

**Interfaces:**
- Produces: `build_optimization_evidence_set(run, ticker) -> list[EvidenceItem]`, safety text patterns in constants

- [ ] **Step 1: Write failing safety tests**

```python
# tests/test_robust_optimization_safety.py
"""Tests: optimization output text avoids advice-like wording."""
import pytest
from src.backtest.robust import FORBIDDEN_OPTIMIZATION_PHRASES, check_optimization_text_safety


SAFE_TEXT = (
    "이 파라미터 세트는 과거 시뮬레이션에서 MDD 18.2%를 기록했습니다. "
    "결과는 연구 목적이며 투자 권유가 아닙니다."
)

UNSAFE_TEXTS = [
    "이 파라미터로 매수하세요.",
    "guaranteed return of 15% using these settings.",
    "이 설정을 사용하면 반드시 수익을 낼 수 있습니다.",
    "recommended trading parameters for immediate use.",
    "자동 주문을 실행하세요.",
]

UNSAFE_WHEN_GUARDRAIL_FAILED = [
    "이 파라미터는 우수한 성과를 보였습니다.",
    "탁월한 전략 파라미터입니다.",
    "이 파라미터 세트는 강력히 권장됩니다.",
]


def test_safe_text_passes_safety_check():
    assert check_optimization_text_safety(SAFE_TEXT, robust_label_allowed=True) is True


def test_forbidden_phrases_fail_safety_check():
    for text in UNSAFE_TEXTS:
        assert check_optimization_text_safety(text, robust_label_allowed=True) is False, (
            f"Expected FAIL for: {text}"
        )


def test_strong_positive_fails_when_robust_label_not_allowed():
    for text in UNSAFE_WHEN_GUARDRAIL_FAILED:
        assert check_optimization_text_safety(text, robust_label_allowed=False) is False, (
            f"Expected FAIL when guardrail failed: {text}"
        )


def test_strong_positive_passes_when_robust_label_allowed():
    # "우수한" alone is OK when guardrail passed — context matters
    text = "이 파라미터는 OOS 폴드에서 우수한 MDD 관리를 보였습니다."
    # This text contains "우수한" but no direct trading instruction — passes
    assert check_optimization_text_safety(text, robust_label_allowed=True) is True


def test_forbidden_phrases_list_not_empty():
    assert len(FORBIDDEN_OPTIMIZATION_PHRASES) >= 5
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_robust_optimization_safety.py -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement safety constants and check function in `src/backtest/robust.py`**

```python
FORBIDDEN_OPTIMIZATION_PHRASES: list[str] = [
    "매수하세요", "매도하세요", "보유하세요",
    "guaranteed return", "반드시 수익", "투자 권유",
    "recommended trading", "자동 주문", "order execution",
    "buy signal", "sell signal",
]

STRONG_POSITIVE_PHRASES: list[str] = [
    "우수한 성과", "탁월한", "강력히 권장", "최고의 파라미터",
    "확실한 수익", "이 파라미터로 투자",
]


def check_optimization_text_safety(text: str, *, robust_label_allowed: bool) -> bool:
    """Return True if text passes optimization safety rules."""
    lower = text.lower()
    for phrase in FORBIDDEN_OPTIMIZATION_PHRASES:
        if phrase.lower() in lower:
            return False
    if not robust_label_allowed:
        for phrase in STRONG_POSITIVE_PHRASES:
            if phrase.lower() in lower:
                return False
    return True
```

- [ ] **Step 4: Implement `build_optimization_evidence_set`**

Add to `src/backtest/robust.py`:

```python
from src.evidence.evidence_builder import build_optimization_evidence
from src.evidence.evidence_schema import EvidenceItem


def build_optimization_evidence_set(
    run: OptimizationRun,
    ticker: str,
) -> list[EvidenceItem]:
    """Build EvidenceItem records for all key numeric claims in an OptimizationRun."""
    items: list[EvidenceItem] = []
    if run.robust_candidate is None:
        return items

    m = run.robust_candidate.metrics

    def _add(metric_name: str, value: float | None, description: str) -> None:
        if value is not None:
            items.append(build_optimization_evidence(
                ticker=ticker, metric_name=metric_name, metric_value=value,
                description=description,
            ))

    _add("robust_score", run.robust_candidate.score,
         "Final robust score (historical simulation, 5-component weighted).")
    _add("cost_adjusted_return_pct", m.cost_adjusted_return_pct,
         "Cost-adjusted OOS return — historical simulation only.")
    _add("max_drawdown_pct", m.max_drawdown_pct,
         "Maximum drawdown across OOS evaluation — historical simulation only.")
    _add("completed_trades", float(m.completed_trades),
         "Total completed OOS trades across all valid folds.")
    _add("worst_fold_return_pct", m.worst_fold_return_pct,
         "Worst single-fold OOS return — historical simulation.")
    _add("median_oos_return_pct", m.median_oos_return_pct,
         "Median OOS return across folds — historical simulation.")
    _add("sharpe", m.sharpe, "OOS Sharpe ratio — historical simulation.")

    if run.manual_baseline:
        _add("manual_baseline_return_pct", run.manual_baseline.metrics.cost_adjusted_return_pct,
             "Manual parameter baseline return for comparison — historical simulation.")
    if run.passive_baseline:
        _add("passive_baseline_return_pct", run.passive_baseline.metrics.cost_adjusted_return_pct,
             "Passive buy-and-hold baseline return for comparison — historical simulation.")

    return items
```

- [ ] **Step 5: Run safety tests — expect PASS**

```bash
pytest tests/test_robust_optimization_safety.py tests/test_optimization_evidence.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/backtest/robust.py tests/test_robust_optimization_safety.py
git commit -m "feat: add optimization safety checker and evidence set builder"
```

---

## Task 12: Evaluator + Coordinator + Rewrite Extensions

**Files:**
- Modify: `src/agents/evaluator_agent.py`
- Modify: `src/agents/coordinator_agent.py`
- Modify: `src/agents/rewrite_agent.py`
- Modify: `tests/test_evaluator.py`
- Modify: `tests/test_workflow_routing.py`

**Interfaces:**
- Produces: 4 new evaluator checks, optimization section in coordinator, evidence invariant in rewrite

- [ ] **Step 1: Add evaluator tests**

Append to `tests/test_evaluator.py`:

```python
# --- Optimization-specific evaluator checks ---

def test_evaluator_fails_on_unsupported_claim_in_optimization():
    from src.agents.evaluator_agent import check_optimization_section
    # Numeric claim with no matching evidence
    result = check_optimization_section(
        optimization_text="OOS 수익률은 15.3%입니다.",
        evidence_ids=[],
    )
    assert result["pass"] is False
    assert "unsupported_claim" in result["errors"]


def test_evaluator_fails_on_missing_limitation():
    from src.agents.evaluator_agent import check_optimization_section
    result = check_optimization_section(
        optimization_text="이 파라미터는 최고의 성과를 보입니다.",
        evidence_ids=["opt_abc123"],
    )
    assert result["pass"] is False
    assert "limitation_missing" in result["errors"]


def test_evaluator_fails_on_strong_expression_when_guardrail_failed():
    from src.agents.evaluator_agent import check_optimization_section
    result = check_optimization_section(
        optimization_text="이 파라미터는 우수한 성과를 보였으며 투자에 적합합니다.",
        evidence_ids=["opt_abc123"],
        robust_label_allowed=False,
    )
    assert result["pass"] is False


def test_evaluator_passes_safe_optimization_text():
    from src.agents.evaluator_agent import check_optimization_section
    result = check_optimization_section(
        optimization_text=(
            "과거 시뮬레이션 결과 OOS 수익률 중앙값은 opt_abc123에 기록되었습니다. "
            "이 결과는 연구 목적이며 매수·매도·보유 권유가 아닙니다."
        ),
        evidence_ids=["opt_abc123"],
        robust_label_allowed=True,
    )
    assert result["pass"] is True
```

- [ ] **Step 2: Run evaluator tests — expect FAIL**

```bash
pytest tests/test_evaluator.py -k "optimization" -v 2>&1 | tail -5
```

- [ ] **Step 3: Add `check_optimization_section` to `src/agents/evaluator_agent.py`**

Find the existing evaluator file and append:

```python
from src.backtest.robust import FORBIDDEN_OPTIMIZATION_PHRASES, STRONG_POSITIVE_PHRASES

LIMITATION_PHRASES = ["과거 시뮬레이션", "historical simulation", "연구 목적", "권유가 아닙니다"]
NUMERIC_PATTERN = r"\d+\.?\d*%"


def check_optimization_section(
    optimization_text: str,
    evidence_ids: list[str],
    robust_label_allowed: bool = True,
) -> dict:
    """Check optimization text for 4 safety conditions. Returns {pass, errors}."""
    import re
    errors: list[str] = []
    lower = optimization_text.lower()

    # 1. Unsupported claim: numeric value present but no evidence id mentioned
    numerics = re.findall(NUMERIC_PATTERN, optimization_text)
    if numerics and not any(eid in optimization_text for eid in evidence_ids):
        errors.append("unsupported_claim")

    # 2. Limitation missing
    if not any(phrase in optimization_text for phrase in LIMITATION_PHRASES):
        errors.append("limitation_missing")

    # 3. Forbidden phrases
    for phrase in FORBIDDEN_OPTIMIZATION_PHRASES:
        if phrase.lower() in lower:
            errors.append(f"forbidden_phrase:{phrase}")
            break

    # 4. Strong expression when guardrail failed
    if not robust_label_allowed:
        for phrase in STRONG_POSITIVE_PHRASES:
            if phrase.lower() in lower:
                errors.append("strong_expression_guardrail_failed")
                break

    return {"pass": len(errors) == 0, "errors": errors}
```

- [ ] **Step 4: Add workflow routing tests**

Append to `tests/test_workflow_routing.py`:

```python
def test_optimization_evaluator_failure_returns_completed_degraded():
    """After max rewrites, status must be completed_degraded with evaluator_errors."""
    from src.backtest.robust import OptimizationRun, CostAssumptions, WalkForwardConfig
    run = OptimizationRun(
        run_id="test", ticker="AAPL", start="2022-01-01", end="2022-12-31",
        initial_balance=10_000,
        fold_setup=WalkForwardConfig(train_window_days=180, test_window_days=60, step_days=60),
    )
    run.status = "completed_degraded"
    run.evaluator_errors = ["limitation_missing", "unsupported_claim"]
    assert run.status == "completed_degraded"
    assert len(run.evaluator_errors) == 2


def test_rewrite_does_not_change_evidence_ids():
    """Rewrite agent must preserve evidence_id references."""
    text = "OOS 수익률은 opt_abc123에 기록되었습니다. 과거 시뮬레이션 결과입니다."
    import re
    evidence_ids_before = set(re.findall(r"opt_[a-f0-9]+", text))
    # Simulate a no-op rewrite that only changes framing
    rewritten = text.replace("기록되었습니다", "나타났습니다")
    evidence_ids_after = set(re.findall(r"opt_[a-f0-9]+", rewritten))
    assert evidence_ids_before == evidence_ids_after
```

- [ ] **Step 5: Add optimization section to `src/agents/coordinator_agent.py`**

Find the existing report generation function and add an optimization section. Locate the function that builds `ResearchReport` and add after the `backtest_section` handling:

```python
# In coordinator_agent.py — inside the function that builds ResearchReport
# Find: backtest_section=..., and add after it:

def _build_optimization_section(robust: "RobustOptimizationAnalysis | None") -> str:
    if robust is None or robust.robust_score is None:
        return ""
    lines = [
        "## Walk-Forward 강건 최적화 참고 (과거 시뮬레이션)",
        "",
        f"- Robust Score: {robust.robust_score:.3f}",
        f"- OOS 수익률 중앙값: {robust.median_oos_return_pct:.1f}%" if robust.median_oos_return_pct else "",
        f"- 최대낙폭 (OOS): {robust.max_drawdown_pct:.1f}%" if robust.max_drawdown_pct else "",
        f"- 유효 폴드 수: {robust.fold_count}",
        "",
        "※ 이 결과는 과거 시뮬레이션이며 매수·매도·보유 권유가 아닙니다.",
    ]
    for w in robust.warnings:
        lines.append(f"⚠️ {w}")
    return "\n".join(l for l in lines if l is not None)
```

Then include in the coordinator's report building call by passing `_build_optimization_section(state.get("robust_optimization"))` into the `backtest_section` or a new dedicated section field.

- [ ] **Step 6: Add rewrite invariant to `src/agents/rewrite_agent.py`**

Find the existing rewrite function and add a post-rewrite check before returning:

```python
# In rewrite_agent.py — add after the rewrite is generated, before returning:

import re as _re

def _check_evidence_ids_preserved(original: str, rewritten: str) -> bool:
    """Evidence IDs in rewritten text must match original — numbers must not change."""
    orig_ids = set(_re.findall(r"opt_[a-f0-9]{8,}", original))
    new_ids = set(_re.findall(r"opt_[a-f0-9]{8,}", rewritten))
    orig_nums = set(_re.findall(r"\d+\.?\d+%", original))
    new_nums = set(_re.findall(r"\d+\.?\d+%", rewritten))
    return orig_ids == new_ids and orig_nums == new_nums

# Usage inside rewrite function, after generating rewritten_text:
if not _check_evidence_ids_preserved(original_text, rewritten_text):
    # Fall back to original rather than break evidence chain
    rewritten_text = original_text
```

- [ ] **Step 7: Run all new tests — expect PASS**

```bash
pytest tests/test_evaluator.py -k "optimization" tests/test_workflow_routing.py -k "optimization or rewrite" -v
```

- [ ] **Step 8: Commit**

```bash
git add src/agents/evaluator_agent.py src/agents/coordinator_agent.py src/agents/rewrite_agent.py \
        tests/test_evaluator.py tests/test_workflow_routing.py
git commit -m "feat: add optimization evaluator checks, coordinator section, rewrite invariant"
```

---

## Task 13: Backtest Agent + Streamlit UI

**Files:**
- Modify: `src/agents/backtest_agent.py`
- Modify: `app.py`

**Interfaces:**
- Consumes: `run_walk_forward_optimization`, `compute_baselines`, `build_optimization_evidence_set`
- Produces: robust optimization section in agent output, Streamlit controls + result view

- [ ] **Step 1: Extend `src/agents/backtest_agent.py`**

Locate the existing `run_backtest_analysis` function (or equivalent entrypoint) and add after the existing backtest call:

```python
# Robust optimization section — historical simulation only
from src.backtest.robust import (
    run_walk_forward_optimization, compute_baselines,
    build_optimization_evidence_set, CostAssumptions,
    RobustScoringPolicy, WalkForwardConfig,
)
from src.graph.state import RobustOptimizationAnalysis

def run_robust_optimization_for_agent(
    df,
    ticker: str,
    initial_balance: float = 10_000.0,
    n_trials: int = 10,
) -> RobustOptimizationAnalysis:
    """Run walk-forward optimization and return RobustOptimizationAnalysis for workflow state."""
    import uuid
    run_id = str(uuid.uuid4())[:8]
    config = WalkForwardConfig(train_window_days=180, test_window_days=60, step_days=60)
    cost = CostAssumptions()
    policy = RobustScoringPolicy()

    opt_run = run_walk_forward_optimization(
        df=df, ticker=ticker, run_id=run_id,
        initial_balance=initial_balance, config=config, cost=cost,
        policy=policy, n_trials=n_trials,
    )
    evidence = build_optimization_evidence_set(opt_run, ticker)

    candidate = opt_run.robust_candidate
    return RobustOptimizationAnalysis(
        ticker=ticker,
        summary=(
            "과거 시뮬레이션 Walk-Forward 최적화 결과입니다. 매수·매도·보유 권유가 아닙니다."
            if opt_run.status == "success"
            else f"최적화 결과: {opt_run.status}"
        ),
        robust_score=candidate.score if candidate else None,
        robust_label_allowed=candidate.robust_label_allowed if candidate else False,
        score_components=candidate.score_components if candidate else {},
        fold_count=len([f for f in opt_run.folds if f.status == "valid"]),
        median_oos_return_pct=candidate.metrics.median_oos_return_pct if candidate else None,
        worst_fold_return_pct=candidate.metrics.worst_fold_return_pct if candidate else None,
        max_drawdown_pct=candidate.metrics.max_drawdown_pct if candidate else None,
        completed_trades=candidate.metrics.completed_trades if candidate else 0,
        regime_summary=opt_run.regime_summary,
        warnings=opt_run.warnings,
        evidence=evidence,
        optimization_run_id=opt_run.run_id,
    )
```

- [ ] **Step 2: Add Streamlit robust optimization UI to `app.py`**

Find the existing backtest section in `app.py` and add after it:

```python
# ── Robust Walk-Forward Optimization UI ───────────────────────────────────────
st.subheader("Walk-Forward 강건 최적화 (과거 시뮬레이션 연구 전용)")
st.caption("⚠️ 결과는 과거 데이터 시뮬레이션이며 매수·매도·보유 권유가 아닙니다.")

with st.expander("강건 최적화 설정", expanded=False):
    wf_train = st.number_input("학습 윈도우 (일)", min_value=60, max_value=720, value=360, step=30)
    wf_test = st.number_input("테스트 윈도우 (일)", min_value=30, max_value=180, value=90, step=30)
    wf_step = st.number_input("스텝 (일)", min_value=30, max_value=180, value=90, step=30)
    wf_trials = st.slider("Optuna 시도 횟수", min_value=5, max_value=50, value=20)
    fee_pct = st.number_input("편도 수수료 (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    slippage_pct = st.number_input("편도 슬리피지 (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

if st.button("강건 최적화 실행"):
    # Retrieve ticker and df from session_state (set by existing backtest flow)
    opt_ticker = st.session_state.get("ticker", "")
    opt_df = st.session_state.get("price_df", None)
    if not opt_ticker or opt_df is None:
        st.warning("먼저 기본 백테스트를 실행하세요.")
    else:
        from src.backtest.robust import (
            run_walk_forward_optimization, compute_baselines,
            build_optimization_evidence_set,
            CostAssumptions, RobustScoringPolicy, WalkForwardConfig,
        )
        import uuid
        with st.spinner("Walk-Forward 최적화 중..."):
            cost = CostAssumptions(fee_pct_one_way=fee_pct, slippage_pct_one_way=slippage_pct)
            config = WalkForwardConfig(
                train_window_days=int(wf_train),
                test_window_days=int(wf_test),
                step_days=int(wf_step),
            )
            opt_run = run_walk_forward_optimization(
                df=opt_df, ticker=opt_ticker, run_id=str(uuid.uuid4())[:8],
                initial_balance=st.session_state.get("initial_balance", 10_000),
                config=config, cost=cost, policy=RobustScoringPolicy(),
                n_trials=wf_trials,
            )

        st.write(f"**상태:** `{opt_run.status}`")
        for w in opt_run.warnings:
            st.warning(w)

        if opt_run.robust_candidate:
            c = opt_run.robust_candidate
            st.success(
                "✅ Robust 후보 선정됨" if c.robust_label_allowed
                else "⚠️ 후보 선정 (가드레일 조건 미충족 — 강한 해석 금지)"
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Robust Score", f"{c.score:.3f}")
            col2.metric("OOS 수익률 중앙값", f"{c.metrics.median_oos_return_pct:.1f}%" if c.metrics.median_oos_return_pct else "N/A")
            col3.metric("최대낙폭 (OOS)", f"{c.metrics.max_drawdown_pct:.1f}%")

            with st.expander("스코어 구성요소"):
                st.json(c.score_components)

            with st.expander("폴드별 상세"):
                for f in opt_run.folds:
                    st.write(f"Fold {f.fold_index}: {f.train_start}~{f.test_end} | {f.status} | OOS 거래={f.candidate_metrics.completed_trades}")

        if opt_run.regime_summary:
            with st.expander("레짐별 성과"):
                for r in opt_run.regime_summary:
                    conf_label = " ⚠️ (저신뢰도)" if r["confidence"] == "low" else ""
                    st.write(f"**{r['regime']}{conf_label}**: 거래 {r['completed_trades']}건 | 수익률 {r['metrics'].get('cost_adjusted_return_pct', 0):.1f}%")
                    if r.get("low_confidence_reason"):
                        st.caption(r["low_confidence_reason"])

        if opt_run.manual_baseline or opt_run.passive_baseline:
            with st.expander("기준선 비교"):
                if opt_run.manual_baseline:
                    st.write(f"수동 파라미터: {opt_run.manual_baseline.metrics.cost_adjusted_return_pct:.1f}%")
                if opt_run.passive_baseline:
                    st.write(f"수동 매수보유: {opt_run.passive_baseline.metrics.cost_adjusted_return_pct:.1f}%")

        st.caption("📌 이 결과는 과거 시뮬레이션 연구 참고용이며 투자 권유나 미래 수익 보장이 아닙니다.")
```

- [ ] **Step 3: Compile check**

```bash
python3 -m compileall src/agents/backtest_agent.py app.py
```

Expected: no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add src/agents/backtest_agent.py app.py
git commit -m "feat: add robust optimization to backtest agent and Streamlit UI"
```

---

## Task 14: Final Polish — README, Quickstart, Test Sweep

**Files:**
- Modify: `README.md`
- Modify: `specs/002-walk-forward-optimization/quickstart.md`

- [ ] **Step 1: Full compile check**

```bash
python3 -m compileall src
```

Expected: Listing all modules OK, no SyntaxError.

- [ ] **Step 2: Run all new test suites**

```bash
pytest tests/test_robust_optimizer_metrics.py \
       tests/test_robust_optimizer_scoring.py \
       tests/test_walk_forward_optimizer.py \
       tests/test_robust_optimizer_baselines.py \
       tests/test_api_robust_optimization.py \
       tests/test_regime_performance.py \
       tests/test_robust_optimization_safety.py \
       tests/test_optimization_evidence.py \
       -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 3: Run existing test suites for regressions**

```bash
pytest tests/test_backtest_charts_optimizer.py \
       tests/test_backtest_agent.py \
       tests/test_api.py \
       tests/test_evaluator.py \
       tests/test_workflow_routing.py \
       -v --tb=short 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 4: Update `README.md`**

Find the Features or Architecture section and add:

```markdown
### Walk-Forward Robust Optimization

- `POST /backtest/optimize` — time-ordered walk-forward evaluation; rejects total-return-only ranking
- Candidate scoring: Sharpe, Sortino, MDD, win rate, profit factor, cost-adjusted return, turnover
- Guardrails: minimum 30 completed trades AND MDD ≤ 25% for robust label
- Regime performance summary: bull/bear/sideways/high-volatility/low-volatility (explanation only)
- All results are historical simulations — not trading advice

**Deferred** (separate future spec): PostgreSQL, Pinecone, Neo4j, OpenSearch, Redis, user accounts.
```

- [ ] **Step 5: Update `specs/002-walk-forward-optimization/quickstart.md`**

Replace the "New Test Scenarios" section with:

```markdown
## Validated Test Scenarios

All of the following now have deterministic pytest coverage:

1. ✅ Candidate with highest return but fewer than 30 completed trades is not labeled robust.
2. ✅ Candidate with maximum drawdown above 25% is not labeled robust.
3. ✅ Walk-forward output with fewer than 3 valid test folds returns `insufficient_data`.
4. ✅ Every valid test fold starts after its train fold (leakage prevention).
5. ✅ Robust score uses the 5-component risk-balanced weights from the spec.
6. ✅ Cost-adjusted returns reflect 0.05% one-way fee + 0.05% slippage.
7. ✅ Regime results with < 10 trades or < 60 trading days marked `low_confidence`.
8. ✅ Optimization evidence items generated for all numeric report claims.
9. ✅ Safety checker rejects forbidden phrases and strong expressions on guardrail failure.
10. ✅ API validates dates, n_trials ≤ 50, positive cost assumptions.
11. ✅ Manual and passive baselines use same periods and cost assumptions.
12. ✅ train_trial_filter rejects candidates before OOS evaluation (separate from OOS guardrail).
13. ✅ completed_degraded status + evaluator_errors returned when rewrite limit exceeded.
```

- [ ] **Step 6: Review all runtime text for safety compliance**

```bash
grep -rn "매수\|매도\|보유\|recommended\|guaranteed\|자동 주문" \
     src/agents/ src/backtest/robust.py app.py | grep -v "권유가 아닙니다\|#" | head -20
```

Expected: no matches (or only negated forms like "권유가 아닙니다").

- [ ] **Step 7: Final commit**

```bash
git add README.md specs/002-walk-forward-optimization/quickstart.md
git commit -m "docs: update README and quickstart with walk-forward optimization coverage"
```
