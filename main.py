"""FastAPI entrypoint for the financial research workflow."""

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, model_validator

from src.config import REPORT_DIR
from src.graph.workflow import run_research_workflow
from src.observability.metrics import get_metrics
from src.backtest.data_loader import load_price_history
from src.storage.report_store import load_report_json
from src.storage.run_store import get_run


app = FastAPI(title="Financial Research Multi-Agent API")

# A ticker is alphanumeric plus dot/dash (e.g. ``AAPL``, ``225010.KQ``, ``BRK-B``).
# Bounding length and character set keeps unvalidated strings from reaching the
# outbound yfinance request.
TICKER_PATTERN = r"^[A-Za-z0-9.\-]{1,20}$"
# Cap how far back a single backtest may look so one request cannot trigger a
# decade-long download + O(n·window) kernel-regression loop on the server.
MAX_BACKTEST_LOOKBACK_DAYS = 1825  # ~5 years


class AnalyzeRequest(BaseModel):
    """Request body for workflow analysis."""

    ticker: str = Field(min_length=1, max_length=20, pattern=TICKER_PATTERN)
    investment_horizon: Literal["단기", "중기", "장기"]
    risk_profile: Literal["보수형", "중립형", "공격형"]


class AnalyzeResponse(BaseModel):
    """API response for workflow analysis."""

    run_id: str
    status: str
    final_report: Any = None
    evaluation_result: Any = None
    report_path: str | None = None
    errors: list[Any] = Field(default_factory=list)


class BacktestParamsInput(BaseModel):
    """Bounded strategy hyper-parameters accepted from the public API.

    Mirrors ``src.backtest.strategy.BacktestParams`` but every field carries an
    explicit range so a caller cannot push degenerate values (huge windows,
    negative periods) into the kernel-regression simulation.
    """

    model_config = {"extra": "forbid"}

    rsi_period: int = Field(default=14, ge=5, le=30)
    kr_window: int = Field(default=30, ge=10, le=100)
    kr_bandwidth: float = Field(default=5.0, ge=0.1, le=20.0)
    bb_k: float = Field(default=2.0, ge=0.1, le=5.0)
    extrema_order: int = Field(default=5, ge=1, le=20)
    rsi_oversold: float = Field(default=30.0, ge=10.0, le=45.0)
    rsi_overbought: float = Field(default=70.0, ge=55.0, le=90.0)

    @model_validator(mode="after")
    def validate_rsi_bounds(self) -> "BacktestParamsInput":
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("rsi_oversold must be below rsi_overbought")
        return self


class BacktestRequest(BaseModel):
    """Request body for a research run that includes the backtest reference node."""

    ticker: str = Field(min_length=1, max_length=20, pattern=TICKER_PATTERN)
    investment_horizon: Literal["단기", "중기", "장기"]
    risk_profile: Literal["보수형", "중립형", "공격형"]
    start: str | None = None
    end: str | None = None
    params: BacktestParamsInput | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "BacktestRequest":
        """Reject malformed, future, inverted, or excessively wide date ranges."""
        try:
            end_date = date.fromisoformat(self.end) if self.end else date.today()
            start_date = (
                date.fromisoformat(self.start)
                if self.start
                else end_date - timedelta(days=MAX_BACKTEST_LOOKBACK_DAYS)
            )
        except ValueError as exc:
            raise ValueError("start/end must be ISO dates (YYYY-MM-DD)") from exc

        if end_date > date.today():
            raise ValueError("end date cannot be in the future")
        if start_date >= end_date:
            raise ValueError("start date must be before end date")
        if (end_date - start_date).days > MAX_BACKTEST_LOOKBACK_DAYS:
            raise ValueError(
                f"date range cannot exceed {MAX_BACKTEST_LOOKBACK_DAYS} days"
            )
        return self


class BacktestResponse(AnalyzeResponse):
    """API response that also surfaces the backtest analysis."""

    backtest_analysis: Any = None


class HealthResponse(BaseModel):
    """Health response."""

    status: str


class MetricsResponse(BaseModel):
    """Runtime metrics response."""

    total_runs: int
    successful_runs: int
    failed_runs: int
    average_evaluation_score: float


def _infer_status(result: dict[str, Any]) -> str:
    if result.get("status"):
        return str(result["status"])

    evaluation = result.get("evaluation_result")
    if evaluation is not None:
        passed = (
            evaluation.get("overall_pass")
            if isinstance(evaluation, dict)
            else getattr(evaluation, "overall_pass", False)
        )
        return "success" if passed else "failed"

    if result.get("errors"):
        return "failed"
    return "success" if result.get("final_report") is not None else "failed"


def _find_report_path(run_id: str) -> Path | None:
    run_record = get_run(run_id)
    if run_record and run_record.get("report_path"):
        path = Path(str(run_record["report_path"]))
        if path.exists():
            return path

    matches = sorted(REPORT_DIR.glob(f"{run_id}_*.json"))
    return matches[-1] if matches else None


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> dict[str, Any]:
    """Return in-memory workflow metrics."""
    return get_metrics()


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """Run the financial research workflow."""
    result = run_research_workflow(
        ticker=request.ticker,
        investment_horizon=request.investment_horizon,
        risk_profile=request.risk_profile,
    )
    response = {
        "run_id": result.get("run_id"),
        "status": _infer_status(result),
        "final_report": result.get("final_report"),
        "evaluation_result": result.get("evaluation_result"),
        "report_path": result.get("report_path"),
        "errors": result.get("errors", []),
    }
    return jsonable_encoder(response)


@app.post("/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest) -> dict[str, Any]:
    """Run the research workflow with the historical-backtest reference node.

    The backtest output is framed as a past simulation and is never a buy/sell/
    hold recommendation; it flows through the same Coordinator and Evaluator
    safety checks as every other evidence source.

    Latency note: this is a synchronous endpoint (FastAPI runs it in the thread
    pool). The handler performs a blocking yfinance download plus an O(n*window)
    kernel-regression simulation, so expect multi-second latency. The cost is
    bounded by the request model: the date range is capped at
    MAX_BACKTEST_LOOKBACK_DAYS and strategy params are range-limited, so a single
    request cannot monopolise a worker indefinitely. Heavy concurrent use should
    be moved to a background task/queue.
    """
    result = run_research_workflow(
        ticker=request.ticker,
        investment_horizon=request.investment_horizon,
        risk_profile=request.risk_profile,
        enable_backtest=True,
        backtest_start=request.start,
        backtest_end=request.end,
        backtest_params=request.params.model_dump() if request.params else None,
    )
    response = {
        "run_id": result.get("run_id"),
        "status": _infer_status(result),
        "final_report": result.get("final_report"),
        "evaluation_result": result.get("evaluation_result"),
        "backtest_analysis": result.get("backtest_analysis"),
        "report_path": result.get("report_path"),
        "errors": result.get("errors", []),
    }
    return jsonable_encoder(response)


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
    import uuid as _uuid
    from src.backtest.robust import (
        CostAssumptions, RobustScoringPolicy, WalkForwardConfig,
        run_walk_forward_optimization, compute_baselines,
    )
    from src.storage.report_store import save_optimization_run

    run_id = str(_uuid.uuid4())[:8]

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


@app.get("/reports/{run_id}")
def get_report(run_id: str) -> dict[str, Any]:
    """Load a saved report by run id."""
    report_path = _find_report_path(run_id)
    if report_path is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return load_report_json(report_path)
