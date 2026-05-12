"""FastAPI entrypoint for the financial research workflow."""

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from src.config import REPORT_DIR
from src.graph.workflow import run_research_workflow
from src.observability.metrics import get_metrics
from src.storage.report_store import load_report_json
from src.storage.run_store import get_run


app = FastAPI(title="Financial Research Multi-Agent API")


class AnalyzeRequest(BaseModel):
    """Request body for workflow analysis."""

    ticker: str = Field(min_length=1)
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


@app.get("/reports/{run_id}")
def get_report(run_id: str) -> dict[str, Any]:
    """Load a saved report by run id."""
    report_path = _find_report_path(run_id)
    if report_path is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return load_report_json(report_path)
