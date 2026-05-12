"""Fundamental analysis agent."""

from math import isfinite
from typing import Any

from src.evidence.evidence_builder import build_fundamental_evidence
from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import FundamentalAnalysis, GraphState, WorkflowError
from src.tools.financial_data import FINANCIAL_FIELDS, fetch_basic_financials


FUNDAMENTAL_NODE = "fundamental_agent"

VALUATION_METRICS = ["marketCap", "trailingPE", "priceToBook"]
PROFITABILITY_METRICS = ["returnOnEquity", "profitMargins", "freeCashflow"]
STABILITY_METRICS = ["debtToEquity", "sector", "industry"]

METRIC_LABELS = {
    "longName": "회사명",
    "sector": "섹터",
    "industry": "산업",
    "marketCap": "시가총액",
    "trailingPE": "후행 PER",
    "priceToBook": "P/B",
    "returnOnEquity": "ROE",
    "profitMargins": "순이익률",
    "debtToEquity": "부채비율",
    "freeCashflow": "잉여현금흐름",
}


def _ticker_from_state(state: GraphState) -> str:
    ticker = state.get("ticker")
    if ticker:
        return ticker.strip().upper()

    user_input = state.get("user_input")
    if user_input is not None:
        return user_input.ticker.strip().upper()

    return ""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float):
        return not isfinite(value)
    return False


def _format_metric(value: Any) -> str:
    if _is_missing(value):
        return "데이터 부족"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _build_summary(metrics: dict[str, Any], metric_names: list[str], title: str) -> str:
    parts = [
        f"{METRIC_LABELS.get(metric_name, metric_name)} {_format_metric(metrics.get(metric_name))}"
        for metric_name in metric_names
    ]
    return f"{title}: " + ", ".join(parts) + "."


def _build_evidence(ticker: str, metrics: dict[str, Any]) -> list[EvidenceItem]:
    source_url = f"https://finance.yahoo.com/quote/{ticker}"
    evidence: list[EvidenceItem] = []
    for metric_name in FINANCIAL_FIELDS:
        metric_value = metrics.get(metric_name)
        if _is_missing(metric_value):
            continue

        label = METRIC_LABELS.get(metric_name, metric_name)
        evidence.append(
            build_fundamental_evidence(
                ticker=ticker,
                metric_name=metric_name,
                metric_value=metric_value,
                description=f"{ticker} {label}: {_format_metric(metric_value)}",
                source_url=source_url,
            )
        )
    return evidence


def run_fundamental_agent(state: GraphState) -> dict:
    """Fetch fundamental data and return evidence-grounded analysis."""
    ticker = _ticker_from_state(state)
    if not ticker:
        error = WorkflowError(
            node=FUNDAMENTAL_NODE,
            message="Ticker is missing for fundamental analysis.",
            recoverable=False,
        )
        return {"status": "failed", "errors": [*state.get("errors", []), error]}

    try:
        metrics = fetch_basic_financials(ticker)
    except Exception as exc:
        note = f"펀더멘털 데이터를 가져오지 못했습니다: {exc}"
        analysis = FundamentalAnalysis(
            ticker=ticker,
            summary="펀더멘털 데이터 수집 실패로 기업 기초 분석을 제한적으로만 제공합니다.",
            missing_data_notes=[note],
        )
        error = WorkflowError(
            node=FUNDAMENTAL_NODE,
            message=note,
            error_type="fundamental_data_error",
            recoverable=True,
        )
        return {
            "status": "degraded",
            "fundamental_analysis": analysis,
            "errors": [*state.get("errors", []), error],
            "warnings": [*state.get("warnings", []), note],
        }

    missing_data_notes = [
        f"{METRIC_LABELS.get(metric_name, metric_name)} 데이터가 제공되지 않았습니다."
        for metric_name in FINANCIAL_FIELDS
        if _is_missing(metrics.get(metric_name))
    ]
    evidence = _build_evidence(ticker, metrics)

    company_name = metrics.get("longName") or ticker
    analysis = FundamentalAnalysis(
        ticker=ticker,
        summary=(
            f"{company_name}의 공개 재무 지표를 기준으로 가치평가, 수익성, 안정성을 분리해 "
            "점검했습니다. 이 내용은 특정 투자 행동을 지시하지 않는 참고 분석입니다."
        ),
        valuation_summary=_build_summary(metrics, VALUATION_METRICS, "가치평가 지표"),
        profitability_summary=_build_summary(metrics, PROFITABILITY_METRICS, "수익성 지표"),
        stability_summary=_build_summary(metrics, STABILITY_METRICS, "안정성 및 업종 지표"),
        evidence=evidence,
        missing_data_notes=missing_data_notes,
    )
    return {
        "status": "success",
        "fundamental_analysis": analysis,
        "evidence": [*state.get("evidence", []), *evidence],
    }
