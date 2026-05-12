"""Market analysis agent."""

from math import isfinite
import pandas as pd

from src.evidence.evidence_builder import build_market_evidence
from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import GraphState, MarketAnalysis, WorkflowError
from src.indicators.technicals import enrich_market_indicators
from src.tools.market_data import fetch_price_history


MARKET_NODE = "market_agent"


def _ticker_from_state(state: GraphState) -> str:
    ticker = state.get("ticker")
    if ticker:
        return ticker.strip().upper()

    user_input = state.get("user_input")
    if user_input is not None:
        return user_input.ticker.strip().upper()

    return ""


def _latest_value(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None

    numeric_value = float(value)
    if not isfinite(numeric_value):
        return None
    return round(numeric_value, 4)


def _score_trend(metrics: dict[str, float | None]) -> float:
    close = metrics.get("latest_close")
    ma20 = metrics.get("MA20")
    ma60 = metrics.get("MA60")
    ma120 = metrics.get("MA120")
    comparisons = [
        None if close is None or ma20 is None else close >= ma20,
        None if ma20 is None or ma60 is None else ma20 >= ma60,
        None if ma60 is None or ma120 is None else ma60 >= ma120,
    ]
    available = [comparison for comparison in comparisons if comparison is not None]
    if not available:
        return 0.0
    return round(sum(1 for comparison in available if comparison) / len(available), 2)


def _score_momentum(metrics: dict[str, float | None]) -> float:
    rsi = metrics.get("RSI")
    macd = metrics.get("MACD")
    macd_signal = metrics.get("MACD_signal")

    components: list[float] = []
    if rsi is not None:
        if 45 <= rsi <= 60:
            components.append(0.75)
        elif 35 <= rsi < 45 or 60 < rsi <= 70:
            components.append(0.55)
        elif rsi < 35:
            components.append(0.35)
        else:
            components.append(0.4)

    if macd is not None and macd_signal is not None:
        components.append(0.7 if macd >= macd_signal else 0.35)

    if not components:
        return 0.0
    return round(sum(components) / len(components), 2)


def _score_volatility_risk(metrics: dict[str, float | None]) -> float:
    close = metrics.get("latest_close")
    atr = metrics.get("ATR")
    if close is None or atr is None or close == 0:
        return 0.0

    atr_ratio = atr / close
    if atr_ratio < 0.02:
        return 0.25
    if atr_ratio < 0.04:
        return 0.5
    if atr_ratio < 0.07:
        return 0.75
    return 1.0


def _format_metric(value: float | None) -> str:
    return "데이터 부족" if value is None else f"{value:.2f}"


def _build_metric_evidence(ticker: str, metrics: dict[str, float | None]) -> list[EvidenceItem]:
    source_url = f"https://finance.yahoo.com/quote/{ticker}"
    descriptions = {
        "latest_close": "최근 종가",
        "MA20": "20일 단순이동평균",
        "MA60": "60일 단순이동평균",
        "MA120": "120일 단순이동평균",
        "RSI": "14일 RSI",
        "MACD": "MACD 지표",
        "ATR": "14일 ATR",
    }
    return [
        build_market_evidence(
            ticker=ticker,
            metric_name=metric_name,
            metric_value=metric_value,
            description=f"{ticker} {description}: {_format_metric(metric_value)}",
            source_url=source_url,
        )
        for metric_name, metric_value in metrics.items()
        if metric_name in descriptions
        for description in [descriptions[metric_name]]
    ]


def run_market_agent(state: GraphState) -> dict:
    """Fetch market data, calculate indicators, and return market analysis."""
    ticker = _ticker_from_state(state)
    if not ticker:
        error = WorkflowError(
            node=MARKET_NODE,
            message="Ticker is missing for market analysis.",
            recoverable=False,
        )
        return {"status": "failed", "errors": [*state.get("errors", []), error]}

    try:
        price_history = fetch_price_history(ticker)
        enriched = enrich_market_indicators(price_history)
    except Exception as exc:
        note = f"시장 가격 데이터를 가져오지 못했습니다: {exc}"
        analysis = MarketAnalysis(
            ticker=ticker,
            summary="시장 데이터 수집 실패로 기술적 분석을 제한적으로만 제공합니다.",
            missing_data_notes=[note],
        )
        error = WorkflowError(
            node=MARKET_NODE,
            message=note,
            error_type="market_data_error",
            recoverable=True,
        )
        return {
            "status": "degraded",
            "market_analysis": analysis,
            "errors": [*state.get("errors", []), error],
            "warnings": [*state.get("warnings", []), note],
        }

    if enriched.empty:
        note = "시장 가격 데이터가 비어 있어 기술적 지표를 계산하지 못했습니다."
        analysis = MarketAnalysis(
            ticker=ticker,
            summary="시장 데이터가 없어 기술적 분석을 제한적으로만 제공합니다.",
            missing_data_notes=[note],
        )
        return {
            "status": "degraded",
            "market_analysis": analysis,
            "warnings": [*state.get("warnings", []), note],
        }

    latest = enriched.iloc[-1]
    metrics = {
        "latest_close": _latest_value(latest, "Close"),
        "MA20": _latest_value(latest, "MA20"),
        "MA60": _latest_value(latest, "MA60"),
        "MA120": _latest_value(latest, "MA120"),
        "RSI": _latest_value(latest, "RSI"),
        "MACD": _latest_value(latest, "MACD"),
        "ATR": _latest_value(latest, "ATR"),
        "MACD_signal": _latest_value(latest, "MACD_signal"),
    }
    missing_data_notes = [
        f"{metric_name} 계산에 필요한 기간의 가격 데이터가 부족합니다."
        for metric_name, metric_value in metrics.items()
        if metric_name != "MACD_signal" and metric_value is None
    ]

    trend_score = _score_trend(metrics)
    momentum_score = _score_momentum(metrics)
    volatility_risk = _score_volatility_risk(metrics)
    atr_ratio = (
        None
        if metrics["latest_close"] in (None, 0) or metrics["ATR"] is None
        else round(metrics["ATR"] / metrics["latest_close"], 4)
    )

    trend_summary = (
        f"MA 정렬 기반 추세 점수는 {trend_score:.2f}입니다. "
        f"종가 {_format_metric(metrics['latest_close'])}, MA20 {_format_metric(metrics['MA20'])}, "
        f"MA60 {_format_metric(metrics['MA60'])}, MA120 {_format_metric(metrics['MA120'])} 기준입니다."
    )
    momentum_summary = (
        f"RSI/MACD 기반 모멘텀 점수는 {momentum_score:.2f}입니다. "
        f"RSI {_format_metric(metrics['RSI'])}, MACD {_format_metric(metrics['MACD'])}를 참고했습니다."
    )
    volatility_summary = (
        f"ATR 기반 변동성 위험 점수는 {volatility_risk:.2f}입니다. "
        f"ATR/종가 비율은 {'데이터 부족' if atr_ratio is None else f'{atr_ratio:.2%}'}입니다."
    )

    evidence = _build_metric_evidence(ticker, metrics)
    analysis = MarketAnalysis(
        ticker=ticker,
        summary=(
            "기술적 지표는 가격 흐름, 모멘텀, 변동성 위험을 비교하기 위한 참고 정보입니다. "
            "특정 매수, 매도, 보유 판단을 직접 지시하지 않습니다."
        ),
        trend_summary=trend_summary,
        momentum_summary=momentum_summary,
        volatility_summary=volatility_summary,
        evidence=evidence,
        missing_data_notes=missing_data_notes,
    )
    return {
        "status": "success",
        "market_analysis": analysis,
        "evidence": [*state.get("evidence", []), *evidence],
    }
