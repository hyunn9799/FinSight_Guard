"""Backtest agent: runs a historical strategy simulation as reference evidence.

Positioning: this node produces a *historical simulation* of a fixed technical
rule set (kernel regression + RSI divergence + Bollinger bands). Its outputs are
technical reference material describing the past, never buy/sell/hold advice and
never a forward-looking performance promise. The Coordinator and Evaluator treat
the resulting EvidenceItems exactly like any other source-grounded evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.backtest.data_loader import load_price_history
from src.backtest.strategy import BacktestParams, run_backtest
from src.evidence.evidence_builder import build_backtest_evidence
from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import BacktestAnalysis, GraphState, WorkflowError


BACKTEST_NODE = "backtest_agent"
DEFAULT_LOOKBACK_DAYS = 400
DEFAULT_INITIAL_BALANCE = 10_000.0
NO_ADVICE_NOTE = (
    "백테스트 결과는 고정된 기술적 규칙의 과거 시뮬레이션입니다. "
    "미래 수익을 보장하지 않으며 매수·매도·보유를 권유하지 않습니다."
)


def _ticker_from_state(state: GraphState) -> str:
    ticker = state.get("ticker")
    if ticker:
        return ticker.strip().upper()
    user_input = state.get("user_input")
    if user_input is not None:
        return user_input.ticker.strip().upper()
    return ""


def _date_range(state: GraphState) -> tuple[str, str]:
    end = state.get("backtest_end")
    start = state.get("backtest_start")
    today = datetime.now(UTC).date()
    if not end:
        end = today.isoformat()
    if not start:
        start = (today - timedelta(days=DEFAULT_LOOKBACK_DAYS)).isoformat()
    return str(start), str(end)


def _degraded(state: GraphState, ticker: str, note: str) -> dict:
    analysis = BacktestAnalysis(
        ticker=ticker,
        summary="백테스트 데이터를 사용할 수 없어 과거 시뮬레이션을 제공하지 못했습니다.",
        missing_data_notes=[note, NO_ADVICE_NOTE],
    )
    error = WorkflowError(
        node=BACKTEST_NODE,
        message=note,
        error_type="backtest_data_error",
        recoverable=True,
    )
    return {
        "status": "degraded",
        "backtest_analysis": analysis,
        "errors": [*state.get("errors", []), error],
        "warnings": [*state.get("warnings", []), note],
    }


def _build_evidence(ticker: str, result, start: str, end: str) -> list[EvidenceItem]:
    bullish = sum(1 for div in result.divergences if div[2] == "bullish")
    bearish = sum(1 for div in result.divergences if div[2] == "bearish")
    metrics = [
        (
            "backtest_profit_pct",
            round(result.profit_pct, 2),
            f"{ticker} {start}~{end} 과거 시뮬레이션 누적 수익률 {result.profit_pct:.2f}% "
            "(고정 규칙 기준, 미래 수익 보장 아님)",
        ),
        (
            "backtest_trade_count",
            result.trade_count,
            f"{ticker} 시뮬레이션 구간에서 발생한 규칙 기반 매매 횟수 {result.trade_count}회",
        ),
        (
            "backtest_divergence_count",
            len(result.divergences),
            f"{ticker} RSI 다이버전스 신호 {len(result.divergences)}건 (강세 {bullish}, 약세 {bearish})",
        ),
        (
            "backtest_final_value",
            round(result.final_value, 2),
            f"{ticker} 초기 자본 대비 시뮬레이션 종료 시점 평가금액 {result.final_value:.2f}",
        ),
    ]
    source_url = f"https://finance.yahoo.com/quote/{ticker}/history"
    return [
        build_backtest_evidence(
            ticker=ticker,
            metric_name=metric_name,
            metric_value=metric_value,
            description=description,
            source_url=source_url,
        )
        for metric_name, metric_value, description in metrics
    ]


def run_backtest_agent(state: GraphState) -> dict:
    """Run a historical backtest and return reference evidence + analysis."""
    ticker = _ticker_from_state(state)
    if not ticker:
        error = WorkflowError(
            node=BACKTEST_NODE,
            message="Ticker is missing for backtest.",
            recoverable=False,
        )
        return {"status": "failed", "errors": [*state.get("errors", []), error]}

    start, end = _date_range(state)
    try:
        price_history = load_price_history(ticker, start, end)
    except Exception as exc:  # noqa: BLE001 - degrade instead of failing the run
        return _degraded(state, ticker, f"백테스트용 가격 데이터를 가져오지 못했습니다: {exc}")

    params = BacktestParams.from_dict(state.get("backtest_params") or {})
    try:
        result = run_backtest(price_history, params, initial_balance=DEFAULT_INITIAL_BALANCE)
    except Exception as exc:  # noqa: BLE001 - degrade instead of failing the run
        return _degraded(state, ticker, f"백테스트 시뮬레이션 실행에 실패했습니다: {exc}")

    if result.profit_pct == -100.0 and result.trade_count == 0 and result.enriched.empty:
        note = "백테스트 구간의 가격 데이터가 부족해 시뮬레이션을 수행하지 못했습니다."
        return _degraded(state, ticker, note)

    bullish = sum(1 for div in result.divergences if div[2] == "bullish")
    bearish = sum(1 for div in result.divergences if div[2] == "bearish")
    evidence = _build_evidence(ticker, result, start, end)
    analysis = BacktestAnalysis(
        ticker=ticker,
        summary=(
            "고정된 기술적 규칙(커널 회귀 + RSI 다이버전스 + 볼린저 밴드)을 과거 가격에 적용한 "
            "시뮬레이션 참고 정보입니다. 특정 매수·매도·보유 판단을 지시하지 않습니다."
        ),
        period_summary=(
            f"시뮬레이션 구간: {start} ~ {end}, 사용된 거래일 {len(result.enriched)}일."
        ),
        performance_summary=(
            f"과거 시뮬레이션 누적 수익률은 {result.profit_pct:.2f}%이며 "
            f"규칙 기반 매매는 {result.trade_count}회 발생했습니다. {NO_ADVICE_NOTE}"
        ),
        signal_summary=(
            f"RSI 다이버전스 신호는 총 {len(result.divergences)}건"
            f"(강세 {bullish}건, 약세 {bearish}건) 확인되었습니다. "
            "신호는 과거 패턴 설명을 위한 기술적 참고 지표입니다."
        ),
        evidence=evidence,
        missing_data_notes=[NO_ADVICE_NOTE],
    )
    return {
        "status": "success",
        "backtest_analysis": analysis,
        "evidence": [*state.get("evidence", []), *evidence],
    }
