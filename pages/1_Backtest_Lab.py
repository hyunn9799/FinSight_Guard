"""Streamlit page: interactive strategy backtest lab (historical simulation).

This page exposes the ported kernel-regression + RSI-divergence + Bollinger
strategy for manual parameter tuning, charting, and Optuna optimization. Every
result describes a *past simulation* of a fixed rule set and is never a buy/sell/
hold recommendation or a forward-looking performance promise.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.backtest.charts import build_backtest_figure, configure_korean_font
from src.backtest.data_loader import load_price_history
from src.backtest.optimizer import SearchSpace, optimize_backtest
from src.backtest.strategy import BacktestParams, run_backtest

INITIAL_BALANCE_DEFAULT = 10_000
FEE = 0.001

STOCK_PRESETS = {
    "미국 주식": ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "NVDA", "NFLX"],
    "암호화폐": ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD"],
    "지수/상품": ["SPY", "QQQ", "GLD"],
    "한국 주식": ["005930.KS", "000660.KS", "035420.KS", "051910.KS"],
}


def _render_result(result, ticker: str, params: BacktestParams) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("최종 평가금액 (USD)", f"{result.final_value:,.2f}")
    col2.metric("누적 수익률 (과거 시뮬레이션)", f"{result.profit_pct:.2f}%")
    col3.metric("규칙 기반 매매 횟수", f"{result.trade_count}")

    if not result.trades.empty:
        st.subheader("거래 내역 (시뮬레이션)")
        st.dataframe(result.trades, use_container_width=True)
    else:
        st.info("이 구간에서는 규칙 조건을 만족하는 매매가 발생하지 않았습니다.")

    st.subheader("차트 (가격 · 신호 · RSI)")
    fig = build_backtest_figure(
        result.enriched,
        result.divergences,
        ticker=ticker,
        rsi_oversold=params.rsi_oversold,
        rsi_overbought=params.rsi_overbought,
    )
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="전략 백테스트 Lab", layout="wide")
    configure_korean_font()

    st.title("전략 백테스트 Lab (과거 시뮬레이션)")
    st.warning(
        "이 페이지의 모든 결과는 고정된 기술적 규칙을 과거 가격에 적용한 시뮬레이션입니다. "
        "미래 수익을 보장하지 않으며, 특정 종목의 매수·매도·보유를 권유하지 않습니다."
    )

    with st.sidebar:
        st.header("백테스트 입력")
        category = st.selectbox("카테고리", list(STOCK_PRESETS.keys()))
        preset = st.selectbox("종목", STOCK_PRESETS[category])
        ticker = st.text_input("티커 직접 입력", value=preset).strip().upper()
        start_date = st.date_input("시작일", pd.to_datetime("2024-01-01"))
        end_date = st.date_input("종료일", pd.to_datetime("2025-08-26"))
        initial_balance = st.number_input(
            "시작 자본금 (USD)", min_value=100, value=INITIAL_BALANCE_DEFAULT, step=100
        )

        st.subheader("전략 파라미터")
        kr_window = st.slider("커널 회귀 윈도우 (일)", 10, 100, 50, 1)
        kr_bandwidth = st.slider("커널 회귀 대역폭", 0.1, 10.0, 5.0, 0.1)
        bb_k = st.slider("볼린저 밴드 k", 0.1, 3.0, 0.7, 0.1)
        rsi_period = st.slider("RSI 기간 (일)", 5, 30, 14, 1)
        extrema_order = st.slider("다이버전스 감지 오더", 1, 10, 5, 1)
        rsi_oversold = st.slider("RSI 과매도", 10, 40, 30, 1)
        rsi_overbought = st.slider("RSI 과매수", 60, 90, 70, 1)

    manual_params = BacktestParams(
        rsi_period=rsi_period,
        kr_window=kr_window,
        kr_bandwidth=kr_bandwidth,
        bb_k=bb_k,
        extrema_order=extrema_order,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
    )

    if not ticker:
        st.info("사이드바에서 티커를 입력하세요.")
        return

    @st.cache_data(show_spinner=False)
    def _load(symbol: str, start, end) -> pd.DataFrame:
        return load_price_history(symbol, start, end)

    try:
        price_history = _load(ticker, start_date, end_date)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        st.error(f"가격 데이터를 불러오지 못했습니다: {exc}")
        return

    st.caption(f"{ticker} · {start_date} ~ {end_date} · 거래일 {len(price_history)}일")

    tab_manual, tab_optuna = st.tabs(["수동 백테스트", "Optuna 최적화"])

    with tab_manual:
        st.subheader("수동 파라미터 백테스트")
        if st.button("백테스트 실행", type="primary"):
            with st.spinner("시뮬레이션 실행 중..."):
                result = run_backtest(price_history, manual_params, float(initial_balance), FEE)
            _render_result(result, ticker, manual_params)

    with tab_optuna:
        st.subheader("Optuna 베이지안 최적화")
        st.write(
            "과거 시뮬레이션 수익률을 최대화하는 파라미터 조합을 탐색합니다. "
            "과적합 가능성이 있으므로 결과는 참고용입니다."
        )
        n_trials = st.number_input("최적화 시도 횟수", min_value=10, value=50, step=10)
        if st.button(f"최적화 시작 ({n_trials}회)"):
            progress = st.progress(0)
            status = st.empty()

            def _on_progress(done: int, total: int) -> None:
                progress.progress(done / total)
                status.info(f"{done} / {total} 시도 완료")

            with st.spinner("최적화 진행 중..."):
                opt = optimize_backtest(
                    price_history,
                    initial_balance=float(initial_balance),
                    fee=FEE,
                    n_trials=int(n_trials),
                    search_space=SearchSpace(),
                    progress_callback=_on_progress,
                )
            progress.empty()
            status.empty()
            st.success("최적화 완료")

            st.write(f"최대 과거 시뮬레이션 수익률: **{opt.best_profit_pct:.2f}%**")
            st.json(opt.best_params)

            st.subheader("최적 파라미터 백테스트")
            best_params = BacktestParams.from_dict(opt.best_params)
            with st.spinner("최적 파라미터로 재실행 중..."):
                best_result = run_backtest(
                    price_history, best_params, float(initial_balance), FEE
                )
            _render_result(best_result, ticker, best_params)


main()
