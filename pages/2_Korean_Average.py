"""Streamlit page: average historical-simulation return across Korean stocks.

Ported from the standalone project's "average profit calculator" page. Runs a
fixed parameter set over a basket of Korean tickers and reports the average
historical-simulation return. Results are past simulations for comparison only,
never investment advice or a forward-looking guarantee.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.backtest.batch import run_batch_backtest
from src.backtest.charts import configure_korean_font
from src.backtest.strategy import BacktestParams

FEE = 0.001
INITIAL_BALANCE_DEFAULT = 10_000

# (ticker, 표시 이름). 넥슨게임즈는 yfinance 데이터가 없어 제외합니다.
KOREAN_STOCKS = [
    ("005930.KS", "삼성전자"),
    ("000250.KQ", "삼천당제약"),
    ("036570.KS", "엔씨소프트"),
    ("051910.KS", "LG화학"),
    ("066700.KQ", "테라젠이텍스"),
    ("066970.KQ", "엘앤에프"),
    ("068270.KQ", "셀트리온제약"),
    ("078600.KQ", "대주전자재료"),
    ("096770.KS", "SK이노베이션"),
    ("128940.KQ", "한미약품"),
    ("185750.KQ", "종근당"),
    ("192080.KQ", "더블유게임즈"),
    ("207940.KS", "삼성바이오로직스"),
    ("251270.KQ", "넷마블"),
    ("293490.KQ", "카카오게임즈"),
    ("373220.KS", "LG에너지솔루션"),
    ("247540.KQ", "에코프로비엠"),
    ("225010.KQ", "넥슨게임즈"),
]

FIXED_PARAMS = BacktestParams(
    kr_window=50,
    kr_bandwidth=5.0,
    bb_k=0.7,
    rsi_period=14,
    extrema_order=5,
    rsi_oversold=30,
    rsi_overbought=70,
)


def main() -> None:
    st.set_page_config(page_title="한국 주식 평균 수익률", layout="wide")
    configure_korean_font()

    st.title("한국 주식 평균 수익률 (과거 시뮬레이션)")
    st.warning(
        "아래 결과는 고정된 기술적 규칙을 과거 가격에 적용한 시뮬레이션의 평균입니다. "
        "미래 수익을 보장하지 않으며, 특정 종목의 매수·매도·보유를 권유하지 않습니다."
    )

    with st.sidebar:
        st.header("배치 설정")
        start_date = st.date_input("시작일", pd.to_datetime("1990-01-01"))
        end_date = st.date_input("종료일", pd.to_datetime("2023-05-31"))
        initial_balance = st.number_input(
            "시작 자본금 (USD)", min_value=100, value=INITIAL_BALANCE_DEFAULT, step=100
        )

    st.subheader("고정 분석 파라미터")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"- 커널 회귀 윈도우: **{FIXED_PARAMS.kr_window}일**\n"
            f"- 커널 회귀 대역폭: **{FIXED_PARAMS.kr_bandwidth}**\n"
            f"- 볼린저 밴드 k: **{FIXED_PARAMS.bb_k}**\n"
            f"- RSI 기간: **{FIXED_PARAMS.rsi_period}일**"
        )
    with col2:
        st.markdown(
            f"- 다이버전스 감지 오더: **{FIXED_PARAMS.extrema_order}**\n"
            f"- RSI 과매도: **{FIXED_PARAMS.rsi_oversold}**\n"
            f"- RSI 과매수: **{FIXED_PARAMS.rsi_overbought}**"
        )

    st.subheader(f"분석 대상 종목 ({len(KOREAN_STOCKS)}개)")
    st.caption(", ".join(label for _, label in KOREAN_STOCKS))

    if not st.button("평균 수익률 계산 시작", type="primary"):
        return

    progress = st.progress(0)
    status = st.empty()

    def _on_progress(done: int, total: int, item) -> None:
        progress.progress(done / total)
        if item.status == "success":
            status.info(f"[{done}/{total}] {item.label} ({item.ticker}) 수익률 {item.profit_pct:.2f}%")
        else:
            status.warning(f"[{done}/{total}] {item.label} ({item.ticker}) — {item.note}")

    with st.spinner("모든 종목의 과거 시뮬레이션을 실행 중입니다..."):
        batch = run_batch_backtest(
            KOREAN_STOCKS,
            start=start_date,
            end=end_date,
            params=FIXED_PARAMS,
            initial_balance=float(initial_balance),
            fee=FEE,
            progress_callback=_on_progress,
        )
    progress.empty()
    status.empty()

    if batch.average_profit_pct is None:
        st.warning("계산 가능한 종목 데이터가 없습니다. 기간을 조정하고 다시 시도하세요.")
        return

    st.subheader("최종 결과")
    st.success(
        f"성공 {batch.successful_count}개 종목 기준 평균 과거 시뮬레이션 수익률: "
        f"**{batch.average_profit_pct:.2f}%**"
    )

    table = batch.as_dataframe()
    if not table.empty:
        st.dataframe(table, use_container_width=True)
        st.bar_chart(table.set_index("종목")["수익률 (%)"])

    skipped = [item for item in batch.results if item.status != "success"]
    if skipped:
        with st.expander(f"제외된 종목 ({len(skipped)}개)"):
            for item in skipped:
                st.write(f"- {item.label} ({item.ticker}): {item.note}")


main()
