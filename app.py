"""Streamlit entrypoint for the financial research workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.graph.workflow import run_research_workflow


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _render_text_block(title: str, text: Any) -> None:
    st.subheader(title)
    if text:
        if isinstance(text, list):
            for item in text:
                st.write(f"- {item}")
        else:
            st.markdown(str(text))
    else:
        st.info("표시할 데이터가 없습니다.")


def _render_analysis(analysis: Any, fields: list[tuple[str, str]]) -> None:
    data = _as_dict(analysis)
    if not data:
        st.info("분석 결과가 없습니다. 워크플로우를 실행하거나 경고 메시지를 확인하세요.")
        return

    for label, key in fields:
        value = data.get(key)
        if value:
            _render_text_block(label, value)

    notes = data.get("missing_data_notes") or []
    if notes:
        st.warning("데이터 유의사항")
        for note in notes:
            st.write(f"- {note}")

    evidence = data.get("evidence") or []
    if evidence:
        with st.expander("연결된 EvidenceItem"):
            st.dataframe(evidence, use_container_width=True)


def _render_report(report: Any, report_path: str | None) -> None:
    data = _as_dict(report)
    if not data:
        st.info("최종 보고서가 아직 생성되지 않았습니다.")
        return

    st.header(data.get("title", "종합 보고서"))
    st.caption(f"Ticker: {data.get('ticker', '-')} | Data date: {data.get('data_date', '-')}")

    sections = [
        ("요약", "executive_summary"),
        ("시장 분석", "market_section"),
        ("재무 분석", "fundamental_section"),
        ("뉴스 분석", "news_section"),
        ("시나리오 분석", "scenario_analysis"),
        ("리스크", "risk_factors"),
        ("한계", "limitations"),
        ("근거 요약", "evidence_summary"),
    ]
    for title, key in sections:
        value = data.get(key)
        if value:
            _render_text_block(title, value)

    disclaimer = data.get("disclaimer")
    if disclaimer:
        st.warning(disclaimer)

    if report_path:
        st.success(f"저장된 보고서 경로: {report_path}")


def _render_evaluator(evaluation: Any) -> None:
    data = _as_dict(evaluation)
    if not data:
        st.info("Evaluator 검수 결과가 없습니다.")
        return

    passed = data.get("overall_pass") is True
    st.success("PASS") if passed else st.error("FAIL")

    score_fields = [
        ("Source grounding", "source_grounding_score"),
        ("Numeric consistency", "numeric_consistency_score"),
        ("Safety", "safety_score"),
        ("Risk disclosure", "risk_disclosure_score"),
        ("Freshness", "freshness_score"),
    ]
    columns = st.columns(len(score_fields))
    for column, (label, key) in zip(columns, score_fields, strict=False):
        column.metric(label, f"{float(data.get(key, 0.0)):.2f}")

    issues = data.get("issues") or []
    suggestions = data.get("revision_suggestions") or []

    st.subheader("Issues")
    if issues:
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.write("검수 이슈가 없습니다.")

    st.subheader("Revision suggestions")
    if suggestions:
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
    else:
        st.write("추가 수정 제안이 없습니다.")


def _render_meta(result: dict[str, Any]) -> None:
    st.subheader("실행 메타정보")
    st.json(
        {
            "run_id": result.get("run_id"),
            "user_input": result.get("user_input"),
            "report_path": result.get("report_path"),
        }
    )

    errors = result.get("errors") or []
    st.subheader("오류")
    if errors:
        for error in errors:
            data = _as_dict(error)
            st.error(data.get("message", str(error)))
            with st.expander("오류 상세"):
                st.json(data or {"error": str(error)})
    else:
        st.write("기록된 오류가 없습니다.")

    log_path = Path("logs/app.log")
    st.subheader("로그 파일")
    if log_path.exists():
        st.code("\n".join(log_path.read_text(encoding="utf-8").splitlines()[-20:]), language="json")
    else:
        st.info("아직 로그 파일이 생성되지 않았습니다.")


def _render_intro() -> None:
    st.title("금융 리서치 Multi-Agent Workflow")
    st.markdown(
        """
        이 앱은 LangGraph 기반 역할 분리형 에이전트 워크플로우입니다.
        시장 데이터, 펀더멘털 데이터, 뉴스 근거를 수집하고 Evaluator Agent가 안전성과 근거성을 검수합니다.
        """
    )
    st.info(
        "안전 고지: 이 시스템은 교육 및 정보 제공 목적의 리서치 보조 도구입니다. "
        "특정 종목의 매수, 매도, 보유를 권유하지 않으며 수익을 보장하지 않습니다."
    )
    st.markdown(
        """
        ```text
        START
          -> Input Validator
          -> Market Agent
          -> Fundamental Agent
          -> News Agent
          -> Coordinator Agent
          -> Evaluator Agent
             -> PASS: Save Report -> END
             -> FAIL: Rewrite Agent -> Evaluator Agent
             -> FAIL after max retries: Save Failed Report -> END
        ```
        """
    )


def main() -> None:
    """Run the Streamlit UI."""
    st.set_page_config(
        page_title="금융 리서치 Multi-Agent Workflow",
        layout="wide",
    )

    with st.sidebar:
        st.header("분석 입력")
        ticker = st.text_input("Ticker", value="AAPL", placeholder="예: AAPL, MSFT, NVDA")
        investment_horizon = st.selectbox("투자 기간", ["단기", "중기", "장기"], index=1)
        risk_profile = st.selectbox("위험 성향", ["보수형", "중립형", "공격형"], index=1)
        run_button = st.button("워크플로우 실행", type="primary", use_container_width=True)

    _render_intro()

    if run_button:
        try:
            with st.spinner("Multi-Agent 워크플로우를 실행 중입니다."):
                st.session_state["workflow_result"] = run_research_workflow(
                    ticker=ticker,
                    investment_horizon=investment_horizon,
                    risk_profile=risk_profile,
                )
        except Exception as exc:
            st.session_state["workflow_result"] = None
            st.error("워크플로우 실행 중 오류가 발생했습니다. 입력값과 로그를 확인해 주세요.")
            with st.expander("오류 상세"):
                st.exception(exc)

    result = st.session_state.get("workflow_result")
    if not result:
        st.info("사이드바에서 입력값을 선택한 뒤 워크플로우를 실행하세요.")
        return

    errors = result.get("errors") or []
    final_report = result.get("final_report")
    if final_report and not errors:
        st.success("워크플로우가 완료되었습니다.")
    elif final_report and errors:
        st.warning("워크플로우가 degraded mode로 완료되었습니다. 오류/경고 탭을 확인하세요.")
    else:
        st.error("워크플로우가 보고서 생성 전에 종료되었습니다. 오류/경고 탭을 확인하세요.")

    tabs = st.tabs(
        [
            "시장 분석",
            "재무 분석",
            "뉴스 분석",
            "종합 보고서",
            "Evaluator 검수",
            "실행 로그/메타정보",
        ]
    )

    with tabs[0]:
        _render_analysis(
            result.get("market_analysis"),
            [
                ("시장 요약", "summary"),
                ("추세", "trend_summary"),
                ("모멘텀", "momentum_summary"),
                ("변동성", "volatility_summary"),
            ],
        )
    with tabs[1]:
        _render_analysis(
            result.get("fundamental_analysis"),
            [
                ("재무 요약", "summary"),
                ("가치평가", "valuation_summary"),
                ("수익성", "profitability_summary"),
                ("안정성", "stability_summary"),
            ],
        )
    with tabs[2]:
        _render_analysis(
            result.get("news_analysis"),
            [
                ("뉴스 요약", "summary"),
                ("긍정 요인", "positive_factors"),
                ("부정 요인", "negative_factors"),
                ("이벤트 리스크", "event_risks"),
            ],
        )
    with tabs[3]:
        _render_report(result.get("final_report"), result.get("report_path"))
    with tabs[4]:
        _render_evaluator(result.get("evaluation_result"))
    with tabs[5]:
        _render_meta(result)


if __name__ == "__main__":
    main()
