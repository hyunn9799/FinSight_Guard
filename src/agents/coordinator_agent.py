"""Coordinator agent for report drafting."""

from datetime import date

from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import GraphState, ResearchReport, WorkflowError
from src.safety.forbidden_phrases import REQUIRED_DISCLAIMER


COORDINATOR_NODE = "coordinator_agent"


def _ticker_from_state(state: GraphState) -> str:
    ticker = state.get("ticker")
    if ticker:
        return ticker.strip().upper()

    user_input = state.get("user_input")
    if user_input is not None:
        return user_input.ticker.strip().upper()

    return ""


def _data_date(evidence: list[EvidenceItem]) -> date:
    if not evidence:
        return date.today()
    return max(item.collected_at.date() for item in evidence)


def _join_nonempty(parts: list[str], fallback: str) -> str:
    clean_parts = [part.strip() for part in parts if part and part.strip()]
    return "\n".join(clean_parts) if clean_parts else fallback


def _format_notes(notes: list[str]) -> str:
    if not notes:
        return ""
    return "데이터 유의사항: " + " ".join(notes)


def _collect_evidence(state: GraphState) -> list[EvidenceItem]:
    evidence = list(state.get("evidence", []))
    for analysis_key in ("market_analysis", "fundamental_analysis", "news_analysis"):
        analysis = state.get(analysis_key)
        if analysis is not None:
            evidence.extend(analysis.evidence)

    seen_ids: set[str] = set()
    unique_evidence: list[EvidenceItem] = []
    for item in evidence:
        if item.evidence_id in seen_ids:
            continue
        seen_ids.add(item.evidence_id)
        unique_evidence.append(item)
    return unique_evidence


def _evidence_summary(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return (
            "현재 연결된 EvidenceItem이 없어 근거 요약이 제한됩니다. "
            "시장, 재무, 뉴스 데이터 수집 상태를 확인할 필요가 있습니다."
        )

    lines = []
    for index, item in enumerate(evidence[:10], start=1):
        value = "" if item.metric_value is None else f" 값={item.metric_value}"
        lines.append(
            f"{index}. [{item.source_type}] {item.metric_name}{value} - {item.description}"
        )
    if len(evidence) > 10:
        lines.append(f"추가 EvidenceItem {len(evidence) - 10}건은 내부 평가에 참고할 수 있습니다.")
    return "\n".join(lines)


def _market_section(state: GraphState) -> str:
    market = state.get("market_analysis")
    if market is None:
        return "시장 분석 데이터가 없어 해당 섹션은 제한적으로 검토할 수 있습니다."
    return _join_nonempty(
        [
            market.summary,
            market.trend_summary,
            market.momentum_summary,
            market.volatility_summary,
            _format_notes(market.missing_data_notes),
        ],
        "시장 분석 결과가 비어 있어 추가 데이터 확인이 필요합니다.",
    )


def _fundamental_section(state: GraphState) -> str:
    fundamental = state.get("fundamental_analysis")
    if fundamental is None:
        return "펀더멘털 분석 데이터가 없어 해당 섹션은 제한적으로 검토할 수 있습니다."
    return _join_nonempty(
        [
            fundamental.summary,
            fundamental.valuation_summary,
            fundamental.profitability_summary,
            fundamental.stability_summary,
            _format_notes(fundamental.missing_data_notes),
        ],
        "펀더멘털 분석 결과가 비어 있어 추가 데이터 확인이 필요합니다.",
    )


def _news_section(state: GraphState) -> str:
    news = state.get("news_analysis")
    if news is None:
        return "뉴스 분석 데이터가 없어 해당 섹션은 제한적으로 검토할 수 있습니다."

    positives = "긍정 요인: " + "; ".join(news.positive_factors) if news.positive_factors else ""
    negatives = "부정 요인: " + "; ".join(news.negative_factors) if news.negative_factors else ""
    risks = "이벤트 리스크: " + "; ".join(news.event_risks) if news.event_risks else ""
    return _join_nonempty(
        [news.summary, positives, negatives, risks, _format_notes(news.missing_data_notes)],
        "뉴스 분석 결과가 비어 있어 추가 데이터 확인이 필요합니다.",
    )


def _scenario_analysis(ticker: str) -> str:
    return "\n".join(
        [
            (
                "1. 관망 시나리오: 현재 확인된 시장, 재무, 뉴스 근거를 바탕으로 "
                f"{ticker}의 추가 데이터와 이벤트를 계속 점검하는 시나리오입니다. "
                "가격 변동성과 실적 관련 정보가 갱신될 때까지 참고할 수 있습니다."
            ),
            (
                "2. 분할 접근 시나리오: 단일 판단에 의존하지 않고 여러 데이터 시점의 "
                "근거를 나누어 검토할 수 있습니다. 이는 실행 지시가 아니라 변동성 관리 관점의 "
                "검토 시나리오입니다."
            ),
            (
                "3. 리스크 회피 시나리오: 근거 부족, 변동성 확대, 부정적 이벤트가 확인될 경우 "
                "노출 수준과 의사결정 시점을 보수적으로 검토할 수 있습니다. 이 역시 투자 행동을 "
                "권유하지 않는 리스크 점검 시나리오입니다."
            ),
        ]
    )


def _risk_factors(state: GraphState) -> str:
    notes: list[str] = []
    for analysis_key in ("market_analysis", "fundamental_analysis", "news_analysis"):
        analysis = state.get(analysis_key)
        if analysis is not None:
            notes.extend(analysis.missing_data_notes)

    base = [
        "시장 가격 변동성, 유동성 변화, 실적 발표, 금리와 거시 환경 변화가 결과 해석에 영향을 줄 수 있습니다.",
        "뉴스 이벤트는 출처와 시점에 따라 해석이 달라질 수 있으므로 후속 확인이 필요합니다.",
    ]
    if notes:
        base.append("누락 또는 제한 데이터: " + " ".join(notes))
    return "\n".join(base)


def _limitations(state: GraphState) -> str:
    warnings = state.get("warnings", [])
    parts = [
        "본 보고서는 공개 데이터와 현재 워크플로우에서 수집된 EvidenceItem에 기반한 교육용 분석입니다.",
        "데이터 지연, 누락 필드, mock 뉴스 fallback, 단순 키워드 분류로 인해 실제 상황을 완전히 반영하지 못할 수 있습니다.",
    ]
    if warnings:
        parts.append("워크플로우 경고: " + " ".join(warnings))
    return "\n".join(parts)


def run_coordinator_agent(state: GraphState) -> dict:
    """Combine agent outputs into a Korean scenario-based research report."""
    ticker = _ticker_from_state(state)
    if not ticker:
        error = WorkflowError(
            node=COORDINATOR_NODE,
            message="Ticker is missing for report coordination.",
            recoverable=False,
        )
        return {"status": "failed", "errors": [*state.get("errors", []), error]}

    user_input = state.get("user_input")
    user_context = ""
    if user_input is not None and user_input.user_query:
        user_context = f"사용자 질의 관점: {user_input.user_query.strip()}"

    evidence = _collect_evidence(state)
    report = ResearchReport(
        title=f"{ticker} 증거 기반 AI 리서치 보고서",
        ticker=ticker,
        data_date=_data_date(evidence),
        executive_summary=_join_nonempty(
            [
                (
                    f"{ticker}에 대해 시장, 펀더멘털, 뉴스 근거를 종합해 "
                    "관망, 분할 접근, 리스크 회피 관점의 시나리오를 검토할 수 있습니다."
                ),
                user_context,
                "본 보고서는 특정 투자 행동을 권유하지 않고, 확인 가능한 근거와 한계를 함께 제시합니다.",
            ],
            f"{ticker}에 대한 증거 기반 요약입니다.",
        ),
        market_section=_market_section(state),
        fundamental_section=_fundamental_section(state),
        news_section=_news_section(state),
        scenario_analysis=_scenario_analysis(ticker),
        risk_factors=_risk_factors(state),
        limitations=_limitations(state),
        evidence_summary=_evidence_summary(evidence),
        disclaimer=REQUIRED_DISCLAIMER,
    )
    return {
        "status": "success",
        "draft_report": report,
    }
