"""Coordinator agent for report drafting."""

from datetime import date

from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import GraphState, ResearchReport, WorkflowError
from src.graph_rag.graph_schema import GraphContext, GraphEdge
from src.safety.forbidden_phrases import REQUIRED_DISCLAIMER


COORDINATOR_NODE = "coordinator_agent"
AGENT_LABELS = {
    "market": "시장 분석",
    "fundamental": "재무 분석",
    "news": "뉴스 분석",
}
QUESTION_TYPE_LABELS = {
    "technical_analysis": "기술적 분석",
    "fundamental_analysis": "펀더멘털 분석",
    "news_risk_analysis": "뉴스/리스크 분석",
    "comprehensive_report": "종합 리서치",
    "safety_or_unclear": "안전 우선 검토",
}


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


def _evidence_ref(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return ""
    return f" [근거: {evidence[0].evidence_id}]"


def _analysis_evidence(state: GraphState, analysis_key: str) -> list[EvidenceItem]:
    analysis = state.get(analysis_key)
    if analysis is None:
        return []
    return analysis.evidence


def _supervisor_plan(state: GraphState):
    return state.get("supervisor_plan")


def _skipped_agents(state: GraphState) -> list[str]:
    supervisor_plan = _supervisor_plan(state)
    skipped = list(state.get("skipped_agents", []))
    if supervisor_plan is not None:
        skipped.extend(supervisor_plan.skipped_agents)
    return _dedupe_strings(skipped)


def _failed_agents(state: GraphState) -> list[str]:
    supervisor_plan = _supervisor_plan(state)
    failed = list(state.get("failed_agents", []))
    if supervisor_plan is not None:
        failed.extend(supervisor_plan.failed_agents)
    return _dedupe_strings(failed)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _scope_limitation_for_agent(state: GraphState, agent: str) -> str | None:
    if agent not in _skipped_agents(state):
        return None
    plan = _supervisor_plan(state)
    question_label = (
        QUESTION_TYPE_LABELS.get(plan.question_type, "선택 분석")
        if plan is not None and plan.question_type is not None
        else "선택 분석"
    )
    agent_label = AGENT_LABELS.get(agent, agent)
    return (
        f"이번 질문은 {question_label} 중심으로 분류되어 "
        f"{agent_label}은 상세 실행 대상에서 제외되었습니다."
    )


def _failure_note_for_agent(state: GraphState, agent: str) -> str | None:
    if agent not in _failed_agents(state):
        return None
    agent_label = AGENT_LABELS.get(agent, agent)
    return f"{agent_label} 데이터 수집이 제한되어 해당 섹션은 제한적으로 해석해야 합니다."


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
            "사용 가능한 근거 데이터가 제한적입니다. "
            "현재 연결된 EvidenceItem이 없어 핵심 주장과 수치의 출처 검증이 제한됩니다."
        )

    lines = []
    for index, item in enumerate(evidence[:10], start=1):
        value = "없음" if item.metric_value is None else str(item.metric_value)
        description = item.description.strip()
        if len(description) > 120:
            description = description[:117] + "..."
        lines.append(
            f"{index}. evidence_id={item.evidence_id}; source_type={item.source_type}; "
            f"metric_name={item.metric_name}; metric_value={value}; description={description}"
        )
    if len(evidence) > 10:
        lines.append(f"추가 EvidenceItem {len(evidence) - 10}건은 내부 평가에 참고할 수 있습니다.")
    return "\n".join(lines)


def _market_section(state: GraphState) -> str:
    scope_note = _scope_limitation_for_agent(state, "market")
    if scope_note:
        return scope_note
    failure_note = _failure_note_for_agent(state, "market")
    if failure_note:
        return failure_note
    market = state.get("market_analysis")
    if market is None:
        return "시장 분석 데이터가 없어 해당 섹션은 제한적으로 검토할 수 있습니다."
    evidence_ref = _evidence_ref(_analysis_evidence(state, "market_analysis"))
    return _join_nonempty(
        [
            market.summary + evidence_ref if market.summary else "",
            market.trend_summary + evidence_ref if market.trend_summary else "",
            market.momentum_summary + evidence_ref if market.momentum_summary else "",
            market.volatility_summary + evidence_ref if market.volatility_summary else "",
            _format_notes(market.missing_data_notes),
        ],
        "시장 분석 결과가 비어 있어 추가 데이터 확인이 필요합니다.",
    )


def _fundamental_section(state: GraphState) -> str:
    scope_note = _scope_limitation_for_agent(state, "fundamental")
    if scope_note:
        return scope_note
    failure_note = _failure_note_for_agent(state, "fundamental")
    if failure_note:
        return failure_note
    fundamental = state.get("fundamental_analysis")
    if fundamental is None:
        return "펀더멘털 분석 데이터가 없어 해당 섹션은 제한적으로 검토할 수 있습니다."
    evidence_ref = _evidence_ref(_analysis_evidence(state, "fundamental_analysis"))
    return _join_nonempty(
        [
            fundamental.summary + evidence_ref if fundamental.summary else "",
            fundamental.valuation_summary + evidence_ref if fundamental.valuation_summary else "",
            fundamental.profitability_summary + evidence_ref if fundamental.profitability_summary else "",
            fundamental.stability_summary + evidence_ref if fundamental.stability_summary else "",
            _format_notes(fundamental.missing_data_notes),
        ],
        "펀더멘털 분석 결과가 비어 있어 추가 데이터 확인이 필요합니다.",
    )


def _news_section(state: GraphState) -> str:
    scope_note = _scope_limitation_for_agent(state, "news")
    if scope_note:
        return scope_note
    failure_note = _failure_note_for_agent(state, "news")
    if failure_note:
        return failure_note
    news = state.get("news_analysis")
    if news is None:
        return "뉴스 분석 데이터가 없어 해당 섹션은 제한적으로 검토할 수 있습니다."

    evidence_ref = _evidence_ref(_analysis_evidence(state, "news_analysis"))
    positives = "긍정 요인: " + "; ".join(news.positive_factors) if news.positive_factors else ""
    negatives = "부정 요인: " + "; ".join(news.negative_factors) if news.negative_factors else ""
    risks = "이벤트 리스크: " + "; ".join(news.event_risks) if news.event_risks else ""
    return _join_nonempty(
        [
            news.summary + evidence_ref if news.summary else "",
            positives + evidence_ref if positives else "",
            negatives + evidence_ref if negatives else "",
            risks + evidence_ref if risks else "",
            _format_notes(news.missing_data_notes),
        ],
        "뉴스 분석 결과가 비어 있어 추가 데이터 확인이 필요합니다.",
    )


def _question_type(state: GraphState) -> str:
    plan = _supervisor_plan(state)
    if plan is None or plan.question_type is None:
        return "comprehensive_report"
    return plan.question_type


def _scenario_analysis(ticker: str, state: GraphState) -> str:
    question_type = _question_type(state)
    if question_type == "technical_analysis":
        emphasis = (
            "기술적 분석 중심 질문이므로 가격 추세, 모멘텀, 단기 변동성을 우선 참고할 수 있습니다. "
            "단기 신호는 빠르게 바뀔 수 있어 리스크를 확인해야 합니다."
        )
    elif question_type == "fundamental_analysis":
        emphasis = (
            "펀더멘털 분석 중심 질문이므로 밸류에이션, 수익성, 안정성, 장기 불확실성을 함께 검토할 수 있습니다."
        )
    elif question_type == "news_risk_analysis":
        emphasis = (
            "뉴스/리스크 중심 질문이므로 최근 이벤트, 시장 반응, 불확실성 확대 여부를 우선 참고할 수 있습니다."
        )
    elif question_type == "safety_or_unclear":
        emphasis = (
            "질문 의도가 직접 투자 판단에 가까울 수 있어 보수적인 언어로 한계와 리스크를 먼저 확인해야 합니다."
        )
    else:
        emphasis = "시장, 재무, 뉴스 근거를 균형 있게 참고할 수 있습니다."

    return "\n".join(
        [
            emphasis,
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
    graph_context = state.get("graph_context")
    if graph_context is not None and graph_context.risk_relations:
        evidence_ref = ""
        if graph_context.risk_relations[0].evidence_id:
            evidence_ref = f" [근거: {graph_context.risk_relations[0].evidence_id}]"
        base.append(
            f"관계 그래프에서 리스크 관계 {len(graph_context.risk_relations)}건이 확인되어 세부 근거를 함께 참고할 수 있습니다."
            f"{evidence_ref}"
        )
    if notes:
        base.append("누락 또는 제한 데이터: " + " ".join(notes))
    return "\n".join(base)


def _limitations(state: GraphState) -> str:
    warnings = state.get("warnings", [])
    evidence = _collect_evidence(state)
    supervisor_plan = _supervisor_plan(state)
    skipped_notes = [
        note
        for agent in _skipped_agents(state)
        for note in [_scope_limitation_for_agent(state, agent)]
        if note
    ]
    failed_notes = [
        note
        for agent in _failed_agents(state)
        for note in [_failure_note_for_agent(state, agent)]
        if note
    ]
    parts = [
        "본 보고서는 공개 데이터와 현재 워크플로우에서 수집된 EvidenceItem에 기반한 교육용 분석입니다.",
        "데이터 지연, 누락 필드, mock 뉴스 fallback, 단순 키워드 분류로 인해 실제 상황을 완전히 반영하지 못할 수 있습니다.",
    ]
    if not evidence:
        parts.append("사용 가능한 근거 데이터가 제한적입니다.")
    if supervisor_plan is not None and supervisor_plan.execution_mode == "selective":
        parts.append("Supervisor 계획에 따라 일부 분석만 수행되어 근거 범위가 제한적입니다.")
    parts.extend(skipped_notes)
    parts.extend(failed_notes)
    if warnings:
        parts.append("워크플로우 경고: " + " ".join(warnings))
    return "\n".join(parts)


def _edge_label(edge: GraphEdge) -> str:
    evidence = f", evidence_id={edge.evidence_id}" if edge.evidence_id else ""
    description = f" - {edge.description}" if edge.description else ""
    return f"{edge.source_id} -> {edge.target_id}: {edge.relation_type}{evidence}{description}"


def _graph_context_section(graph_context: GraphContext | None) -> str:
    title = "관계 기반 리스크 및 근거 요약"
    if graph_context is None:
        return (
            f"{title}\n"
            "관계 그래프 컨텍스트가 생성되지 않아 EvidenceItem 기반 관계 요약은 제한됩니다."
        )

    lines = [title]
    if graph_context.key_relations_summary:
        lines.append("핵심 관계:")
        lines.extend(f"- {summary}" for summary in graph_context.key_relations_summary[:5])
    else:
        lines.append("핵심 관계: 현재 사용 가능한 EvidenceItem만으로는 별도 관계 요약이 제한됩니다.")

    if graph_context.risk_relations:
        lines.append("리스크 관계:")
        lines.extend(f"- {_edge_label(edge)}" for edge in graph_context.risk_relations[:5])
    else:
        lines.append("리스크 관계: 명시적으로 추출된 리스크 관계가 없습니다.")

    if graph_context.positive_relations:
        lines.append("긍정 관계:")
        lines.extend(f"- {_edge_label(edge)}" for edge in graph_context.positive_relations[:5])
    else:
        lines.append("긍정 관계: 명시적으로 추출된 긍정 관계가 없습니다.")

    if graph_context.evidence_ids:
        lines.append("연결 EvidenceItem ID: " + ", ".join(graph_context.evidence_ids[:10]))
    else:
        lines.append("연결 EvidenceItem ID: 없음")
    return "\n".join(lines)


def _available_evidence_ids(evidence: list[EvidenceItem], graph_context: GraphContext | None) -> set[str]:
    evidence_ids = {item.evidence_id for item in evidence if item.evidence_id}
    if graph_context is not None:
        evidence_ids.update(evidence_id for evidence_id in graph_context.evidence_ids if evidence_id)
    return evidence_ids


def _traceability_note(evidence: list[EvidenceItem], graph_context: GraphContext | None) -> str:
    evidence_ids = sorted(_available_evidence_ids(evidence, graph_context))
    if not evidence_ids:
        return "보고서 본문에 연결할 수 있는 EvidenceItem ID가 없습니다."
    return "보고서 본문은 사용 가능한 EvidenceItem ID만 근거로 표시합니다: " + ", ".join(evidence_ids)


def _combined_evidence_summary(evidence: list[EvidenceItem], graph_context: GraphContext | None) -> str:
    parts = [_evidence_summary(evidence), _traceability_note(evidence, graph_context)]
    if graph_context is not None:
        parts.append(_graph_context_section(graph_context))
    return "\n\n".join(parts)


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
    graph_context = state.get("graph_context")
    graph_context_section = _graph_context_section(graph_context)
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
        graph_context_section=graph_context_section,
        scenario_analysis=_scenario_analysis(ticker, state),
        risk_factors=_risk_factors(state),
        limitations=_limitations(state),
        evidence_summary=_combined_evidence_summary(evidence, graph_context),
        disclaimer=REQUIRED_DISCLAIMER,
    )
    return {
        "status": "success",
        "draft_report": report,
    }
