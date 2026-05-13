"""Supervisor planning helpers for deterministic and optional LLM routing."""

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src import config
from src.graph.state import AgentName, GraphState, NextNode, QuestionType, SupervisorPlan


AGENT_ORDER: list[AgentName] = ["market", "fundamental", "news"]
QUESTION_TYPES: list[QuestionType] = [
    "technical_analysis",
    "fundamental_analysis",
    "news_risk_analysis",
    "comprehensive_report",
    "safety_or_unclear",
]
AGENT_ANALYSIS_KEYS: dict[AgentName, str] = {
    "market": "market_analysis",
    "fundamental": "fundamental_analysis",
    "news": "news_analysis",
}
QUESTION_TYPE_CONFIG: dict[QuestionType, dict[str, object]] = {
    "technical_analysis": {
        "planned_agent_order": ["market", "news"],
        "execution_mode": "selective",
        "risk_focus": False,
        "needs_graph_context": False,
        "routing_reason": "기술적/차트 중심 질문으로 Market Agent와 News Agent를 우선 실행합니다.",
    },
    "fundamental_analysis": {
        "planned_agent_order": ["fundamental", "news"],
        "execution_mode": "selective",
        "risk_focus": False,
        "needs_graph_context": True,
        "routing_reason": "재무/밸류에이션 중심 질문으로 Fundamental Agent와 News Agent를 실행합니다.",
    },
    "news_risk_analysis": {
        "planned_agent_order": ["news", "market", "fundamental"],
        "execution_mode": "selective",
        "risk_focus": True,
        "needs_graph_context": True,
        "routing_reason": "뉴스와 리스크 중심 질문으로 News Agent를 먼저 실행합니다.",
    },
    "comprehensive_report": {
        "planned_agent_order": ["market", "fundamental", "news"],
        "execution_mode": "full",
        "risk_focus": True,
        "needs_graph_context": True,
        "routing_reason": "종합 리서치 요청으로 모든 분석 Agent를 실행합니다.",
    },
    "safety_or_unclear": {
        "planned_agent_order": ["market", "fundamental", "news"],
        "execution_mode": "full",
        "risk_focus": True,
        "needs_graph_context": True,
        "routing_reason": "직접 투자 조언 또는 불명확한 질문으로 안전 검토를 위해 전체 분석을 실행합니다.",
    },
}
TECHNICAL_KEYWORDS = (
    "차트",
    "기술적",
    "단기",
    "진입",
    "추세",
    "rsi",
    "macd",
    "이동평균",
    "돌파",
    "지지",
    "저항",
)
FUNDAMENTAL_KEYWORDS = (
    "저평가",
    "재무",
    "per",
    "pbr",
    "roe",
    "실적",
    "장기",
    "펀더멘털",
    "밸류에이션",
    "이익",
    "현금흐름",
)
NEWS_RISK_KEYWORDS = (
    "뉴스",
    "악재",
    "호재",
    "리스크",
    "위험",
    "이슈",
    "규제",
    "소송",
    "환율",
    "최근",
    "공급망",
)
COMPREHENSIVE_KEYWORDS = (
    "종합",
    "전체",
    "분석해줘",
    "보고서",
    "리서치",
    "전반적으로",
)
DIRECT_ADVICE_PHRASES = (
    "무조건",
    "사야 해",
    "사야해",
    "팔아야 해",
    "팔아야해",
    "수익 보장",
    "손실 없음",
)
NODE_TO_AGENT: dict[NextNode, AgentName | None] = {
    "market_node": "market",
    "fundamental_node": "fundamental",
    "news_node": "news",
    "coordinator_node": None,
}
AGENT_ERROR_NODES: dict[AgentName, set[str]] = {
    "market": {"market_agent", "market_node"},
    "fundamental": {"fundamental_agent", "fundamental_node"},
    "news": {"news_agent", "news_node"},
}
AGENT_ERROR_TYPES: dict[AgentName, set[str]] = {
    "market": {"market_data_error"},
    "fundamental": {"fundamental_data_error"},
    "news": {"news_search_error"},
}
AGENT_ERROR_MARKERS: dict[AgentName, tuple[str, ...]] = {
    "market": ("market_agent", "market_node", "market analysis", "시장 가격 데이터"),
    "fundamental": (
        "fundamental_agent",
        "fundamental_node",
        "fundamental analysis",
        "펀더멘털 데이터",
    ),
    "news": ("news_agent", "news_node", "news analysis", "뉴스 검색"),
}


class LLMSupervisorRouting(BaseModel):
    """Strict schema for LLM routing output."""

    question_type: QuestionType
    planned_agent_order: list[AgentName] = Field(min_length=1)
    primary_agents: list[AgentName] = Field(default_factory=list)
    secondary_agents: list[AgentName] = Field(default_factory=list)
    execution_mode: str
    risk_focus: bool
    needs_graph_context: bool
    routing_reason: str = ""
    confidence: float = 0.0


def _dedupe_agents(agents: list[AgentName]) -> list[AgentName]:
    """Deduplicate agents in the default execution order."""
    seen = set(agents)
    return [agent for agent in AGENT_ORDER if agent in seen]


def _dedupe_agents_preserve_order(agents: list[AgentName]) -> list[AgentName]:
    """Deduplicate agents while preserving a custom planned order."""
    seen: set[AgentName] = set()
    deduped: list[AgentName] = []
    for agent in agents:
        if agent in seen:
            continue
        if agent not in AGENT_ORDER:
            continue
        seen.add(agent)
        deduped.append(agent)
    return deduped


def _question_text(user_input: Any) -> str:
    if user_input is None:
        return ""
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, dict):
        return str(user_input.get("user_query") or user_input.get("query") or "")
    return str(getattr(user_input, "user_query", "") or "")


def _is_generated_context(text: str) -> bool:
    clean_text = text.strip()
    return clean_text.startswith("투자기간:") and "위험성향:" in clean_text


def _keyword_score(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _contains_direct_advice(text: str) -> bool:
    return any(phrase in text for phrase in DIRECT_ADVICE_PHRASES)


def _clamp_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def classify_question_type(user_input: Any) -> QuestionType:
    """Classify the user question with deterministic Korean/English keyword rules."""
    text = _question_text(user_input).strip().lower()
    if not text:
        return "safety_or_unclear"
    if _is_generated_context(text):
        return "comprehensive_report"
    if _contains_direct_advice(text):
        return "safety_or_unclear"

    scores: list[tuple[QuestionType, int]] = [
        ("technical_analysis", _keyword_score(text, TECHNICAL_KEYWORDS)),
        ("fundamental_analysis", _keyword_score(text, FUNDAMENTAL_KEYWORDS)),
        ("news_risk_analysis", _keyword_score(text, NEWS_RISK_KEYWORDS)),
        ("comprehensive_report", _keyword_score(text, COMPREHENSIVE_KEYWORDS)),
    ]
    best_question_type, best_score = max(scores, key=lambda item: item[1])
    if best_score == 0:
        return "safety_or_unclear"
    return best_question_type


def _error_value(error: object, field: str) -> str:
    """Read a field from a WorkflowError model or dict-like test object."""
    if isinstance(error, dict):
        return str(error.get(field) or "")
    return str(getattr(error, field, "") or "")


def infer_completed_agents(state: GraphState) -> list[AgentName]:
    """Infer completed agents from existing analysis outputs and state metadata."""
    completed = list(state.get("completed_agents", []))
    for agent in AGENT_ORDER:
        if state.get(AGENT_ANALYSIS_KEYS[agent]) is not None:
            completed.append(agent)
    return _dedupe_agents(completed)


def infer_failed_agents(state: GraphState) -> list[AgentName]:
    """Infer failed agents from explicit state and clear agent-specific errors."""
    failed = list(state.get("failed_agents", []))
    for error in state.get("errors", []):
        node = _error_value(error, "node")
        error_type = _error_value(error, "error_type")
        message = _error_value(error, "message").lower()
        for agent in AGENT_ORDER:
            markers = tuple(marker.lower() for marker in AGENT_ERROR_MARKERS[agent])
            if (
                node in AGENT_ERROR_NODES[agent]
                or error_type in AGENT_ERROR_TYPES[agent]
                or any(marker in message for marker in markers)
            ):
                failed.append(agent)
    return _dedupe_agents(failed)


def determine_next_node(state: GraphState) -> NextNode:
    """Route through planned agents, then to coordinator."""
    completed_agents = infer_completed_agents(state)
    failed_agents = infer_failed_agents(state)
    supervisor_plan = state.get("supervisor_plan")
    planned_agent_order = (
        supervisor_plan.planned_agent_order
        if supervisor_plan is not None and supervisor_plan.planned_agent_order
        else list(AGENT_ORDER)
    )

    for agent in planned_agent_order:
        if agent not in completed_agents and agent not in failed_agents:
            return f"{agent}_node"  # type: ignore[return-value]
    return "coordinator_node"


def build_supervisor_plan(state: GraphState) -> SupervisorPlan:
    """Build a deterministic SupervisorPlan without external calls or LLM routing."""
    question_type = classify_question_type(state.get("user_input"))
    config = QUESTION_TYPE_CONFIG[question_type]
    planned_agent_order = _dedupe_agents_preserve_order(
        list(config["planned_agent_order"])  # type: ignore[arg-type]
    )
    completed_agents = infer_completed_agents(state)
    failed_agents = infer_failed_agents(state)
    unplanned_agents = [agent for agent in AGENT_ORDER if agent not in planned_agent_order]
    skipped_agents = _dedupe_agents(
        [
            *list(state.get("skipped_agents", [])),
            *[agent for agent in unplanned_agents if agent not in completed_agents],
        ]
    )
    state_with_plan: GraphState = {
        **state,
        "supervisor_plan": SupervisorPlan(
            next_node="coordinator_node",
            planned_agent_order=planned_agent_order,
        ),
    }
    next_node = determine_next_node(state_with_plan)
    next_agent = NODE_TO_AGENT[next_node]
    if next_agent is None:
        rationale = "계획된 분석이 모두 완료되어 Coordinator Agent로 이동합니다."
    elif next_agent == "market":
        rationale = "market_analysis가 없어 Market Agent를 먼저 실행합니다."
    elif next_agent == "fundamental":
        rationale = "fundamental_analysis가 없어 Fundamental Agent를 실행합니다."
    else:
        rationale = "news_analysis가 없어 News Agent를 실행합니다."

    return SupervisorPlan(
        required_agents=list(AGENT_ORDER),
        completed_agents=completed_agents,
        failed_agents=failed_agents,
        skipped_agents=skipped_agents,
        next_node=next_node,
        rationale=rationale,
        allow_degraded_report=True,
        question_type=question_type,
        execution_mode=config["execution_mode"],  # type: ignore[arg-type]
        risk_focus=bool(config["risk_focus"]),
        needs_graph_context=bool(config["needs_graph_context"]),
        routing_reason=str(config["routing_reason"]),
        planned_agent_order=planned_agent_order,
        confidence=1.0,
        used_llm=False,
        fallback_used=False,
    )


def build_llm_supervisor_prompt(
    user_input: Any,
    allowed_agents: list[AgentName],
    allowed_question_types: list[QuestionType],
) -> str:
    """Build a constrained prompt for LLM-only routing decisions."""
    question = _question_text(user_input)
    allowed_agents_text = ", ".join(allowed_agents)
    allowed_question_types_text = ", ".join(allowed_question_types)
    return (
        "You are a routing planner for a financial research multi-agent workflow.\n"
        "You do not provide investment advice.\n"
        "You only decide which analysis agents should run.\n"
        f"Allowed agents are: {allowed_agents_text}.\n"
        f"Allowed question types are: {allowed_question_types_text}.\n"
        "Return only valid JSON.\n"
        "Do not include markdown.\n"
        "Do not include explanations outside JSON.\n"
        "Do not create financial facts, metrics, source URLs, evidence, or buy/sell/hold recommendations.\n"
        "Expected JSON schema:\n"
        "{\n"
        '  "question_type": "...",\n'
        '  "planned_agent_order": ["..."],\n'
        '  "primary_agents": ["..."],\n'
        '  "secondary_agents": ["..."],\n'
        '  "execution_mode": "full|selective|degraded",\n'
        '  "risk_focus": true,\n'
        '  "needs_graph_context": true,\n'
        '  "routing_reason": "...",\n'
        '  "confidence": 0.0\n'
        "}\n"
        f"User input: {question}"
    )


def _fallback_plan(state: GraphState, warning: str | None = None) -> dict:
    plan = build_supervisor_plan(state).model_copy(
        update={
            "used_llm": False,
            "fallback_used": True,
        }
    )
    result = _plan_result(plan)
    if warning:
        result["warnings"] = [*state.get("warnings", []), warning]
    return result


def _plan_result(plan: SupervisorPlan) -> dict:
    next_agent = NODE_TO_AGENT[plan.next_node]
    return {
        "supervisor_plan": plan,
        "completed_agents": plan.completed_agents,
        "failed_agents": plan.failed_agents,
        "skipped_agents": plan.skipped_agents,
        "next_agent": next_agent,
    }


def parse_llm_supervisor_response(text: str, state: GraphState | None = None) -> SupervisorPlan:
    """Parse and validate LLM routing JSON into a SupervisorPlan."""
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM supervisor response must be a JSON object.")
    parsed = LLMSupervisorRouting.model_validate(payload)
    if parsed.execution_mode not in {"full", "selective", "degraded"}:
        raise ValueError("Unknown execution_mode.")

    planned_agent_order = _dedupe_agents_preserve_order(parsed.planned_agent_order)
    if not planned_agent_order or len(planned_agent_order) != len(set(parsed.planned_agent_order)):
        raise ValueError("Invalid planned_agent_order.")

    working_state: GraphState = state or {}
    completed_agents = infer_completed_agents(working_state)
    failed_agents = infer_failed_agents(working_state)
    unplanned_agents = [agent for agent in AGENT_ORDER if agent not in planned_agent_order]
    skipped_agents = _dedupe_agents(
        [
            *list(working_state.get("skipped_agents", [])),
            *[agent for agent in unplanned_agents if agent not in completed_agents],
        ]
    )
    state_with_plan: GraphState = {
        **working_state,
        "supervisor_plan": SupervisorPlan(
            next_node="coordinator_node",
            planned_agent_order=planned_agent_order,
        ),
    }
    next_node = determine_next_node(state_with_plan)
    return SupervisorPlan(
        required_agents=list(AGENT_ORDER),
        completed_agents=completed_agents,
        failed_agents=failed_agents,
        skipped_agents=skipped_agents,
        next_node=next_node,
        rationale=parsed.routing_reason,
        allow_degraded_report=True,
        question_type=parsed.question_type,
        execution_mode=parsed.execution_mode,  # type: ignore[arg-type]
        risk_focus=parsed.risk_focus,
        needs_graph_context=parsed.needs_graph_context,
        routing_reason=parsed.routing_reason,
        planned_agent_order=planned_agent_order,
        confidence=_clamp_confidence(parsed.confidence),
        used_llm=True,
        fallback_used=False,
    )


def _llm_supervisor_enabled() -> bool:
    return bool(config.ENABLE_LLM_SUPERVISOR and config.OPENAI_API_KEY)


def _call_llm_supervisor(prompt: str) -> str:
    """Call the configured LLM. Tests monkeypatch this to avoid live APIs."""
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.LLM_SUPERVISOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def run_llm_supervisor_agent(state: GraphState) -> dict:
    """Try LLM routing when enabled, otherwise use deterministic fallback."""
    if not _llm_supervisor_enabled():
        plan = build_supervisor_plan(state)
        return _plan_result(plan)
    if _contains_direct_advice(_question_text(state.get("user_input")).strip().lower()):
        plan = build_supervisor_plan(state)
        return _plan_result(plan)

    prompt = build_llm_supervisor_prompt(
        state.get("user_input"),
        allowed_agents=list(AGENT_ORDER),
        allowed_question_types=list(QUESTION_TYPES),
    )
    try:
        response_text = _call_llm_supervisor(prompt)
        plan = parse_llm_supervisor_response(response_text, state)
        return _plan_result(plan)
    except (ValueError, ValidationError, json.JSONDecodeError, Exception) as exc:
        return _fallback_plan(
            state,
            warning=f"LLM Supervisor fallback used: {exc}",
        )


def run_supervisor_agent(state: GraphState) -> dict:
    """Return supervisor planning metadata for future graph routing phases."""
    if _llm_supervisor_enabled():
        return run_llm_supervisor_agent(state)
    plan = build_supervisor_plan(state)
    return _plan_result(plan)
