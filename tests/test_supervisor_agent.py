"""Tests for deterministic Supervisor Agent planning."""

from src.agents.supervisor_agent import (
    build_supervisor_plan,
    build_llm_supervisor_prompt,
    classify_question_type,
    determine_next_node,
    parse_llm_supervisor_response,
    run_supervisor_agent,
)
from src.graph.state import FundamentalAnalysis, MarketAnalysis, NewsAnalysis, UserInput


def _market_analysis() -> MarketAnalysis:
    return MarketAnalysis(ticker="AAPL", summary="시장 분석")


def _fundamental_analysis() -> FundamentalAnalysis:
    return FundamentalAnalysis(ticker="AAPL", summary="펀더멘털 분석")


def _news_analysis() -> NewsAnalysis:
    return NewsAnalysis(ticker="AAPL", summary="뉴스 분석")


def test_empty_state_routes_to_market_node() -> None:
    plan = build_supervisor_plan({})

    assert plan.next_node == "market_node"
    assert plan.question_type == "safety_or_unclear"
    assert plan.planned_agent_order == ["market", "fundamental", "news"]
    assert plan.completed_agents == []
    assert plan.failed_agents == []
    assert plan.allow_degraded_report is True


def test_market_analysis_routes_to_fundamental_node() -> None:
    plan = build_supervisor_plan({"market_analysis": _market_analysis()})

    assert plan.next_node == "fundamental_node"
    assert plan.completed_agents == ["market"]


def test_market_and_fundamental_analysis_routes_to_news_node() -> None:
    plan = build_supervisor_plan(
        {
            "market_analysis": _market_analysis(),
            "fundamental_analysis": _fundamental_analysis(),
        }
    )

    assert plan.next_node == "news_node"
    assert plan.completed_agents == ["market", "fundamental"]


def test_all_analysis_outputs_route_to_coordinator_node() -> None:
    plan = build_supervisor_plan(
        {
            "market_analysis": _market_analysis(),
            "fundamental_analysis": _fundamental_analysis(),
            "news_analysis": _news_analysis(),
        }
    )

    assert plan.next_node == "coordinator_node"
    assert plan.completed_agents == ["market", "fundamental", "news"]
    assert "Coordinator Agent" in plan.rationale


def test_failed_market_routes_to_fundamental_node_for_degraded_report() -> None:
    plan = build_supervisor_plan({"failed_agents": ["market"]})

    assert plan.next_node == "fundamental_node"
    assert plan.failed_agents == ["market"]
    assert plan.allow_degraded_report is True


def test_run_supervisor_agent_returns_plan_and_next_agent() -> None:
    result = run_supervisor_agent({"market_analysis": _market_analysis()})

    assert result["supervisor_plan"].next_node == "fundamental_node"
    assert result["completed_agents"] == ["market"]
    assert result["failed_agents"] == []
    assert result["skipped_agents"] == []
    assert result["next_agent"] == "fundamental"


def test_technical_question_uses_market_then_news_plan() -> None:
    user_input = UserInput(ticker="AAPL", user_query="차트상 단기 진입 괜찮아?")

    plan = build_supervisor_plan({"user_input": user_input})

    assert classify_question_type(user_input) == "technical_analysis"
    assert plan.question_type == "technical_analysis"
    assert plan.execution_mode == "selective"
    assert plan.risk_focus is False
    assert plan.needs_graph_context is False
    assert plan.planned_agent_order == ["market", "news"]
    assert plan.skipped_agents == ["fundamental"]
    assert plan.next_node == "market_node"


def test_fundamental_question_uses_fundamental_then_news_plan() -> None:
    user_input = UserInput(ticker="AAPL", user_query="장기적으로 저평가야?")

    plan = build_supervisor_plan({"user_input": user_input})

    assert classify_question_type(user_input) == "fundamental_analysis"
    assert plan.question_type == "fundamental_analysis"
    assert plan.execution_mode == "selective"
    assert plan.risk_focus is False
    assert plan.needs_graph_context is True
    assert plan.planned_agent_order == ["fundamental", "news"]
    assert plan.skipped_agents == ["market"]
    assert plan.next_node == "fundamental_node"


def test_news_risk_question_uses_news_first_plan() -> None:
    user_input = UserInput(ticker="AAPL", user_query="최근 악재 때문에 위험해?")

    plan = build_supervisor_plan({"user_input": user_input})

    assert classify_question_type(user_input) == "news_risk_analysis"
    assert plan.question_type == "news_risk_analysis"
    assert plan.execution_mode == "selective"
    assert plan.risk_focus is True
    assert plan.needs_graph_context is True
    assert plan.planned_agent_order == ["news", "market", "fundamental"]
    assert plan.skipped_agents == []
    assert plan.next_node == "news_node"


def test_comprehensive_question_uses_all_agents_plan() -> None:
    user_input = UserInput(ticker="AAPL", user_query="종합 보고서 만들어줘")

    plan = build_supervisor_plan({"user_input": user_input})

    assert classify_question_type(user_input) == "comprehensive_report"
    assert plan.question_type == "comprehensive_report"
    assert plan.execution_mode == "full"
    assert plan.risk_focus is True
    assert plan.needs_graph_context is True
    assert plan.planned_agent_order == ["market", "fundamental", "news"]
    assert plan.skipped_agents == []


def test_direct_advice_question_uses_safety_or_unclear_full_plan() -> None:
    user_input = UserInput(ticker="AAPL", user_query="무조건 사야 해?")

    plan = build_supervisor_plan({"user_input": user_input})

    assert classify_question_type(user_input) == "safety_or_unclear"
    assert plan.question_type == "safety_or_unclear"
    assert plan.execution_mode == "full"
    assert plan.risk_focus is True
    assert plan.needs_graph_context is True
    assert plan.planned_agent_order == ["market", "fundamental", "news"]


def test_route_follows_planned_order_after_first_planned_agent_completes() -> None:
    user_input = UserInput(ticker="AAPL", user_query="차트상 단기 진입 괜찮아?")
    state = {
        "user_input": user_input,
        "market_analysis": _market_analysis(),
    }
    plan = build_supervisor_plan(state)

    assert plan.next_node == "news_node"
    assert determine_next_node({**state, "supervisor_plan": plan}) == "news_node"


def test_unplanned_agent_is_not_skipped_if_already_completed() -> None:
    user_input = UserInput(ticker="AAPL", user_query="차트상 단기 진입 괜찮아?")

    plan = build_supervisor_plan(
        {
            "user_input": user_input,
            "fundamental_analysis": _fundamental_analysis(),
        }
    )

    assert plan.planned_agent_order == ["market", "news"]
    assert plan.completed_agents == ["fundamental"]
    assert plan.skipped_agents == []


def test_llm_prompt_is_constrained_to_routing_only() -> None:
    prompt = build_llm_supervisor_prompt(
        UserInput(ticker="AAPL", user_query="차트와 뉴스 리스크를 봐줘"),
        ["market", "fundamental", "news"],
        [
            "technical_analysis",
            "fundamental_analysis",
            "news_risk_analysis",
            "comprehensive_report",
            "safety_or_unclear",
        ],
    )

    assert "You do not provide investment advice." in prompt
    assert "You only decide which analysis agents should run." in prompt
    assert "Return only valid JSON." in prompt
    assert "market, fundamental, news" in prompt


def test_valid_llm_json_returns_plan_with_used_llm(monkeypatch) -> None:
    import src.agents.supervisor_agent as supervisor

    monkeypatch.setattr(supervisor.config, "ENABLE_LLM_SUPERVISOR", True)
    monkeypatch.setattr(supervisor.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        supervisor,
        "_call_llm_supervisor",
        lambda prompt: (
            '{"question_type":"technical_analysis",'
            '"planned_agent_order":["market","news"],'
            '"primary_agents":["market"],'
            '"secondary_agents":["news"],'
            '"execution_mode":"selective",'
            '"risk_focus":false,'
            '"needs_graph_context":false,'
            '"routing_reason":"차트 중심 질문입니다.",'
            '"confidence":1.2}'
        ),
    )

    result = run_supervisor_agent(
        {"user_input": UserInput(ticker="AAPL", user_query="차트상 단기 흐름 어때?")}
    )
    plan = result["supervisor_plan"]

    assert plan.used_llm is True
    assert plan.fallback_used is False
    assert plan.confidence == 1.0
    assert plan.question_type == "technical_analysis"
    assert plan.planned_agent_order == ["market", "news"]
    assert result["next_agent"] == "market"


def test_invalid_llm_json_falls_back_to_rule_based_plan(monkeypatch) -> None:
    import src.agents.supervisor_agent as supervisor

    monkeypatch.setattr(supervisor.config, "ENABLE_LLM_SUPERVISOR", True)
    monkeypatch.setattr(supervisor.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(supervisor, "_call_llm_supervisor", lambda prompt: "not json")

    result = run_supervisor_agent(
        {"user_input": UserInput(ticker="AAPL", user_query="장기적으로 저평가야?")}
    )
    plan = result["supervisor_plan"]

    assert plan.used_llm is False
    assert plan.fallback_used is True
    assert plan.question_type == "fundamental_analysis"
    assert plan.planned_agent_order == ["fundamental", "news"]
    assert result["warnings"]


def test_unknown_agent_in_llm_output_falls_back(monkeypatch) -> None:
    import src.agents.supervisor_agent as supervisor

    monkeypatch.setattr(supervisor.config, "ENABLE_LLM_SUPERVISOR", True)
    monkeypatch.setattr(supervisor.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        supervisor,
        "_call_llm_supervisor",
        lambda prompt: (
            '{"question_type":"technical_analysis",'
            '"planned_agent_order":["market","trading"],'
            '"primary_agents":["trading"],'
            '"secondary_agents":[],'
            '"execution_mode":"selective",'
            '"risk_focus":false,'
            '"needs_graph_context":false,'
            '"routing_reason":"invalid",'
            '"confidence":0.8}'
        ),
    )

    result = run_supervisor_agent(
        {"user_input": UserInput(ticker="AAPL", user_query="차트상 단기 진입 괜찮아?")}
    )
    plan = result["supervisor_plan"]

    assert plan.used_llm is False
    assert plan.fallback_used is True
    assert "trading" not in plan.planned_agent_order
    assert set(plan.planned_agent_order).issubset({"market", "fundamental", "news"})


def test_disabled_or_missing_api_key_uses_rule_based_plan(monkeypatch) -> None:
    import src.agents.supervisor_agent as supervisor

    monkeypatch.setattr(supervisor.config, "ENABLE_LLM_SUPERVISOR", False)
    monkeypatch.setattr(supervisor.config, "OPENAI_API_KEY", "")

    result = run_supervisor_agent(
        {"user_input": UserInput(ticker="AAPL", user_query="최근 악재 때문에 위험해?")}
    )
    plan = result["supervisor_plan"]

    assert plan.used_llm is False
    assert plan.fallback_used is False
    assert plan.question_type == "news_risk_analysis"
    assert plan.planned_agent_order == ["news", "market", "fundamental"]


def test_direct_advice_uses_safety_or_unclear_even_when_llm_enabled(monkeypatch) -> None:
    import src.agents.supervisor_agent as supervisor

    called = False

    def fake_call(prompt: str) -> str:
        nonlocal called
        called = True
        return "{}"

    monkeypatch.setattr(supervisor.config, "ENABLE_LLM_SUPERVISOR", True)
    monkeypatch.setattr(supervisor.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(supervisor, "_call_llm_supervisor", fake_call)

    result = run_supervisor_agent(
        {"user_input": UserInput(ticker="AAPL", user_query="무조건 사야 해?")}
    )
    plan = result["supervisor_plan"]

    assert called is False
    assert plan.question_type == "safety_or_unclear"
    assert plan.planned_agent_order == ["market", "fundamental", "news"]


def test_parse_llm_supervisor_response_rejects_routes_outside_allowed_agents() -> None:
    try:
        parse_llm_supervisor_response(
            '{"question_type":"technical_analysis",'
            '"planned_agent_order":["market","broker"],'
            '"primary_agents":["broker"],'
            '"secondary_agents":[],'
            '"execution_mode":"selective",'
            '"risk_focus":false,'
            '"needs_graph_context":false,'
            '"routing_reason":"invalid",'
            '"confidence":0.4}'
        )
    except Exception as exc:
        assert exc is not None
    else:
        raise AssertionError("Unknown LLM route was accepted.")
