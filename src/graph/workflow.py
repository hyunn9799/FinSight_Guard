"""LangGraph workflow assembly."""

from datetime import UTC, datetime
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from src.agents.coordinator_agent import run_coordinator_agent
from src.agents.evaluator_agent import run_evaluator_agent
from src.agents.fundamental_agent import run_fundamental_agent
from src.agents.market_agent import run_market_agent
from src.agents.news_agent import run_news_agent
from src.agents.rewrite_agent import run_rewrite_agent
from src.graph.routing import route_after_evaluation, route_after_validation
from src.graph.state import GraphState, ResearchReport, UserInput, WorkflowError
from src.observability.logger import log_node_error, log_node_start, log_node_success
from src.observability.metrics import record_run
from src.storage.report_store import save_report_json, save_report_markdown
from src.storage.run_store import save_run


VALID_INVESTMENT_HORIZONS = {"단기", "중기", "장기"}
VALID_RISK_PROFILES = {"보수형", "중립형", "공격형"}


class ResearchWorkflowState(GraphState, total=False):
    """Workflow-local state keys not stored in the shared contract yet."""

    investment_horizon: str
    risk_profile: str
    report_path: str


def _workflow_error(
    *,
    node: str,
    message: str,
    error_type: str = "workflow_error",
    recoverable: bool = True,
) -> WorkflowError:
    return WorkflowError(
        node=node,
        message=message,
        error_type=error_type,
        recoverable=recoverable,
    )


def _evaluation_score(state: ResearchWorkflowState) -> float:
    evaluation = state.get("evaluation_result")
    if evaluation is None:
        return 0.0
    scores = [
        evaluation.source_grounding_score,
        evaluation.numeric_consistency_score,
        evaluation.safety_score,
        evaluation.risk_disclosure_score,
        evaluation.freshness_score,
    ]
    return sum(scores) / len(scores)


def _safe_node(node_name: str, fn, state: ResearchWorkflowState) -> dict:
    run_id = state.get("run_id", "")
    ticker = state.get("ticker", "")
    start = perf_counter()
    log_node_start(run_id, node_name, ticker)
    try:
        result = fn(state)
        log_node_success(run_id, node_name, (perf_counter() - start) * 1000)
        return result
    except Exception as exc:
        log_node_error(run_id, node_name, exc)
        error = _workflow_error(
            node=node_name,
            message=f"{node_name} failed: {exc}",
            error_type="node_runtime_error",
            recoverable=True,
        )
        return {
            "status": "degraded",
            "errors": [*state.get("errors", []), error],
            "warnings": [*state.get("warnings", []), error.message],
        }


def input_validator_node(state: ResearchWorkflowState) -> dict:
    """Validate user inputs and stop early on critical errors."""
    run_id = state.get("run_id", "")
    start = perf_counter()
    log_node_start(run_id, "input_validator_node", state.get("ticker", ""))

    ticker = state.get("ticker", "").strip().upper()
    investment_horizon = state.get("investment_horizon", "").strip()
    risk_profile = state.get("risk_profile", "").strip()

    errors: list[WorkflowError] = []
    if not ticker:
        errors.append(
            _workflow_error(
                node="input_validator_node",
                message="ticker must not be empty.",
                error_type="validation_error",
                recoverable=False,
            )
        )
    if investment_horizon not in VALID_INVESTMENT_HORIZONS:
        errors.append(
            _workflow_error(
                node="input_validator_node",
                message='investment_horizon must be one of ["단기", "중기", "장기"].',
                error_type="validation_error",
                recoverable=False,
            )
        )
    if risk_profile not in VALID_RISK_PROFILES:
        errors.append(
            _workflow_error(
                node="input_validator_node",
                message='risk_profile must be one of ["보수형", "중립형", "공격형"].',
                error_type="validation_error",
                recoverable=False,
            )
        )

    if errors:
        for error in errors:
            log_node_error(run_id, "input_validator_node", error.message)
        record_run(False, 0.0)
        log_node_success(run_id, "input_validator_node", (perf_counter() - start) * 1000)
        return {
            "status": "failed",
            "errors": [*state.get("errors", []), *errors],
        }

    result = {
        "status": "running",
        "ticker": ticker,
        "user_input": UserInput(
            ticker=ticker,
            user_query=f"투자기간: {investment_horizon}; 위험성향: {risk_profile}",
        ),
        "started_at": state.get("started_at") or datetime.now(UTC),
        "warnings": state.get("warnings", []),
        "errors": state.get("errors", []),
        "rewrite_attempts": state.get("rewrite_attempts", 0),
    }
    log_node_success(run_id, "input_validator_node", (perf_counter() - start) * 1000)
    return result


def market_node(state: ResearchWorkflowState) -> dict:
    """Run Market Agent with degraded-mode exception handling."""
    return _safe_node("market_node", run_market_agent, state)


def fundamental_node(state: ResearchWorkflowState) -> dict:
    """Run Fundamental Agent with degraded-mode exception handling."""
    return _safe_node("fundamental_node", run_fundamental_agent, state)


def news_node(state: ResearchWorkflowState) -> dict:
    """Run News Agent with degraded-mode exception handling."""
    return _safe_node("news_node", run_news_agent, state)


def coordinator_node(state: ResearchWorkflowState) -> dict:
    """Run Coordinator Agent with degraded-mode exception handling."""
    return _safe_node("coordinator_node", run_coordinator_agent, state)


def evaluator_node(state: ResearchWorkflowState) -> dict:
    """Run Evaluator Agent with degraded-mode exception handling."""
    return _safe_node("evaluator_node", run_evaluator_agent, state)


def rewrite_node(state: ResearchWorkflowState) -> dict:
    """Run Rewrite Agent with degraded-mode exception handling."""
    return _safe_node("rewrite_node", run_rewrite_agent, state)


def _report_payload(report: ResearchReport, state: ResearchWorkflowState) -> dict:
    return {
        "run_id": state.get("run_id"),
        "status": state.get("status"),
        "evaluation_passed": (
            state.get("evaluation_result").overall_pass
            if state.get("evaluation_result") is not None
            else None
        ),
        "report": report.model_dump(mode="json"),
    }


def save_report_node(state: ResearchWorkflowState) -> dict:
    """Persist the latest report and set final workflow status."""
    run_id = state.get("run_id") or uuid4().hex
    start = perf_counter()
    log_node_start(run_id, "save_report_node", state.get("ticker", ""))

    report = state.get("draft_report") or state.get("final_report")
    if report is None:
        error = _workflow_error(
            node="save_report_node",
            message="No report is available to save.",
            error_type="report_save_error",
            recoverable=False,
        )
        log_node_error(run_id, "save_report_node", error.message)
        record_run(False, _evaluation_score(state))
        log_node_success(run_id, "save_report_node", (perf_counter() - start) * 1000)
        return {
            "status": "failed",
            "errors": [*state.get("errors", []), error],
            "completed_at": datetime.now(UTC),
        }

    evaluation = state.get("evaluation_result")
    final_status = "success" if evaluation is not None and evaluation.overall_pass else "failed"
    payload = _report_payload(report, {**state, "status": final_status, "run_id": run_id})
    report_path = save_report_json(run_id, payload)
    markdown_path = save_report_markdown(run_id, report)
    evaluation_score = _evaluation_score(state)
    success = final_status == "success"
    record_run(success, evaluation_score)
    save_run(
        run_id,
        {
            "ticker": state.get("ticker"),
            "status": final_status,
            "evaluation_score": evaluation_score,
            "report_path": report_path,
            "markdown_path": markdown_path,
        },
    )
    log_node_success(run_id, "save_report_node", (perf_counter() - start) * 1000)

    return {
        "run_id": run_id,
        "status": final_status,
        "final_report": report,
        "report_path": report_path,
        "completed_at": datetime.now(UTC),
    }


def build_research_graph():
    """Build and compile the LangGraph research workflow."""
    graph = StateGraph(ResearchWorkflowState)
    graph.add_node("input_validator_node", input_validator_node)
    graph.add_node("market_node", market_node)
    graph.add_node("fundamental_node", fundamental_node)
    graph.add_node("news_node", news_node)
    graph.add_node("coordinator_node", coordinator_node)
    graph.add_node("evaluator_node", evaluator_node)
    graph.add_node("rewrite_node", rewrite_node)
    graph.add_node("save_report_node", save_report_node)

    graph.add_edge(START, "input_validator_node")
    graph.add_conditional_edges(
        "input_validator_node",
        route_after_validation,
        {
            "continue": "market_node",
            "stop": END,
        },
    )
    graph.add_edge("market_node", "fundamental_node")
    graph.add_edge("fundamental_node", "news_node")
    graph.add_edge("news_node", "coordinator_node")
    graph.add_edge("coordinator_node", "evaluator_node")
    graph.add_conditional_edges(
        "evaluator_node",
        route_after_evaluation,
        {
            "save_report_node": "save_report_node",
            "rewrite_node": "rewrite_node",
        },
    )
    graph.add_edge("rewrite_node", "evaluator_node")
    graph.add_edge("save_report_node", END)
    return graph.compile()


def run_research_workflow(
    ticker: str,
    investment_horizon: str,
    risk_profile: str,
) -> dict:
    """Run the full research workflow and return API-friendly state fields."""
    run_id = uuid4().hex
    initial_state: ResearchWorkflowState = {
        "run_id": run_id,
        "ticker": ticker,
        "investment_horizon": investment_horizon,
        "risk_profile": risk_profile,
        "status": "pending",
        "errors": [],
        "warnings": [],
        "rewrite_attempts": 0,
        "started_at": datetime.now(UTC),
    }
    final_state = build_research_graph().invoke(initial_state)
    if "run_id" not in final_state:
        final_state["run_id"] = run_id

    user_input = {
        "ticker": final_state.get("ticker", ticker),
        "investment_horizon": investment_horizon,
        "risk_profile": risk_profile,
        "user_query": (
            final_state["user_input"].user_query
            if final_state.get("user_input") is not None
            else None
        ),
    }

    return {
        "run_id": final_state.get("run_id"),
        "user_input": user_input,
        "market_analysis": final_state.get("market_analysis"),
        "fundamental_analysis": final_state.get("fundamental_analysis"),
        "news_analysis": final_state.get("news_analysis"),
        "final_report": final_state.get("final_report"),
        "evaluation_result": final_state.get("evaluation_result"),
        "errors": final_state.get("errors", []),
        "report_path": final_state.get("report_path"),
    }
