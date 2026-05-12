"""Conditional routing helpers for the LangGraph workflow."""

from typing import Literal

from src.graph.state import GraphState


ValidationRoute = Literal["continue", "stop"]
EvaluationRoute = Literal["save_report_node", "rewrite_node"]

MAX_REWRITE_ATTEMPTS = 2


def route_after_validation(state: GraphState) -> ValidationRoute:
    """Stop early on critical validation failure, otherwise continue."""
    if state.get("status") == "failed":
        return "stop"
    return "continue"


def route_after_evaluation(state: GraphState) -> EvaluationRoute:
    """Route passed reports to save, failed reports to rewrite until the limit."""
    evaluation_result = state.get("evaluation_result")
    if evaluation_result is not None and evaluation_result.overall_pass:
        return "save_report_node"

    rewrite_attempts = state.get("rewrite_attempts", 0)
    if rewrite_attempts < MAX_REWRITE_ATTEMPTS:
        return "rewrite_node"

    return "save_report_node"
