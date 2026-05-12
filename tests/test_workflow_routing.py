"""Tests for LangGraph routing behavior."""

from src.graph.routing import route_after_evaluation
from src.graph.state import EvaluationResult
from src.graph.workflow import run_research_workflow


def _evaluation(overall_pass: bool) -> EvaluationResult:
    return EvaluationResult(
        overall_pass=overall_pass,
        source_grounding_score=1.0 if overall_pass else 0.2,
        numeric_consistency_score=1.0,
        safety_score=1.0 if overall_pass else 0.0,
        risk_disclosure_score=1.0,
        freshness_score=1.0,
        issues=[] if overall_pass else ["failed"],
        revision_suggestions=[] if overall_pass else ["rewrite"],
    )


def test_evaluator_pass_routes_to_save() -> None:
    state = {"evaluation_result": _evaluation(True), "rewrite_attempts": 0}

    assert route_after_evaluation(state) == "save_report_node"


def test_evaluator_fail_routes_to_rewrite() -> None:
    state = {"evaluation_result": _evaluation(False), "rewrite_attempts": 1}

    assert route_after_evaluation(state) == "rewrite_node"


def test_rewrite_count_at_limit_routes_to_save_failed_report() -> None:
    state = {"evaluation_result": _evaluation(False), "rewrite_attempts": 2}

    assert route_after_evaluation(state) == "save_report_node"


def test_invalid_ticker_returns_validation_error() -> None:
    result = run_research_workflow("", "중기", "중립형")

    assert result["run_id"]
    assert result["final_report"] is None
    assert result["report_path"] is None
    assert result["errors"]
    assert result["errors"][0].error_type == "validation_error"
