"""Tests for lightweight GraphContext building."""

from datetime import UTC, datetime

from src.evidence.evidence_builder import build_market_evidence, build_news_evidence
from src.graph_rag.graph_context_builder import build_graph_context
from src.graph.state import MarketAnalysis, SupervisorPlan
from src.graph.workflow import graph_context_node


def _collected_at() -> datetime:
    return datetime(2026, 5, 12, tzinfo=UTC)


def _use_tmp_logs(monkeypatch, tmp_path) -> None:
    import src.observability.logger as project_logger

    monkeypatch.setattr(project_logger, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(project_logger, "_CONFIGURED", False)


def test_build_graph_context_from_mock_market_evidence() -> None:
    evidence = build_market_evidence(
        ticker="AAPL",
        metric_name="RSI",
        metric_value=54.2,
        description="AAPL RSI 지표는 단기 추세 확인에 사용됩니다.",
        collected_at=_collected_at(),
    )

    context = build_graph_context("AAPL", [evidence], focus="technical")

    assert context.ticker == "AAPL"
    assert context.focus == "technical"
    assert any(node.node_id == "company:AAPL" for node in context.nodes)
    assert any(node.node_type == "metric" and node.name == "RSI" for node in context.nodes)
    assert any(edge.relation_type == "metric_signal" for edge in context.edges)
    assert evidence.evidence_id in context.evidence_ids


def test_build_graph_context_from_mock_news_risk_evidence() -> None:
    evidence = build_news_evidence(
        ticker="AAPL",
        source_name="Mock News",
        source_url="https://example.com/aapl-risk",
        description="AAPL 관련 규제 리스크와 공급망 불확실성이 확대되었습니다.",
        collected_at=_collected_at(),
    )

    context = build_graph_context("AAPL", [evidence], focus="news_risk")

    assert any(node.node_id == "company:AAPL" for node in context.nodes)
    assert any(node.node_type == "risk" and node.name == "규제" for node in context.nodes)
    assert any(edge.relation_type == "negative_risk" for edge in context.edges)
    assert context.risk_relations


def test_positive_keyword_creates_positive_driver_edge() -> None:
    evidence = build_news_evidence(
        ticker="AAPL",
        source_name="Mock News",
        source_url="https://example.com/aapl-positive",
        description="AAPL 신제품 호재와 수요 증가 기대가 성장 요인으로 언급되었습니다.",
        collected_at=_collected_at(),
    )

    context = build_graph_context("AAPL", [evidence], focus="news_risk")

    assert any(edge.relation_type == "positive_driver" for edge in context.edges)
    assert context.positive_relations


def test_graph_context_deduplicates_nodes_and_edges() -> None:
    evidence = build_market_evidence(
        ticker="AAPL",
        metric_name="RSI",
        metric_value=54.2,
        description="AAPL RSI 지표는 단기 추세 확인에 사용됩니다.",
        collected_at=_collected_at(),
    )

    context = build_graph_context("AAPL", [evidence, evidence], focus="technical")
    node_ids = [node.node_id for node in context.nodes]
    edge_keys = [
        (edge.source_id, edge.target_id, edge.relation_type, edge.evidence_id)
        for edge in context.edges
    ]

    assert len(node_ids) == len(set(node_ids))
    assert len(edge_keys) == len(set(edge_keys))


def test_empty_evidence_returns_valid_graph_context() -> None:
    context = build_graph_context("AAPL", [], focus="unknown")

    assert context.ticker == "AAPL"
    assert context.focus == "unknown"
    assert len(context.nodes) == 1
    assert context.nodes[0].node_id == "company:AAPL"
    assert context.edges == []
    assert context.key_relations_summary == []
    assert context.risk_relations == []
    assert context.positive_relations == []
    assert context.evidence_ids == []


def test_graph_context_node_creates_comprehensive_graph_context(monkeypatch, tmp_path) -> None:
    _use_tmp_logs(monkeypatch, tmp_path)
    evidence = build_market_evidence(
        ticker="AAPL",
        metric_name="RSI",
        metric_value=54.2,
        description="AAPL RSI 지표는 단기 추세 확인에 사용됩니다.",
        collected_at=_collected_at(),
    )

    result = graph_context_node(
        {
            "ticker": "AAPL",
            "evidence": [evidence],
            "supervisor_plan": SupervisorPlan(
                next_node="coordinator_node",
                question_type="comprehensive_report",
            ),
        }
    )

    assert result["graph_context"].focus == "comprehensive"
    assert evidence.evidence_id in result["graph_context"].evidence_ids


def test_graph_context_node_supports_partial_technical_evidence(monkeypatch, tmp_path) -> None:
    _use_tmp_logs(monkeypatch, tmp_path)
    evidence = build_market_evidence(
        ticker="AAPL",
        metric_name="RSI",
        metric_value=54.2,
        description="AAPL RSI 지표는 단기 추세 확인에 사용됩니다.",
        collected_at=_collected_at(),
    )

    result = graph_context_node(
        {
            "ticker": "AAPL",
            "market_analysis": MarketAnalysis(
                ticker="AAPL",
                evidence=[evidence],
            ),
            "supervisor_plan": SupervisorPlan(
                next_node="coordinator_node",
                question_type="technical_analysis",
                planned_agent_order=["market", "news"],
                skipped_agents=["fundamental"],
            ),
        }
    )

    assert result["graph_context"].focus == "technical"
    assert any(edge.relation_type == "metric_signal" for edge in result["graph_context"].edges)


def test_graph_context_node_handles_empty_evidence(monkeypatch, tmp_path) -> None:
    _use_tmp_logs(monkeypatch, tmp_path)
    result = graph_context_node(
        {
            "ticker": "AAPL",
            "supervisor_plan": SupervisorPlan(
                next_node="coordinator_node",
                question_type="safety_or_unclear",
            ),
        }
    )

    assert result["graph_context"].ticker == "AAPL"
    assert result["graph_context"].focus == "comprehensive"
    assert len(result["graph_context"].nodes) == 1
    assert result["graph_context"].edges == []


def test_graph_context_node_failure_records_warning_and_continues(monkeypatch) -> None:
    import src.graph.workflow as workflow
    import tempfile
    from pathlib import Path

    _use_tmp_logs(monkeypatch, Path(tempfile.mkdtemp()))

    def failing_builder(*args, **kwargs):
        raise RuntimeError("graph context failed")

    monkeypatch.setattr(workflow, "build_graph_context", failing_builder)

    result = workflow.graph_context_node(
        {
            "ticker": "AAPL",
            "supervisor_plan": SupervisorPlan(
                next_node="coordinator_node",
                question_type="comprehensive_report",
            ),
        }
    )

    assert result["status"] == "degraded"
    assert result["errors"]
    assert result["warnings"]
    assert "graph_context_node failed" in result["warnings"][0]
    assert "graph_context" not in result
