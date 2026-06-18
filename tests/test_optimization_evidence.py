"""Tests: optimization evidence items are traceable and correctly typed."""
import pytest
from src.evidence.evidence_builder import build_optimization_evidence
from src.evidence.evidence_schema import EvidenceItem


def test_build_optimization_evidence_returns_evidence_item():
    item = build_optimization_evidence(
        ticker="AAPL",
        metric_name="max_drawdown_pct",
        metric_value=18.2,
        description="Historical simulation MDD for robust candidate.",
    )
    assert isinstance(item, EvidenceItem)
    assert item.source_type == "backtest"
    assert item.ticker == "AAPL"
    assert item.metric_name == "max_drawdown_pct"
    assert item.metric_value == 18.2


def test_optimization_evidence_id_starts_with_opt():
    item = build_optimization_evidence(ticker="MSFT", metric_name="sharpe", metric_value=0.9,
                                       description="Sharpe ratio.")
    assert item.evidence_id.startswith("opt_")


def test_optimization_evidence_description_non_empty():
    item = build_optimization_evidence(ticker="TSLA", metric_name="win_rate_pct",
                                       metric_value=54.0, description="Win rate.")
    assert len(item.description) > 0


def test_optimization_evidence_source_name_contains_simulation():
    item = build_optimization_evidence(ticker="AAPL", metric_name="cost_adjusted_return_pct",
                                       metric_value=8.4, description="Cost-adjusted return.")
    assert "simulation" in item.source_name.lower() or "backtest" in item.source_name.lower()
