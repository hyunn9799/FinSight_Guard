"""Tests for evidence schemas and builders."""

from datetime import UTC, datetime

from src.evidence.evidence_builder import (
    build_fundamental_evidence,
    build_market_evidence,
    build_news_evidence,
    generate_evidence_id,
)
from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import EvaluationResult, MarketAnalysis, ResearchReport


def test_generate_evidence_id_uses_prefix() -> None:
    evidence_id = generate_evidence_id("Market Data")

    assert evidence_id.startswith("market_data_")


def test_build_market_evidence_returns_typed_item() -> None:
    collected_at = datetime(2026, 5, 11, tzinfo=UTC)

    evidence = build_market_evidence(
        ticker="AAPL",
        metric_name="MA20",
        metric_value=185.2,
        description="20-day moving average from adjusted close prices.",
        collected_at=collected_at,
    )

    assert isinstance(evidence, EvidenceItem)
    assert evidence.source_type == "market"
    assert evidence.ticker == "AAPL"
    assert evidence.metric_value == 185.2
    assert evidence.collected_at == collected_at


def test_build_fundamental_and_news_evidence_source_types() -> None:
    fundamental = build_fundamental_evidence(
        ticker="MSFT",
        metric_name="trailing_pe",
        metric_value=32.5,
        description="Trailing P/E ratio from company financial data.",
    )
    news = build_news_evidence(
        ticker="MSFT",
        source_name="Mock News",
        source_url="https://example.com/news",
        description="Article describing a product launch.",
    )

    assert fundamental.source_type == "fundamental"
    assert news.source_type == "news"
    assert news.source_url == "https://example.com/news"


def test_analysis_and_report_models_accept_evidence() -> None:
    evidence = build_market_evidence(
        ticker="NVDA",
        metric_name="RSI",
        metric_value=58.0,
        description="RSI calculated from recent close prices.",
    )
    analysis = MarketAnalysis(
        ticker="NVDA",
        summary="Market summary",
        evidence=[evidence],
        missing_data_notes=["ATR unavailable"],
    )
    report = ResearchReport(
        title="NVDA Research Report",
        ticker="NVDA",
        data_date=datetime(2026, 5, 11, tzinfo=UTC).date(),
        executive_summary="Summary",
        market_section="Market",
        fundamental_section="Fundamental",
        news_section="News",
        scenario_analysis="Scenarios",
        risk_factors="Risks",
        limitations="Limitations",
        evidence_summary="Evidence",
        disclaimer="Disclaimer",
    )
    evaluation = EvaluationResult(
        overall_pass=True,
        source_grounding_score=1.0,
        numeric_consistency_score=1.0,
        safety_score=1.0,
        risk_disclosure_score=1.0,
        freshness_score=1.0,
    )

    assert analysis.evidence == [evidence]
    assert analysis.missing_data_notes == ["ATR unavailable"]
    assert report.ticker == "NVDA"
    assert evaluation.overall_pass is True
