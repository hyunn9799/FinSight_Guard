"""Helpers for building typed evidence items from agent outputs."""

from datetime import UTC, datetime
from uuid import uuid4

from src.evidence.evidence_schema import EvidenceItem, EvidenceMetricValue


def generate_evidence_id(prefix: str) -> str:
    """Generate a stable-format evidence identifier with a caller-owned prefix."""
    clean_prefix = prefix.strip().lower().replace(" ", "_") or "evidence"
    return f"{clean_prefix}_{uuid4().hex[:12]}"


def _collected_at_or_now(collected_at: datetime | None) -> datetime:
    return collected_at or datetime.now(UTC)


def build_market_evidence(
    *,
    ticker: str,
    metric_name: str,
    metric_value: EvidenceMetricValue,
    description: str,
    source_name: str = "yfinance",
    source_url: str | None = None,
    collected_at: datetime | None = None,
) -> EvidenceItem:
    """Build evidence for market prices or technical indicators."""
    return EvidenceItem(
        evidence_id=generate_evidence_id("market"),
        source_type="market",
        source_name=source_name,
        source_url=source_url,
        collected_at=_collected_at_or_now(collected_at),
        ticker=ticker,
        metric_name=metric_name,
        metric_value=metric_value,
        description=description,
    )


def build_fundamental_evidence(
    *,
    ticker: str,
    metric_name: str,
    metric_value: EvidenceMetricValue,
    description: str,
    source_name: str = "yfinance",
    source_url: str | None = None,
    collected_at: datetime | None = None,
) -> EvidenceItem:
    """Build evidence for company profile, valuation, or financial metrics."""
    return EvidenceItem(
        evidence_id=generate_evidence_id("fundamental"),
        source_type="fundamental",
        source_name=source_name,
        source_url=source_url,
        collected_at=_collected_at_or_now(collected_at),
        ticker=ticker,
        metric_name=metric_name,
        metric_value=metric_value,
        description=description,
    )


def build_backtest_evidence(
    *,
    ticker: str,
    metric_name: str,
    metric_value: EvidenceMetricValue,
    description: str,
    source_name: str = "FinSight 전략 백테스트 (과거 시뮬레이션)",
    source_url: str | None = None,
    collected_at: datetime | None = None,
) -> EvidenceItem:
    """Build evidence for a historical strategy-backtest simulation result.

    Positioning: backtest outputs are technical reference material describing a
    past simulation of a fixed rule set. They are never buy/sell/hold advice.
    """
    return EvidenceItem(
        evidence_id=generate_evidence_id("backtest"),
        source_type="backtest",
        source_name=source_name,
        source_url=source_url,
        collected_at=_collected_at_or_now(collected_at),
        ticker=ticker,
        metric_name=metric_name,
        metric_value=metric_value,
        description=description,
    )


def build_optimization_evidence(
    *,
    ticker: str,
    metric_name: str,
    metric_value: EvidenceMetricValue,
    description: str,
    source_name: str = "FinSight robust optimization (historical simulation)",
    source_url: str | None = None,
    collected_at: datetime | None = None,
) -> EvidenceItem:
    """Build evidence for robust optimization numeric claims."""
    return EvidenceItem(
        evidence_id=generate_evidence_id("opt"),
        source_type="backtest",
        source_name=source_name,
        source_url=source_url,
        collected_at=_collected_at_or_now(collected_at),
        ticker=ticker,
        metric_name=metric_name,
        metric_value=metric_value,
        description=description,
    )


def build_news_evidence(
    *,
    ticker: str,
    source_name: str,
    description: str,
    source_url: str | None = None,
    metric_name: str = "news_item",
    metric_value: EvidenceMetricValue = None,
    collected_at: datetime | None = None,
) -> EvidenceItem:
    """Build evidence for a news item or external article."""
    return EvidenceItem(
        evidence_id=generate_evidence_id("news"),
        source_type="news",
        source_name=source_name,
        source_url=source_url,
        collected_at=_collected_at_or_now(collected_at),
        ticker=ticker,
        metric_name=metric_name,
        metric_value=metric_value,
        description=description,
    )
