"""User Story 1 contract tests (deterministic, no live calls)."""

import pytest
from pydantic import ValidationError

from src.providers.enums import (
    DegradationStatus,
    NormalizationStatus,
    ProviderKind,
    RawResponseRef,
    Warning,
)


def test_enums_have_required_members():
    assert ProviderKind.NEWS.value == "news"
    assert ProviderKind.FINANCIAL.value == "financial"
    assert ProviderKind.MARKET_DATA.value == "market_data"
    assert NormalizationStatus.SUCCESS.value == "success"
    assert NormalizationStatus.PARTIAL_SUCCESS.value == "partial_success"
    assert NormalizationStatus.DEGRADED.value == "degraded"
    assert NormalizationStatus.FAILED.value == "failed"
    assert NormalizationStatus.UNSUPPORTED_FIELD.value == "unsupported_field"
    assert NormalizationStatus.INSUFFICIENT_DATA.value == "insufficient_data"
    assert DegradationStatus.COMPLETE.value == "complete"
    assert DegradationStatus.INSUFFICIENT_DATA.value == "insufficient_data"


def test_lineage_ref_forbids_extra_fields():
    RawResponseRef(raw_response_id="r1")
    with pytest.raises(ValidationError):
        RawResponseRef(raw_response_id="r1", bogus="x")


def test_warning_is_structured():
    w = Warning(code="missing_url", message="news event has no source url")
    assert w.code == "missing_url"


# Task 3 (T006): Normalized & derived entity contracts
from datetime import UTC, datetime

from src.providers.entities import (
    CompanyProfile,
    FinancialMetric,
    NewsEvent,
    TechnicalAnalysisResult,
    WaveAnalysisResult,
)
from src.providers.enums import RuleStatus


def test_news_event_minimal_and_forbids_extra():
    ev = NewsEvent(
        request_id="req1",
        ticker_id="tk1",
        raw_response_id="raw1",
        title="Acme beats earnings",
        normalization_status=NormalizationStatus.SUCCESS,
    )
    assert ev.title == "Acme beats earnings"
    assert ev.warnings == []
    with pytest.raises(ValidationError):
        NewsEvent(
            request_id="req1",
            ticker_id="tk1",
            raw_response_id="raw1",
            title="x",
            normalization_status=NormalizationStatus.SUCCESS,
            content="RAW PROVIDER FIELD",  # not in contract -> rejected
        )


def test_derived_results_trace_to_market_data_not_raw():
    tar = TechnicalAnalysisResult(
        request_id="req1",
        ticker_id="tk1",
        source_market_data_refs=["md1"],
        indicator_values={"rsi_14": 55.0},
        normalization_or_derivation_status=NormalizationStatus.SUCCESS,
    )
    assert tar.source_market_data_refs == ["md1"]
    # derived contract has no raw_response_id field at all
    assert "raw_response_id" not in TechnicalAnalysisResult.model_fields

    war = WaveAnalysisResult(
        request_id="req1",
        ticker_id="tk1",
        source_market_data_refs=["md1"],
        rule_refs=["rule_neowave_1"],
        rule_statuses={"rule_neowave_1": RuleStatus.NEEDS_HUMAN_REVIEW},
    )
    assert war.rule_statuses["rule_neowave_1"] is RuleStatus.NEEDS_HUMAN_REVIEW


def test_company_and_financial_profiles():
    cp = CompanyProfile(
        request_id="req1", ticker_id="tk1", raw_response_id="raw1",
        company_name="Acme Corp", normalization_status=NormalizationStatus.SUCCESS,
    )
    assert cp.sector is None
    fm = FinancialMetric(
        request_id="req1", ticker_id="tk1", raw_response_id="raw1",
        metric_name="revenue", metric_value=1234.5, period="FY2025",
        normalization_status=NormalizationStatus.SUCCESS,
    )
    assert fm.metric_value == 1234.5
