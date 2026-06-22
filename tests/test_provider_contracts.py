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


# Task 4 (T005): Provider interface Protocols + result models
from src.providers.interfaces import (
    FinancialProvider,
    FinancialProviderResult,
    MarketDataProvider,
    MarketDataProviderResult,
    NewsProvider,
    NewsProviderResult,
)


def test_provider_results_carry_normalized_objects_only():
    res = NewsProviderResult(
        raw_response_ref="raw1",
        normalization_status=NormalizationStatus.SUCCESS,
        news_events=[],
    )
    assert res.news_events == []
    assert res.warnings == []
    # result must not allow a raw payload field
    with pytest.raises(ValidationError):
        NewsProviderResult(
            raw_response_ref="raw1",
            normalization_status=NormalizationStatus.SUCCESS,
            news_events=[],
            payload_body={"x": 1},
        )


def test_protocols_are_runtime_checkable():
    class _FakeNews:
        def fetch_news(self, request):  # noqa: ANN001
            return NewsProviderResult(
                raw_response_ref="raw1",
                normalization_status=NormalizationStatus.SUCCESS,
                news_events=[],
            )

    assert isinstance(_FakeNews(), NewsProvider)


# Task 5 (T048): Token-based no-trading-field safety contract
from pydantic import BaseModel

from src.providers.safety import (
    FORBIDDEN_TOKENS,
    SAFETY_CHECKED_CONTRACTS,
    assert_no_trading_fields,
    find_trading_fields,
)


def test_token_matching_rejects_trading_field_names():
    class Bad(BaseModel):
        buy_signal: int = 0

    class Bad2(BaseModel):
        target_price: float = 0.0

    class Bad3(BaseModel):
        order_action: str = ""

    for m in (Bad, Bad2, Bad3):
        assert find_trading_fields(m), f"{m.__name__} should be flagged"
        with pytest.raises(ValueError):
            assert_no_trading_fields(m)


def test_substrings_do_not_false_positive():
    class Ok(BaseModel):
        threshold: float = 0.0   # contains "hold" as substring -> must NOT match
        household_count: int = 0  # token "household" != "hold"
        metric_name: str = ""     # value could be "price"; names are clean

    assert find_trading_fields(Ok) == []
    assert_no_trading_fields(Ok)  # no raise


def test_all_mvp_contracts_are_clean():
    assert len(SAFETY_CHECKED_CONTRACTS) >= 5
    for contract in SAFETY_CHECKED_CONTRACTS:
        assert_no_trading_fields(contract)


# Task 6 (T007): Normalization result containers + helper seams
from src.providers.normalization import (
    NormalizationResult,
    RawNewsItem,
    normalize_news,
)


def test_normalization_result_container_shape():
    result = NormalizationResult(
        status=NormalizationStatus.SUCCESS, records=[], warnings=[], errors=[]
    )
    assert result.records == []


def test_normalize_news_signature_exists_but_unimplemented():
    # US1 (T012) implements the body; here we only assert the seam exists.
    with pytest.raises(NotImplementedError):
        normalize_news(
            raw_items=[RawNewsItem(headline="x")],
            request_id="req1",
            ticker_id="tk1",
            raw_response_id="raw1",
        )
