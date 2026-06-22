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
