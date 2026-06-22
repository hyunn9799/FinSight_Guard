"""US3 ScenarioReportInput + VectorReference contract tests."""

import pytest
from pydantic import ValidationError

from src.providers.entities import CompanyProfile, NewsEvent
from src.providers.enums import DegradationStatus, NormalizationStatus
from src.providers.scenario_input import ScenarioReportInput, VectorReference


def test_vector_reference_is_lightweight_canonical_only():
    vr = VectorReference(source_kind="news_original", canonical_ref_id="src_doc_1")
    assert vr.source_uri is None
    # no score/store/embedding fields exist on the contract
    for forbidden in ("score", "embedding", "vector_store", "store"):
        assert forbidden not in VectorReference.model_fields
    # dangling reference (empty canonical id) is invalid
    with pytest.raises(ValidationError):
        VectorReference(source_kind="news_original", canonical_ref_id="")
    # unknown source_kind rejected
    with pytest.raises(ValidationError):
        VectorReference(source_kind="raw_payload", canonical_ref_id="x")


def test_scenario_report_input_excludes_raw_and_carries_required_fields():
    sri = ScenarioReportInput(
        request_id="req1", ticker="ACME",
        company_profile=None, news_events=[], financial_metrics=[],
        technical_analysis_results=[], wave_analysis_results=[],
        graph_context={}, evidence_ids=["ev1"],
        vector_references=[VectorReference(source_kind="report_chunk", canonical_ref_id="ev1")],
        missing_data_notes=["no market data"],
        degradation_status=DegradationStatus.PARTIAL_PROVIDER_FAILURE,
    )
    assert sri.degradation_status == DegradationStatus.PARTIAL_PROVIDER_FAILURE
    # raw payload fields cannot be attached
    with pytest.raises(ValidationError):
        ScenarioReportInput(
            request_id="req1", ticker="ACME", degradation_status=DegradationStatus.COMPLETE,
            payload_body={"x": 1},
        )
