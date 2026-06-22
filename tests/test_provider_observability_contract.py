"""US3 observability contract tests (FR-017, SER-006). No live services."""

from src.providers.entities import TechnicalAnalysisResult
from src.providers.enums import DegradationStatus, NormalizationStatus
from src.providers.scenario_input import build_scenario_report_input
from tests.fixtures.provider_contracts import scenario_inputs_complete


def test_complete_inputs_produce_complete_status():
    cp, news, metrics = scenario_inputs_complete()
    sri = build_scenario_report_input(
        request_id="req1", ticker="ACME", company_profile=cp,
        news_events=news, financial_metrics=metrics,
        technical_analysis_results=[TechnicalAnalysisResult(
            request_id="req1", ticker_id="tk1", source_market_data_refs=["md1"],
            normalization_or_derivation_status=NormalizationStatus.SUCCESS,
        )],
        wave_analysis_results=[], graph_context={"company": "ACME"}, vector_references=[],
    )
    assert sri.degradation_status == DegradationStatus.COMPLETE
    assert sri.missing_data_notes == []


def test_missing_categories_surface_as_structured_warnings():
    sri = build_scenario_report_input(
        request_id="req1", ticker="ACME", company_profile=None,
        news_events=[], financial_metrics=[], technical_analysis_results=[],
        wave_analysis_results=[], graph_context={}, vector_references=[],
    )
    assert sri.degradation_status in (
        DegradationStatus.INSUFFICIENT_DATA, DegradationStatus.PARTIAL_PROVIDER_FAILURE,
    )
    assert sri.missing_data_notes  # explicit, non-empty
    assert any(w.code for w in sri.warnings)


def test_missing_graph_context_marks_degraded():
    cp, news, metrics = scenario_inputs_complete()
    sri = build_scenario_report_input(
        request_id="req1", ticker="ACME", company_profile=cp, news_events=news,
        financial_metrics=metrics, technical_analysis_results=[], wave_analysis_results=[],
        graph_context={}, vector_references=[],  # empty graph context
    )
    assert any("graph" in note.lower() for note in sri.missing_data_notes)
