"""US3 graph mapping eligibility tests (no live graph)."""

from src.graph_rag.graph_context_builder import build_contract_graph_context
from src.graph_rag.mapping_contracts import (
    GraphEligibleSpec, build_eligible_specs, is_graph_eligible,
)
from src.providers.entities import NewsEvent
from src.providers.enums import NormalizationStatus


def _news():
    return NewsEvent(
        request_id="req1", ticker_id="tk1", raw_response_id="raw1",
        title="Acme beats earnings", source_url="https://x/y", evidence_id="ev1",
        normalization_status=NormalizationStatus.SUCCESS,
    )


def test_normalized_records_are_graph_eligible():
    specs = build_eligible_specs([_news()])
    assert all(isinstance(s, GraphEligibleSpec) for s in specs)
    assert specs[0].graph_node_type == "NewsEvent"
    assert specs[0].source_canonical_ref  # must resolve to canonical reference


def test_raw_and_row_level_data_not_projected():
    # raw payloads / candles / sentences / financial rows must be ineligible
    assert not is_graph_eligible({"raw_response_id": "raw1", "payload_body": {}})
    assert not is_graph_eligible({"candle": {"o": 1}})
    specs = build_eligible_specs([{"raw_candle": True}])
    assert specs == []


def test_graph_context_degrades_when_projection_missing():
    specs = build_eligible_specs([_news()])
    ok = build_contract_graph_context(specs, projection_status="ready")
    assert ok["degraded"] is False and ok["nodes"]

    stale = build_contract_graph_context(specs, projection_status="stale")
    assert stale["degraded"] is True
    assert any("stale" in w.lower() or "missing" in w.lower() for w in stale["warnings"])
