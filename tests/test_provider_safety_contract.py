"""Full structural + instance safety sweep over SAFETY_CHECKED_CONTRACTS (SC-006, SER-001)."""

from src.providers.enums import DegradationStatus
from src.providers.safety import (
    SAFETY_CHECKED_CONTRACTS, assert_no_trading_fields, find_trading_fields,
)
from src.providers.scenario_input import ScenarioReportInput, VectorReference


def test_structural_no_contract_declares_trading_fields():
    assert ScenarioReportInput in SAFETY_CHECKED_CONTRACTS
    for contract in SAFETY_CHECKED_CONTRACTS:
        assert find_trading_fields(contract) == [], contract.__name__
        assert_no_trading_fields(contract)


def test_token_matching_examples():
    from pydantic import BaseModel

    class Buy(BaseModel):
        buy_signal: int = 0

    class Tgt(BaseModel):
        target_price: float = 0.0

    assert find_trading_fields(Buy) == ["buy_signal"]
    assert find_trading_fields(Tgt) == ["target_price"]

    class Clean(BaseModel):
        threshold: float = 0.0
        household_segment: str = ""

    assert find_trading_fields(Clean) == []


def test_instance_scenario_report_input_exposes_no_trading_fields():
    sri = ScenarioReportInput(
        request_id="req1", ticker="ACME", degradation_status=DegradationStatus.COMPLETE,
        vector_references=[VectorReference(source_kind="report_chunk", canonical_ref_id="ev1")],
    )
    dumped = sri.model_dump()
    from src.providers.safety import _field_violates
    assert not any(_field_violates(k) for k in dumped)
