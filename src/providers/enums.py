"""Status/degradation enums and lineage reference primitives (006).

These are the shared vocabulary for normalization outcomes and the canonical
references that tie raw responses, normalized records, evidence, market data,
and NEoWave rules together. No DB/graph/provider imports here.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProviderKind(str, Enum):
    NEWS = "news"
    FINANCIAL = "financial"
    MARKET_DATA = "market_data"


class NormalizationStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNSUPPORTED_FIELD = "unsupported_field"
    INSUFFICIENT_DATA = "insufficient_data"


class DegradationStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL_PROVIDER_FAILURE = "partial_provider_failure"
    PARTIAL_NORMALIZATION_FAILURE = "partial_normalization_failure"
    GRAPH_MAPPING_DEGRADED = "graph_mapping_degraded"
    INSUFFICIENT_DATA = "insufficient_data"


class RuleStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class _Contract(BaseModel):
    """Base for all 006 contract models: reject unknown fields."""

    model_config = ConfigDict(extra="forbid")


class RawResponseRef(_Contract):
    raw_response_id: str


class EvidenceRef(_Contract):
    evidence_id: str


class MarketDataRef(_Contract):
    market_data_ref_id: str


class RuleRef(_Contract):
    rule_id: str


class Warning(_Contract):
    code: str
    message: str = Field(min_length=1)
    field: str | None = None


class ProviderError(_Contract):
    code: str
    message: str = Field(min_length=1)
    provider_name: str | None = None
