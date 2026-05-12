"""Schemas for evidence-grounded financial research outputs."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


EvidenceSourceType = Literal["market", "fundamental", "news", "system"]
EvidenceMetricValue = str | float | int | None


class EvidenceItem(BaseModel):
    """A single source-backed factual or numeric claim used in a report."""

    evidence_id: str
    source_type: EvidenceSourceType
    source_name: str
    source_url: str | None = None
    collected_at: datetime
    ticker: str
    metric_name: str
    metric_value: EvidenceMetricValue = None
    description: str = Field(min_length=1)
