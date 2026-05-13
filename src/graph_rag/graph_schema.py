"""Pydantic schemas for lightweight EvidenceItem-based graph context."""

from typing import Literal

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """Node extracted from available evidence."""

    node_id: str
    node_type: Literal["company", "sector", "event", "risk", "metric", "source", "unknown"]
    name: str
    description: str | None = None


class GraphEdge(BaseModel):
    """Relationship between graph nodes supported by optional evidence."""

    source_id: str
    target_id: str
    relation_type: Literal[
        "mentions",
        "affects",
        "positive_driver",
        "negative_risk",
        "peer_comparison",
        "metric_signal",
        "source_supports",
    ]
    evidence_id: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    description: str | None = None


class GraphContext(BaseModel):
    """Partial graph context assembled from whatever evidence is available."""

    ticker: str
    focus: Literal["technical", "fundamental", "news_risk", "comprehensive", "unknown"]
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    key_relations_summary: list[str] = Field(default_factory=list)
    risk_relations: list[GraphEdge] = Field(default_factory=list)
    positive_relations: list[GraphEdge] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
