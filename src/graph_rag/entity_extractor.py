"""Rule-based entity and relation extraction from EvidenceItem objects."""

from typing import Iterable

from src.evidence.evidence_schema import EvidenceItem
from src.graph_rag.graph_schema import GraphEdge, GraphNode


RISK_KEYWORDS = (
    "규제",
    "소송",
    "환율",
    "지정학",
    "공급망",
    "실적 쇼크",
    "수출",
    "불확실성",
    "둔화",
)
EVENT_KEYWORDS = (
    "실적발표",
    "가이던스",
    "capex",
    "신제품",
    "수요 증가",
    "수요 둔화",
    "호재",
    "악재",
)
METRIC_KEYWORDS = ("rsi", "macd", "per", "pbr", "roe", "atr", "ma20", "ma60", "ma120")
SECTOR_KEYWORDS = ("반도체", "ai", "메모리", "금융", "전기차", "클라우드", "바이오")
POSITIVE_KEYWORDS = ("성장", "개선", "수요 증가", "호재", "상승")
NEGATIVE_KEYWORDS = (*RISK_KEYWORDS, "악재", "리스크", "위험")


def normalize_text(text: str) -> str:
    """Normalize text for deterministic keyword matching."""
    return " ".join((text or "").strip().lower().split())


def _safe_node_id(prefix: str, value: str) -> str:
    normalized = normalize_text(value).replace(" ", "_")
    return f"{prefix}:{normalized}"


def _append_unique(nodes: list[GraphNode], node: GraphNode) -> None:
    if all(existing.node_id != node.node_id for existing in nodes):
        nodes.append(node)


def _contains_any(text: str, keywords: Iterable[str]) -> list[str]:
    return [keyword for keyword in keywords if normalize_text(keyword) in text]


def _company_name_from_description(ticker: str, description: str) -> str | None:
    """Extract a conservative company label if the description names one explicitly."""
    text = description.strip()
    if not text:
        return None
    first_token = text.split()[0].strip(".,:;()[]")
    if first_token and first_token.upper() != ticker.upper() and len(first_token) <= 80:
        return first_token
    return None


def extract_entities_from_evidence(evidence_item: EvidenceItem) -> list[GraphNode]:
    """Extract lightweight graph nodes from a single EvidenceItem."""
    ticker = evidence_item.ticker.strip().upper()
    description = evidence_item.description or ""
    text = normalize_text(f"{evidence_item.metric_name} {evidence_item.metric_value} {description}")
    nodes: list[GraphNode] = []

    _append_unique(
        nodes,
        GraphNode(
            node_id=f"company:{ticker}",
            node_type="company",
            name=ticker,
            description=f"{ticker} 관련 기업 노드",
        ),
    )
    company_name = _company_name_from_description(ticker, description)
    if company_name:
        _append_unique(
            nodes,
            GraphNode(
                node_id=_safe_node_id("company", company_name),
                node_type="company",
                name=company_name,
                description="증거 설명에서 추출한 기업명 후보",
            ),
        )

    metric_name = normalize_text(evidence_item.metric_name)
    if metric_name in METRIC_KEYWORDS:
        _append_unique(
            nodes,
            GraphNode(
                node_id=_safe_node_id("metric", evidence_item.metric_name),
                node_type="metric",
                name=evidence_item.metric_name,
                description=f"{ticker} {evidence_item.metric_name} 지표",
            ),
        )

    for keyword in _contains_any(text, RISK_KEYWORDS):
        _append_unique(
            nodes,
            GraphNode(
                node_id=_safe_node_id("risk", keyword),
                node_type="risk",
                name=keyword,
                description=f"{keyword} 관련 리스크",
            ),
        )

    for keyword in _contains_any(text, EVENT_KEYWORDS):
        _append_unique(
            nodes,
            GraphNode(
                node_id=_safe_node_id("event", keyword),
                node_type="event",
                name=keyword,
                description=f"{keyword} 관련 이벤트",
            ),
        )

    for keyword in _contains_any(text, SECTOR_KEYWORDS):
        _append_unique(
            nodes,
            GraphNode(
                node_id=_safe_node_id("sector", keyword),
                node_type="sector",
                name=keyword,
                description=f"{keyword} 관련 섹터",
            ),
        )

    if evidence_item.source_name:
        _append_unique(
            nodes,
            GraphNode(
                node_id=_safe_node_id("source", evidence_item.source_name),
                node_type="source",
                name=evidence_item.source_name,
                description=evidence_item.source_url,
            ),
        )

    return nodes


def infer_relations_from_evidence(
    evidence_item: EvidenceItem,
    nodes: list[GraphNode],
) -> list[GraphEdge]:
    """Infer lightweight graph edges from evidence and extracted nodes."""
    ticker = evidence_item.ticker.strip().upper()
    company_id = f"company:{ticker}"
    text = normalize_text(f"{evidence_item.metric_name} {evidence_item.metric_value} {evidence_item.description}")
    edges: list[GraphEdge] = []

    for node in nodes:
        if node.node_id == company_id:
            continue
        if node.node_type == "metric":
            edges.append(
                GraphEdge(
                    source_id=node.node_id,
                    target_id=company_id,
                    relation_type="metric_signal",
                    evidence_id=evidence_item.evidence_id,
                    confidence=0.8,
                    description=f"{node.name} 지표가 {ticker} 분석을 뒷받침합니다.",
                )
            )
        elif node.node_type == "risk":
            edges.append(
                GraphEdge(
                    source_id=node.node_id,
                    target_id=company_id,
                    relation_type="negative_risk",
                    evidence_id=evidence_item.evidence_id,
                    confidence=0.7,
                    description=f"{node.name} 요인이 {ticker} 리스크로 언급되었습니다.",
                )
            )
        elif node.node_type == "event":
            relation_type = (
                "positive_driver"
                if any(keyword in text for keyword in POSITIVE_KEYWORDS)
                else "negative_risk"
                if any(keyword in text for keyword in NEGATIVE_KEYWORDS)
                else "affects"
            )
            edges.append(
                GraphEdge(
                    source_id=node.node_id,
                    target_id=company_id,
                    relation_type=relation_type,
                    evidence_id=evidence_item.evidence_id,
                    confidence=0.65,
                    description=f"{node.name} 이벤트가 {ticker}와 관련됩니다.",
                )
            )
        elif node.node_type == "sector":
            edges.append(
                GraphEdge(
                    source_id=company_id,
                    target_id=node.node_id,
                    relation_type="mentions",
                    evidence_id=evidence_item.evidence_id,
                    confidence=0.6,
                    description=f"{ticker} 증거가 {node.name} 섹터를 언급합니다.",
                )
            )
        elif node.node_type == "source":
            edges.append(
                GraphEdge(
                    source_id=node.node_id,
                    target_id=company_id,
                    relation_type="source_supports",
                    evidence_id=evidence_item.evidence_id,
                    confidence=0.75,
                    description=f"{node.name} 출처가 {ticker} 관련 근거를 제공합니다.",
                )
            )

    if any(keyword in text for keyword in POSITIVE_KEYWORDS):
        positive_node = GraphNode(
            node_id="event:positive_signal",
            node_type="event",
            name="positive_signal",
            description="긍정 요인",
        )
        if all(node.node_id != positive_node.node_id for node in nodes):
            nodes.append(positive_node)
        edges.append(
            GraphEdge(
                source_id=positive_node.node_id,
                target_id=company_id,
                relation_type="positive_driver",
                evidence_id=evidence_item.evidence_id,
                confidence=0.6,
                description=f"{ticker}에 대한 긍정 요인이 언급되었습니다.",
            )
        )

    return edges
