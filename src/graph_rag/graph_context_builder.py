"""Build lightweight GraphContext objects from EvidenceItem lists."""

from src.evidence.evidence_schema import EvidenceItem
from src.graph_rag.entity_extractor import (
    POSITIVE_KEYWORDS,
    RISK_KEYWORDS,
    extract_entities_from_evidence,
    infer_relations_from_evidence,
    normalize_text,
)
from src.graph_rag.graph_schema import GraphContext, GraphEdge, GraphNode


def _company_node(ticker: str) -> GraphNode:
    clean_ticker = ticker.strip().upper()
    return GraphNode(
        node_id=f"company:{clean_ticker}",
        node_type="company",
        name=clean_ticker,
        description=f"{clean_ticker} 관련 기업 노드",
    )


def _dedupe_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    deduped: dict[str, GraphNode] = {}
    for node in nodes:
        deduped.setdefault(node.node_id, node)
    return list(deduped.values())


def _dedupe_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    deduped: dict[tuple[str, str, str, str | None], GraphEdge] = {}
    for edge in edges:
        key = (edge.source_id, edge.target_id, edge.relation_type, edge.evidence_id)
        deduped.setdefault(key, edge)
    return list(deduped.values())


def _relation_summary(edge: GraphEdge, nodes_by_id: dict[str, GraphNode]) -> str:
    source = nodes_by_id.get(edge.source_id)
    target = nodes_by_id.get(edge.target_id)
    source_name = source.name if source is not None else edge.source_id
    target_name = target.name if target is not None else edge.target_id
    return f"{source_name} -> {target_name}: {edge.relation_type}"


def _extra_keyword_edges(
    ticker: str,
    evidence_item: EvidenceItem,
    nodes: list[GraphNode],
) -> list[GraphEdge]:
    """Add direct keyword relation edges when no specific node was extracted."""
    company_id = f"company:{ticker.strip().upper()}"
    text = normalize_text(evidence_item.description)
    edges: list[GraphEdge] = []

    if any(keyword in text for keyword in RISK_KEYWORDS):
        risk_node = GraphNode(
            node_id="risk:general_risk",
            node_type="risk",
            name="general_risk",
            description="일반 리스크",
        )
        if all(node.node_id != risk_node.node_id for node in nodes):
            nodes.append(risk_node)
        edges.append(
            GraphEdge(
                source_id=risk_node.node_id,
                target_id=company_id,
                relation_type="negative_risk",
                evidence_id=evidence_item.evidence_id,
                confidence=0.55,
                description="증거 설명에 리스크 키워드가 포함되었습니다.",
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
                confidence=0.55,
                description="증거 설명에 긍정 키워드가 포함되었습니다.",
            )
        )

    return edges


def build_graph_context(
    ticker: str,
    evidence_items: list[EvidenceItem],
    focus: str = "unknown",
) -> GraphContext:
    """Build a lightweight graph context from available evidence only."""
    allowed_focus = {"technical", "fundamental", "news_risk", "comprehensive", "unknown"}
    normalized_focus = focus if focus in allowed_focus else "unknown"
    clean_ticker = ticker.strip().upper()
    nodes: list[GraphNode] = [_company_node(clean_ticker)]
    edges: list[GraphEdge] = []

    for evidence_item in evidence_items:
        item_nodes = extract_entities_from_evidence(evidence_item)
        nodes.extend(item_nodes)
        edges.extend(infer_relations_from_evidence(evidence_item, item_nodes))
        edges.extend(_extra_keyword_edges(clean_ticker, evidence_item, item_nodes))
        nodes.extend(item_nodes)

    nodes = _dedupe_nodes(nodes)
    edges = _dedupe_edges(edges)
    risk_relations = [edge for edge in edges if edge.relation_type == "negative_risk"]
    positive_relations = [edge for edge in edges if edge.relation_type == "positive_driver"]
    nodes_by_id = {node.node_id: node for node in nodes}
    key_relations_summary = [_relation_summary(edge, nodes_by_id) for edge in edges[:10]]

    return GraphContext(
        ticker=clean_ticker,
        focus=normalized_focus,  # type: ignore[arg-type]
        nodes=nodes,
        edges=edges,
        key_relations_summary=key_relations_summary,
        risk_relations=risk_relations,
        positive_relations=positive_relations,
        evidence_ids=[item.evidence_id for item in evidence_items if item.evidence_id],
    )
