"""Provider-agnostic MCP contract layer (006).

Owns provider interface contracts, normalization contracts, lineage
requirements, and graph-mapping eligibility. Does NOT own canonical tables
(004) or the graph model (005). No live MCP/API/Neo4j/vector calls live here.

Stable public exports are populated in T008.
"""
