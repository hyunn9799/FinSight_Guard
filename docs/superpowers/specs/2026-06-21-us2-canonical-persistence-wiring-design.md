# Design: US2 Canonical Persistence Wiring

**Date**: 2026-06-21
**Status**: Approved (brainstorming) — pending implementation plan
**Feature lineage**: Next slice of `specs/003-platform-storage-architecture` / completes US2 of `specs/004-postgresql-table-schema`

## Problem

The `/analyze` LangGraph workflow builds a `GraphContext` in memory at
`graph_context_node` (`src/graph/workflow.py:285`, stored as `state.graph_context`)
but never persists it. The US2 building blocks that should consume it —
`build_evidence_path_spec()` (`src/graph_rag/graph_context_builder.py:138`) and
`GraphRepository.persist_evidence_path_from_spec()`
(`src/db/repositories/graph_repository.py:160`) — exist and are unit-tested, but
have **zero live callers**. The canonical graph evidence path produced by every
research run is discarded.

## Goal

Make a real `/analyze` run persist its US2 canonical graph evidence path into
PostgreSQL, and mark that path `pending` in `index_projection_status` so a future
derived-index (Neo4j) projection adapter has a durable work queue to consume.

Scope is deliberately narrow: this completes the US2 wiring gap only. It does
**not** build any external index adapter (Pinecone/Neo4j/OpenSearch) — that is
the next slice of spec 003. It does **not** touch US3 (settings / notifications /
portfolio exposure), which is sequenced after this slice.

## Non-Goals

- No external service calls. No Pinecone/Neo4j/OpenSearch/Redis client.
- No new API endpoints or Streamlit pages.
- No change to US1 report/evidence persistence behavior or response shapes.
- No source-document/chunk persistence (separately unwired; out of scope here).

## Architecture & Data Flow

```text
graph_context_node  →  state.graph_context (GraphContext)
        ↓ (coordinator → evaluator → save_report_node)
save_report_node
        ↓ persist_research_run(..., graph_context=state.graph_context)
session_scope() (single atomic transaction)
   ├─ existing: ticker, request, node runs, report, version, evidence, citations
   ├─ NEW: build evidence_id_to_uuid map during the evidence loop
   ├─ NEW: GraphRepository.persist_evidence_path_from_spec(
   │          build_evidence_path_spec(graph_context),
   │          evidence_id_to_uuid=…, request_id, ticker_id)
   └─ NEW: if a path was created →
            ProjectionRepository.upsert_status(
               source_table="evidence_paths", source_id=path.id,
               target_system="neo4j", projection_type="graph_evidence_path",
               projection_key=f"evidence_path:{path.id}",
               idempotency_key=f"evidence_paths:{path.id}:neo4j:graph_evidence_path",
               status="pending")
```

### Projection-status vocabulary

`upsert_status` requires **both** `projection_key` and `idempotency_key`
(uniqueness is enforced on `(target_system, projection_type, idempotency_key)`,
not on `source`). This slice sets:

- `idempotency_key = f"evidence_paths:{path.id}:neo4j:graph_evidence_path"`
- `projection_key   = f"evidence_path:{path.id}"`

The status lifecycle follows the **code**, not the README prose: rows are written
`pending` here and later transition `pending → success / failed / stale` via
`mark_success` / `mark_failure` / `mark_stale` in a future projection adapter.
(The README's "projected" wording is stale; this slice does not adopt it.)

## Components & Changes

### (a) `src/db/persistence.py` — `persist_research_run`

- Add `graph_context=None` keyword parameter.
- Inside the existing evidence loop, accumulate
  `evidence_id_to_uuid[item.evidence_id] = row.id`.
- After the evidence loop (still inside `session_scope()`):
  - `spec = build_evidence_path_spec(graph_context)` when `graph_context` is not None.
  - `path = GraphRepository(session).persist_evidence_path_from_spec(spec, evidence_id_to_uuid=…, request_id=request.id, ticker_id=ticker_row.id)`.
    Returns `None` when there is no graph context, no grounded edges, or no edge
    whose `evidence_id` resolves to a persisted evidence row — composes cleanly.
  - When `path` is not None:
    `ProjectionRepository(session).upsert_status(source_table="evidence_paths", source_id=path.id, target_system="neo4j", projection_type="graph_evidence_path", projection_key=f"evidence_path:{path.id}", idempotency_key=f"evidence_paths:{path.id}:neo4j:graph_evidence_path", status="pending")`.
- Add `evidence_path_id` (the `path.id` or `None`) to the returned result dict.

### (b) `src/graph/workflow.py` — `save_report_node`

- Pass `graph_context=state.get("graph_context")` into `persist_research_run`.
- Thread `evidence_path_id` from the result into the `save_run(...)` metadata and
  the node's internal returned state, alongside the existing `request_id`.
- **Do not** add `evidence_path_id` to the public `AnalyzeResponse` model
  (`main.py:38`). That model selects named fields only, so leaving it untouched
  keeps the `/analyze` response shape unchanged even though the field exists in
  internal workflow state — honoring the "response shapes unchanged" non-goal.

### (c) Tests — `tests/test_storage_postgres_research_flow.py`

Deterministic, no external services (consistent with existing 004 tests):

1. A run **with** a `GraphContext` containing evidence-backed edges persists an
   `EvidencePath` plus ordered `EvidencePathStep` rows, each `node_id` resolving
   to a persisted `evidence_items` UUID, with `step_index` ordering preserved.
2. The same run records exactly one `index_projection_status` row with
   `target_system="neo4j"`, `projection_type="graph_evidence_path"`,
   `status="pending"`, referencing the new path.
3. A run with **no** graph context (or a context with no evidence-backed edges)
   persists **no** evidence path and **no** projection-status row, and the rest
   of the run persists unchanged.
4. A failure during evidence-path persistence rolls back the whole run
   transaction and the workflow still degrades gracefully (existing file-export
   path), confirming canonical atomicity.

## Error Handling

All new writes share the existing `session_scope()` transaction in
`persist_research_run`. A failure anywhere rolls back the entire run persistence
atomically; `save_report_node` already wraps the call in `try/except` and falls
back to JSON/Markdown file export with a deterministic `persistence_error`
degraded warning (`src/graph/workflow.py:373`). No partial canonical state. The
`pending` projection marker is local bookkeeping — it makes no external call, so
it introduces no new failure mode and no live-service dependency.

## Testing Strategy

Extend the existing deterministic PostgreSQL flow test module. Reuse the
established DB session fixture and rollback cleanup from `tests/conftest.py`. No
live providers, no derived index services — same constraints the 004 suite
already enforces.

## Risks & Mitigations

- **Risk**: evidence/path UUID mismatch if the evidence loop and path step
  resolution drift. **Mitigation**: `persist_evidence_path_from_spec` already
  filters steps to those whose `evidence_id` is present in
  `evidence_id_to_uuid`, so unmatched edges are dropped rather than producing
  dangling references; test (1) asserts every step resolves.
- **Risk**: speculative `pending` rows accumulate with no consumer yet.
  **Mitigation**: this is the documented US2 purpose (a durable projection
  queue); `upsert_status` is idempotent on
  `(target_system, projection_type, idempotency_key)`, and the per-path
  `idempotency_key` is deterministic, so re-runs update in place rather than
  duplicate.

## Definition of Done

- `persist_research_run` persists the graph evidence path + steps and a `pending`
  projection-status row when a grounded `GraphContext` is present.
- `save_report_node` passes `graph_context` and surfaces `evidence_path_id`.
- New deterministic tests (1)–(4) pass; existing 004 storage + regression suites
  still pass.
- No external service dependency introduced; `/analyze` response shape
  (`AnalyzeResponse`) unchanged.
- `specs/004-postgresql-table-schema/tasks.md` **T050** (canonical evidence-path
  persistence integration) marked complete once this slice lands.

## Process note

No formal `specs/005-…` SDD bundle is created. This is wiring of an already-
specified 004/US2 capability, not a new feature, so this single design doc plus a
lightweight implementation plan (`docs/superpowers/plans/…`) is the right weight.
