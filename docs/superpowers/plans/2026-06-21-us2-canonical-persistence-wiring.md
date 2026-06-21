# US2 Canonical Persistence Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the in-memory `GraphContext` evidence path produced by every `/analyze` run into PostgreSQL, plus a `pending` `index_projection_status` marker, so the already-built US2 repositories stop being dead code.

**Architecture:** Thread `state.graph_context` from `save_report_node` into `persist_research_run`. Inside the existing `session_scope()` transaction, build an `evidence_id → row.id` map during the evidence loop, then call the existing `GraphRepository.persist_evidence_path_from_spec(build_evidence_path_spec(graph_context), …)` and write one `ProjectionRepository.upsert_status(... status="pending")` row for the new path. All writes are atomic with the run; failures fall through the existing degrade-to-file path.

**Tech Stack:** Python, SQLAlchemy 2.x, pytest. No new dependencies. No external services.

**Design doc:** `docs/superpowers/specs/2026-06-21-us2-canonical-persistence-wiring-design.md`

## Global Constraints

- No external service calls; no Pinecone/Neo4j/OpenSearch/Redis client. The Neo4j projection-status row is local bookkeeping only.
- All new writes share the existing `session_scope()` transaction in `persist_research_run` — atomic with the run, no partial canonical state.
- `/analyze` response shape (`AnalyzeResponse`, `main.py:38`) MUST remain unchanged — do **not** add `evidence_path_id` to that model.
- `upsert_status` requires **both** `projection_key` and `idempotency_key`; uniqueness is on `(target_system, projection_type, idempotency_key)`. Use exactly:
  - `idempotency_key = f"evidence_paths:{path.id}:neo4j:graph_evidence_path"`
  - `projection_key   = f"evidence_path:{path.id}"`
- Projection status vocabulary follows code: `pending` → `success`/`failed`/`stale`. Never write `"projected"`.
- Tests are deterministic and DB-gated by `TEST_DATABASE_URL` (the `@REQUIRES_DB` marker); no live providers.

---

## File Structure

- **Modify** `src/db/persistence.py` — add `graph_context` param to `persist_research_run`; build evidence-id→uuid map; persist evidence path + projection marker; return `evidence_path_id`. (Task 1, 2)
- **Modify** `src/graph/workflow.py:362-422` — pass `graph_context` into the call; add `evidence_path_id` to `save_run` metadata. (Task 3)
- **Modify** `tests/test_storage_postgres_research_flow.py` — add evidence-path persistence + projection-marker tests, and a no-graph-context negative test. (Task 1, 2)
- **Modify** `specs/004-postgresql-table-schema/tasks.md` — mark T050 complete. (Task 4)

The existing degrade-on-persistence-error test (`test_save_report_node_degrades_on_persistence_error`, lines 183-216) already covers atomic rollback behavior — no new test needed for that path.

---

### Task 1: Persist evidence path inside `persist_research_run`

**Files:**
- Modify: `src/db/persistence.py`
- Test: `tests/test_storage_postgres_research_flow.py`

**Interfaces:**
- Consumes:
  - `build_evidence_path_spec(graph_context: GraphContext) -> dict | None` from `src/graph_rag/graph_context_builder.py`
  - `GraphRepository(session).persist_evidence_path_from_spec(spec, *, evidence_id_to_uuid: dict, request_id=None, ticker_id=None) -> EvidencePath | None` from `src/db/repositories/graph_repository.py`
  - `EvidenceItem.evidence_id: str` and `evidence_repo.add_evidence(item, ...) -> row` with `row.id: UUID`
- Produces:
  - `persist_research_run(..., graph_context=None)` keyword param
  - `result["evidence_path_id"]: UUID | None` in the returned dict

- [ ] **Step 1: Write the failing test**

Add to `tests/test_storage_postgres_research_flow.py`. Place a shared helper near the top (after `_sample_evidence`), then the test:

```python
def _sample_graph_context(ticker="AAPL"):
    from src.graph_rag.graph_schema import GraphContext, GraphEdge, GraphNode
    return GraphContext(
        ticker=ticker,
        focus="comprehensive",
        nodes=[
            GraphNode(node_id="ev-source-1", node_type="event", name="실적 발표"),
            GraphNode(node_id=f"company:{ticker}", node_type="company", name=ticker),
        ],
        edges=[
            GraphEdge(
                source_id="ev-source-1",
                target_id=f"company:{ticker}",
                relation_type="positive_driver",
                evidence_id="ev-1",
                description="실적 호조가 주가를 견인",
            )
        ],
        key_relations_summary=["실적 호조 → 주가 상승"],
        evidence_ids=["ev-1"],
    )


@REQUIRES_DB
def test_persist_run_stores_graph_evidence_path(db_session, monkeypatch):
    import src.db.persistence as persistence
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield db_session

    monkeypatch.setattr(persistence, "session_scope", _scope)

    out = persistence.persist_research_run(
        run_id="run-gc", ticker="AAPL", status="success",
        report=_sample_report(), evidence=_sample_evidence(),
        evaluation={"overall_pass": True, "source_grounding_score": 0.9},
        graph_context=_sample_graph_context(),
    )

    from src.db.models import EvidencePath, EvidencePathStep, EvidenceItemRecord

    path = db_session.get(EvidencePath, out["evidence_path_id"])
    assert path is not None
    assert path.path_type == "graph_context"
    assert path.request_id == out["request_id"]

    steps = (
        db_session.query(EvidencePathStep)
        .filter(EvidencePathStep.evidence_path_id == path.id)
        .order_by(EvidencePathStep.step_index)
        .all()
    )
    assert len(steps) == 1
    assert steps[0].step_index == 0
    assert steps[0].node_table == "evidence_items"
    assert steps[0].relationship_type == "positive_driver"
    # the step's node_id resolves to a persisted evidence row
    assert db_session.get(EvidenceItemRecord, steps[0].node_id) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py::test_persist_run_stores_graph_evidence_path -v`
Expected: FAIL — `persist_research_run() got an unexpected keyword argument 'graph_context'` (or `KeyError: 'evidence_path_id'` if the kwarg is silently accepted). If `TEST_DATABASE_URL` is unset the test SKIPS — set it first (see `tests/conftest.py` / `tests/fixtures/postgres.py`).

- [ ] **Step 3: Write minimal implementation**

In `src/db/persistence.py`, add the imports near the existing repository imports (top of file):

```python
from src.db.repositories.graph_repository import GraphRepository
from src.graph_rag.graph_context_builder import build_evidence_path_spec
```

Change the signature of `persist_research_run` to accept `graph_context`:

```python
def persist_research_run(
    *,
    run_id: str,
    ticker: str,
    status: str,
    report,
    evidence,
    evaluation,
    node_runs: list[dict] | None = None,
    missing_data_notes: list[str] | None = None,
    graph_context=None,
) -> dict:
```

Add `evidence_path_id` to the early-return result dict (so the report-is-None path stays consistent):

```python
        result = {
            "request_id": request.id,
            "report_id": None,
            "report_version_id": None,
            "evidence_path_id": None,
        }
        if report is None:
            return result
```

Replace the existing evidence loop to capture the id map, then persist the path after it. The current loop is:

```python
        for item in evidence:
            row = evidence_repo.add_evidence(item, request_id=request.id, ticker_id=ticker_row.id)
            evidence_repo.add_citation(
                version.id, row.id, section_name="evidence_summary", claim_text=item.description
            )
```

Replace it with:

```python
        evidence_id_to_uuid: dict = {}
        for item in evidence:
            row = evidence_repo.add_evidence(item, request_id=request.id, ticker_id=ticker_row.id)
            evidence_repo.add_citation(
                version.id, row.id, section_name="evidence_summary", claim_text=item.description
            )
            if getattr(item, "evidence_id", None):
                evidence_id_to_uuid[item.evidence_id] = row.id

        evidence_path = GraphRepository(session).persist_evidence_path_from_spec(
            build_evidence_path_spec(graph_context) if graph_context is not None else None,
            evidence_id_to_uuid=evidence_id_to_uuid,
            request_id=request.id,
            ticker_id=ticker_row.id,
        )
```

Set the result before returning:

```python
        result["report_id"] = report_row.id
        result["report_version_id"] = version.id
        result["evidence_path_id"] = evidence_path.id if evidence_path is not None else None
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py::test_persist_run_stores_graph_evidence_path -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/persistence.py tests/test_storage_postgres_research_flow.py
git commit -m "feat(db): persist graph evidence path from research run (US2 wiring)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Write `pending` projection-status marker + negative case

**Files:**
- Modify: `src/db/persistence.py`
- Test: `tests/test_storage_postgres_research_flow.py`

**Interfaces:**
- Consumes: `ProjectionRepository(session).upsert_status(*, source_table, source_id, target_system, projection_type, projection_key, idempotency_key, status="pending") -> IndexProjectionStatus` from `src/db/repositories/projection_repository.py`
- Produces: one `index_projection_status` row per persisted evidence path; no row when no path is created.

- [ ] **Step 1: Write the failing tests**

Add two tests to `tests/test_storage_postgres_research_flow.py`:

```python
@REQUIRES_DB
def test_persist_run_marks_projection_pending(db_session, monkeypatch):
    import src.db.persistence as persistence
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield db_session

    monkeypatch.setattr(persistence, "session_scope", _scope)

    out = persistence.persist_research_run(
        run_id="run-proj", ticker="AAPL", status="success",
        report=_sample_report(), evidence=_sample_evidence(),
        evaluation={"overall_pass": True, "source_grounding_score": 0.9},
        graph_context=_sample_graph_context(),
    )

    from src.db.models import IndexProjectionStatus

    rows = (
        db_session.query(IndexProjectionStatus)
        .filter(IndexProjectionStatus.source_id == out["evidence_path_id"])
        .all()
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.source_table == "evidence_paths"
    assert row.target_system == "neo4j"
    assert row.projection_type == "graph_evidence_path"
    assert row.status == "pending"
    assert row.idempotency_key == f"evidence_paths:{out['evidence_path_id']}:neo4j:graph_evidence_path"
    assert row.projection_key == f"evidence_path:{out['evidence_path_id']}"


@REQUIRES_DB
def test_persist_run_without_graph_context_skips_path_and_projection(db_session, monkeypatch):
    import src.db.persistence as persistence
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield db_session

    monkeypatch.setattr(persistence, "session_scope", _scope)

    out = persistence.persist_research_run(
        run_id="run-nogc", ticker="AAPL", status="success",
        report=_sample_report(), evidence=_sample_evidence(),
        evaluation={"overall_pass": True, "source_grounding_score": 0.9},
        graph_context=None,
    )

    from src.db.models import EvidencePath, IndexProjectionStatus

    assert out["evidence_path_id"] is None
    assert db_session.query(EvidencePath).count() == 0
    assert db_session.query(IndexProjectionStatus).count() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py::test_persist_run_marks_projection_pending tests/test_storage_postgres_research_flow.py::test_persist_run_without_graph_context_skips_path_and_projection -v`
Expected: `test_persist_run_marks_projection_pending` FAILS (`assert len(rows) == 1` → 0 rows, no marker written yet). The negative test should already PASS (no path means no marker) — that is fine; it locks the behavior in.

- [ ] **Step 3: Write minimal implementation**

In `src/db/persistence.py`, add the import alongside the others:

```python
from src.db.repositories.projection_repository import ProjectionRepository
```

Right after the `evidence_path = GraphRepository(...).persist_evidence_path_from_spec(...)` block from Task 1, add:

```python
        if evidence_path is not None:
            ProjectionRepository(session).upsert_status(
                source_table="evidence_paths",
                source_id=evidence_path.id,
                target_system="neo4j",
                projection_type="graph_evidence_path",
                projection_key=f"evidence_path:{evidence_path.id}",
                idempotency_key=f"evidence_paths:{evidence_path.id}:neo4j:graph_evidence_path",
                status="pending",
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py::test_persist_run_marks_projection_pending tests/test_storage_postgres_research_flow.py::test_persist_run_without_graph_context_skips_path_and_projection -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add src/db/persistence.py tests/test_storage_postgres_research_flow.py
git commit -m "feat(db): mark graph evidence path pending for neo4j projection (US2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire `graph_context` through `save_report_node`

**Files:**
- Modify: `src/graph/workflow.py:362-422`
- Test: `tests/test_storage_postgres_research_flow.py`

**Interfaces:**
- Consumes: `state.get("graph_context")` (a `GraphContext | None` placed in state by `graph_context_node`, `src/graph/workflow.py:298`); `persisted["evidence_path_id"]` from Task 1.
- Produces: `evidence_path_id` key in the `save_run(...)` metadata dict.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_storage_postgres_research_flow.py`. This asserts the node passes `graph_context` through and records the path id in run metadata:

```python
@REQUIRES_DB
def test_save_report_node_persists_graph_path_and_metadata(db_session, monkeypatch, tmp_path):
    import src.db.persistence as persistence
    import src.graph.workflow as workflow
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        yield db_session

    saved_meta = {}
    monkeypatch.setattr(persistence, "session_scope", _scope)
    monkeypatch.setattr(workflow, "save_report_json", lambda run_id, payload: str(tmp_path / "r.json"))
    monkeypatch.setattr(workflow, "save_report_markdown", lambda run_id, report: str(tmp_path / "r.md"))
    monkeypatch.setattr(workflow, "save_run", lambda run_id, meta: saved_meta.update(meta))

    from src.graph.state import EvaluationResult
    evaluation = EvaluationResult(
        overall_pass=True, source_grounding_score=1.0, numeric_consistency_score=1.0,
        safety_score=1.0, risk_disclosure_score=1.0, freshness_score=1.0,
    )
    state = {
        "run_id": "run-node-gc", "ticker": "AAPL", "status": "success",
        "draft_report": _sample_report(), "evidence": _sample_evidence(),
        "evaluation_result": evaluation, "graph_context": _sample_graph_context(),
    }
    out = workflow.save_report_node(state)
    assert out["status"] == "success"

    from src.db.models import EvidencePath
    assert saved_meta.get("evidence_path_id") is not None
    assert db_session.get(EvidencePath, saved_meta["evidence_path_id"]) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py::test_save_report_node_persists_graph_path_and_metadata -v`
Expected: FAIL — `saved_meta.get("evidence_path_id")` is `None` because the node neither passes `graph_context` nor records the id.

- [ ] **Step 3: Write minimal implementation**

In `src/graph/workflow.py`, find the `persist_research_run(...)` call inside `save_report_node` (around line 364) and add the `graph_context` argument:

```python
        persisted = persist_research_run(
            run_id=run_id,
            ticker=state.get("ticker", ""),
            status=final_status,
            report=report,
            evidence=state.get("evidence", []),
            evaluation=evaluation,
            missing_data_notes=state.get("missing_data_notes", []),
            graph_context=state.get("graph_context"),
        )
```

Then in the success-path `save_run(...)` call (around line 412-422), add the `evidence_path_id` key to the metadata dict:

```python
        save_run(
            run_id,
            {
                "ticker": state.get("ticker"),
                "status": final_status,
                "evaluation_score": evaluation_score,
                "report_path": report_path,
                "markdown_path": markdown_path,
                "request_id": str(persisted.get("request_id")) if persisted.get("request_id") else None,
                "evidence_path_id": str(persisted.get("evidence_path_id")) if persisted.get("evidence_path_id") else None,
            },
        )
```

Do **not** change the returned node-state dict's public-facing fields beyond what already exists, and do **not** touch `AnalyzeResponse` in `main.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py::test_save_report_node_persists_graph_path_and_metadata -v`
Expected: PASS

- [ ] **Step 5: Run the full flow + regression suite**

Run:
```bash
.venv/bin/python -m pytest tests/test_storage_postgres_research_flow.py tests/test_storage_postgres_projection_status.py -v
.venv/bin/python -m pytest tests/test_workflow_routing.py tests/test_api.py -v
```
Expected: PASS (DB-gated tests SKIP cleanly if `TEST_DATABASE_URL` is unset; regression tests pass either way).

- [ ] **Step 6: Commit**

```bash
git add src/graph/workflow.py tests/test_storage_postgres_research_flow.py
git commit -m "feat(graph): pass graph_context to persistence + record path id in run metadata (US2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Mark 004 T050 complete

**Files:**
- Modify: `specs/004-postgresql-table-schema/tasks.md`

**Interfaces:**
- Consumes: nothing. Produces: nothing. Documentation-only bookkeeping.

- [ ] **Step 1: Update the task line**

In `specs/004-postgresql-table-schema/tasks.md`, change the T050 line from:

```
- [ ] T050 [US2] Integrate canonical evidence path persistence with graph context builder output in `src/graph_rag/graph_context_builder.py`
```

to (check the box and note the live wiring):

```
- [x] T050 [US2] Integrate canonical evidence path persistence with graph context builder output in `src/graph_rag/graph_context_builder.py` (live-wired into `persist_research_run` via `save_report_node`; see docs/superpowers/plans/2026-06-21-us2-canonical-persistence-wiring.md)
```

- [ ] **Step 2: Commit**

```bash
git add specs/004-postgresql-table-schema/tasks.md
git commit -m "chore(sdd): mark 004 T050 complete — evidence path live-wired (US2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage** (against the design doc):
- Design (a) `persist_research_run` graph_context param + evidence map + path persist + `evidence_path_id` → Task 1. ✅
- Design (a) projection `pending` marker with pinned keys → Task 2. ✅
- Design (b) `save_report_node` passes `graph_context`, records `evidence_path_id` in `save_run` metadata, leaves `AnalyzeResponse` untouched → Task 3. ✅
- Design tests (1) path+ordered steps resolving to evidence UUID → Task 1 Step 1. (2) projection pending row → Task 2 Step 1. (3) no graph context → no path/marker → Task 2 Step 1 negative test. (4) graceful degrade on persistence failure → already covered by existing `test_save_report_node_degrades_on_persistence_error` (noted in File Structure). ✅
- Design DoD: T050 marked complete → Task 4. ✅
- Non-goal "no external service": only a local `pending` row is written; no client. ✅

**2. Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step shows full code. ✅

**3. Type consistency:** `evidence_path_id` used consistently across Tasks 1–3. `upsert_status` called with all six required kwargs (`source_table`, `source_id`, `target_system`, `projection_type`, `projection_key`, `idempotency_key`) + `status`. `persist_evidence_path_from_spec` called with `spec`, `evidence_id_to_uuid`, `request_id`, `ticker_id` matching its signature. `build_evidence_path_spec` takes a `GraphContext` and returns `dict | None`; guarded with `if graph_context is not None`. ✅
