# US2 Canonical Persistence Wiring — Progress

Plan: docs/superpowers/plans/2026-06-21-us2-canonical-persistence-wiring.md
Design: docs/superpowers/specs/2026-06-21-us2-canonical-persistence-wiring-design.md
Branch: feature/005-us2-canonical-persistence-wiring (from main @ 7ad2f1d)
Base commit before Task 1: b712067

## Test DB / env (export before pytest)
- export DATABASE_URL="postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test"
- export TEST_DATABASE_URL="postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test"
- runner: .venv/bin/python (3.12); postgres via `docker compose up -d db`, container finsight_guard-db-1, finsight_test DB created.
- Smoke check: existing tests/test_storage_postgres_research_flow.py = 6 passed.

## Scope
Wire built-but-uncalled US2 graph evidence-path persistence into persist_research_run + save_report_node, plus a pending neo4j index_projection_status marker. No external services. /analyze response shape unchanged.

## Tasks
- Task 1: persist evidence path inside persist_research_run — IN PROGRESS (RED test added, implementation NOT done)
- Task 2: pending projection marker + negative case — pending
- Task 3: wire graph_context through save_report_node — pending
- Task 4: mark 004 T050 complete — pending

## RESUME HERE (next session)
- Task 1 RED test `test_persist_run_stores_graph_evidence_path` + `_sample_graph_context` helper
  are committed in tests/test_storage_postgres_research_flow.py but EXPECTED TO FAIL — the
  persist_research_run implementation (graph_context param + evidence_id map + GraphRepository
  call + evidence_path_id in result) is NOT yet written. See plan Task 1 Step 3 for exact code.
- Resume by implementing plan Task 1 Step 3 in src/db/persistence.py, then continue Tasks 2-4.
- Before pytest, re-export DATABASE_URL + TEST_DATABASE_URL (see above) and ensure
  `docker compose up -d db` is running. Recreate .venv + pip install -r requirements.txt if gone.

## Minor findings (for final review triage)
- (none yet)

## Log
- Env bootstrapped (.venv created, requirements installed, postgres up, finsight_test created).
- Task 1 implementer added the RED test only; dispatch interrupted before implementation.
  Committed as WIP and pushed to origin to preserve progress remotely.
