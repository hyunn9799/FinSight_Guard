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
- Task 1: persist evidence path inside persist_research_run — COMPLETE (commit 74610f7, review SPEC ✅ + QUALITY approved; 7 passed, no regressions)
- Task 2: pending projection marker + negative case — COMPLETE (commit e07ca87, review SPEC ✅ + QUALITY approved; 9 passed)
- Task 3: wire graph_context through save_report_node — COMPLETE (commit 3bbabe5, review SPEC ✅ + QUALITY approved; 49 passed full regression)
- Task 4: mark 004 T050 complete — COMPLETE (commit a174563, doc-only)

## RESUME HERE (next task)
- ALL TASKS COMPLETE (1-4). Branch main..HEAD = 59bfa34..a174563. Final whole-branch review pending,
  then superpowers:finishing-a-development-branch (PR/merge). No implementation work remains.
- Before pytest, re-export DATABASE_URL + TEST_DATABASE_URL (see above) and ensure
  `docker compose up -d db` is running. Recreate .venv + pip install -r requirements.txt if gone.

## Minor findings (for final review triage)
- Task 1: `evidence_id_to_uuid: dict` hint unparameterized (mirrors brief; prefer `dict[str, UUID]`). Non-blocking.
- Task 1: None-guard applied to spec arg inline (`build_evidence_path_spec(gc) if gc is not None else None`)
  rather than wrapping the repo call; functionally identical — repo no-ops on falsy spec (graph_repository.py:170). OK.

## Log
- Env bootstrapped (.venv created, requirements installed, postgres up, finsight_test created).
- Task 1 (RED test) committed as WIP 93a28b7 in a prior session.
- Plan refined + verified against live code (ab3f1fc).
- Task 1 implemented (74610f7): graph_context param + evidence_id map + path persist + evidence_path_id. Reviewed clean. 7 passed.
- Task 2 implemented (e07ca87): pending neo4j projection marker + 2 tests (pinned keys verified char-by-char). Reviewed clean. 9 passed.
- Task 3 implemented (3bbabe5): save_report_node passes graph_context + records evidence_path_id in run metadata + 1 test. Reviewed clean. 49 passed (full regression).
- Task 4 (a174563): marked 004 T050 complete (doc only).
