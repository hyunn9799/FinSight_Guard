# DB Indexes + Atomic Upserts — Progress

Plan: docs/superpowers/plans/2026-06-19-db-indexes-and-atomic-upserts.md
Issue: #85
Branch: feature/004-db-indexes-and-atomic-upserts (from main, has US1+US2+US3)
Test DB: postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test
Env: export DATABASE_URL + TEST_DATABASE_URL before pytest; .venv/bin/python (3.12). Migration head before = 07c7492119e4 (US3).

## Scope decisions
- Part 1: 6 evidence-based indexes (portfolios.user_id, portfolio_items.portfolio_id, evidence_items.request_id, index_projection_status.status, notifications(user_id,created_at), index_projection_status(source_table,source_id)). ticker_id NOT indexed (unused). Leading-col-of-uq columns NOT indexed (already covered).
- Part 2: atomic ON CONFLICT only for side-effect-free get-or-create: upsert_term, get_or_add_chunk, link_scenario_rule. upsert_status / upsert_setting kept select-then-write (conditional side-logic).

## Tasks
- Task 1: FK/query indexes + migration + index test — pending
- Task 2: atomic ON CONFLICT for 3 get-or-create upserts — pending

## Log
- Branch from main. Plan + issue #85 created.
