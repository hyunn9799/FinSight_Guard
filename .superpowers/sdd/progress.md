# US3 PostgreSQL UX Tables — Progress

Plan: docs/superpowers/plans/2026-06-19-postgresql-research-ledger-us3.md
Branch: feature/004-postgresql-us3 (from main, which has US2 merged via #83)
Test DB: postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test (docker compose `db`)
Env: export DATABASE_URL + TEST_DATABASE_URL before pytest; run via .venv/bin/python (3.12). Migration head before US3 = d305f4d1ff77 (US2).

## Tasks
- Task 1: 4 ORM models + constants + make_user fixture + migration + schema test — complete (commit 837bd98, review Approved no fixes; migration 07c7492119e4 chains after d305f4d1ff77)
- Task 2: SettingsRepository (user_settings) — complete (commit 6f15226, review Approved no fixes)
- Task 3: NotificationRepository (in-app only) — complete (commit f253004, review Approved; ⚠️ resolved: session.query is house style across all repos, not an inconsistency)
- Task 4: PortfolioRepository (portfolios + items) — complete (commit 8ffe4a1, review Approved; "Important" item_metadata={} is moot — model column is nullable=False default=dict, so {} is required not a silent write)
- Task 5: UserRepository + FR-019 anonymization — complete (commit dafa78d, review Approved; analysis_requests not imported → audit preservation structurally guaranteed). Minor (fix at final): unused make_user import in test_storage_postgres_privacy.py (genuinely dead — file complete at Task 5).
- Task 6: Safety presence assertion + full validation + docs — complete (commit b2c7b62, review Approved; full suite 268 passed / 0 failed / 0 skipped, US3-related failures 0). Minor (fix at final): US3_TABLES constant defined mid-file not at module top.

## FINAL WHOLE-BRANCH REVIEW (opus): Ready to merge = YES
- Range 502a810..b2c7b62 (8 commits). Critical 0, Important 0, Minor only.
- FR-019 anonymize verified correct: PII cleared, UX hard-deleted child-first (items before portfolios), analysis_requests untouched & audit-preservation proven by real-DB test. FR-020 (no delivery fields) + SER-001 (no order fields) honored. 4 tables, migration chains after d305f4d1ff77 touching only the 4. flush-not-commit. Full suite 268 passed / 0 / 0.
- Cleanups applied post-review (commit after b2c7b62): dropped dead make_user import in privacy test; hoisted US3_TABLES to module top.

## Log
- US3 branch created from main@502a810 (US2 merged). Baseline US1+US2 schema+safety: 10 passed. Plan committed.
- Task 1: complete (commit 837bd98). Minor (deferred to final): US3 schema test engine.dispose() not in try/finally (pre-existing US2 pattern).
- Task 2: complete (commit 6f15226). Minor (transient): make_ticker imported in test_storage_postgres_ux.py, unused until Task 3/4 (same file). No lint gate. Confirm used by end of Task 4.
