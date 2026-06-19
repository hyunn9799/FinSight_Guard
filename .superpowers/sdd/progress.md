# US3 PostgreSQL UX Tables — Progress

Plan: docs/superpowers/plans/2026-06-19-postgresql-research-ledger-us3.md
Branch: feature/004-postgresql-us3 (from main, which has US2 merged via #83)
Test DB: postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test (docker compose `db`)
Env: export DATABASE_URL + TEST_DATABASE_URL before pytest; run via .venv/bin/python (3.12). Migration head before US3 = d305f4d1ff77 (US2).

## Tasks
- Task 1: 4 ORM models + constants + make_user fixture + migration + schema test — pending
- Task 2: SettingsRepository (user_settings) — pending
- Task 3: NotificationRepository (in-app only) — pending
- Task 4: PortfolioRepository (portfolios + items) — pending
- Task 5: UserRepository + FR-019 anonymization — pending
- Task 6: Safety presence assertion + full validation + docs — pending

## Log
- US3 branch created from main@502a810 (US2 merged). Baseline US1+US2 schema+safety: 10 passed. Plan committed.
