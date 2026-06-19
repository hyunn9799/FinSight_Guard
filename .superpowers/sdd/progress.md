# US2 PostgreSQL Research Ledger — Progress

Plan: docs/superpowers/plans/2026-06-19-postgresql-research-ledger-us2.md
Branch: feature/004-postgresql-us2
Test DB: postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test (docker compose `db`)

## Tasks
- Task 1: ORM models + constants + migration — complete (commits cad33dc..112292d, review clean after 1 fix)
- Task 2: ProjectionRepository status lifecycle — complete (commits de57d99..905c405, review clean after 1 fix)
- Task 3: ProjectionRepository keyword terms — complete (commit d83ce15, review clean, Approved no fixes)
- Task 4: Idempotent document chunks — complete (commit 8ad925f, review Approved no fixes; ⚠️ resolved: DocumentChunk has UniqueConstraint(source_document_id, chunk_index) so one_or_none cannot raise)
- Task 5: GraphRepository rules/scenarios/conditions/joins — complete (commits aedb41d..d828cf0, review Approved; fix: savepoint-isolated IntegrityError test for pristine output)
- Task 6: GraphRepository evidence paths + steps — complete (commit 4430aa6, review Approved no fixes; savepoint pattern applied to step-index uniqueness test)
- Task 7: graph_context_builder spec + persistence — complete (commit 3a6054c, review Approved; purity boundary confirmed no DB import; added required ticker="AAPL" to brief's EvidenceItem test)
- Task 8: Safety scan + full validation + docs — complete (commit bd5b558, review Approved; full suite 254 passed/0 failed/0 skipped, US2-related failures 0)

## Log
- Env: venv (py3.12) created, requirements installed, docker `db` up, finsight_test DB created & migrated to US1 head 40be5b7bbb1d. Baseline US1 schema+safety tests: 8 passed.
- Task 1: complete (commits cad33dc..112292d, review clean after 1 fix — restored exact-set metadata contract test)
- Task 2: complete (commits de57d99..905c405; fix added test pinning re-upsert status reset semantics)

## FINAL WHOLE-BRANCH REVIEW (opus): Ready to merge = YES
- Range 2461648..bd5b558 (13 commits). Critical 0, Important 0, Minor only.
- Confirmed: 8 tables (no US3 leak); migration chains after US1 head 40be5b7bbb1d, creates/drops only the 8, backward compatible; pure builder/repository boundary held (no DB import); SER-008 non-mutation real (failure_warning pure-read + refresh assertions + skip-unresolvable); SER-001/SC-005 full-schema token scan; IntegrityError tests savepoint-isolated; full suite 254 passed / 0 failed / 0 skipped.
- All "forward import" / forward-reference concerns resolve clean at HEAD (no unused import remains).
- Optional non-blocking tidy-ups suggested: move US2_EXPECTED_TABLES above its first use; add failure_warning non-failed guard/docstring.

## Minor findings deferred to final whole-branch review
- No lint config / CI present in repo (ruff/flake8/F401 not enforced).
- Task 2: `KeywordTerm` imported in projection_repository.py but unused until Task 3 (same file). Plan-verbatim forward import. Confirm used by end of Task 3.
- Task 2: `make_request, make_ticker` imported in test_storage_postgres_projection_status.py but unused until Task 5/6/7 (same file). Plan-verbatim forward import. Confirm used by end of Task 7.
- Task 2 (reviewer Minor, not fixed): list_by_status untested; failure_warning has no guard against non-failed status; last_attempt_at not asserted. Triage at final review.
- Task 3 (reviewer Minor, not fixed): list_terms lacks secondary sort key (nondeterministic for same normalized_term); upsert_term ignores `term` display-form on collision (latest spelling does not win); in-function import style. Triage at final review.
