# 006 Provider-Agnostic MCP Contracts — Progress

Plan: docs/superpowers/plans/2026-06-22-provider-mcp-contracts.md
Branch: 006-provider-mcp-contracts (from main)
Base commit before Task 1: 45a3fdd

## Environment (verified 2026-06-22)
- Runner: `.venv/bin/python -m pytest` (NOT `uv run` — no pyproject.toml, only requirements.txt).
- venv bootstrapped via `uv venv .venv --python 3.12` + `uv pip install --python .venv/bin/python -r requirements.txt`.
- pydantic 2.13.4, sqlalchemy 2.0.51, alembic, pytest all present.
- PostgreSQL (US2 only): needs `docker compose up -d db` + `export TEST_DATABASE_URL="postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test"` (and DATABASE_URL same). DB not yet started — bring up before Phase 4.
- Compile gate: `.venv/bin/python -m compileall src`. Lint: `.venv/bin/python -m ruff check src tests` (verify ruff present; else `uv run ruff`).

## Execution units (group test+impl into green-ending dispatches; preserves T-ID mapping)
- Unit 1 = Plan Task 1 (T001-T003) scaffolding
- Unit 2 = Plan Task 2 (T004) enums
- Unit 3 = Plan Task 3 (T006) entities
- Unit 4 = Plan Task 4 (T005) interfaces
- Unit 5 = Plan Task 5 (T048) safety
- Unit 6 = Plan Task 6 (T007) normalization seams
- Unit 7 = Plan Task 7 (T008) exports + Phase-2 checkpoint
- Unit 8 = Plan Tasks 8+9 (T009+T012) news normalization (test+impl together, green)
- Unit 9 = Plan Task 10 (T010+T013+T014) company/financial/market norm
- Unit 10 = Plan Task 11 (T011) degradation/status tests
- Unit 11 = Plan Task 12 (T015-T017) agent boundary fns
- Unit 12 = Plan Tasks 14+15 (T021,T022,T023) ORM models + migration (DB up)
- Unit 13 = Plan Tasks 13+16 (T018,T019,T024,T025) persistence tests + repo (green)
- Unit 14 = Plan Task 17 (T020+T026) orchestration helper + partial/failed tests (green)
- Unit 15 = Plan Task 18 (T027,T028,T030) graph mapping eligibility
- Unit 16 = Plan Task 19 (T029,T031,T042,T044) VectorReference + ScenarioReportInput
- Unit 17 = Plan Task 20 (T032,T043,T045) SRI builder + observability
- Unit 18 = Plan Task 21 (T033) graph-context boundary
- Unit 19 = Plan Task 22 (T034) coordinator boundary
- Unit 20 = Plan Task 23 (T046) full safety sweep
- Unit 21 = Plan Task 24 (T035,T036) docs
- Unit 22 = Plan Task 25 (T037-T041,T047) final validation gate

## Tasks
- Unit 1 (Plan Task 1, T001-T003): complete (commit 45a3fdd..79b5ff4, review clean)
- Unit 2 (Plan Task 2, T004 enums): complete (commit c78c85c, review clean; 3 passed)
- Unit 3 (Plan Task 3, T006 entities): complete (commit a70e006, review clean; 6 passed)
- Unit 4 (Plan Task 4, T005 interfaces): complete (commit 94fcbbc, review clean; 8 passed)
- Unit 5 (Plan Task 5, T048 safety): complete (commit 07234d9, review clean; 11 passed; matching hand-traced)
- Unit 6 (Plan Task 6, T007 normalization seams): complete (commit e8242d5, review clean; 13 passed)
- Unit 7 (Plan Task 7, T008 exports): complete (commit 359a149, review clean; 14 passed; Phase-2 checkpoint green). PHASE 2 DONE.
- Unit 8 (Plan Tasks 8+9, T009+T012 news norm): complete (commits 186632f, 6d1237c; review clean; 15 passed; logic hand-traced)
- Unit 9 (Plan Task 10, T010+T013+T014 company/financial/market norm): complete (commit 8921280; review clean; 17 passed; boundary verified)
- Unit 10 (Plan Task 11, T011 degradation tests): complete (commit 71a0edc; review clean; 20 passed; test-only, no impl change)
- Unit 11 (Plan Task 12, T015-T017 agent boundary fns): complete (commit a759345; review clean; 23 passed + 16 agent regression). US1 (MVP) DONE.
- Unit 12 (Plan Tasks 14+15, T021/T022/T023 ORM models + migration): complete (commits 0b9ddea, 1845a81; migration downgrade-order fix in follow-up commit). Verified: alembic upgrade head → downgrade -1 → upgrade head round-trips clean (FK children dropped before raw_provider_responses); 23 passed; compileall clean. Postgres up (finsight_guard-db-1 healthy).
- Unit 13 (Plan Tasks 13+16 = T018/T019/T020 + T024/T025, persistence lineage tests + ProviderRepository): complete (commit b3ebd7e; review Spec ✅ + quality Approved; 26 passed with TEST_DATABASE_URL). Controller bundled ALL 3 of Task 13's tests here (so Unit 14 = Task 17 only). IMPORTANT downstream note: ORM column is `normalization_status` (NOT `status`) on technical/wave/etc — brief drafts that say `status=` are wrong; use `normalization_status=`.

- Unit 14 (Plan Task 17, T026 persist_normalization orchestration helper): complete (commit 8a663cc; review Spec ✅ + quality Approved; 4/4 + 27/27 no-regression). Imports at top of persistence.py (no E402); ownership boundary hard-enforced (006 tables only). US2 (Phase 4) DONE.

- Unit 15 (Plan Task 18, T027/T028/T030 graph mapping eligibility): complete (commit 1869861; review Spec ✅ + quality Approved; 2/2 passed). Eligibility specs only (005 owns graph); raw/rows/dicts never eligible; both models subclass _Contract.

- Unit 16 (Plan Task 19, T029/T031/T042/T044 VectorReference + ScenarioReportInput schema + tests): complete (commit 7a68462; TDD RED→GREEN; 25/25 passed). VectorReference: lightweight (source_kind Literal 3 values, canonical_ref_id non-empty validator, optional source_uri/chunk_id, NO score/store/embedding); ScenarioReportInput: full normalized input contract; both subclass _Contract (extra=forbid). ScenarioReportInput appended to SAFETY_CHECKED_CONTRACTS (tuple rebuilt); both exported from __init__.py. No circular imports. Compile gate clean.

- Unit 17 (Plan Task 20, T032/T043/T045 SRI builder + observability): complete (commit 7972599; review Spec ✅ + quality Approved; 28 passed). build_scenario_report_input degradation precedence verified against all 3 obs tests; missing categories → notes+warnings+status, never silent.

- Unit 18 (Plan Task 21, T033 build_contract_graph_context): complete (commit 00a0b16; review Spec ✅ + quality Approved; 3/3 passed). Pure boundary fn, no Neo4j; reflects projection_status into degraded/warnings.

- Unit 19 (Plan Task 22, T034 coordinator scenario boundary): complete (commit aa7b772; review Spec ✅ + quality Approved; 12/12 incl 9/9 coordinator no-regression). Additive boundary fn; import at top (no E402); exposes only normalized fields, no raw payloads. US3 (Phase 5) DONE.

- Unit 20 (Plan Task 23, T046 full safety contract test): complete (commit 1ab93c7; review Spec ✅ + quality Approved; 3/3 passed). Structural sweep over all SAFETY_CHECKED_CONTRACTS (incl ScenarioReportInput) + token-matching pos/neg examples + instance check.

- Unit 21 (Plan Task 24, T035/T036 docs): complete (commit f110a73; review Spec ✅ + quality Approved). quickstart.md validation section corrected to `.venv/bin/python -m ...` (dropped erroneous uv run) + all 6 test files + ruff; PROJECT_PLAN.md "006 Provider Contract Boundary" subsection (ownership split + migration 8f1a2b3c4d5e). Docs-only, lossless.

- Unit 22 (Plan Task 25, T037-T041/T047 FINAL VALIDATION GATE): complete (commit f8e3155). compile clean; 39/39 provider suites; 006 ruff surface clean; full suite 314/314. Lint fixes: mid-file imports relocated to top in 3 agent files + normalization.py + fixtures/provider_contracts.py; unused imports removed from test_provider_contracts.py (FORBIDDEN_TOKENS, FinancialProvider, MarketDataProvider, FinancialProviderResult, MarketDataProviderResult, UTC, datetime, raw_market_data fixture from second import), test_provider_graphrag_mapping_contracts.py (CompanyProfile), test_scenario_report_input_contract.py (CompanyProfile, NewsEvent, NormalizationStatus). test_storage_postgres_schema.py updated with US006_EXPECTED_TABLES to match new provider ORM tables. Pre-existing ruff issues (27 errors in backtest/graph/ux/storage files not touched by 006) left untouched. PHASE 6 (Polish) DONE. Review (Sonnet, b23c583..f8e3155): Spec ✅ + quality Approved — import relocations verified behavior-preserving, removed imports genuinely unused, schema test tightened not weakened, no 006 runtime logic altered.

## ALL 22 UNITS COMPLETE + FINAL REVIEW DONE — AWAITING USER GO TO MERGE
- All implementation + per-unit reviews done. Plan file now committed to branch.
- Final whole-branch review (Opus, 1a5e4b4..530a021): **READY TO MERGE = YES.** Critical 0, Important 0. All 5 accumulated Minors triaged ACCEPT. Surfaced 1 latent correctness bug + 1 defensive gap → both FIXED (commit b6f0f1f, review Spec ✅ + Approved):
  - normalize_news per-item warning scoping (was leaking earlier items' warnings to later NewsEvents in multi-item batches) + 2-item regression test.
  - persist_normalization now raises TypeError on unknown record types (was silent drop).
- Verification: 40 provider-suite tests pass; full suite was 314/314 at Unit 22; migration round-trips clean; 006 ruff surface clean.
- NEXT (AWAITING USER GO): superpowers:finishing-a-development-branch — PR to main or merge. 27 pre-existing ruff errors in non-006 files (backtest/graph/storage) remain out of scope.

## Minor findings (whole-branch review triage)
- Unit 13: `AnalysisRepository` added to `src/db/repositories/__init__.py` export (not requested by brief; additive, non-breaking; reviewer accepted). Final review may keep or drop.
- Unit 14: test uses function-body imports (lines ~117-119) — plan-mandated style across all 006 tests; not E402. Final review note only.
- Unit 14: `persist_normalization` silently drops record types outside {NewsEvent,CompanyProfile,FinancialMetric} (no else-branch). Defensive-only; brief did not require handling. Final review may add `else: raise TypeError`.
- Unit 15: `tests/test_provider_graphrag_mapping_contracts.py` imports `CompanyProfile` unused (F401) — brief artifact. Unit 22 ruff gate must remove. Also `is_graph_eligible` uses `type(x) in ...` (exact, not isinstance) — brief-mandated, fine.
- Unit 17: `scenario_inputs_complete()` fixture uses function-body import (brief-mandated, fixture file already had one at line 39); `any([...])` list vs generator — trivial. Both polish-only, non-blocking.

## Known/accepted Minors (for final review triage)
- Contract model `Warning` (src/providers/enums.py) shadows the builtin `Warning` exception. Functionally harmless (never raised/caught); accepted by plan. Final review may rename to `ProviderWarning` if desired.
- normalization.py imports CompanyProfile/FinancialMetric/NewsEvent currently unused at end of Unit 6 (F401 risk). RESOLVED: Units 8-9 implemented seam bodies that use them.
- SYSTEMIC ruff E402 risk: plan code appends new `import` statements next to appended functions in EXISTING files (mid-file), not at top. Ruff default enables E402. Affected so far: src/agents/news_agent.py, fundamental_agent.py, market_agent.py (Unit 11). Future file-modifying units (persistence.py, coordinator_agent.py, graph_context_builder.py, scenario_input.py) likewise. MITIGATION: future implementers editing existing files told to put imports at TOP. Unit 22 (T041 ruff gate) must relocate any remaining mid-file imports to top-of-file and re-run ruff.

## Log
- 2026-06-22: env bootstrapped (.venv created via uv, requirements installed, deps verified). Plan written. Fresh 006 ledger created (prior ledger was for completed US2 wiring feature).
