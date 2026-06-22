# Unit 22 Report — Final Validation Gate (T037-T041, T047)

Date: 2026-06-22
Branch: 006-provider-mcp-contracts
Commit: f8e3155

---

## Step 1 (T037): Compile

```
.venv/bin/python -m compileall -q src
```

Result: **CLEAN** — no output, zero errors.

---

## Step 2 (T038/T039/T040/T047): Six Provider Test Suites (before lint fixes)

```
export DATABASE_URL="postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test"
export TEST_DATABASE_URL="$DATABASE_URL"
.venv/bin/python -m pytest tests/test_provider_contracts.py tests/test_provider_persistence_contracts.py tests/test_provider_graphrag_mapping_contracts.py tests/test_scenario_report_input_contract.py tests/test_provider_observability_contract.py tests/test_provider_safety_contract.py -q
```

Result: **39 passed, 1 warning** (Alembic path_separator deprecation, benign).

Breakdown:
- test_provider_contracts.py: 18 tests
- test_provider_persistence_contracts.py: 2 tests (real DB, TEST_DATABASE_URL set)
- test_provider_graphrag_mapping_contracts.py: 5 tests
- test_scenario_report_input_contract.py: 3 tests
- test_provider_observability_contract.py: 8 tests
- test_provider_safety_contract.py: 3 tests

---

## Step 3 (T041): Ruff — Full Output Before Fixes

Running `ruff check src tests` produced 58 errors total:

### 006-surface errors (30, all fixed):

| File | Line | Code | Issue |
|------|------|------|-------|
| src/agents/fundamental_agent.py | 153 | E402 | `from src.providers.entities import CompanyProfile, FinancialMetric` mid-file |
| src/agents/market_agent.py | 231 | E402 | `from src.providers.entities import TechnicalAnalysisResult` mid-file |
| src/agents/news_agent.py | 241 | E402 | `from src.providers.entities import NewsEvent as ContractNewsEvent` mid-file |
| src/providers/normalization.py | 80 | E402 | `from datetime import datetime` mid-file |
| tests/fixtures/provider_contracts.py | 39 | E402 | `from src.providers.normalization import RawCompanyPayload, RawFinancialRow, RawMarketData` mid-file |
| tests/test_provider_contracts.py | 41 | E402 | `from datetime import UTC, datetime` mid-file |
| tests/test_provider_contracts.py | 41 | F401 | `datetime.UTC` unused |
| tests/test_provider_contracts.py | 41 | F401 | `datetime.datetime` unused |
| tests/test_provider_contracts.py | 43 | E402 | entity imports mid-file |
| tests/test_provider_contracts.py | 50 | E402 | `from src.providers.enums import RuleStatus` mid-file |
| tests/test_provider_contracts.py | 111 | E402 | interface imports mid-file |
| tests/test_provider_contracts.py | 112-115 | F401 | FinancialProvider, FinancialProviderResult, MarketDataProvider, MarketDataProviderResult unused |
| tests/test_provider_contracts.py | 152 | E402 | `from pydantic import BaseModel` mid-file |
| tests/test_provider_contracts.py | 154 | E402 | safety imports mid-file |
| tests/test_provider_contracts.py | 155 | F401 | `FORBIDDEN_TOKENS` unused |
| tests/test_provider_contracts.py | 195-200 | E402 | normalization imports mid-file |
| tests/test_provider_contracts.py | 262-267 | E402 | normalize_company/financials/market + fixture imports mid-file |
| tests/test_provider_contracts.py | 323-326 | E402 | agent boundary imports mid-file |
| tests/test_provider_contracts.py | 326 | F811 | `raw_company_payload` redefined (imported again on line 326, previously on 266) |
| tests/test_provider_contracts.py | 326 | F811 | `raw_financial_rows` redefined |
| tests/test_provider_graphrag_mapping_contracts.py | 7 | F401 | `CompanyProfile` unused |
| tests/test_scenario_report_input_contract.py | 6 | F401 | `CompanyProfile` unused |
| tests/test_scenario_report_input_contract.py | 6 | F401 | `NewsEvent` unused |
| tests/test_scenario_report_input_contract.py | 7 | F401 | `NormalizationStatus` unused |

### Pre-existing errors (28, NOT fixed — files unrelated to 006):

| File | Count | Issues |
|------|-------|--------|
| src/backtest/regime.py | 1 | F401 numpy unused |
| src/backtest/robust.py | 2 | F401 TYPE_CHECKING unused; F821 BacktestResult undefined |
| src/db/migrations/versions/56464a69bd55_add_query_indexes.py | 1 | F401 sqlalchemy unused |
| src/graph/workflow.py | 1 | F401 TypedDict unused |
| tests/test_api_robust_optimization.py | 4 | F401 pytest+3 backtest imports unused |
| tests/test_optimization_evidence.py | 1 | F401 pytest unused |
| tests/test_regime_performance.py | 1 | F401 pytest unused |
| tests/test_robust_optimization_safety.py | 1 | F401 pytest unused |
| tests/test_robust_optimizer_baselines.py | 1 | F401 pytest unused |
| tests/test_robust_optimizer_metrics.py | 7 | F401 numpy+CandidateMetrics+OptimizationRun unused; E402 4x mid-file imports |
| tests/test_robust_optimizer_scoring.py | 1 | F401 pytest unused |
| tests/test_storage_postgres_privacy.py | 1 | F841 `note` assigned but unused |
| tests/test_storage_postgres_ux.py | 1 | F841 `unread` assigned but unused |
| tests/test_walk_forward_optimizer.py | 2 | E402 2x mid-file imports |
| tests/test_workflow_e2e.py | 1 | F401 EvaluationResult unused |

---

## Step 4 (T041): Fixes Applied

### Files modified and what changed:

**src/agents/fundamental_agent.py**
- Moved `from src.providers.entities import CompanyProfile, FinancialMetric` from line 153 to top import block (after existing imports).
- Removed the orphaned mid-file import line.

**src/agents/market_agent.py**
- Moved `from src.providers.entities import TechnicalAnalysisResult` from line 231 to top import block.
- Removed the orphaned mid-file import line.

**src/agents/news_agent.py**
- Moved `from src.providers.entities import NewsEvent as ContractNewsEvent` from line 241 to top import block.
- Removed the orphaned mid-file import line.

**src/providers/normalization.py**
- Moved `from datetime import datetime` from line 80 (below class definitions) to the top import block.
- Removed the orphaned mid-file import line.

**tests/fixtures/provider_contracts.py**
- Merged the two `from src.providers.normalization import ...` statements (lines 10 and 39) into a single import at the top: `from src.providers.normalization import RawCompanyPayload, RawFinancialRow, RawMarketData, RawNewsItem`.
- Removed the mid-file second import.

**tests/test_provider_contracts.py**
- Complete restructure: all incremental mid-file imports (lines 41, 43, 50, 111, 152, 154, 195, 262, 265, 323, 324, 325, 326) moved to a unified import block at the top of the file.
- Removed unused imports: `datetime.UTC`, `datetime.datetime`, `FinancialProvider`, `FinancialProviderResult`, `MarketDataProvider`, `MarketDataProviderResult`, `FORBIDDEN_TOKENS`.
- Fixed F811 redefinition: `raw_company_payload` and `raw_financial_rows` were imported twice (lines 266 and 326); merged into single import at top.
- All test function bodies are identical; only the import structure changed.

**tests/test_provider_graphrag_mapping_contracts.py**
- Removed `CompanyProfile` from `from src.providers.entities import CompanyProfile, NewsEvent` (CompanyProfile was unused).

**tests/test_scenario_report_input_contract.py**
- Removed entire `from src.providers.entities import CompanyProfile, NewsEvent` line (both unused).
- Removed `NormalizationStatus` from the enums import (unused), keeping only `DegradationStatus`.

**tests/test_storage_postgres_schema.py** (006-caused test update)
- Added `US006_EXPECTED_TABLES` set with the 6 new ORM table names added by T021/T022:
  `raw_provider_responses`, `provider_company_profiles`, `provider_news_events`,
  `provider_financial_metrics`, `provider_technical_analysis_results`, `provider_wave_analysis_results`.
- Updated `test_metadata_has_exactly_us1_us2_us3_tables` to include `| US006_EXPECTED_TABLES` in the exact-match assertion.
- This was a new failure introduced by 006's ORM models (T021/T022 commit 0b9ddea) which register into `Base.metadata`. The pre-existing test used `==` exact match and was not updated in Units 12/13. This is a test-maintenance fix, not a logic change.

---

## Step 5 (T041): Ruff After Fixes — 006 Surface

```
.venv/bin/ruff check src/providers src/graph_rag/mapping_contracts.py ... [006 files]
```

Result: **All checks passed!** — 006 surface fully clean.

---

## Step 6: Six Provider Suites After Lint Fixes

```
.venv/bin/python -m pytest tests/test_provider_contracts.py tests/test_provider_persistence_contracts.py tests/test_provider_graphrag_mapping_contracts.py tests/test_scenario_report_input_contract.py tests/test_provider_observability_contract.py tests/test_provider_safety_contract.py -q
```

Result: **39 passed, 1 warning** — identical to before-fix run. Import relocations did not break any imports.

---

## Step 7 (T047 spirit): Full Suite Regression

### First run (before test_storage_postgres_schema.py fix):
- **1 FAILED**: `tests/test_storage_postgres_schema.py::test_metadata_has_exactly_us1_us2_us3_tables`
- Classification: **NEW failure caused by 006** (T021/T022 ORM models added to Base.metadata; pre-existing test used exact-match `==` and was not updated).
- Resolution: Updated test to add `US006_EXPECTED_TABLES` (fix is in `tests/test_storage_postgres_schema.py`).

### Second run (after fix):
```
.venv/bin/python -m pytest -q
```

Result: **314 passed, 4 warnings** — zero failures.

### New vs Pre-existing breakdown:
- **New failures introduced by 006:** 1 (`test_metadata_has_exactly_us1_us2_us3_tables`) — RESOLVED by adding `US006_EXPECTED_TABLES` to the expected-tables assertion. This is a test-maintenance fix; the 006 ORM models are correct and intentional.
- **Pre-existing failures:** 0 (none found in the final run after fix).
- **Warnings (all pre-existing, benign):**
  - Alembic path_separator deprecation (1) — pre-existing alembic config issue.
  - SQLAlchemy `transaction already deassociated from connection` (2 tests) — pre-existing DB session handling.
  - `httpx` with `starlette.testclient` (1) — pre-existing dep version mismatch.

### Pre-existing ruff issues (27 errors, 15 files, NOT fixed):
All in files unrelated to 006: `src/backtest/`, `src/graph/workflow.py`, `src/db/migrations/versions/56464a69bd55_*.py`, `tests/test_api_robust_optimization.py`, `tests/test_optimization_evidence.py`, `tests/test_regime_performance.py`, `tests/test_robust_optimizer_*.py`, `tests/test_storage_postgres_privacy.py`, `tests/test_storage_postgres_ux.py`, `tests/test_walk_forward_optimizer.py`, `tests/test_workflow_e2e.py`. These pre-date 006 and are out of scope for this gate.

---

## Summary

| Gate | Result |
|------|--------|
| T037 compile | PASS — clean |
| T038 test_provider_contracts | PASS — 18/18 |
| T039 test_provider_persistence_contracts | PASS — 2/2 (real DB) |
| T040 test_provider_graphrag + test_scenario | PASS — 8/8 |
| T047 test_provider_observability + test_provider_safety | PASS — 11/11 |
| T041 ruff 006 surface | PASS — clean after fixes |
| Full suite regression | PASS — 314/314 |

**Commit:** f8e3155 — `chore(006): final compile/test/ruff validation gate (T037-T041,T047)`

**Branch status:** All 006 tasks complete. Ready for final whole-branch review.
