# Tasks: Walk-Forward Strategy Optimization

**Input**: Design documents from `specs/002-walk-forward-optimization/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [storage-architecture.md](./storage-architecture.md), [contracts/robust-optimization-api.md](./contracts/robust-optimization-api.md), [quickstart.md](./quickstart.md)

**Tests**: Required. The spec and constitution require deterministic tests for optimization scoring, walk-forward routing, evidence/report safety, storage, API behavior, and external-store failure handling. Tests must not depend on live yfinance, Pinecone, Neo4j, OpenSearch, Redis, PostgreSQL, Tavily, Firecrawl, or OpenAI services.

**Organization**: Tasks are grouped by user story to enable independently testable increments.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare dependencies, directories, and deterministic test helpers without changing user-visible behavior.

- [ ] T001 Add PostgreSQL, Redis, Pinecone, Neo4j, OpenSearch, and SQL migration/client dependencies to `requirements.txt`
- [ ] T002 Create storage and index package skeletons in `src/db/__init__.py`, `src/db/repositories/__init__.py`, `src/indexes/__init__.py`, and `src/backtest/robust.py`
- [ ] T003 [P] Create deterministic optimization fixture helpers in `tests/fixtures/optimization_data.py`
- [ ] T004 [P] Create fake external store clients for PostgreSQL, Redis, Pinecone, Neo4j, and OpenSearch in `tests/fakes/external_stores.py`
- [ ] T005 [P] Document local environment variables for optional PostgreSQL, Redis, Pinecone, Neo4j, and OpenSearch clients in `.env.example`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared data contracts, storage boundaries, and adapter behavior required by all user stories.

**Critical**: No user story implementation should begin until this phase is complete.

- [ ] T006 Define robust optimization Pydantic/domain models for cost assumptions, scoring policy, candidate metrics, folds, regimes, baselines, and run status in `src/backtest/robust.py`
- [ ] T007 Define PostgreSQL source-of-truth schema constants or migration SQL for users, tickers, analysis requests, analysis results, reports, evidence items, robust optimization runs, walk-forward folds, market regime results, settings, notifications, and portfolios in `src/db/schema.py`
- [ ] T008 Implement PostgreSQL connection/session facade with safe no-live-service fallback behavior in `src/db/postgres.py`
- [ ] T009 [P] Implement Redis runtime facade for cache, queue, rate-limit, session, and run-status keys in `src/db/redis.py`
- [ ] T010 [P] Implement repository interfaces for users, tickers, analysis requests, reports, analysis results, evidence items, optimization runs, settings, notifications, and portfolios in `src/db/repositories/base.py`
- [ ] T011 Implement in-memory deterministic repository implementations for tests and local fallback in `src/db/repositories/memory.py`
- [ ] T012 [P] Implement Pinecone semantic index facade with idempotent upsert/query method signatures in `src/indexes/pinecone_index.py`
- [ ] T013 [P] Implement Neo4j graph facade for wave rules, scenarios, invalidation conditions, and evidence paths in `src/indexes/neo4j_graph.py`
- [ ] T014 [P] Implement OpenSearch facade for news, reports, logs, and keyword indexes in `src/indexes/opensearch_index.py`
- [ ] T015 Implement index orchestration that persists PostgreSQL first and reports degraded/index-pending warnings for derived index failures in `src/indexes/indexing_service.py`
- [ ] T016 [P] Add deterministic tests for PostgreSQL repository contracts and source-of-truth rules in `tests/test_storage_postgres.py`
- [ ] T017 [P] Add deterministic tests for Pinecone, Neo4j, and OpenSearch adapter failure handling in `tests/test_indexes.py`
- [ ] T018 [P] Add deterministic tests proving Redis runtime state is ephemeral and not the only durable store in `tests/test_redis_runtime.py`
- [ ] T019 Add optimization evidence builder helper using the existing `EvidenceItem` schema in `src/evidence/evidence_builder.py`
- [ ] T020 Extend `GraphState` and analysis models to carry robust optimization output without replacing existing backtest analysis in `src/graph/state.py`

**Checkpoint**: Storage/index boundaries, robust optimization data contracts, and evidence contracts are ready for story work.

---

## Phase 3: User Story 1 - Optimize With Risk Controls (Priority: P1) MVP

**Goal**: Reject total-return-only ranking and select/downgrade candidates using drawdown, risk-adjusted metrics, trade count, cost-adjusted return, turnover, and the clarified robust score policy.

**Independent Test**: Run optimization on deterministic sample data where the highest-return candidate breaches drawdown or trade-sample limits and confirm it is not selected as the robust candidate.

### Tests for User Story 1

- [ ] T021 [P] [US1] Add deterministic tests for candidate metrics including MDD, Sharpe, Sortino, win rate, profit factor, average holding period, turnover, and cost-adjusted return in `tests/test_robust_optimizer_metrics.py`
- [ ] T022 [P] [US1] Add deterministic tests rejecting candidates with fewer than 30 completed trades or MDD above 25% in `tests/test_robust_optimizer_scoring.py`
- [ ] T023 [P] [US1] Add deterministic tests for risk-balanced score weights and score component explanations in `tests/test_robust_optimizer_scoring.py`
- [ ] T024 [P] [US1] Add deterministic tests for default 0.05% one-way fee and 0.05% one-way slippage behavior in `tests/test_robust_optimizer_metrics.py`

### Implementation for User Story 1

- [ ] T025 [US1] Implement candidate metric calculation from `BacktestResult.trades` and enriched price data in `src/backtest/robust.py`
- [ ] T026 [US1] Implement transaction-cost-adjusted metric handling with user-adjustable fee and slippage defaults in `src/backtest/robust.py`
- [ ] T027 [US1] Implement robust label guardrails for minimum 30 completed trades and MDD 25% or lower in `src/backtest/robust.py`
- [ ] T028 [US1] Implement risk-balanced robust score calculation and score component explanations in `src/backtest/robust.py`
- [ ] T029 [US1] Extend `src/backtest/optimizer.py` with a compatibility-preserving robust candidate scoring entrypoint that does not alter `optimize_backtest`
- [ ] T030 [US1] Persist robust optimization candidate metrics and evidence through repository interfaces in `src/db/repositories/optimization.py`
- [ ] T031 [US1] Expose robust candidate scoring in the Backtest Agent while preserving historical-simulation wording in `src/agents/backtest_agent.py`

**Checkpoint**: User Story 1 is independently testable and can be demoed as robust scoring without walk-forward folds.

---

## Phase 4: User Story 2 - Validate With Walk-Forward Evaluation (Priority: P2)

**Goal**: Split historical data into time-ordered train/test folds, evaluate out-of-sample performance, report every fold, and return insufficient-data status when fewer than 3 valid test folds exist.

**Independent Test**: Use deterministic dated sample data and confirm every test window starts after its train window, each fold is reported, and aggregates include median, worst-fold, and stability measures.

### Tests for User Story 2

- [ ] T032 [P] [US2] Add deterministic tests for time-ordered train/test fold generation in `tests/test_walk_forward_optimizer.py`
- [ ] T033 [P] [US2] Add deterministic tests for fewer than 3 valid test folds returning insufficient-data status in `tests/test_walk_forward_optimizer.py`
- [ ] T034 [P] [US2] Add deterministic tests for median OOS return, worst-fold return, drawdown, and fold-to-fold variability aggregation in `tests/test_walk_forward_optimizer.py`
- [ ] T035 [P] [US2] Add FastAPI contract tests for `POST /backtest/optimize` success and insufficient-data responses in `tests/test_api_robust_optimization.py`

### Implementation for User Story 2

- [ ] T036 [US2] Implement walk-forward fold generation with train/test/step windows and chronological validation in `src/backtest/robust.py`
- [ ] T037 [US2] Implement fold-level robust optimization orchestration using `run_backtest` and candidate scoring in `src/backtest/optimizer.py`
- [ ] T038 [US2] Implement insufficient-data and invalid-fold warnings without labeling output robust in `src/backtest/optimizer.py`
- [ ] T039 [US2] Persist walk-forward fold records and aggregate summaries through repositories in `src/db/repositories/optimization.py`
- [ ] T040 [US2] Add Pydantic request/response models for robust optimization API contract in `main.py`
- [ ] T041 [US2] Implement `POST /backtest/optimize` endpoint using deterministic validation, robust optimization service, storage, and warnings in `main.py`
- [ ] T042 [US2] Add Streamlit controls for robust optimization date range, folds, trials, and cost assumptions in `app.py`
- [ ] T043 [US2] Add Streamlit result view for score components, fold breakdown, baselines, warnings, and evidence in `app.py`

**Checkpoint**: User Story 2 is independently testable through the optimizer, API endpoint, and UI result view.

---

## Phase 5: User Story 3 - Explain Regime-Specific Strengths and Weaknesses (Priority: P3)

**Goal**: Classify evaluated periods into regimes and report strategy behavior by bull, bear, sideways, high-volatility, and low-volatility segments with low-confidence flags for thin samples.

**Independent Test**: Label deterministic sample periods by regime and confirm separate performance summaries exist for each available regime and low-confidence rules trigger below 10 trades or 60 trading days.

### Tests for User Story 3

- [ ] T044 [P] [US3] Add deterministic tests for bull, bear, sideways, high-volatility, and low-volatility regime classification in `tests/test_regime_performance.py`
- [ ] T045 [P] [US3] Add deterministic tests for regime performance summaries and low-confidence thresholds in `tests/test_regime_performance.py`
- [ ] T046 [P] [US3] Add deterministic tests for weak bear or sideways regime warnings in `tests/test_regime_performance.py`

### Implementation for User Story 3

- [ ] T047 [US3] Implement market regime classifier for trend and volatility regimes in `src/backtest/regime.py`
- [ ] T048 [US3] Implement per-regime candidate metrics and confidence labeling in `src/backtest/regime.py`
- [ ] T049 [US3] Integrate regime summaries into robust optimization output in `src/backtest/optimizer.py`
- [ ] T050 [US3] Persist market regime results through repositories in `src/db/repositories/optimization.py`
- [ ] T051 [US3] Surface regime-specific strengths, weaknesses, and low-confidence warnings in `src/agents/backtest_agent.py`
- [ ] T052 [US3] Add regime summary rendering to Streamlit robust optimization results in `app.py`

**Checkpoint**: User Story 3 can be tested independently with labeled deterministic sample data.

---

## Phase 6: User Story 4 - Keep Optimization Research-Only (Priority: P4)

**Goal**: Ensure robust optimization output remains educational, evidence-grounded, non-prescriptive, and compatible with Coordinator/Evaluator report safety.

**Independent Test**: Check final optimization-containing output for required limitations, no-recommendation framing, evidence references, Korean disclaimer, and absence of forbidden advice language.

### Tests for User Story 4

- [ ] T053 [P] [US4] Add safety tests for robust optimization text avoiding advice-like wording and forbidden phrases in `tests/test_robust_optimization_safety.py`
- [ ] T054 [P] [US4] Add evaluator pass/fail tests for reports that present robust parameters as guaranteed, advice-like, or automatic-execution-ready in `tests/test_evaluator.py`
- [ ] T055 [P] [US4] Add evidence tests verifying numeric optimization report claims map to `EvidenceItem` records in `tests/test_optimization_evidence.py`
- [ ] T056 [P] [US4] Add workflow routing tests for optimization evidence flowing through Coordinator, Evaluator, Rewrite, and report storage in `tests/test_workflow_routing.py`

### Implementation for User Story 4

- [ ] T057 [US4] Add optimization evidence item generation for robust score, cost-adjusted return, MDD, trade count, worst fold, and regime metrics in `src/evidence/evidence_builder.py`
- [ ] T058 [US4] Extend Coordinator report generation to include optimization evidence, limitations, scenario framing, and required disclaimer in `src/agents/coordinator_agent.py`
- [ ] T059 [US4] Extend Evaluator checks to fail advice-like robust parameter language, guaranteed optimization claims, or missing optimization evidence in `src/agents/evaluator_agent.py`
- [ ] T060 [US4] Extend Rewrite Agent to remove unsafe optimization wording while preserving evidence-based reasoning in `src/agents/rewrite_agent.py`
- [ ] T061 [US4] Persist final optimization-containing reports and evaluation results through PostgreSQL repositories while keeping local export compatibility in `src/storage/report_store.py`
- [ ] T062 [US4] Index persisted reports, evidence paths, semantic chunks, and logs through OpenSearch, Neo4j, Pinecone, and Redis orchestration warnings in `src/indexes/indexing_service.py`

**Checkpoint**: User Story 4 can be verified through deterministic report/evaluator/workflow tests with no live external services.

---

## Final Phase: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, verification, and integration cleanup across all stories.

- [ ] T063 [P] Update README architecture, storage role separation, safety limitations, and robust optimization usage in `README.md`
- [ ] T064 [P] Update quickstart validation commands and expected dependency notes in `specs/002-walk-forward-optimization/quickstart.md`
- [ ] T065 [P] Add Docker Compose or documented local service wiring for PostgreSQL, Redis, Neo4j, OpenSearch, and fake/optional Pinecone in `docker-compose.yml`
- [ ] T066 Run static compile validation with `python3 -m compileall src` and record result in `specs/002-walk-forward-optimization/quickstart.md`
- [ ] T067 Run focused deterministic pytest suites for robust optimization, API, storage, indexes, safety, evaluator, and workflow routing in `tests/`
- [ ] T068 Review all new runtime text for no buy/sell/hold recommendations, no guaranteed returns, and exact required Korean disclaimer in `src/agents/` and `app.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- Phase 1 Setup has no dependencies.
- Phase 2 Foundational depends on Phase 1 and blocks all user stories.
- Phase 3 US1 depends on Phase 2 and is the MVP.
- Phase 4 US2 depends on Phase 3 because walk-forward uses robust scoring and candidate metrics.
- Phase 5 US3 depends on Phase 4 because regime summaries apply to evaluated folds and robust candidates.
- Phase 6 US4 depends on Phase 3 for evidence/safety basics and should be finalized after US2/US3 if optimization metrics reach reports.
- Final Phase depends on the selected user stories for release/demo.

### User Story Dependencies

- US1: No user-story dependency after foundational work.
- US2: Depends on US1 scoring primitives.
- US3: Depends on US2 fold/candidate outputs.
- US4: Can start after US1 for safety/evidence scaffolding, but full validation depends on US2 and US3 outputs.

### Parallel Opportunities

- T003, T004, and T005 can run in parallel after T001.
- T009, T010, T012, T013, T014, T016, T017, and T018 can run in parallel after T006 and T007 contracts are understood.
- Test tasks within each user story are parallelizable and should be written before implementation tasks.
- Adapter implementations for Pinecone, Neo4j, OpenSearch, and Redis can proceed in parallel once foundational interfaces exist.
- README, quickstart, and Docker Compose polish tasks can proceed in parallel after the relevant implementation paths stabilize.

---

## Parallel Execution Examples

### User Story 1

```bash
Task: "T021 Add deterministic tests for candidate metrics in tests/test_robust_optimizer_metrics.py"
Task: "T022 Add deterministic tests rejecting weak robust candidates in tests/test_robust_optimizer_scoring.py"
Task: "T023 Add deterministic tests for risk-balanced score weights in tests/test_robust_optimizer_scoring.py"
Task: "T024 Add deterministic tests for default transaction costs in tests/test_robust_optimizer_metrics.py"
```

### User Story 2

```bash
Task: "T032 Add deterministic tests for fold generation in tests/test_walk_forward_optimizer.py"
Task: "T035 Add FastAPI contract tests for POST /backtest/optimize in tests/test_api_robust_optimization.py"
```

### User Story 3

```bash
Task: "T044 Add deterministic regime classification tests in tests/test_regime_performance.py"
Task: "T045 Add regime confidence threshold tests in tests/test_regime_performance.py"
```

### User Story 4

```bash
Task: "T053 Add robust optimization safety wording tests in tests/test_robust_optimization_safety.py"
Task: "T055 Add optimization evidence mapping tests in tests/test_optimization_evidence.py"
Task: "T056 Add workflow routing tests in tests/test_workflow_routing.py"
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 only.
3. Validate US1 with deterministic metrics/scoring/cost tests.
4. Demo robust scoring that rejects total-return-only candidates without presenting trading advice.

### Incremental Delivery

1. Add US1 robust risk controls.
2. Add US2 walk-forward validation and API/UI exposure.
3. Add US3 regime-specific explanations.
4. Add US4 full report/evaluator/rewrite/storage/index integration.
5. Finish cross-cutting docs, Docker/local service wiring, compile, and pytest verification.

### Storage Rollout

1. PostgreSQL repository interfaces and in-memory fakes first.
2. Redis runtime facade second.
3. Derived index adapters for Pinecone, Neo4j, and OpenSearch third.
4. Index orchestration last, always treating PostgreSQL as authoritative.
