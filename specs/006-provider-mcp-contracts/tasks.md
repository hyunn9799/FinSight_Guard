# Tasks: Provider-Agnostic MCP Contracts

**Input**: Design documents from `/specs/006-provider-mcp-contracts/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/provider-normalization-contract.md`, `quickstart.md`

**Tests**: Deterministic tests are required for this feature because it changes provider fallback behavior, canonical lineage, GraphRAG mapping eligibility, and ScenarioReportInput construction. Tests must not call live MCP servers, yfinance, Tavily, Firecrawl, OpenAI, Neo4j, vector databases, broker APIs, or paper-trading APIs.

**Organization**: Tasks are grouped by user story so each story can be implemented and validated independently.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the provider contract package and deterministic fixture surface used by all stories.

- [ ] T001 Create provider contract package marker in `src/providers/__init__.py`
- [ ] T002 [P] Add deterministic provider fixture module skeleton in `tests/fixtures/provider_contracts.py`
- [ ] T003 [P] Expose provider fixture helpers from `tests/fixtures/__init__.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define shared contract primitives before any story-specific implementation.

**CRITICAL**: No user story work should begin until this phase is complete.

- [ ] T004 [P] Define provider status enums, degradation enums, lineage reference models, and shared warnings/errors in `src/providers/enums.py`
- [ ] T005 [P] Define `NewsProvider`, `FinancialProvider`, and `MarketDataProvider` protocol interfaces in `src/providers/interfaces.py`
- [ ] T006 [P] Define `CompanyProfile`, `NewsEvent`, `FinancialMetric`, `TechnicalAnalysisResult`, and `WaveAnalysisResult` Pydantic contracts with `model_config = ConfigDict(extra="forbid")` in `src/providers/entities.py`
- [ ] T048 [P] Define `FORBIDDEN_TOKENS`, `FORBIDDEN_TOKEN_PHRASES`, `SAFETY_CHECKED_CONTRACTS`, and the token-based `assert_no_trading_fields(model)` helper (snake_case token/phrase matching over field names, not substrings) in `src/providers/safety.py`
- [ ] T007 Define normalization result containers and helper signatures in `src/providers/normalization.py`
- [ ] T008 Export stable public provider contracts, safety helpers, and normalization helpers from `src/providers/__init__.py`

**Checkpoint**: Provider contract primitives exist and can be imported without touching database, graph, or live provider clients.

---

## Phase 3: User Story 1 - Normalize Provider Data For Agents (Priority: P1) MVP

**Goal**: Provider-specific raw shapes normalize into stable agent-facing objects before NewsAgent, FinancialAgent, MarketDataAgent, or ScenarioAgent consume provider-derived data.

**Independent Test**: Use two deterministic raw fixtures with different shapes for the same ticker and verify they produce the same normalized object types with no provider-specific raw fields exposed.

### Tests for User Story 1

- [ ] T009 [US1] Add deterministic tests for equivalent raw news shapes normalizing to stable `NewsEvent` objects in `tests/test_provider_contracts.py`
- [ ] T010 [US1] Add deterministic tests for company profile, financial metric, and market-data provider outputs preserving the provider/derived boundary in `tests/test_provider_contracts.py`
- [ ] T011 [US1] Add deterministic tests for provider failure, partial success, unsupported fields, and insufficient data statuses in `tests/test_provider_contracts.py`

### Implementation for User Story 1

- [ ] T012 [US1] Implement raw news fixture normalization into `NewsEvent` contracts in `src/providers/normalization.py`
- [ ] T013 [US1] Implement raw company and financial fixture normalization into `CompanyProfile` and `FinancialMetric` contracts in `src/providers/normalization.py`
- [ ] T014 [US1] Implement market-data provider normalization as normalized market data references, not technical or wave results, in `src/providers/normalization.py`
- [ ] T015 [US1] Update `NewsAgent` input handling to consume normalized `NewsEvent` contracts without raw provider fields in `src/agents/news_agent.py`
- [ ] T016 [US1] Update `FundamentalAgent` input handling to consume `CompanyProfile` and `FinancialMetric` contracts without raw provider fields in `src/agents/fundamental_agent.py`
- [ ] T017 [US1] Update `MarketAgent` input handling to keep provider market data separate from derived `TechnicalAnalysisResult` records in `src/agents/market_agent.py`

**Checkpoint**: User Story 1 is complete when `tests/test_provider_contracts.py` passes without live provider calls.

---

## Phase 4: User Story 2 - Trace Raw And Normalized Persistence (Priority: P2)

**Goal**: Raw provider responses, provider-normalized records, and internal derived records are persistable and traceable through PostgreSQL without confusing 006 contracts with 004 storage ownership.

**Independent Test**: Ingest deterministic fixture records and verify raw responses remain separate, normalized records trace to raw responses, and technical/wave results trace to normalized inputs and evidence references.

### Tests for User Story 2

- [ ] T018 [US2] Add persistence tests for `raw_provider_responses`, `company_profiles`, `news_events`, and `financial_metrics` lineage in `tests/test_provider_persistence_contracts.py`
- [ ] T019 [US2] Add persistence tests for `technical_analysis_results` and `wave_analysis_results` derived lineage to normalized market data, evidence, and rule references in `tests/test_provider_persistence_contracts.py`
- [ ] T020 [US2] Add persistence tests for partial normalization warnings, provider errors, and recoverable failed normalization metadata in `tests/test_provider_persistence_contracts.py`

### Implementation for User Story 2

- [ ] T021 [US2] Add SQLAlchemy models for raw provider responses and normalized provider records in `src/db/models.py`
- [ ] T022 [US2] Add SQLAlchemy models for derived technical and wave analysis contract records in `src/db/models.py`
- [ ] T023 [US2] Create Alembic migration for provider contract records in `src/db/migrations/versions/8f1a2b3c4d5e_provider_contract_records.py`
- [ ] T024 [US2] Implement `ProviderRepository` create/read lineage methods in `src/db/repositories/provider_repository.py`
- [ ] T025 [US2] Export `ProviderRepository` from `src/db/repositories/__init__.py`
- [ ] T026 [US2] Add raw/normalized persistence orchestration helpers that preserve 004 source-of-truth boundaries in `src/db/persistence.py`

**Checkpoint**: User Story 2 is complete when `tests/test_provider_persistence_contracts.py` passes against the test PostgreSQL database with deterministic fixtures.

---

## Phase 5: User Story 3 - Map Normalized Data To GraphRAG Context (Priority: P3)

**Goal**: ScenarioAgent receives a provider-agnostic `ScenarioReportInput` built from normalized records, evidence IDs, missing-data notes, and Company/Ticker-centered graph context.

**Independent Test**: Map deterministic normalized objects to graph-eligible specs and verify `ScenarioReportInput` contains no raw provider payload fields and degrades explicitly when graph context is missing or stale.

### Tests for User Story 3

- [ ] T027 [US3] Add GraphRAG mapping eligibility tests for normalized and derived records in `tests/test_provider_graphrag_mapping_contracts.py`
- [ ] T028 [US3] Add tests that `RawProviderResponse`, raw candles, raw news sentences, and raw financial rows are not graph-projected in `tests/test_provider_graphrag_mapping_contracts.py`
- [ ] T029 [US3] Add `ScenarioReportInput` construction and degraded graph-context tests in `tests/test_scenario_report_input_contract.py`
- [ ] T042 [US3] Add vector reference contract tests verifying news originals, NEoWave rule explanations, and report chunks are referenced from `ScenarioReportInput` via canonical source references (not raw payloads or vector-store internals) in `tests/test_scenario_report_input_contract.py` (covers FR-018)
- [ ] T043 [US3] Add observability contract tests verifying provider degradation status and graph-context warnings are surfaced as structured warning fields on `ScenarioReportInput` and in persistence metadata in `tests/test_provider_observability_contract.py` (covers FR-017, SER-006)

### Implementation for User Story 3

- [ ] T030 [US3] Implement graph mapping rule and graph-eligible spec builders in `src/graph_rag/mapping_contracts.py`
- [ ] T031 [US3] Implement `ScenarioReportInput` schema and validation in `src/providers/scenario_input.py`
- [ ] T032 [US3] Implement `ScenarioReportInput` builder with missing-data notes and degradation status in `src/providers/scenario_input.py`
- [ ] T033 [US3] Add graph-context warning and stale/missing projection handling at the contract boundary in `src/graph_rag/graph_context_builder.py`
- [ ] T034 [US3] Update `CoordinatorAgent` scenario input handling to consume `ScenarioReportInput` without provider-specific raw fields in `src/agents/coordinator_agent.py`
- [ ] T044 [US3] Define the vector retrieval reference contract via a `VectorReference` object (`source_kind` ∈ news_original/neowave_rule_explanation/report_chunk, `canonical_ref_id`, optional `source_uri`/`chunk_id`; no score/store/embedding fields) on `ScenarioReportInput` in `src/providers/scenario_input.py`, with mapping eligibility expressed in `src/graph_rag/mapping_contracts.py`. Every reference MUST resolve to a canonical PG source/evidence reference (covers FR-018)
- [ ] T045 [US3] Surface provider degradation status and graph-context warnings as structured warning fields on `ScenarioReportInput` in `src/providers/scenario_input.py` and at the graph-context boundary in `src/graph_rag/graph_context_builder.py` (covers FR-017, SER-006)

**Checkpoint**: User Story 3 is complete when `tests/test_provider_graphrag_mapping_contracts.py`, `tests/test_scenario_report_input_contract.py`, and `tests/test_provider_observability_contract.py` pass without live graph or vector services.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the contract slice end-to-end and keep docs aligned with the 006 ownership boundaries.

- [ ] T046 Add a deterministic safety contract test in `tests/test_provider_safety_contract.py` (covers SC-006, SER-001) that, over `SAFETY_CHECKED_CONTRACTS` only: (1) structural — asserts no contract model's snake_case field-name tokens hit `FORBIDDEN_TOKENS` or `FORBIDDEN_TOKEN_PHRASES` (e.g. `buy_signal`, `order_action`, `target_price` rejected; `threshold`/`household` not false-positived); (2) instance — asserts a built `ScenarioReportInput` fixture exposes none. Uses the shared constants/helper from `src/providers/safety.py`. Note: future Phase 2/3 simulation contracts (`SignalCandidate`, `StrategyRule`, `PaperTradingExecution`) are a separate namespace with their own allowlist and are intentionally NOT in scope here; they stay simulated-only and MUST NOT carry live order-execution fields.
- [ ] T035 [P] Update provider contract quickstart notes after implementation in `specs/006-provider-mcp-contracts/quickstart.md`
- [ ] T036 [P] Update project plan references to the provider contract boundary in `PROJECT_PLAN.md`
- [ ] T037 Run `uv run python -m compileall src` for `src/`
- [ ] T038 Run `uv run pytest tests/test_provider_contracts.py` for `tests/test_provider_contracts.py`
- [ ] T039 Run `uv run pytest tests/test_provider_persistence_contracts.py` for `tests/test_provider_persistence_contracts.py`
- [ ] T040 Run `uv run pytest tests/test_provider_graphrag_mapping_contracts.py tests/test_scenario_report_input_contract.py` for `tests/test_provider_graphrag_mapping_contracts.py` and `tests/test_scenario_report_input_contract.py`
- [ ] T047 Run `uv run pytest tests/test_provider_observability_contract.py tests/test_provider_safety_contract.py` for the observability and safety contract tests
- [ ] T041 Run `uv run ruff check src tests` for `src/` and `tests/`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Phase 1 and blocks all user stories.
- **US1 Normalize Provider Data (Phase 3)**: Depends on Phase 2.
- **US2 Trace Persistence (Phase 4)**: Depends on Phase 2; can use US1 fixtures but remains independently testable.
- **US3 GraphRAG Context (Phase 5)**: Depends on Phase 2; can use US1 normalized fixtures and US2 canonical references when available.
- **Polish (Phase 6)**: Depends on the selected story scope.

### User Story Dependencies

- **US1 (P1)**: MVP slice. No dependency on US2 or US3 after foundational contracts.
- **US2 (P2)**: Can begin after foundational contracts; repository tests may reuse US1 fixture builders.
- **US3 (P3)**: Can begin after foundational contracts; full integration is strongest after US1 and US2 are complete.

### Within Each User Story

- Tests should be written first and fail before implementation.
- Contract models before normalizers.
- Models and migrations before repositories.
- Mapping rules before ScenarioReportInput integration.
- Agent integration after contract and builder behavior passes.

---

## Parallel Opportunities

- T002 and T003 can run in parallel after T001 is clear.
- T004 (`enums.py`), T005 (`interfaces.py`), T006 (`entities.py`), and T048 (`safety.py`) live in separate files and ARE parallelizable `[P]`. T007 (`src/providers/normalization.py`) and T008 (`__init__.py` exports) follow once the contract primitives exist.
- US1 test tasks T009, T010, and T011 are in one file and should be sequenced by the same implementer.
- US2 test tasks T018, T019, and T020 are in one file and should be sequenced by the same implementer.
- US3 graph mapping tests T027/T028 and ScenarioReportInput tests T029 can run in parallel because they use different test files.
- US3 implementation T030 can run in parallel with T031/T032 until integration at T033/T034.
- Polish documentation tasks T035 and T036 can run in parallel with validation commands after implementation is complete.

## Parallel Example: User Story 3

```bash
# Parallel contract test work:
Task: "T027 [US3] Add GraphRAG mapping eligibility tests in tests/test_provider_graphrag_mapping_contracts.py"
Task: "T029 [US3] Add ScenarioReportInput construction tests in tests/test_scenario_report_input_contract.py"

# Parallel implementation work:
Task: "T030 [US3] Implement graph mapping rule builders in src/graph_rag/mapping_contracts.py"
Task: "T031 [US3] Implement ScenarioReportInput schema in src/providers/scenario_input.py"
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational contracts.
3. Complete US1 normalization tests and implementation.
4. Validate with `uv run pytest tests/test_provider_contracts.py`.

This establishes the core provider-agnostic boundary before storage and graph projection are expanded.

### Incremental Delivery

1. US1: Stable normalized contracts for agents.
2. US2: PostgreSQL lineage for raw, normalized, and derived records.
3. US3: GraphRAG mapping eligibility and ScenarioReportInput construction.
4. Polish: Compile, focused pytest, ruff, and documentation alignment.

### Risk Controls

- Keep 006 as the owner of contracts only; treat database additions as explicit 004-compatible extensions.
- Keep 005 as the owner of graph semantics; 006 only emits graph-eligible mapping specs.
- Keep `TechnicalAnalysisResult` and `WaveAnalysisResult` as internal derived outputs, not provider-normalized payloads.
- Do not introduce real MCP clients, API keys, live external calls, real order execution, or buy/sell/hold recommendation fields in this feature.
