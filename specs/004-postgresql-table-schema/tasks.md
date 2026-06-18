# Tasks: PostgreSQL Source-of-Truth Table Schema

**Input**: Design documents from `/specs/004-postgresql-table-schema/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/postgresql-storage-contract.md, quickstart.md

**Tests**: Deterministic tests are REQUIRED because this feature affects report storage, workflow persistence, evidence grounding, FastAPI analyze/backtest persistence behavior, observability, and financial-safety auditability. Tests MUST NOT depend on live external APIs or live derived index services.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Single Python project rooted at `src/`
- Tests rooted at `tests/`
- PostgreSQL persistence boundary rooted at `src/db/`
- Database migration assets rooted at `src/db/migrations/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add database dependencies, configuration, and directory structure needed by every story.

- [ ] T001 Add SQLAlchemy 2.x, Alembic, and psycopg dependencies to `requirements.txt`
- [ ] T002 Add PostgreSQL environment variables and local/demo defaults to `.env.example`
- [ ] T003 Add PostgreSQL service and healthcheck placeholders to `docker-compose.yml`
- [ ] T004 Create database package structure in `src/db/__init__.py`
- [ ] T005 [P] Create repository package structure in `src/db/repositories/__init__.py`
- [ ] T006 [P] Create migrations package placeholder in `src/db/migrations/README.md`
- [ ] T007 [P] Create PostgreSQL test fixture module in `tests/fixtures/postgres.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core DB infrastructure that MUST be complete before any user story can persist data.

**CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T008 Create SQLAlchemy engine/session factory and safe configuration loading in `src/db/postgres.py`
- [ ] T009 Define shared SQLAlchemy metadata, UUID/timestamp helpers, and naming conventions in `src/db/models.py`
- [ ] T010 Define shared status vocabularies and validation constants in `src/db/constants.py`
- [ ] T011 Create Alembic configuration entrypoint for project metadata in `src/db/migrations/env.py`
- [ ] T012 Create initial Alembic revision file for the full MVP schema in `src/db/migrations/versions/0001_postgresql_source_of_truth.py`
- [ ] T013 Create base repository transaction helper in `src/db/repositories/base.py`
- [ ] T014 [P] Add deterministic DB session fixture and rollback cleanup in `tests/conftest.py`
- [ ] T015 [P] Add schema import smoke test for metadata loading in `tests/test_storage_postgres_schema.py`
- [ ] T016 [P] Add repository contract test scaffolding in `tests/test_storage_postgres_repositories.py`
- [ ] T017 [P] Add workflow persistence test scaffolding in `tests/test_storage_postgres_research_flow.py`
- [ ] T018 [P] Add privacy and user-owned UX test scaffolding in `tests/test_storage_postgres_privacy.py`
- [ ] T019 [P] Add projection and graph persistence test scaffolding in `tests/test_storage_postgres_projection_status.py`

**Checkpoint**: Foundation ready - schema metadata, migrations, sessions, and test fixtures are available.

---

## Phase 3: User Story 1 - Define Canonical Research Records (Priority: P1) MVP

**Goal**: Persist a complete research workflow ledger covering requests, node runs, analysis results, reports, report versions, evidence, citations, warnings, degraded states, and source document chunks.

**Independent Test**: A deterministic sample research run can store and fetch ticker, request, node attempts, market/fundamental/news results, evidence, draft/final report versions, evaluator result, citations, source documents, chunks, and degraded warnings without live provider calls.

### Tests for User Story 1

- [ ] T020 [P] [US1] Add schema catalog completeness assertions for core research tables, purpose, primary keys, columns, relationships, retention_role, uniqueness rules, and deferred table absence in `tests/test_storage_postgres_schema.py`
- [ ] T021 [P] [US1] Add ticker, analysis request, node run, and analysis result repository tests in `tests/test_storage_postgres_repositories.py`
- [ ] T022 [P] [US1] Add evidence, report version, and citation repository tests in `tests/test_storage_postgres_repositories.py`
- [ ] T023 [P] [US1] Add complete deterministic research ledger integration test in `tests/test_storage_postgres_research_flow.py`
- [ ] T024 [P] [US1] Add degraded provider persistence test with missing-data notes in `tests/test_storage_postgres_research_flow.py`

### Implementation for User Story 1

- [ ] T025 [P] [US1] Implement `User`, `Ticker`, `AnalysisRequest`, and `WorkflowNodeRun` models in `src/db/models.py`
- [ ] T026 [P] [US1] Implement `AnalysisResult`, `Report`, `ReportVersion`, `EvidenceItemRecord`, and `ReportEvidenceCitation` models in `src/db/models.py`
- [ ] T027 [P] [US1] Implement `SourceDocument`, `DocumentChunk`, and `StructuredLogEvent` models, including source document revision lineage fields, in `src/db/models.py`
- [ ] T028 [US1] Implement ticker and analysis request repository methods in `src/db/repositories/analysis_repository.py`
- [ ] T029 [US1] Implement workflow node run and analysis result repository methods in `src/db/repositories/analysis_repository.py`
- [ ] T030 [US1] Implement evidence insert/fetch/list/citation methods in `src/db/repositories/evidence_repository.py`
- [ ] T031 [US1] Implement report container/version/current-version/safety-status methods in `src/db/repositories/report_repository.py`
- [ ] T032 [US1] Implement source document, corrected/recrawled document version, and ordered chunk methods in `src/db/repositories/source_document_repository.py`
- [ ] T033 [US1] Implement structured log event persistence helper in `src/db/repositories/analysis_repository.py`
- [ ] T034 [US1] Add compatibility wrapper that exports PostgreSQL-backed reports to existing JSON/Markdown files in `src/storage/report_store.py`
- [ ] T035 [US1] Add compatibility wrapper that mirrors PostgreSQL run metadata for existing callers in `src/storage/run_store.py`
- [ ] T036 [US1] Integrate request/report persistence into successful and degraded workflow paths in `src/graph/workflow.py`
- [ ] T037 [US1] Ensure `/analyze`, `/backtest`, and `/backtest/optimize` responses remain backward compatible while using persisted request/report IDs in `main.py`
- [ ] T038 [US1] Add persistence failure handling that returns degraded warning or user-facing persistence error in `src/graph/workflow.py`

**Checkpoint**: User Story 1 is fully functional and testable independently as the PostgreSQL research-ledger MVP.

---

## Phase 4: User Story 2 - Rebuild Derived Search And Graph Indexes (Priority: P2)

**Goal**: Persist canonical source/chunk/projection and wave graph records so Pinecone, Neo4j, and OpenSearch projections can be rebuilt and failures can be audited.

**Independent Test**: A sample news/report/evidence/wave scenario can create source documents, chunks, projection statuses, wave rules, wave scenarios, invalidation conditions, evidence paths, and ordered path steps; projection failures do not remove canonical records.

### Tests for User Story 2

- [ ] T039 [P] [US2] Add projection status lifecycle tests for pending, success, failed, and stale states in `tests/test_storage_postgres_projection_status.py`
- [ ] T040 [P] [US2] Add source document, recrawl/correction revision, and document chunk idempotency tests in `tests/test_storage_postgres_projection_status.py`
- [ ] T041 [P] [US2] Add graph repository tests for wave rules, scenarios, invalidation conditions, and scenario-rule joins in `tests/test_storage_postgres_projection_status.py`
- [ ] T042 [P] [US2] Add evidence path ordering and canonical node reference tests in `tests/test_storage_postgres_projection_status.py`

### Implementation for User Story 2

- [ ] T043 [P] [US2] Implement `IndexProjectionStatus` and `KeywordTerm` models in `src/db/models.py`
- [ ] T044 [P] [US2] Implement `WaveRule`, `WaveScenario`, `WaveInvalidationCondition`, and `WaveScenarioRule` models in `src/db/models.py`
- [ ] T045 [P] [US2] Implement `EvidencePath` and `EvidencePathStep` models in `src/db/models.py`
- [ ] T046 [US2] Implement projection status upsert/success/failure/list methods in `src/db/repositories/projection_repository.py`
- [ ] T047 [US2] Implement keyword term repository methods in `src/db/repositories/projection_repository.py`
- [ ] T048 [US2] Implement wave rule, scenario, invalidation, and scenario-rule methods in `src/db/repositories/graph_repository.py`
- [ ] T049 [US2] Implement evidence path and ordered path step methods in `src/db/repositories/graph_repository.py`
- [ ] T050 [US2] Integrate canonical evidence path persistence with graph context builder output in `src/graph_rag/graph_context_builder.py`
- [ ] T051 [US2] Add projection failure warning behavior that never mutates canonical records in `src/db/repositories/projection_repository.py`

**Checkpoint**: User Story 2 can rebuild derived projection inputs and preserve canonical graph records independently of derived services.

---

## Phase 5: User Story 3 - Support User History, Settings, In-App Notifications, And Portfolios (Priority: P3)

**Goal**: Persist local/demo user ownership, settings, in-app notifications, and research-only portfolio context without implementing login, external notification delivery, brokerage fields, or order execution.

**Independent Test**: A local/demo user can own settings, in-app notifications, portfolios, and portfolio items; user deletion anonymizes or removes PII/UX records while preserving report/evidence/audit records under anonymous ownership.

### Tests for User Story 3

- [ ] T052 [P] [US3] Add local/demo user repository tests without email, password, SSO, or session fields in `tests/test_storage_postgres_privacy.py`
- [ ] T053 [P] [US3] Add user setting create/update/delete-or-anonymize tests in `tests/test_storage_postgres_privacy.py`
- [ ] T054 [P] [US3] Add in-app notification create/list/mark-read/delete tests with forbidden phrase checks in `tests/test_storage_postgres_privacy.py`
- [ ] T055 [P] [US3] Add research-only portfolio and portfolio item tests excluding brokerage/order fields in `tests/test_storage_postgres_privacy.py`
- [ ] T056 [P] [US3] Add user anonymization test preserving reports, evidence, node runs, and audit references in `tests/test_storage_postgres_privacy.py`

### Implementation for User Story 3

- [ ] T057 [P] [US3] Implement `UserSetting`, `Notification`, `Portfolio`, and `PortfolioItem` models in `src/db/models.py`
- [ ] T058 [US3] Implement local/demo user create/update/anonymize/soft-delete methods in `src/db/repositories/user_repository.py`
- [ ] T059 [US3] Implement setting create/update/list/delete-or-anonymize methods in `src/db/repositories/user_repository.py`
- [ ] T060 [US3] Implement in-app notification create/list/mark-read/archive-delete methods in `src/db/repositories/notification_repository.py`
- [ ] T061 [US3] Add notification safety validation using forbidden phrases in `src/db/repositories/notification_repository.py`
- [ ] T062 [US3] Implement portfolio create/update/list/soft-delete methods in `src/db/repositories/portfolio_repository.py`
- [ ] T063 [US3] Implement portfolio item add/update/remove/list methods in `src/db/repositories/portfolio_repository.py`
- [ ] T064 [US3] Implement user anonymization workflow across settings, notifications, portfolios, and portfolio items in `src/db/repositories/user_repository.py`
- [ ] T065 [US3] Add safeguards preventing brokerage, order, execution, guaranteed target, or direct recommendation fields in `src/db/models.py`

**Checkpoint**: User Story 3 is independently testable as local/demo user UX persistence without real authentication or trading behavior.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, compatibility, and final validation across all selected stories.

- [ ] T066 [P] Update README storage architecture and PostgreSQL local setup notes in `README.md`
- [ ] T067 [P] Update Docker/local service documentation for PostgreSQL source-of-truth setup in `docker-compose.yml`
- [ ] T068 [P] Document migration and repository boundaries in `specs/004-postgresql-table-schema/quickstart.md`
- [ ] T069 Add repository-backed metrics preservation check for `/health` and `/metrics` in `tests/test_api.py`
- [ ] T070 Run `python -m compileall src`
- [ ] T071 Run deterministic PostgreSQL storage tests with `.venv/bin/python -m pytest tests/test_storage_postgres_schema.py tests/test_storage_postgres_repositories.py tests/test_storage_postgres_research_flow.py tests/test_storage_postgres_privacy.py tests/test_storage_postgres_projection_status.py`
- [ ] T072 Run relevant existing regression tests with `.venv/bin/python -m pytest tests/test_report_store.py tests/test_workflow_routing.py tests/test_api.py tests/test_safety_checker.py`
- [ ] T073 Verify no live external providers or derived index services are required by PostgreSQL storage tests in `tests/`
- [ ] T074 Review all new persistence paths for forbidden investment advice, brokerage, order execution, and guaranteed outcome fields in `src/db/`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies, can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational; suggested MVP slice.
- **User Story 2 (Phase 4)**: Depends on Foundational; can run in parallel with US1 after shared models/repositories are stable, but integrates best after US1 source/evidence tables exist.
- **User Story 3 (Phase 5)**: Depends on Foundational; can run in parallel with US1/US2 after user/ticker base models exist.
- **Polish (Phase 6)**: Depends on all selected user stories.

### User Story Dependencies

- **US1 - Define Canonical Research Records**: Starts after Phase 2 and is the recommended MVP implementation target.
- **US2 - Rebuild Derived Search And Graph Indexes**: Starts after Phase 2; source document/evidence relationships benefit from US1 completion.
- **US3 - Support User History, Settings, In-App Notifications, And Portfolios**: Starts after Phase 2; user model is shared with US1 but remains independently testable.

### Within Each User Story

- Write deterministic tests first and confirm they fail for missing behavior.
- Implement models before repositories.
- Implement repositories before workflow/API integration.
- Run story-specific tests before starting polish tasks.

---

## Parallel Execution Examples

### User Story 1

```bash
Task: "T020 [P] [US1] Add schema assertions for core research tables and uniqueness rules in tests/test_storage_postgres_schema.py"
Task: "T021 [P] [US1] Add ticker, analysis request, node run, and analysis result repository tests in tests/test_storage_postgres_repositories.py"
Task: "T022 [P] [US1] Add evidence, report version, and citation repository tests in tests/test_storage_postgres_repositories.py"
Task: "T023 [P] [US1] Add complete deterministic research ledger integration test in tests/test_storage_postgres_research_flow.py"
```

### User Story 2

```bash
Task: "T039 [P] [US2] Add projection status lifecycle tests for pending, success, failed, and stale states in tests/test_storage_postgres_projection_status.py"
Task: "T041 [P] [US2] Add graph repository tests for wave rules, scenarios, invalidation conditions, and scenario-rule joins in tests/test_storage_postgres_projection_status.py"
Task: "T043 [P] [US2] Implement IndexProjectionStatus and KeywordTerm models in src/db/models.py"
Task: "T044 [P] [US2] Implement WaveRule, WaveScenario, WaveInvalidationCondition, and WaveScenarioRule models in src/db/models.py"
```

### User Story 3

```bash
Task: "T052 [P] [US3] Add local/demo user repository tests without email, password, SSO, or session fields in tests/test_storage_postgres_privacy.py"
Task: "T054 [P] [US3] Add in-app notification create/list/mark-read/delete tests with forbidden phrase checks in tests/test_storage_postgres_privacy.py"
Task: "T055 [P] [US3] Add research-only portfolio and portfolio item tests excluding brokerage/order fields in tests/test_storage_postgres_privacy.py"
Task: "T057 [P] [US3] Implement UserSetting, Notification, Portfolio, and PortfolioItem models in src/db/models.py"
```

---

## Implementation Strategy

### First Delivery Slice: US1

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational DB infrastructure.
3. Complete Phase 3 User Story 1.
4. Stop and validate `tests/test_storage_postgres_schema.py`, `tests/test_storage_postgres_repositories.py`, and `tests/test_storage_postgres_research_flow.py`.
5. Demo a persisted research run while keeping local JSON/Markdown export as compatibility output.

**004 Completion Criteria**: The full `004-postgresql-table-schema` feature is complete only after US1, US2, and US3 are implemented and validated.

### Incremental Delivery

1. Add PostgreSQL research ledger (US1).
2. Add projection and graph canonical records (US2).
3. Add local/demo user UX persistence (US3).
4. Run polish and cross-cutting validation.

### Safety Boundaries

- Do not add login, password, SSO, durable sessions, brokerage fields, order tables, trade execution fields, guaranteed target fields, or external notification delivery in this feature.
- Keep all tests deterministic and free of live external provider calls.
- Preserve the required Korean disclaimer behavior in final report persistence.
