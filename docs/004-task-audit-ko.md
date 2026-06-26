# 004 PostgreSQL tasks 상태 감사

## 이번 단계의 목적

`specs/004-postgresql-table-schema/tasks.md`의 체크박스와 실제 코드 상태가
맞지 않아 다음 구현 범위를 정하기 어렵다. 이 문서는 T001부터 T074까지를
실제 파일, 테스트, 문서에 대조해 현재 상태를 한국어로 정리한다.

## 판단 기준

- **완료**: 요구한 파일 또는 동등한 구현이 존재하고, 정적 확인상 task 목적을 충족한다.
- **부분 완료**: 핵심 구현은 있으나 task 문구 일부가 빠졌거나 범위가 다르다.
- **검증 필요**: 구현/테스트 파일은 있으나 현재 환경에서 pytest 실행 검증을 못 했다.
- **미완료**: 요구한 파일이나 동작이 확인되지 않는다.

## 실행 검증 상태

- `python3 -m compileall src`: 통과
- `git diff --check`: 통과
- DB 불필요 storage safety test:
  - 명령: `uv --cache-dir /private/tmp/uv-cache run --with pytest --with sqlalchemy --with alembic --with 'psycopg[binary]' pytest tests/test_storage_postgres_safety.py`
  - 결과: 3 passed
- PostgreSQL 관련 pytest:
  - 명령: `uv --cache-dir /private/tmp/uv-cache run --with-requirements requirements.txt pytest tests/test_storage_postgres_schema.py tests/test_storage_postgres_repositories.py tests/test_storage_postgres_research_flow.py tests/test_storage_postgres_privacy.py tests/test_storage_postgres_projection_status.py`
  - 결과: 51 passed, 3 warnings
- PostgreSQL UX/safety pytest:
  - 명령: `uv --cache-dir /private/tmp/uv-cache run --with-requirements requirements.txt pytest tests/test_storage_postgres_ux.py tests/test_storage_postgres_safety.py`
  - 결과: 11 passed, 1 warning
- 기존 regression pytest:
  - 명령: `uv --cache-dir /private/tmp/uv-cache run --with-requirements requirements.txt pytest tests/test_report_store.py tests/test_workflow_routing.py tests/test_api.py tests/test_safety_checker.py`
  - 결과: 43 passed, 1 warning
- 환경 제약:
  - 시스템 `python3`에는 pytest가 없다.
  - 일반 PATH에는 `docker`가 없지만 OrbStack의 Docker CLI를 사용해 PostgreSQL을 실행했다.
  - Docker 명령은 `/Applications/OrbStack.app/Contents/MacOS/xbin/docker-compose`를 사용했다.

## 전체 요약

현재 004의 주요 구현은 대부분 존재한다. 특히 `src/db/models.py`,
`src/db/repositories/*`, `src/db/migrations/versions/*`,
`tests/test_storage_postgres_*.py`가 이미 있다.

감사 후 `specs/004-postgresql-table-schema/tasks.md`를 실제 구현/테스트
상태에 맞게 정리했다. T035와 T037은 추가 DB 테이블을 만들기보다 현재
compatibility contract를 명확히 하는 방향으로 정리했다. 현재 004 tasks.md의
T001부터 T074까지는 모두 완료 상태다.

## Phase 1: Setup

| Task | 판정 | 근거 | 다음 조치 |
| --- | --- | --- | --- |
| T001 | 완료 | `requirements.txt`에 `sqlalchemy`, `alembic`, `psycopg[binary]` 존재 | 체크 가능 |
| T002 | 완료 | `.env.example`에 `DATABASE_URL`, `TEST_DATABASE_URL` 존재 | 체크 가능 |
| T003 | 완료 | `docker-compose.yml`에 `db` service와 healthcheck 존재 | 체크 가능 |
| T004 | 완료 | `src/db/__init__.py` 존재 | 체크 가능 |
| T005 | 완료 | `src/db/repositories/__init__.py` 존재 | 체크 가능 |
| T006 | 미완료 | `src/db/migrations/README.md` 없음 | README를 만들거나 task를 실제 구조 기준으로 수정 |
| T007 | 완료 | `tests/fixtures/postgres.py` 존재 | 체크 가능 |

## Phase 2: Foundational

| Task | 판정 | 근거 | 다음 조치 |
| --- | --- | --- | --- |
| T008 | 완료 | `src/db/postgres.py`에 `get_engine`, `SessionLocal`, `session_scope` 존재 | 체크 가능 |
| T009 | 완료 | `src/db/models.py`에 `Base`, naming convention, UUID/timestamp mixin 존재 | 체크 가능 |
| T010 | 완료 | `src/db/constants.py`에 status vocabulary 존재 | 체크 가능 |
| T011 | 완료 | `src/db/migrations/env.py`가 `Base.metadata`를 Alembic에 연결 | 체크 가능 |
| T012 | 부분 완료 | `0001_postgresql_source_of_truth.py`는 없고 US1/US2/US3로 분리된 migration 존재 | task 문구를 실제 migration 구조에 맞게 수정 |
| T013 | 완료 | `src/db/repositories/base.py`에 `BaseRepository` 존재 | 체크 가능 |
| T014 | 완료 | `tests/conftest.py`의 Alembic/db_session fixture가 실제 DB 테스트에서 동작 | 51 passed |
| T015 | 완료 | metadata/session smoke test와 migration schema tests 통과 | 51 passed |
| T016 | 완료 | repository contract tests 통과 | 51 passed |
| T017 | 완료 | workflow persistence tests 통과 | 51 passed |
| T018 | 완료 | privacy tests 통과 | 51 passed |
| T019 | 완료 | projection/graph persistence tests 통과 | 51 passed |

## Phase 3: US1 - Canonical Research Records

| Task | 판정 | 근거 | 다음 조치 |
| --- | --- | --- | --- |
| T020 | 완료 | schema catalog/migration/index assertions 통과 | 51 passed |
| T021 | 완료 | ticker/request/node/result repository tests 통과 | 51 passed |
| T022 | 완료 | evidence/report/citation repository tests 통과 | 51 passed |
| T023 | 완료 | deterministic research ledger integration tests 통과 | 51 passed |
| T024 | 완료 | degraded persistence missing-data tests 통과 | 51 passed |
| T025 | 완료 | `User`, `Ticker`, `AnalysisRequest`, `WorkflowNodeRun` 모델 존재 | 체크 가능 |
| T026 | 완료 | `AnalysisResult`, `Report`, `ReportVersion`, `EvidenceItemRecord`, `ReportEvidenceCitation` 모델 존재 | 체크 가능 |
| T027 | 완료 | `SourceDocument`, `DocumentChunk`, `StructuredLogEvent` 모델 존재 | 체크 가능 |
| T028 | 완료 | `AnalysisRepository.upsert_ticker`, `create_request` 등 존재 | 체크 가능 |
| T029 | 완료 | `record_node_run`, `add_result`, request status update 메서드 존재 | 체크 가능 |
| T030 | 완료 | `EvidenceRepository.add_evidence`, `list_for_request`, `add_citation` 존재 | 체크 가능 |
| T031 | 완료 | `ReportRepository.create_report`, `add_version`, `set_status` 존재 | 체크 가능 |
| T032 | 완료 | `SourceDocumentRepository`가 document, correction, chunk 메서드 제공 | 체크 가능 |
| T033 | 완료 | `AnalysisRepository.record_log_event` 존재 | 체크 가능 |
| T034 | 완료 | `save_report_node`가 PostgreSQL 저장 후 JSON/Markdown export 수행 | 51 passed + regression passed |
| T035 | 완료 | `save_report_node`가 PostgreSQL `request_id`/`evidence_path_id`를 기존 `save_run` metadata로 mirror한다. `run_store.py`는 compatibility in-memory store로 유지 | workflow/API tests 통과 |
| T036 | 완료 | `src/graph/workflow.py`의 `save_report_node`가 `persist_research_run` 호출 | 51 passed + workflow routing passed |
| T037 | 완료 | `/analyze`와 `/backtest`는 persisted `request_id`를 전달한다. `/backtest/optimize`는 별도 optimization-run export contract로 유지 | API tests 통과 |
| T038 | 완료 | PostgreSQL 저장 실패 시 degraded 상태로 JSON/Markdown export 후 warning/error 반환 | 51 passed + regression passed |

## Phase 4: US2 - Projection And Graph Records

| Task | 판정 | 근거 | 다음 조치 |
| --- | --- | --- | --- |
| T039 | 완료 | projection status lifecycle tests 통과 | 51 passed |
| T040 | 완료 | source document revision/chunk idempotency tests 통과 | 51 passed |
| T041 | 완료 | wave rule/scenario/condition/join tests 통과 | 51 passed |
| T042 | 완료 | evidence path ordering/canonical node reference tests 통과 | 51 passed |
| T043 | 완료 | `IndexProjectionStatus`, `KeywordTerm` 모델 존재 | 체크 가능 |
| T044 | 완료 | `WaveRule`, `WaveScenario`, `WaveInvalidationCondition`, `WaveScenarioRule` 모델 존재 | 체크 가능 |
| T045 | 완료 | `EvidencePath`, `EvidencePathStep` 모델 존재 | 체크 가능 |
| T046 | 완료 | `ProjectionRepository`에 upsert/success/failure/list 메서드 존재 | 체크 가능 |
| T047 | 완료 | `ProjectionRepository.upsert_term`, `list_terms` 존재 | 체크 가능 |
| T048 | 완료 | `GraphRepository`에 wave rule/scenario/condition/join 메서드 존재 | 체크 가능 |
| T049 | 완료 | `GraphRepository`에 evidence path/step 메서드 존재 | 체크 가능 |
| T050 | 완료 | graph evidence path persistence와 projection pending 생성 tests 통과 | 51 passed |
| T051 | 완료 | `ProjectionRepository.failure_warning` non-mutating warning behavior tests 통과 | 51 passed |

## Phase 5: US3 - User UX Persistence

| Task | 판정 | 근거 | 다음 조치 |
| --- | --- | --- | --- |
| T052 | 완료 | local/demo user privacy tests 통과 | 51 passed |
| T053 | 완료 | settings create/update/delete tests 통과. 구현은 `SettingsRepository`로 분리됨 | 10 passed, task 경로 문구 정리 권장 |
| T054 | 완료 | notification lifecycle/list tests와 forbidden phrase rejection tests 통과 | 11 passed |
| T055 | 완료 | research-only portfolio/item tests 통과 | 10 passed |
| T056 | 완료 | user anonymization preserving audit tests 통과 | 51 passed |
| T057 | 완료 | `UserSettings`, `Notification`, `Portfolio`, `PortfolioItem` 모델 존재 | 체크 가능 |
| T058 | 완료 | `UserRepository.create_user`, `anonymize_user`, soft-delete 계열 동작 tests 통과 | 51 passed |
| T059 | 완료 | settings 구현은 `src/db/repositories/settings_repository.py`에 존재하고 UX tests 통과 | 10 passed, task 경로 문구 정리 권장 |
| T060 | 완료 | `NotificationRepository`에 create/list/mark-read/archive/delete 메서드 존재 | 체크 가능 |
| T061 | 완료 | `NotificationRepository.create`가 title/body/payload의 forbidden phrase를 거부 | 11 passed |
| T062 | 완료 | `PortfolioRepository.create_portfolio`, list, soft-delete 존재 | 체크 가능 |
| T063 | 완료 | `PortfolioRepository.add_item`, list item 계열 메서드 존재 | 체크 가능 |
| T064 | 완료 | `UserRepository.anonymize_user`가 settings/notifications/portfolios/items 정리. privacy tests 통과 | 51 passed |
| T065 | 완료 | `tests/test_storage_postgres_safety.py`가 forbidden column token을 검사 | 3 passed |

## Phase 6: Polish & Cross-Cutting

| Task | 판정 | 근거 | 다음 조치 |
| --- | --- | --- | --- |
| T066 | 완료 | README에 PostgreSQL source-of-truth setup 설명 존재 | 체크 가능 |
| T067 | 완료 | `docker-compose.yml`에 PostgreSQL service와 app dependency 존재 | 체크 가능 |
| T068 | 검증 필요 | `specs/004-postgresql-table-schema/quickstart.md` 존재 | migration/repository boundary 최신성 확인 필요 |
| T069 | 완료 | persisted request ID가 붙은 `/analyze` 후에도 `/health`, `/metrics` 계약 보존 테스트 통과 | 43 passed |
| T070 | 완료 | `python3 -m compileall src` 통과 | 체크 가능 |
| T071 | 완료 | PostgreSQL storage tests 실제 DB로 실행 | 51 passed |
| T072 | 완료 | `test_report_store`, `test_workflow_routing`, `test_api`, `test_safety_checker`: 43 passed | 체크 가능 |
| T073 | 완료 | PostgreSQL storage tests가 실제 DB에서 외부 provider/derived index service 없이 통과 | 51 passed |
| T074 | 완료 | `tests/test_storage_postgres_safety.py`가 DB schema forbidden token 검사 | 3 passed |

## 핵심 설계 결정

1. 먼저 별도 audit 문서로 근거를 남긴 뒤 tasks.md를 정리했다.
   - 이유: 완료 체크를 먼저 바꾸면 어떤 근거로 완료 처리했는지 학습자가
     추적하기 어렵기 때문이다.
2. migration은 `0001` 단일 파일이 아니라 US1/US2/US3/provider extension으로
   분리된 실제 구조를 인정했다.
   - 이 구조는 feature 확장 이력을 보기 쉽지만, tasks.md의 T012 문구와는 맞지 않는다.
3. settings repository는 `user_repository.py`가 아니라
   `settings_repository.py`에 분리되어 있다.
   - 이 구조는 관심사 분리에 더 맞지만, T053/T059의 파일 경로와 다르다.
4. `run_store.py`는 아직 in-memory다.
   - PostgreSQL request/report ID를 metadata로 mirror하긴 하지만, repository-backed
     run store라고 보기는 어렵다.

## 검토한 대안

- **대안 A: tasks.md 체크박스를 바로 수정**
  - 장점: 보기에는 즉시 깔끔해진다.
  - 단점: pytest 실행 검증이 안 된 상태에서 완료 체크를 하면 학습 기록이 부정확해진다.

- **대안 B: 실제 코드만 보고 바로 다음 구현 시작**
  - 장점: 빠르게 기능 구현으로 넘어갈 수 있다.
  - 단점: 이미 구현된 작업을 중복하거나, 실제 미완료 작업을 놓칠 가능성이 크다.

- **선택한 방식: 별도 audit 문서 작성**
  - 이유: 학습 목적에 맞게 “왜 완료/부분완료/검증필요인지”를 먼저 볼 수 있다.

## 반드시 읽어야 할 부분

1. 이 문서의 `전체 요약`
2. `Phase 3: US1 - Canonical Research Records`
3. `Phase 6: Polish & Cross-Cutting`
4. `핵심 설계 결정`

## 추천 다음 작업

다음 작업은 004를 마무리 검증하는 것이다.

1. 필요하면 아래 명령으로 전체 004 DB 테스트를 재실행한다.

```bash
uv --cache-dir /private/tmp/uv-cache run --with-requirements requirements.txt pytest tests/test_storage_postgres_schema.py tests/test_storage_postgres_repositories.py tests/test_storage_postgres_research_flow.py tests/test_storage_postgres_privacy.py tests/test_storage_postgres_projection_status.py
```

2. API/regression 테스트를 재실행한다.

```bash
uv --cache-dir /private/tmp/uv-cache run --with-requirements requirements.txt pytest tests/test_report_store.py tests/test_workflow_routing.py tests/test_api.py tests/test_safety_checker.py
```
