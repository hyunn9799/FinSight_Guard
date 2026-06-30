# 작업 목록: Spec Kit 문서 한국어화

**입력**: `specs/007-korean-spec-docs/`의 설계 문서

**선행 문서**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/translation-validation-contract.md](./contracts/translation-validation-contract.md), [quickstart.md](./quickstart.md)

**검증 방침**: 이 기능은 제품 실행 코드를 변경하지 않으므로 별도 런타임 테스트 파일을 만들지 않는다. 각 번역 배치는 `translation-validation-contract.md`의 G0~G5 검증을 결정적으로 수행하며 외부 API를 사용하지 않는다. 저장소 공통 완료 게이트로 `python3 -m compileall src`를 실행한다.

**구성 원칙**: 작업은 사용자 스토리별로 묶으며, 각 스토리는 독립적인 사용자 가치를 제공하고 별도의 완료 기준으로 검증할 수 있다.

## 작업 형식: `[ID] [P?] [Story] 설명과 파일 경로`

- **[P]**: 서로 다른 파일을 수정하고 완료되지 않은 선행 task에 의존하지 않아 병렬 실행할 수 있음
- **[Story]**: 사용자 스토리 추적 표기인 `[US1]`, `[US2]`, `[US3]`
- 모든 작업은 실제 대상 또는 검증 결과를 기록할 정확한 파일 경로를 포함함

## 학습 단계 브리핑

**이번 단계의 목적**: 계획의 번역 배치와 G0~G5 게이트를 사용자가 한 번에 선택해 실행하고 검증할 수 있는 크기의 task로 바꾼다.

**이번 단계에서 확인할 판단**:

- 공통 용어와 기준선을 모든 번역보다 먼저 완성한다.
- US1은 핵심 명세와 요구사항 체크리스트만으로 독립적인 MVP를 제공한다.
- US2는 조사·모델·계약·계획·가이드·작업 문서를 기능 단위로 완료한다.
- US3는 향후 문서가 처음부터 한국어로 작성되는 유지 규칙을 검증한다.
- 같은 용어 또는 선행 기능에 의존하는 task는 순차 실행하고, 실제로 충돌하지 않는 파일만 `[P]`로 표시한다.

**가정**:

- 구현 단계에서는 사용자가 명시한 task 또는 task 범위만 실행하며 전체 목록을 기본 실행하지 않는다.
- 번역 원문은 Git `HEAD`에서 조회하며 영문 사본이나 영문 주석을 새로 만들지 않는다.
- 번역 중 발견한 설계 문제는 같은 task에서 수정하지 않고 `translation-inventory.md`의 발견 사항으로 기록한다.
- 템플릿의 전면 한국어화는 이번 범위가 아니며, US3는 한국어 우선 작성 규칙과 생성 결과 검증으로 달성한다.

## Phase 1: 설정

**목적**: 번역 대상, 용어와 공통 검증 양식을 준비한다.

- [X] T001 현재 `specs/**/*.md` 대상 경로, 산출물 유형, 원문 언어와 초기 상태를 `specs/007-korean-spec-docs/translation-inventory.md`에 작성
- [X] T002 [P] 공통 금융·백테스트·저장소·provider·GraphRAG 용어의 한국어 표현, 원문 유지 여부와 최초 병기 규칙을 `specs/007-korean-spec-docs/glossary.md`에 작성
- [X] T003 [P] G0~G5, 정확 일치·구조·의미 보호 요소와 배치별 결과 기록란을 `specs/007-korean-spec-docs/checklists/translation-review.md`에 작성

---

## Phase 2: 공통 기반

**목적**: 모든 사용자 스토리를 막는 배치 의존성, 용어와 constitution 보호 기준을 확정한다.

**중요**: 이 Phase가 완료되기 전에는 번역을 시작하지 않는다.

- [X] T004 [P] `001 → 002 → 003 → 004 → 006 → 007`의 P1 배치와 `002 → 004 → 006`의 P2 배치, 선행 관계와 직접 의존 문서를 `specs/007-korean-spec-docs/translation-inventory.md`에 매핑
- [X] T005 [P] MUST/MUST NOT, evidence, fallback, degraded, traceability, source of truth, contract 등 의무 강도와 문맥이 중요한 용어를 `specs/007-korean-spec-docs/glossary.md`에서 확정
- [X] T006 [P] 요구사항·작업 ID, 숫자, 링크 목적지, 코드 펜스, 명령어, 정확한 면책문과 constitution 중요 의미의 검증 규칙을 `specs/007-korean-spec-docs/checklists/translation-review.md`에 확정
- [X] T007 `specs/007-korean-spec-docs/contracts/translation-validation-contract.md`에 따라 `specs/007-korean-spec-docs/translation-inventory.md`, `specs/007-korean-spec-docs/glossary.md`, `specs/007-korean-spec-docs/checklists/translation-review.md`의 G0 준비 상태를 검증하고 결과 기록

**체크포인트**: 모든 번역 배치가 같은 용어, 상태 모델과 합격 기준을 사용할 준비가 완료됨

---

## Phase 3: 사용자 스토리 1 - 핵심 명세를 한국어로 학습 (우선순위: P1) 🎯 MVP

**목표**: 기존 모든 기능의 `spec.md`와 요구사항 품질 체크리스트를 한국어 단일 본문으로 제공한다.

**독립 테스트**: 각 P1 배치에서 원문과 번역문을 비교하여 요구사항 ID, 숫자, 링크, 구조와 constitution 중요 의미가 보존되고 미번역 설명 문장이 0건인지 G0~G5로 검증한다.

### P1-A: 001 견고한 백테스트 검증

- [X] T008 [US1] 식별자·숫자·인수 조건을 보존하며 `specs/001-robust-backtest-validation/spec.md`의 설명 본문을 한국어로 번역
- [X] T009 [P] [US1] 체크 항목과 명세 링크를 보존하며 `specs/001-robust-backtest-validation/checklists/requirements.md`를 한국어로 번역
- [X] T010 [US1] P1-A의 G0~G5를 실행하고 `specs/007-korean-spec-docs/translation-inventory.md`와 `specs/007-korean-spec-docs/checklists/translation-review.md`에 승인 또는 실패 결과 기록

### P1-B: 002 Walk-Forward 최적화

- [X] T011 [US1] P1-A 용어와 요구사항 의미를 재사용하여 `specs/002-walk-forward-optimization/spec.md`를 한국어로 번역
- [X] T012 [P] [US1] 명세 링크와 품질 판정 의미를 보존하며 `specs/002-walk-forward-optimization/checklists/requirements.md`를 한국어로 번역
- [X] T013 [US1] P1-B의 G0~G5를 실행하고 `specs/007-korean-spec-docs/translation-inventory.md`와 `specs/007-korean-spec-docs/checklists/translation-review.md`에 승인 또는 실패 결과 기록

### P1-C: 003 플랫폼 저장소 아키텍처

- [X] T014 [US1] 범위 경계, 미래 방향과 비목표를 보존하며 `specs/003-platform-storage-architecture/spec.md`를 한국어로 번역
- [X] T015 [US1] P1-C의 G0~G5를 실행하고 `specs/007-korean-spec-docs/translation-inventory.md`와 `specs/007-korean-spec-docs/checklists/translation-review.md`에 승인 또는 실패 결과 기록

### P1-D: 004 PostgreSQL 기준 테이블 스키마

- [X] T016 [US1] P1-C 저장소 용어, 엔터티 경계와 요구사항 ID를 보존하며 `specs/004-postgresql-table-schema/spec.md`를 한국어로 번역
- [X] T017 [P] [US1] 명세 링크와 품질 판정 의미를 보존하며 `specs/004-postgresql-table-schema/checklists/requirements.md`를 한국어로 번역
- [X] T018 [US1] P1-D의 G0~G5를 실행하고 `specs/007-korean-spec-docs/translation-inventory.md`와 `specs/007-korean-spec-docs/checklists/translation-review.md`에 승인 또는 실패 결과 기록

### P1-E: 006 Provider 독립 MCP 계약

- [ ] T019 [US1] 004 소유권 경계, provider 정규화, GraphRAG, 안전 계약과 식별자를 보존하며 `specs/006-provider-mcp-contracts/spec.md`를 한국어로 번역
- [ ] T020 [P] [US1] 명세 링크와 품질 판정 의미를 보존하며 `specs/006-provider-mcp-contracts/checklists/requirements.md`를 한국어로 번역
- [ ] T021 [US1] P1-E의 G0~G5를 실행하고 `specs/007-korean-spec-docs/translation-inventory.md`와 `specs/007-korean-spec-docs/checklists/translation-review.md`에 승인 또는 실패 결과 기록

### P1-F: 007 기준 문서 정렬

- [ ] T022 [US1] `specs/007-korean-spec-docs/spec.md`와 `specs/007-korean-spec-docs/checklists/requirements.md`를 공통 용어집과 대조하고 P1 전체의 SC-001, SC-003~SC-006 결과를 `specs/007-korean-spec-docs/checklists/translation-review.md`에 기록

**체크포인트**: 모든 핵심 명세와 요구사항 체크리스트를 한국어로 읽고 독립적으로 범위·요구사항·성공 기준을 설명할 수 있음

---

## Phase 4: 사용자 스토리 2 - 설계 과정 전체를 한국어로 추적 (우선순위: P2)

**목표**: 후속 산출물이 있는 `002`, `004`, `006` 기능의 조사부터 작업 목록까지 한국어로 연결한다.

**독립 테스트**: 기능 디렉터리 하나를 선택해 `spec → research → data-model/contract → plan → quickstart → tasks`의 링크와 요구사항·작업 추적성을 따라가며 미번역 설명, 보호 요소 변경과 깨진 참조가 0건인지 검증한다.

### P2-A: 002 전체 산출물

- [ ] T023 [US2] 결정·근거·대안 구조와 확정 용어를 보존하며 `specs/002-walk-forward-optimization/research.md`를 한국어로 번역
- [ ] T024 [US2] 엔터티, 필드, 관계, 검증 규칙과 상태 의미를 보존하며 `specs/002-walk-forward-optimization/data-model.md`를 한국어로 번역
- [ ] T025 [US2] 저장 책임과 구성 요소 경계를 보존하며 `specs/002-walk-forward-optimization/storage-architecture.md`를 한국어로 번역
- [ ] T026 [US2] API 이름, 필드, 요청·응답 예시와 계약 의무를 보존하며 `specs/002-walk-forward-optimization/contracts/robust-optimization-api.md`를 한국어로 번역
- [ ] T027 [US2] constitution 점검, 구조 결정, 참조 링크와 절충안을 보존하며 `specs/002-walk-forward-optimization/plan.md`를 한국어로 번역
- [ ] T028 [US2] 명령어와 기대 결과를 보존하며 `specs/002-walk-forward-optimization/quickstart.md`를 한국어로 번역
- [ ] T029 [US2] T### ID, `[P]`, `[US#]`, 체크박스, 파일 경로와 의존 순서를 보존하며 `specs/002-walk-forward-optimization/tasks.md`를 한국어로 번역
- [ ] T030 [US2] P2-A 전체의 G0~G5와 상대 링크 추적을 실행하고 `specs/007-korean-spec-docs/translation-inventory.md`와 `specs/007-korean-spec-docs/checklists/translation-review.md`에 결과 기록

### P2-B: 004 전체 산출물

- [ ] T031 [US2] 결정·근거·대안과 PostgreSQL 저장 용어를 보존하며 `specs/004-postgresql-table-schema/research.md`를 한국어로 번역
- [ ] T032 [US2] 테이블·필드·관계·제약조건·인덱스 의미를 보존하며 `specs/004-postgresql-table-schema/data-model.md`를 한국어로 번역
- [ ] T033 [US2] 스키마명, 필드명, SQL 예시와 저장 계약 의무를 보존하며 `specs/004-postgresql-table-schema/contracts/postgresql-storage-contract.md`를 한국어로 번역
- [ ] T034 [US2] constitution 점검, 구조 결정, 참조 링크와 003·006 경계를 보존하며 `specs/004-postgresql-table-schema/plan.md`를 한국어로 번역
- [ ] T035 [US2] 명령어, 마이그레이션 순서와 기대 결과를 보존하며 `specs/004-postgresql-table-schema/quickstart.md`를 한국어로 번역
- [ ] T036 [US2] T### ID, `[P]`, `[US#]`, 체크박스, 파일 경로와 의존 순서를 보존하며 `specs/004-postgresql-table-schema/tasks.md`를 한국어로 번역
- [ ] T037 [US2] P2-B 전체의 G0~G5와 상대 링크 추적을 실행하고 `specs/007-korean-spec-docs/translation-inventory.md`와 `specs/007-korean-spec-docs/checklists/translation-review.md`에 결과 기록

### P2-C: 006 전체 산출물

- [ ] T038 [US2] 결정·근거·대안과 provider·GraphRAG 용어를 보존하며 `specs/006-provider-mcp-contracts/research.md`를 한국어로 번역
- [ ] T039 [US2] 계약 엔터티, 필드, 관계, 상태와 004·005 소유권을 보존하며 `specs/006-provider-mcp-contracts/data-model.md`를 한국어로 번역
- [ ] T040 [US2] 계약명, 필드명, 예시, 정규화·계보 의무와 안전 범위를 보존하며 `specs/006-provider-mcp-contracts/contracts/provider-normalization-contract.md`를 한국어로 번역
- [ ] T041 [US2] constitution 점검, 구조 결정, 참조 링크와 004·005·006 경계를 보존하며 `specs/006-provider-mcp-contracts/plan.md`를 한국어로 번역
- [ ] T042 [US2] 명령어, 검증 시나리오와 기대 결과를 보존하며 `specs/006-provider-mcp-contracts/quickstart.md`를 한국어로 번역
- [ ] T043 [US2] T### ID, `[P]`, `[US#]`, 체크박스, 파일 경로와 의존 순서를 보존하며 `specs/006-provider-mcp-contracts/tasks.md`를 한국어로 번역
- [ ] T044 [US2] P2-C 전체의 G0~G5와 상대 링크 추적을 실행하고 `specs/007-korean-spec-docs/translation-inventory.md`와 `specs/007-korean-spec-docs/checklists/translation-review.md`에 결과 기록

**체크포인트**: `002`, `004`, `006`의 설계 흐름과 요구사항 추적을 한국어 문서만으로 끝까지 확인할 수 있음

---

## Phase 5: 사용자 스토리 3 - 향후 산출물을 처음부터 한국어로 작성 (우선순위: P3)

**목표**: 새 Spec Kit 산출물이 한국어 단일 본문과 동일한 용어·검증 기준을 처음부터 적용하도록 한다.

**독립 테스트**: 현재 `007` 산출물을 새 문서 표본으로 사용해 필수 설명 섹션이 한국어이고 허용된 기술 원문만 남으며 영한 중복 본문이 없는지 검증한다.

- [ ] T045 [P] [US3] 향후 산출물의 한국어 우선 작성, 최초 원문 병기와 예외 처리 규칙을 `specs/007-korean-spec-docs/glossary.md`의 유지관리 섹션에 추가
- [ ] T046 [P] [US3] 새 산출물 초안의 한국어 설명, 기술 원문 허용 목록과 영한 중복 금지를 확인하는 항목을 `specs/007-korean-spec-docs/checklists/translation-review.md`에 추가
- [ ] T047 [US3] `specs/007-korean-spec-docs/spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`, `contracts/translation-validation-contract.md`, `tasks.md`를 새 문서 표본으로 검증하고 결과를 `specs/007-korean-spec-docs/translation-inventory.md`에 기록
- [ ] T048 [US3] US3의 SC-007과 G2·G3·G5를 검증하고 승인 결과 또는 템플릿 후속 검토 필요성을 `specs/007-korean-spec-docs/checklists/translation-review.md`에 기록

**체크포인트**: 향후 문서의 한국어 우선 작성 규칙과 검증 절차가 독립적으로 적용 가능함

---

## Phase 6: 마무리 및 교차 검증

**목적**: 완료한 사용자 스토리 범위 전체의 언어, 참조, 보호 요소와 constitution 정합성을 최종 확인한다.

- [ ] T049 전체 `specs/**/*.md`의 영문 후보를 허용된 기술 원문과 미번역 설명으로 분류하고 결과를 `specs/007-korean-spec-docs/translation-inventory.md`에 기록
- [ ] T050 상대 링크, 요구사항·작업 ID, 숫자, 코드 펜스와 명령어의 전역 비교 결과를 `specs/007-korean-spec-docs/checklists/translation-review.md`에 기록
- [ ] T051 영한 중복 본문 0건과 기능 간 용어 일관성을 검토하고 예외·용어 결정을 `specs/007-korean-spec-docs/glossary.md`와 `specs/007-korean-spec-docs/translation-inventory.md`에 반영
- [ ] T052 `specs/007-korean-spec-docs/quickstart.md`의 검증 명령, `git diff --check`, `python3 -m compileall src`를 실행하고 결과를 `specs/007-korean-spec-docs/checklists/translation-review.md`에 기록
- [ ] T053 금융 안전, `EvidenceItem`, 실패·대체 처리, 최대 2회 재작성과 관측성 요구의 의미 보존을 최종 검토하고 완료 상태와 미해결 발견 사항을 `specs/007-korean-spec-docs/translation-inventory.md`에 기록

---

## 의존성과 실행 순서

### Phase 의존성

- **Phase 1 설정**: 즉시 시작할 수 있다.
- **Phase 2 공통 기반**: Phase 1 완료 후 시작하며 모든 사용자 스토리를 차단한다.
- **Phase 3 US1**: Phase 2 완료 후 시작한다.
- **Phase 4 US2**: 공통 기반만으로 독립 검증할 수 있지만, 계획의 용어 의존성 때문에 US1 승인 후 시작한다.
- **Phase 5 US3**: 공통 기반만으로 규칙 작성은 가능하지만, 전체 용어와 실제 산출물을 표본으로 사용하기 위해 US2 승인 후 완료한다.
- **Phase 6 마무리**: 사용자가 선택한 사용자 스토리 범위가 모두 승인된 뒤 수행한다.

### 사용자 스토리 의존성 그래프

```text
Setup → Foundation → US1(P1 핵심 명세) → US2(P2 전체 산출물) → US3(P3 향후 작성 규칙) → Polish
```

### 사용자 스토리별 독립성

- **US1**: 핵심 명세와 체크리스트만 번역해도 독립적인 MVP가 된다.
- **US2**: 기능 디렉터리 하나의 문서 흐름만 선택하여 독립 검증할 수 있지만 공통 용어는 US1 결과를 재사용한다.
- **US3**: 한국어 우선 작성 규칙은 독립 적용할 수 있지만 최종 검증은 완성된 `007` 문서 표본을 사용한다.

### 사용자 스토리 내부 순서

- 번역 전에 G0 기준선과 선행 배치 승인을 확인한다.
- 명세가 체크리스트보다 먼저이며, 동일 배치의 두 파일은 충돌하지 않을 때 병렬 처리할 수 있다.
- 조사 → 데이터 모델 → 추가 아키텍처/계약 → 계획 → 빠른 시작 → 작업 목록 순서로 번역한다.
- 각 배치 번역 뒤 G1~G5를 통과해야 다음 배치로 이동한다.

## 병렬 실행 기회

### 설정과 공통 기반

```text
T001 inventory 작성
T002 glossary 초안 작성
T003 공통 translation-review 체크리스트 작성

Phase 1 완료 후:
T004 배치 의존성 매핑
T005 중요 용어 확정
T006 보호 요소 검증 규칙 확정
```

### US1

각 명세 번역과 해당 요구사항 체크리스트 번역은 같은 선행 배치 승인 후 서로 다른 파일에서 병렬 실행할 수 있다.

```text
T008 001 spec.md 번역
T009 001 requirements.md 번역

T011 002 spec.md 번역
T012 002 requirements.md 번역

T016 004 spec.md 번역
T017 004 requirements.md 번역

T019 006 spec.md 번역
T020 006 requirements.md 번역
```

### US2

US2 산출물은 앞 문서의 용어와 계약을 뒤 문서가 재사용하므로 기능 묶음 내부를 순차 실행한다. 기능 묶음도 `002 → 004 → 006` 의존 순서를 유지한다.

### US3

```text
T045 glossary 유지관리 규칙 추가
T046 translation-review 새 문서 검증 항목 추가
```

## 구현 전략

### MVP 우선: US1만 실행

1. T001~T003으로 목록, 용어집과 검토 양식을 만든다.
2. T004~T007로 공통 기반을 승인한다.
3. T008~T022로 핵심 명세와 체크리스트를 순서대로 번역한다.
4. T022에서 US1 독립 검증을 완료하고 중단한다.
5. 사용자가 승인한 뒤에만 US2로 확장한다.

### 점진적 제공

1. **MVP**: Setup + Foundation + US1 — 모든 핵심 명세를 한국어로 학습
2. **확장 1**: US2의 P2-A — `002` 설계 흐름 전체 한국어화
3. **확장 2**: US2의 P2-B — `004` 저장 설계 흐름 전체 한국어화
4. **확장 3**: US2의 P2-C — `006` provider 계약 흐름 전체 한국어화
5. **완성**: US3 + 마무리 — 향후 한국어 우선 작성과 전역 검증

### task 선택 원칙

- `/speckit-implement`에는 반드시 실행할 task ID 또는 연속 범위를 명시한다.
- 선행 task가 완료되지 않은 범위는 선택하지 않는다.
- 한 번에 하나의 번역 배치와 해당 검증 task를 선택하는 것을 기본으로 한다.
- 검증에 실패하면 다음 task로 넘어가지 않고 같은 배치를 수정한다.

## 참고 사항

- `[P]`는 서로 다른 파일이며 선행 의존성이 없는 경우에만 사용한다.
- 번역은 설계 변경이 아니므로 요구사항 ID, 작업 ID, 수치와 범위를 바꾸지 않는다.
- 금지 문구가 규칙 설명에 등장한 경우와 실제 금융 권유 문장을 구분한다.
- 영어 검색 결과가 0일 필요는 없으며, 설명 문장으로 분류된 미번역 영어만 0건이어야 한다.
- 각 task 전에는 변경 대상 파일과 변경 이유를 설명하고, task 후에는 실제 변경과 검증 결과를 요약한다.

## 단계 완료 리뷰

- **이번 단계의 목적**: 번역 계획을 사용자가 선택 가능한 53개의 순차·병렬 task와 독립 검증 체크포인트로 변환했다.
- **새로 생긴 파일 또는 변경된 파일**: `specs/007-korean-spec-docs/tasks.md`.
- **핵심 설계 결정**: US1을 MVP로 고정하고, 공통 기준선 이후 기능 의존성 순서로 번역하며, 각 배치 뒤 G0~G5 결과를 기록한다.
- **검토한 task 묶음 대안**: 문서 하나당 번역·검증을 모두 묶는 방식, 모든 `spec.md`를 병렬 번역하는 방식, 기능 디렉터리 전체를 한 task로 처리하는 방식을 검토했다.
- **현재 task 구조를 선택한 이유**: 번역 변경과 검증 책임을 분리하면서도 각 task가 한 LLM 컨텍스트에서 처리 가능한 크기이고, 실패한 배치를 정확히 식별할 수 있기 때문이다.
- **반드시 읽어야 할 부분**: Phase 2 공통 기반, Phase 3 US1의 배치별 검증 task, 의존성 그래프, MVP 우선 전략과 task 선택 원칙.
