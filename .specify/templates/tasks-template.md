---

description: "Task list template for feature implementation"
---

# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Deterministic tests are REQUIRED for changes that affect indicators,
safety checks, evaluator behavior, workflow routing, report storage, FastAPI
health/analyze behavior, provider fallback, or report generation. Tests MUST NOT
depend on live external APIs.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**학습 규칙**: tasks를 만들기 전에 작업을 어떻게 나눌지에 대한 판단을
먼저 설명한다. 구현 단계에서는 사용자가 명시한 task 범위만 실행하며,
기본값으로 전체 tasks.md를 실행하지 않는다.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## 학습 단계 브리핑

<!--
  필수 작성: tasks.md를 채우기 전에 작업을 어떻게 묶을지, 어떤 의존성이
  구현을 막는지, 어떤 테스트가 필요한지, 남은 assumptions가 무엇인지,
  어떤 파일이 변경될 것으로 예상되는지 설명한다.
-->

**이번 단계의 목적**: [학습자가 이 task 분해를 왜 읽어야 하는지 설명]

**이번 단계에서 확인할 판단**:
- [Phase 경계 결정]
- [User story 단위 분할 결정]
- [테스트 우선순위와 검증 방식 결정]

**Assumptions**:
- [애매한 요구사항을 임의로 확정하지 말고 assumptions로 기록]

<!--
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.

  The /speckit-tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/

  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment

  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize [language] project with [framework] dependencies
- [ ] T003 [P] Configure linting and formatting tools

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [ ] T004 Setup database schema and migrations framework
- [ ] T005 [P] Implement authentication/authorization framework
- [ ] T006 [P] Setup API routing and middleware structure
- [ ] T007 Create base models/entities that all stories depend on
- [ ] T008 Configure error handling and logging infrastructure
- [ ] T009 Setup environment configuration management
- [ ] T010 Define or update `EvidenceItem` contracts and deterministic provider
      fakes when the story touches research evidence
- [ ] T011 Configure financial safety checks, required disclaimer handling, and
      forbidden phrase validation when report text can change
- [ ] T012 Configure structured logging, report persistence, health checks, and
      metrics when runtime workflow paths can change

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - [Title] (Priority: P1) 🎯 MVP

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 1 ⚠️

> **NOTE: Write required deterministic tests FIRST when this story affects
> constitution-governed behavior. External APIs must be mocked or replaced with
> local fakes.**

- [ ] T013 [P] [US1] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T014 [P] [US1] Integration test for [user journey] in tests/integration/test_[name].py
- [ ] T015 [P] [US1] Safety/evidence/routing test for [behavior] in tests/[area]/test_[name].py

### Implementation for User Story 1

- [ ] T016 [P] [US1] Create [Entity1] model in src/models/[entity1].py
- [ ] T017 [P] [US1] Create [Entity2] model in src/models/[entity2].py
- [ ] T018 [US1] Implement [Service] in src/services/[service].py (depends on T016, T017)
- [ ] T019 [US1] Implement [endpoint/feature] in src/[location]/[file].py
- [ ] T020 [US1] Add validation and error handling
- [ ] T021 [US1] Add structured logging, degraded-data notes, and metrics for user story 1 operations

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - [Title] (Priority: P2)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 2 ⚠️

- [ ] T022 [P] [US2] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T023 [P] [US2] Integration test for [user journey] in tests/integration/test_[name].py
- [ ] T024 [P] [US2] Safety/evidence/routing test for [behavior] in tests/[area]/test_[name].py

### Implementation for User Story 2

- [ ] T025 [P] [US2] Create [Entity] model in src/models/[entity].py
- [ ] T026 [US2] Implement [Service] in src/services/[service].py
- [ ] T027 [US2] Implement [endpoint/feature] in src/[location]/[file].py
- [ ] T028 [US2] Integrate with User Story 1 components (if needed)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - [Title] (Priority: P3)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 3 ⚠️

- [ ] T029 [P] [US3] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T030 [P] [US3] Integration test for [user journey] in tests/integration/test_[name].py
- [ ] T031 [P] [US3] Safety/evidence/routing test for [behavior] in tests/[area]/test_[name].py

### Implementation for User Story 3

- [ ] T032 [P] [US3] Create [Entity] model in src/models/[entity].py
- [ ] T033 [US3] Implement [Service] in src/services/[service].py
- [ ] T034 [US3] Implement [endpoint/feature] in src/[location]/[file].py

**Checkpoint**: All user stories should now be independently functional

---

[Add more user story phases as needed, following the same pattern]

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] TXXX [P] Documentation updates in docs/
- [ ] TXXX Code cleanup and refactoring
- [ ] TXXX Performance optimization across all stories
- [ ] TXXX [P] Additional deterministic unit tests in tests/unit/
- [ ] TXXX Security hardening
- [ ] TXXX Run `python -m compileall src`
- [ ] TXXX Run relevant pytest suites without live external APIs
- [ ] TXXX Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
Task: "Contract test for [endpoint] in tests/contract/test_[name].py"
Task: "Integration test for [user journey] in tests/integration/test_[name].py"

# Launch all models for User Story 1 together:
Task: "Create [Entity1] model in src/models/[entity1].py"
Task: "Create [Entity2] model in src/models/[entity2].py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence

## 단계 완료 리뷰

<!--
  필수 작성: tasks.md를 만들거나 수정한 뒤 학습 관점의 결과를 요약한다.
-->

- **이번 단계의 목적**: [이 단계에서 실행 가능하게 만들려던 것]
- **새로 생긴 파일 또는 변경된 파일**: [생성/수정 파일]
- **핵심 설계 결정**: [결정 목록]
- **검토한 task 묶음 대안**: [대안과 절충점]
- **현재 task 구조를 선택한 이유**: [선택 근거]
- **반드시 읽어야 할 부분**: [학습자가 확인할 구체적인 섹션]
