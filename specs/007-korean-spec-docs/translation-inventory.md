# Spec Kit 문서 번역 목록

## 기준선

- **기준일**: 2026-06-30
- **직접 범위**: `specs/**/*.md`
- **원문 기준**: 기존 추적 파일은 Git `HEAD`, 007 기능에서 새로 생성된 파일은 최초 승인본
- **상태 목적**: 번역 대상, 선행 관계, G0~G5 결과와 발견 사항을 한 문서에서 추적
- **중복 정책**: 영문 사본이나 영문 주석을 만들지 않고 한국어 단일 본문을 유지

## 상태 정의

- `discovered`: 번역 대상만 식별된 상태
- `baselined`: 원문과 보호 요소 기준선을 확인한 상태
- `translating`: 한국어화 진행 중
- `mechanically_verified`: G1·G2 기계 검증 완료
- `semantically_reviewed`: G3·G4 의미 및 constitution 검토 완료
- `approved`: G0~G5 전체 통과
- `excluded`: 직접 범위에서 제외되었으며 사유가 기록된 상태

검토 상태는 `not_started`, `pending`, `pass`, `fail`, `blocked`를 사용한다.

## 문서 현황

| 경로 | 산출물 유형 | 원문 언어 | 번역 상태 | 검토 상태 |
|------|-------------|-----------|-----------|-----------|
| `specs/001-robust-backtest-validation/checklists/requirements.md` | checklist | english | approved | pass |
| `specs/001-robust-backtest-validation/spec.md` | spec | english | approved | pass |
| `specs/002-walk-forward-optimization/checklists/requirements.md` | checklist | english | approved | pass |
| `specs/002-walk-forward-optimization/contracts/robust-optimization-api.md` | contract | english | discovered | not_started |
| `specs/002-walk-forward-optimization/data-model.md` | data-model | english | discovered | not_started |
| `specs/002-walk-forward-optimization/plan.md` | plan | english | discovered | not_started |
| `specs/002-walk-forward-optimization/quickstart.md` | quickstart | english | discovered | not_started |
| `specs/002-walk-forward-optimization/research.md` | research | english | discovered | not_started |
| `specs/002-walk-forward-optimization/spec.md` | spec | english | approved | pass |
| `specs/002-walk-forward-optimization/storage-architecture.md` | other:storage-architecture | english | discovered | not_started |
| `specs/002-walk-forward-optimization/tasks.md` | tasks | english | discovered | not_started |
| `specs/003-platform-storage-architecture/spec.md` | spec | english | approved | pass |
| `specs/004-postgresql-table-schema/checklists/requirements.md` | checklist | english | approved | pass |
| `specs/004-postgresql-table-schema/contracts/postgresql-storage-contract.md` | contract | english | discovered | not_started |
| `specs/004-postgresql-table-schema/data-model.md` | data-model | english | discovered | not_started |
| `specs/004-postgresql-table-schema/plan.md` | plan | english | discovered | not_started |
| `specs/004-postgresql-table-schema/quickstart.md` | quickstart | english | discovered | not_started |
| `specs/004-postgresql-table-schema/research.md` | research | english | discovered | not_started |
| `specs/004-postgresql-table-schema/spec.md` | spec | english | approved | pass |
| `specs/004-postgresql-table-schema/tasks.md` | tasks | english | discovered | not_started |
| `specs/006-provider-mcp-contracts/checklists/requirements.md` | checklist | english | discovered | not_started |
| `specs/006-provider-mcp-contracts/contracts/provider-normalization-contract.md` | contract | english | discovered | not_started |
| `specs/006-provider-mcp-contracts/data-model.md` | data-model | english | discovered | not_started |
| `specs/006-provider-mcp-contracts/plan.md` | plan | english | discovered | not_started |
| `specs/006-provider-mcp-contracts/quickstart.md` | quickstart | english | discovered | not_started |
| `specs/006-provider-mcp-contracts/research.md` | research | english | discovered | not_started |
| `specs/006-provider-mcp-contracts/spec.md` | spec | mixed | discovered | not_started |
| `specs/006-provider-mcp-contracts/tasks.md` | tasks | english | discovered | not_started |
| `specs/007-korean-spec-docs/checklists/requirements.md` | checklist | korean | baselined | pending |
| `specs/007-korean-spec-docs/contracts/translation-validation-contract.md` | contract | korean | baselined | pending |
| `specs/007-korean-spec-docs/data-model.md` | data-model | korean | baselined | pending |
| `specs/007-korean-spec-docs/glossary.md` | other:glossary | korean | approved | pass |
| `specs/007-korean-spec-docs/plan.md` | plan | korean | baselined | pending |
| `specs/007-korean-spec-docs/quickstart.md` | quickstart | korean | baselined | pending |
| `specs/007-korean-spec-docs/research.md` | research | korean | baselined | pending |
| `specs/007-korean-spec-docs/spec.md` | spec | korean | baselined | pending |
| `specs/007-korean-spec-docs/tasks.md` | tasks | korean | baselined | pending |
| `specs/007-korean-spec-docs/translation-inventory.md` | other:translation-inventory | korean | approved | pass |
| `specs/007-korean-spec-docs/checklists/translation-review.md` | checklist | korean | approved | pass |

## 배치 매핑

| 배치 ID | 우선순위 | 대상 산출물 | 선행 배치 | 직접 의존 문서 | 초기 상태 |
|---------|----------|-------------|-----------|-----------------|-----------|
| FOUNDATION | 공통 | `translation-inventory.md`, `glossary.md`, `checklists/translation-review.md` | 없음 | `research.md`, `data-model.md`, `contracts/translation-validation-contract.md`, `quickstart.md` | approved |
| P1-A | P1 | `001/spec.md`, `001/checklists/requirements.md` | FOUNDATION | `glossary.md`, `checklists/translation-review.md`, constitution 관련 원칙 | approved |
| P1-B | P1 | `002/spec.md`, `002/checklists/requirements.md` | P1-A | P1-A 승인 문서, `glossary.md`, constitution 관련 원칙 | approved |
| P1-C | P1 | `003/spec.md` | P1-B | P1-B 승인 용어, `glossary.md` | approved |
| P1-D | P1 | `004/spec.md`, `004/checklists/requirements.md` | P1-C | `003/spec.md`, `glossary.md`, constitution 관련 원칙 | approved |
| P1-E | P1 | `006/spec.md`, `006/checklists/requirements.md` | P1-D | `004/spec.md`, `specs/005-neo4j-graphrag-scenarios` 설계 경계, `glossary.md`, constitution 관련 원칙 | planned |
| P1-F | P1 | `007/spec.md`, `007/checklists/requirements.md` | P1-E | P1-A~P1-E 승인 용어, `glossary.md` | planned |
| P2-A | P2 | `002/research.md`, `002/data-model.md`, `002/storage-architecture.md`, `002/contracts/robust-optimization-api.md`, `002/plan.md`, `002/quickstart.md`, `002/tasks.md` | P1-F | `002/spec.md`, P1 승인 용어, 직접 링크 문서 | planned |
| P2-B | P2 | `004/research.md`, `004/data-model.md`, `004/contracts/postgresql-storage-contract.md`, `004/plan.md`, `004/quickstart.md`, `004/tasks.md` | P2-A | `003/spec.md`, `004/spec.md`, P2-A 승인 용어, 직접 링크 문서 | planned |
| P2-C | P2 | `006/research.md`, `006/data-model.md`, `006/contracts/provider-normalization-contract.md`, `006/plan.md`, `006/quickstart.md`, `006/tasks.md` | P2-B | `004` 승인 문서, `006/spec.md`, `005` 설계 경계, 직접 링크 문서 | planned |
| P3 | P3 | 향후 Spec Kit 산출물의 한국어 우선 작성 규칙 | P2-C | `007` 승인 문서, `glossary.md`, `checklists/translation-review.md` | planned |

### 배치 컨텍스트 제한

각 배치의 AI 컨텍스트에는 다음만 포함한다.

1. 해당 배치의 대상 산출물
2. `glossary.md`의 관련 용어
3. `checklists/translation-review.md`의 보호 기준
4. 위 표의 직접 의존 문서
5. 관련 constitution 원칙

전체 `specs/` 문서를 매번 컨텍스트에 넣지 않는다.

## 검증 결과

### FOUNDATION

| 항목 | 결과 |
|------|------|
| 검토 시점 | 2026-06-30 |
| G0 | pass |
| G1~G5 | not_applicable — 번역 배치가 아니라 공통 기준선 준비 단계 |
| 문서 목록 | 실제 Markdown 39개와 inventory 39개 일치, 누락·초과 0건 |
| 배치 매핑 | FOUNDATION, P1-A~P1-F, P2-A~P2-C, P3 총 11개가 중복 없이 존재 |
| 용어집 | 중요 용어 77개, 중복 0건, 필수 의무·근거·복구 용어 존재 |
| 보호 기준 | G0~G5, 정확 일치·구조·의미, constitution 중요 의미 규칙 존재 |
| 상대 링크 | `007` 기능 문서의 누락된 상대 링크 0건 |
| 자리표시자·형식 | 미해결 자리표시자 0건, `git diff --check` 통과 |
| 발견 사항 | F001 — P1-E/P2-C 시작 전 확인, 현재 Foundation과 P1-A에는 영향 없음 |
| 최종 상태 | approved |

### P1-A

| 항목 | 결과 |
|------|------|
| 검토 시점 | 2026-06-30 |
| 대상 | `specs/001-robust-backtest-validation/spec.md`, `specs/001-robust-backtest-validation/checklists/requirements.md` |
| 선행 배치 | FOUNDATION approved |
| G0 | pass — 원문, 용어집, 보호 규칙과 선행 상태 존재 |
| G1 | pass — ID 24개, 숫자 후보, 제목 13개, 체크 항목 16개, 링크 목적지가 원문과 일치 |
| G2 | pass — 미번역 설명과 영한 중복 본문 0건; 남은 영어는 식별자·기술명·최초 병기·품질 마커 |
| G3 | pass — 사용자 스토리 3개, FR 10개, SER 8개, SC 6개의 의무·부정·수치·범위 의미 보존 |
| G4 | pass — 직접 권유·보장·주문 실행 금지, `EvidenceItem`, 한계 공개, 결정적 실패 처리와 관측성 의미 보존 |
| G5 | pass — 상대 링크 누락 0건, 미완성 자리표시자 0건, `git diff --check`와 `python3 -m compileall src` 통과 |
| 발견 사항 | 없음 |
| 최종 상태 | approved |

### P1-B

| 항목 | 결과 |
|------|------|
| 검토 시점 | 2026-06-30 |
| 대상 | `specs/002-walk-forward-optimization/spec.md`, `specs/002-walk-forward-optimization/checklists/requirements.md` |
| 선행 배치 | P1-A approved |
| G0 | pass — 원문, P1-A 승인 용어, 공통 용어집, 보호 규칙과 선행 상태 존재 |
| G1 | pass — ID 34개, 숫자 후보, 제목 16개, 체크 항목 16개, 링크 목적지가 원문과 일치 |
| G2 | pass — 미번역 설명과 영한 중복 본문 0건; 남은 영어는 사용자 입력·식별자·기술명·최초 병기·품질 마커 |
| G3 | pass — 명확화 5개, 사용자 스토리 4개, FR 17개, SER 9개, SC 8개의 의무·부정·수치·범위 의미 보존 |
| G4 | pass — 총수익률 단독 최적화 금지, 과적합 한계, 직접 권유·보장·자동 실행 금지, `EvidenceItem`, 결정적 실패 처리와 관측성 의미 보존 |
| G5 | pass — 상대 링크 누락 0건, 미완성 자리표시자 0건, `git diff --check`와 `python3 -m compileall src` 통과 |
| 발견 사항 | 없음 |
| 최종 상태 | approved |

### P1-C

| 항목 | 결과 |
|------|------|
| 검토 시점 | 2026-06-30 |
| 대상 | `specs/003-platform-storage-architecture/spec.md` |
| 선행 배치 | P1-B approved |
| G0 | pass — 원문, P1-B 승인 용어, 공통 용어집, 보호 규칙과 선행 상태 존재 |
| G1 | pass — 숫자 후보, 제목 5개, 경로와 코드 펜스가 원문과 일치 |
| G2 | pass — 미번역 설명과 영한 중복 본문 0건; 남은 영어는 제품명·경로·식별자·기술명 |
| G3 | pass — 저장소 5개의 미래 가능성, 비목표 2개, 향후 계획 조건 4개의 의무·부정·범위 의미 보존 |
| G4 | pass — 거래·중개·주문·보장·금융 자문 금지, 외부 인프라 선택성, 유료 실시간 공급자 없는 검증 의미 보존 |
| G5 | pass — 미완성 자리표시자 0건, `git diff --check`와 `python3 -m compileall src` 통과 |
| 발견 사항 | 없음 |
| 최종 상태 | approved |

### P1-D

| 항목 | 결과 |
|------|------|
| 검토 시점 | 2026-06-30 |
| 대상 | `specs/004-postgresql-table-schema/spec.md`, `specs/004-postgresql-table-schema/checklists/requirements.md` |
| 선행 배치 | P1-C approved |
| G0 | pass — 원문, 003 승인 저장소 경계, 공통 용어집, 보호 규칙과 선행 상태 존재 |
| G1 | pass — ID 38개, 숫자 후보 58개, 제목 20개, 목록 99개, 인라인 코드 297개와 체크 항목 16개가 원문과 일치 |
| G2 | pass — 미번역 설명과 영한 중복 본문 0건; 남은 영어는 제품명·식별자·필드명·상태값·품질 마커 |
| G3 | pass — 명확화 5개, 사용자 스토리 3개, FR 20개, SER 8개, SC 10개와 개체 31개의 의무·부정·수치·범위 의미 보존 |
| G4 | pass — PostgreSQL 기준 소유권, 재구축 가능한 투영, Redis 비영속 경계, 금융 안전, `EvidenceItem`, 삭제·익명화, 실패 처리와 관측성 의미 보존 |
| G5 | pass — 상대 링크 누락 0건, 미완성 자리표시자 0건, `git diff --check`와 `python3 -m compileall src` 통과 |
| 발견 사항 | 없음 |
| 최종 상태 | approved |

## 번역 발견 사항

번역 중 확인한 원문의 모순, 오래된 참조, 누락, constitution 충돌과 오탈자를 설계 변경과 분리하여 기록한다.

| ID | 경로 | 분류 | 설명 | 번역 조치 | 후속 조치 |
|----|------|------|------|-----------|-----------|
| F001 | `specs/006-provider-mcp-contracts/spec.md` | 누락된 선행 산출물 | 문서가 `specs/005-neo4j-graphrag-scenarios`를 소유권 경계로 참조하지만 현재 저장소에 해당 디렉터리가 없음 | 원문 의미를 유지하고 P1-E/P2-C의 직접 의존성으로 기록 | 해당 배치 시작 전에 005 산출물 위치 또는 누락 상태를 확인 |
