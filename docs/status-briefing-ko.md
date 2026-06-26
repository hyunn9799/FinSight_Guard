# FinSight Guard 진행상황 브리핑

## 목적

이 문서는 지금 프로젝트가 어디까지 와 있는지 한국어로 빠르게 파악하기
위한 학습용 브리핑이다. 구현 지시서가 아니라, 다음에 무엇을 확인하고
어떤 순서로 진행할지 결정하기 위한 현재 상태 요약이다.

## 현재 큰 그림

FinSight Guard는 LangGraph 기반 금융 리서치 멀티 에이전트 프로젝트다.
목표는 종목을 매수/매도하라고 말하는 시스템이 아니라, 시장 데이터,
재무 데이터, 뉴스 근거를 모아 한국어 시나리오 기반 리서치 보고서를
만들고 Evaluator/Rewrite 루프로 안전성을 검토하는 것이다.

현재 코드에는 다음 축이 이미 존재한다.

- Market/Fundamental/News/Coordinator/Evaluator/Rewrite/Supervisor Agent
- FastAPI 진입점 `main.py`
- Streamlit UI `app.py`
- EvidenceItem 기반 근거 추적
- 금지 문구와 필수 고지문 검사
- 로컬 report/run storage
- PostgreSQL 모델, repository, migration
- provider-agnostic contract 계층
- backtest 및 walk-forward optimization 관련 코드와 테스트

## 가장 헷갈리는 지점

`specs/*/tasks.md`의 체크박스와 실제 코드 상태가 맞지 않는다. 예를 들어
`004-postgresql-table-schema/tasks.md`와 `006-provider-mcp-contracts/tasks.md`에는
아직 `[ ]`로 남은 작업이 많지만, 해당 파일과 테스트는 이미 상당 부분
구현되어 있다.

따라서 지금 바로 새 기능을 구현하기보다, 먼저 tasks.md와 실제 코드의
상태를 맞추는 감사 작업이 필요하다.

## 현재 기준 feature

AGENTS.md는 현재 Spec Kit 계획으로 다음 문서를 가리킨다.

- `specs/004-postgresql-table-schema/plan.md`

즉, 현재 우선순위가 가장 높은 Spec Kit feature는 `004 PostgreSQL
Source-of-Truth Table Schema`다.

## 주요 feature별 상태

### 004 PostgreSQL Source-of-Truth Table Schema

목표는 PostgreSQL을 리서치 결과, evidence, report, source document,
projection status, user UX record의 canonical store로 두는 것이다.

현재 코드에는 `src/db/models.py`, `src/db/repositories/*`,
`src/db/migrations/versions/*`, PostgreSQL 관련 테스트가 이미 존재한다.
하지만 `tasks.md` 체크박스는 대부분 미완료로 남아 있다.

다음에 해야 할 일은 구현이 아니라, T001부터 T074까지 실제 코드와 테스트에
대조해 다음 세 가지로 분류하는 것이다.

- 완료됨
- 구현은 있으나 검증 필요
- 실제 미완료

### 006 Provider-Agnostic MCP Contracts

목표는 future MCP provider를 바로 붙이기 전에 raw provider payload가 agent
logic으로 새지 않도록 normalized contract 계층을 만드는 것이다.

현재 코드에는 `src/providers/*`, `src/graph_rag/mapping_contracts.py`,
`ScenarioReportInput`, provider contract 테스트들이 이미 있다. 이 feature도
tasks.md 체크박스와 실제 구현 상태가 맞지 않는다.

004 상태 감사를 먼저 끝낸 뒤 006도 같은 방식으로 정리하는 것이 좋다.

### 002 Walk-Forward Optimization

backtest와 walk-forward optimization 관련 코드와 테스트가 존재한다. 다만
현재 AGENTS.md가 가리키는 중심 feature는 004이므로, 당장 우선순위는 낮다.

## 추천 다음 단계

다음 작업은 새 구현이 아니라 다음 범위의 감사 작업이다.

`004-postgresql-table-schema/tasks.md`의 T001부터 T074까지를 실제 파일과
테스트에 대조해 상태 표를 만든다.

산출물은 다음 형태가 좋다.

- `docs/004-task-audit-ko.md`

내용은 task ID별로 다음 항목을 기록한다.

- task ID
- 현재 판정: 완료됨 / 검증 필요 / 미완료
- 근거 파일
- 필요한 테스트
- 다음 조치

이 작업이 끝나면 그때부터 구현 범위를 작게 정할 수 있다. 예를 들어
“검증 필요인 task 중 PostgreSQL schema tests만 먼저 정리”처럼 안전하게
범위를 고를 수 있다.

## 반드시 읽어야 할 문서

- `AGENTS.md`
- `PROJECT_PLAN.md`
- `.specify/memory/constitution.md`
- `specs/004-postgresql-table-schema/plan.md`
- `specs/004-postgresql-table-schema/tasks.md`
- `specs/006-provider-mcp-contracts/plan.md`
- `specs/006-provider-mcp-contracts/tasks.md`

## 지금 구현하지 말아야 할 것

다음 작업은 아직 바로 시작하지 않는 편이 좋다.

- 새 provider MCP adapter 구현
- 새 DB 테이블 추가
- Streamlit UI 확장
- FastAPI endpoint 추가
- tasks.md 전체 실행

이유는 현재 task 체크리스트와 실제 코드 상태가 먼저 맞아야, 다음 구현이
중복인지 누락 보완인지 판단할 수 있기 때문이다.
