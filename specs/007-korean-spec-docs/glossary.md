# Spec Kit 한국어 용어집

## 사용 원칙

1. 설명 문장은 한국어 단일 본문으로 작성하고 영문 원문을 주석이나 병렬 본문으로 복제하지 않는다.
2. 기술 개념은 아래 기본 표현을 사용하며, 학습에 필요한 경우 문서별 첫 사용에서만 원문을 병기한다.
3. 코드 식별자, 계약명, 데이터 필드명, 파일 경로, 명령어, URL, 요구사항·작업 ID는 번역하지 않는다.
4. 제품명, 라이브러리명, 표준 약어와 에이전트 역할명은 원형을 유지하고 조사는 자연스럽게 붙인다.
5. 같은 영어 용어가 문맥에 따라 다른 뜻이면 문맥을 명시한 별도 항목으로 관리한다.

## 항목 형식

| 필드 | 의미 |
|------|------|
| 원문 용어 | 원문에서 사용한 영어 표현 |
| 기본 한국어 표현 | 설명 본문에서 사용할 표현 |
| 원형 유지 | 영어 원형을 그대로 유지할지 여부 |
| 최초 사용 | 첫 등장 시 권장 표기 |
| 문맥·예외 | 기본 규칙을 적용하지 않는 경우 |

## 공통 연구와 금융 안전

| 원문 용어 | 기본 한국어 표현 | 원형 유지 | 최초 사용 | 문맥·예외 |
|-----------|------------------|-----------|-----------|-----------|
| evidence | 근거 | 아니요 | 근거(evidence) | `EvidenceItem` 객체명과 필드명은 원형 유지 |
| evidence-grounded | 근거 기반 | 아니요 | 근거 기반(evidence-grounded) | 연구·보고서의 검증 가능성을 설명할 때 사용 |
| source grounding | 출처 근거성 | 아니요 | 출처 근거성(source grounding) | 평가 기준 이름에서는 병기 가능 |
| traceability | 추적성 | 아니요 | 추적성(traceability) | 데이터 계보와 요구사항 연결 모두에 사용 가능 |
| evidence summary | 근거 요약 | 아니요 | 근거 요약(evidence summary) | 보고서 섹션명은 한국어 사용 |
| disclaimer | 면책문 | 아니요 | 면책문(disclaimer) | constitution의 정확한 면책문 본문은 변경 금지 |
| financial advice | 금융 자문 | 아니요 | 금융 자문(financial advice) | 시스템이 제공하지 않는 범위를 설명 |
| scenario framing | 시나리오 기반 표현 | 아니요 | 시나리오 기반 표현(scenario framing) | 직접적인 매수·매도·보유 권유의 대안 |
| risk disclosure | 위험 공개 | 아니요 | 위험 공개(risk disclosure) | `RiskAssessment` 계약명은 원형 유지 |
| limitation | 한계 | 아니요 | 한계(limitation) | 복수형도 `한계`로 통일 |
| data freshness | 데이터 최신성 | 아니요 | 데이터 최신성(data freshness) | 수집 시점과 별도 개념 |
| EvidenceItem | `EvidenceItem` | 예 | `EvidenceItem` | 객체명과 필드명은 항상 원형 유지 |

## 의무 강도와 계약 용어

| 원문 용어 | 기본 한국어 표현 | 원형 유지 | 최초 사용 | 문맥·예외 |
|-----------|------------------|-----------|-----------|-----------|
| MUST | 반드시 ~해야 한다 | 조건부 | 반드시 ~해야 한다(MUST) | 원문 인용, 계약 키워드와 코드 블록에서는 `MUST` 유지 |
| MUST NOT | 절대로 ~해서는 안 된다 | 조건부 | 절대로 ~해서는 안 된다(MUST NOT) | 금지 범위를 완화하는 `권장하지 않는다`로 번역 금지 |
| SHOULD | ~하는 것이 좋다 | 조건부 | ~하는 것이 좋다(SHOULD) | 필수 의무인 `해야 한다`로 강화하지 않음 |
| MAY | ~할 수 있다 | 조건부 | ~할 수 있다(MAY) | 허용과 가능성을 의무로 바꾸지 않음 |
| required | 필수 | 아니요 | 필수(required) | 문서 구조의 필수 섹션과 필수 동작에 사용 |
| contract | 계약 | 아니요 | 계약(contract) | 타입명·파일명·코드 식별자는 원형 유지 |
| requirement | 요구사항 | 아니요 | 요구사항(requirement) | `FR-*`, `SER-*` ID는 원형 유지 |
| acceptance criteria | 인수 조건 | 아니요 | 인수 조건(acceptance criteria) | 시나리오의 전제·행동·결과와 구분해 사용 |
| success criteria | 성공 기준 | 아니요 | 성공 기준(success criteria) | `SC-*` ID는 원형 유지 |
| validation | 검증 | 아니요 | 검증(validation) | 입력 검증과 품질 검증을 문맥으로 구분 |
| evaluation | 평가 | 아니요 | 평가(evaluation) | `Evaluator Agent` 역할명은 원형 유지 |
| pass | 통과 | 아니요 | 통과(pass) | 상태 값 `pass`는 원형 유지 |
| fail | 실패 | 아니요 | 실패(fail) | 상태 값 `fail`, `failed`는 원형 유지 |
| conditional routing | 조건부 라우팅 | 아니요 | 조건부 라우팅(conditional routing) | LangGraph 경로 분기 의미 |
| retry | 재시도 | 아니요 | 재시도(retry) | 횟수와 종료 조건을 함께 보존 |
| rewrite | 재작성 | 아니요 | 재작성(rewrite) | `Rewrite Agent`는 원형 유지 |
| source of truth | 기준 정보원 | 아니요 | 기준 정보원(source of truth) | canonical 저장 소유권의 유일한 기준을 뜻함 |
| degraded report | 성능 저하 보고서 | 아니요 | 성능 저하 보고서(degraded report) | 누락 데이터를 공개하는 보고서 상태 |

### 의무 강도 보존 규칙

- `MUST`와 `required`는 선택 또는 권고 표현으로 낮추지 않는다.
- `MUST NOT`의 금지 범위와 부정문을 삭제하거나 완화하지 않는다.
- `SHOULD`는 필수 의무로 강화하지 않고 권고 수준을 유지한다.
- `MAY`는 허용 또는 가능성을 나타내며 자동 실행 의무로 바꾸지 않는다.
- 재시도 횟수, 최대 재작성 횟수, 성공·실패 종료 조건은 용어 번역과 별개로 숫자를 정확히 보존한다.

## 백테스트와 최적화

| 원문 용어 | 기본 한국어 표현 | 원형 유지 | 최초 사용 | 문맥·예외 |
|-----------|------------------|-----------|-----------|-----------|
| backtest | 백테스트 | 아니요 | 백테스트(backtest) | 코드 심볼에 포함된 `backtest`는 유지 |
| walk-forward optimization | 워크포워드 최적화 | 아니요 | 워크포워드 최적화(walk-forward optimization) | 기능명에서는 `Walk-Forward` 원문 병기 가능 |
| in-sample | 표본 내 | 아니요 | 표본 내(in-sample) | `IS` 약어를 쓰면 최초 병기 |
| out-of-sample | 표본 외 | 아니요 | 표본 외(out-of-sample) | `OOS` 약어를 쓰면 최초 병기 |
| holdout | 홀드아웃 | 아니요 | 홀드아웃(holdout) | 검증 구간 의미일 때 사용 |
| robustness | 견고성 | 아니요 | 견고성(robustness) | 통계적 강건성과 구별이 필요하면 문맥 설명 |
| overfitting | 과적합 | 아니요 | 과적합(overfitting) | 최적화 위험 설명에 사용 |
| risk-adjusted | 위험 조정 | 아니요 | 위험 조정(risk-adjusted) | 수익률·성과와 결합해 사용 |
| maximum drawdown | 최대 낙폭 | 아니요 | 최대 낙폭(maximum drawdown) | 약어 `MDD`는 원형 유지 |
| Sharpe ratio | 샤프 비율 | 아니요 | 샤프 비율(Sharpe ratio) | 고유 지표명 `Sharpe`는 원형 병기 가능 |
| Sortino ratio | 소르티노 비율 | 아니요 | 소르티노 비율(Sortino ratio) | 고유 지표명 `Sortino`는 원형 병기 가능 |
| parameter set | 매개변수 집합 | 아니요 | 매개변수 집합(parameter set) | 코드 필드명은 유지 |
| regime | 시장 국면 | 아니요 | 시장 국면(regime) | 상태 열거형 값은 원형 유지 |
| stress test | 스트레스 테스트 | 아니요 | 스트레스 테스트(stress test) | 파일명·함수명은 유지 |

## 저장소와 데이터 모델

| 원문 용어 | 기본 한국어 표현 | 원형 유지 | 최초 사용 | 문맥·예외 |
|-----------|------------------|-----------|-----------|-----------|
| persistence | 영속성 | 아니요 | 영속성(persistence) | 저장 동작을 일반적으로 설명할 때 사용 |
| canonical | 기준 | 아니요 | 기준(canonical) | 객체명·테이블명에 포함된 `canonical`은 유지 가능 |
| schema | 스키마 | 아니요 | 스키마(schema) | SQL 객체명은 원형 유지 |
| migration | 마이그레이션 | 아니요 | 마이그레이션(migration) | revision ID는 원형 유지 |
| repository | 리포지터리 | 아니요 | 리포지터리(repository) | Git 저장소 의미와 데이터 접근 계층 의미를 문맥으로 구분 |
| lineage | 데이터 계보 | 아니요 | 데이터 계보(lineage) | raw→normalized→derived 추적 관계 |
| retention | 보존 기간 | 아니요 | 보존 기간(retention) | 정책 이름과 설정 키는 원형 유지 |
| constraint | 제약조건 | 아니요 | 제약조건(constraint) | 데이터베이스 제약 이름은 원형 유지 |
| index | 인덱스 | 아니요 | 인덱스(index) | 인덱스 이름은 원형 유지 |
| derived data | 파생 데이터 | 아니요 | 파생 데이터(derived data) | canonical 데이터와 구분 |

## Provider와 정규화

| 원문 용어 | 기본 한국어 표현 | 원형 유지 | 최초 사용 | 문맥·예외 |
|-----------|------------------|-----------|-----------|-----------|
| provider | 공급자 | 조건부 | 공급자(provider) | 인터페이스명·클래스명과 `provider_id` 등 필드명은 원형 유지 |
| provider-agnostic | 공급자 독립적 | 아니요 | 공급자 독립적(provider-agnostic) | 기능명에서는 원문 병기 가능 |
| normalization | 정규화 | 아니요 | 정규화(normalization) | 객체명·함수명은 원형 유지 |
| normalized record | 정규화 레코드 | 아니요 | 정규화 레코드(normalized record) | 계약 타입명은 원형 유지 |
| raw payload | 원시 페이로드 | 아니요 | 원시 페이로드(raw payload) | `raw_payload` 필드명은 유지 |
| raw provider response | 원시 공급자 응답 | 아니요 | 원시 공급자 응답(raw provider response) | `RawProviderResponse`는 유지 |
| adapter | 어댑터 | 아니요 | 어댑터(adapter) | 클래스명은 원형 유지 |
| fallback | 대체 처리 | 아니요 | 대체 처리(fallback) | mock provider 선택과 일반 복구를 모두 포함 |
| degraded | 성능 저하 | 아니요 | 성능 저하 상태(degraded state) | 상태 열거형 값 `degraded`는 유지 |
| partial success | 부분 성공 | 아니요 | 부분 성공(partial success) | 상태 값 `partial_success`는 유지 |
| insufficient data | 데이터 부족 | 아니요 | 데이터 부족(insufficient data) | 상태 값 `insufficient_data`는 유지 |
| mock provider | 모의 공급자 | 아니요 | 모의 공급자(mock provider) | 클래스명은 원형 유지 |
| MCP | `MCP` | 예 | 모델 컨텍스트 프로토콜(MCP) | 약어와 계약명은 원형 유지 |

## GraphRAG와 검색

| 원문 용어 | 기본 한국어 표현 | 원형 유지 | 최초 사용 | 문맥·예외 |
|-----------|------------------|-----------|-----------|-----------|
| GraphRAG | `GraphRAG` | 예 | 그래프 기반 검색 증강 생성(GraphRAG) | 기능명과 기술명은 원형 유지 |
| graph context | 그래프 컨텍스트 | 아니요 | 그래프 컨텍스트(graph context) | `GraphContextBuilder`는 유지 |
| node | 노드 | 아니요 | 노드(node) | 타입명·라벨명은 원형 유지 |
| relationship | 관계 | 아니요 | 관계(relationship) | 관계 타입명은 원형 유지 |
| graph projection | 그래프 투영 | 아니요 | 그래프 투영(graph projection) | 투영 규칙 이름은 원형 유지 |
| retrieval | 검색 | 아니요 | 검색(retrieval) | 함수명·계약명은 원형 유지 |
| vector retrieval | 벡터 검색 | 아니요 | 벡터 검색(vector retrieval) | 저장소 제품명은 원형 유지 |
| embedding | 임베딩 | 아니요 | 임베딩(embedding) | 모델명·필드명은 원형 유지 |
| evidence path | 근거 경로 | 아니요 | 근거 경로(evidence path) | 그래프 관계 기반 근거를 설명 |
| ScenarioReportInput | `ScenarioReportInput` | 예 | `ScenarioReportInput` | 계약 타입명은 항상 원형 유지 |

## 역할명과 고유 기술명

다음 이름은 원형을 유지한다.

- `LangGraph`, `FastAPI`, `Streamlit`, `PostgreSQL`, `Neo4j`, `yfinance`, `Tavily`, `Firecrawl`, `OpenAI`
- `Market Agent`, `Fundamental Agent`, `News Agent`, `Coordinator Agent`, `Evaluator Agent`, `Rewrite Agent`
- `CompanyProfile`, `FinancialMetric`, `NewsEvent`, `MarketData`, `TechnicalAnalysisResult`, `WaveAnalysisResult`, `RiskAssessment`
- `MA20`, `MA60`, `MA120`, `RSI`, `MACD`, `ATR`, `MDD`, `OOS`

역할명을 설명 문장에 사용할 때는 최초 한 번 `시장 에이전트(Market Agent)`처럼 병기할 수 있지만 코드 역할명과 다이어그램 라벨은 원형을 유지한다.

## 예외 기록

새 예외는 원문 용어, 적용 문서, 유지 이유와 승인 배치를 함께 기록한다.

현재 예외: 없음.
