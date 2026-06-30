# 기능 명세: PostgreSQL 기준 정보원 테이블 스키마

**기능 브랜치**: `[004-postgresql-table-schema]`

**작성일**: 2026-06-18

**상태**: 초안

**입력**: 사용자 설명: "PostgreSQL은 사용자, ticker, 분석 요청, 보고서, 분석 결과, 설정, 알림과 포트폴리오의 실제 기준 정보원이다. Pinecone은 뉴스, 재무제표 설명과 파동 이론 자료를 위한 의미 기반 청크 인덱스다. Neo4j는 파동 규칙, 시나리오, 무효화 조건과 근거 경로를 위한 관계 인덱스/지식 그래프다. OpenSearch는 뉴스 원문, 보고서, 로그와 키워드 검색을 위한 키워드/전문 인덱스다. Redis는 임시 캐시, 작업 큐, 요청 속도 제한과 세션 저장소다. PostgreSQL 테이블 명세로 데이터 모델링을 시작한다."

## 명확화

### 세션 2026-06-18

- 질문: 첫 MVP 구현에 포함할 PostgreSQL 테이블 범위는 무엇인가? → 답변: 전체 플랫폼 MVP로 사용자, 설정, 알림과 포트폴리오를 첫 PostgreSQL 구현에 포함한다.
- 질문: MVP 사용자 모델이 지원할 인증 범위는 무엇인가? → 답변: 로컬/데모 사용자를 우선한다. 사용자는 선택적 소유자이고 이메일은 null을 허용하며 로그인, 비밀번호, SSO와 영구 세션 자격 증명 저장은 보류한다.
- 질문: 사용자 소유 데이터에 적용할 삭제 및 보존 정책은 무엇인가? → 답변: 사용자 데이터를 익명화한다. PII와 사용자 소유 UX 레코드는 삭제하거나 익명화하되, 보고서·근거·감사 레코드는 익명 소유자 아래에 보존한다.
- 질문: MVP가 지원할 알림 전달 범위는 무엇인가? → 답변: 앱 내부만 지원한다. MVP 알림은 앱 내부 레코드로 저장하고 이메일, webhook, push 전달과 채널 재시도 처리는 보류한다.
- 질문: 그래프 지식과 근거 경로 테이블을 첫 PostgreSQL MVP에 포함해야 하는가? → 답변: 그래프 테이블을 지금 포함한다. 파동 규칙, 파동 시나리오, 무효화 조건과 근거 경로 테이블은 첫 MVP 구현에 포함한다.

## 사용자 시나리오 및 테스트 *(필수)*

### 사용자 스토리 1 - 기준 리서치 레코드 정의 (우선순위: P1)

제품 책임자와 개발자는 모든 분석 요청, 에이전트 결과, 보고서, 근거 항목과 안전 평가를 영구 저장하고 나중에 감사할 수 있도록 리서치 워크플로 레코드의 완전한 기준 정보원 테이블 목록이 필요하다.

**이 우선순위인 이유**: 기존 MVP는 데이터를 수집하고 보고서를 생성할 수 있지만, 파생 인덱스, 사용자 이력, 알림과 포트폴리오 보기를 안전하게 구현하려면 먼저 영구적인 기준 레코드가 필요하다.

**독립 테스트**: 완전한 리서치 워크플로 실행 하나와 테이블 목록을 대조하여 요청, ticker, 에이전트 출력, 근거, Evaluator 결과, 최종 보고서, 경고와 실행 수명주기 이벤트에 각각 기준 테이블이 지정되어 있는지 검증한다.

**인수 시나리오**:

1. **전제** 리서치 워크플로가 완료되었을 때, **행동** 검토자가 각 워크플로 산출물을 스키마에 매핑하면, **결과** 모든 기준 산출물이 정확히 하나의 주 소유 테이블에 지정된다.
2. **전제** 시장, 기업 또는 뉴스 데이터가 누락된 성능 저하 워크플로가 있을 때, **행동** 검토자가 스키마를 확인하면, **결과** 누락된 사실을 만들어내지 않고 경고, 실패 이유, 재시도와 성능 저하 상태를 저장할 수 있다.

---

### 사용자 스토리 2 - 파생 검색 및 그래프 인덱스 재구축 (우선순위: P2)

개발자는 Pinecone, Neo4j와 OpenSearch를 원본 기준 정보원으로 취급하지 않고 해당 투영을 재구축할 수 있는 기준 레코드가 필요하다.

**이 우선순위인 이유**: 플랫폼 아키텍처는 명확한 역할 분리에 의존한다. PostgreSQL은 원장이며, 의미·그래프·키워드 저장소는 재구축 가능한 투영이고 런타임 저장소는 임시 상태다.

**독립 테스트**: 뉴스 항목, 보고서, 근거 항목과 파동 이론 시나리오를 선택하고, 각 파생 인덱스 레코드가 기준 테이블 레코드와 안정적인 식별자로 역추적되는지 검증한다.

**인수 시나리오**:

1. **전제** 의미 인덱스가 삭제되었거나 오래되었을 때, **행동** 재인덱싱 작업이 기준 출처 문서와 청크를 읽으면, **결과** 안정적인 출처 참조로 의미 인덱스를 다시 채울 수 있다.
2. **전제** 그래프 근거 경로가 사용자에게 표시될 때, **행동** 감사자가 해당 경로를 추적하면, **결과** 인용된 모든 규칙, 시나리오, 조건, 근거 항목, ticker, 요청과 보고서가 기준 레코드로 확인된다.

---

### 사용자 스토리 3 - 사용자 이력, 설정, 앱 내부 알림과 포트폴리오 지원 (우선순위: P3)

데모 사용자는 거래 또는 주문 실행을 추가하지 않고 더 풍부한 리서치 경험을 얻기 위해 영구적인 이력, 선호 설정, 알림과 포트폴리오 관찰 컨텍스트가 필요하다.

**이 우선순위인 이유**: 사용자 대상 영속성은 첫 플랫폼 MVP에 포함되지만 리서치 전용으로 유지되어야 하며 중개 서비스, 실행 또는 투자 권유 동작을 암시해서는 안 된다.

**독립 테스트**: 사용자 프로필, 설정 레코드, 알림, 포트폴리오와 관심 목록 형태의 보유 참조를 생성한 뒤, 주문 실행, 중개 서비스 자격 증명, 보장된 목표 또는 매수·매도 지시를 요구하는 테이블이 없는지 검증한다.

**인수 시나리오**:

1. **전제** 사용자가 선호 설정을 변경할 때, **행동** 선호 설정을 저장하면, **결과** 해당 설정을 사용자에게 추적할 수 있고 과거 보고서를 변경하지 않고 갱신할 수 있다.
2. **전제** 사용자가 포트폴리오 컨텍스트를 유지할 때, **행동** ticker를 포트폴리오에 연결하면, **결과** 해당 연결은 주문 실행이 아니라 리서치와 모니터링만 지원한다.

### 경계 사례

- 서로 다른 시장의 중복 ticker 심볼은 시장/거래소 메타데이터로 구분할 수 있어야 한다.
- 로컬/데모 실행에는 인증된 사용자가 없을 수 있으며, 적절한 경우 사용자 참조가 시스템 또는 익명 소유자를 허용해야 한다.
- MVP 사용자 레코드는 로그인 자격 증명 없는 로컬/데모 소유자일 수 있으며 인증, 비밀번호, SSO와 영구 자격 증명/세션 저장은 계속 보류해야 한다.
- MVP 알림은 외부 전달 주소 없이 존재할 수 있으며 이메일, webhook, push 전달과 채널 재시도 실패는 첫 구현 범위에서 제외한다.
- 공급자가 부분 데이터, 누락된 URL, 지연된 공시 또는 뉴스 없음 상태를 반환할 수 있으며 기준 레코드는 누락 데이터 메모와 출처 상태를 보존해야 한다.
- 생성된 보고서가 Evaluator 검토에 실패할 수 있으며 실패한 초안, 재작성 시도, 최종 실패 상태와 안전 이유를 감사할 수 있어야 한다.
- 출처 문서를 다시 수집하거나 수정할 수 있으며 스키마는 공급자 식별자, 출처 URL, 수집 시각과 버전 관계를 보존해야 한다.
- 기준 데이터를 저장한 뒤 파생 인덱스 쓰기에 실패할 수 있으며, 투영 상태는 출처 레코드를 잃지 않고 대기·성공·실패를 기록해야 한다.
- Redis 런타임 상태가 만료될 수 있으며 영구 실행 상태, 보고서 경로와 최종 결과를 기준 레코드에서 복구할 수 있어야 한다.
- 사용자 삭제가 리서치 감사 가능성을 훼손해서는 안 된다. PII와 사용자 소유 UX 레코드는 삭제하거나 익명화하고, 보고서·근거·출처 문서·워크플로 노드 실행·감사 레코드는 익명 소유자 아래에 유지한다.
- 그래프 투영 실패로 PostgreSQL의 기준 파동 규칙, 시나리오, 무효화 조건 또는 근거 경로 레코드를 제거해서는 안 된다.

## 요구사항 *(필수)*

### 기능 요구사항

- **FR-001**: 시스템은 영구 비즈니스 레코드의 기준 소유자로 PostgreSQL을 반드시 정의해야 한다. 해당 레코드는 사용자, ticker, 분석 요청, 분석 결과, 보고서, 근거, 설정, 알림, 포트폴리오, 출처 문서, 워크플로 이벤트와 인덱스 투영 상태다.
- **FR-002**: 시스템은 모든 영구 리서치 워크플로 산출물을 단일 기준 테이블에 반드시 지정하고, 테이블의 목적, 기본 키, 중요 열, 관계와 보존 역할을 문서화해야 한다.
- **FR-003**: 시스템은 파생 시스템이 기준 레코드를 참조할 수 있도록 요청, 보고서, 근거 항목, 출처 문서, 문서 청크, 그래프 규칙, 그래프 시나리오와 인덱스 투영 레코드의 안정적인 식별자를 반드시 보존해야 한다.
- **FR-004**: 시스템은 로그인 자격 증명을 요구하지 않고 인증되지 않은 로컬/데모 실행과 로컬/데모 사용자 소유권을 반드시 지원해야 한다. `users.email`은 없을 수 있으며 인증, 비밀번호, SSO와 영구 세션 자격 증명 저장은 보류한다.
- **FR-005**: 시스템은 심볼, 시장/거래소 컨텍스트, 표시 이름, 공급자 메타데이터, 활성 상태와 타임스탬프로 ticker 식별자를 반드시 표현해야 한다.
- **FR-006**: 시스템은 분석 요청 입력, 요청 유형, 수명주기 상태, 타임스탬프, 재시도/성능 저하 상태, 요청 매개변수와 사용자 대상 오류 또는 경고 요약을 반드시 기록해야 한다.
- **FR-007**: 시스템은 노드 이름, 시작/종료 시각, 상태, 재시도 횟수, 성능 저하 이유, 오류 메시지와 가능한 경우 평가 점수를 포함한 에이전트 및 워크플로 노드 실행 상세를 반드시 기록해야 한다.
- **FR-008**: 시스템은 시장, 기업, 뉴스, 백테스트, 최적화, 그래프 컨텍스트, 평가와 재작성 결과를 독립적으로 확인할 수 있도록 에이전트 분석 결과를 생성된 보고서와 분리하여 반드시 기록해야 한다.
- **FR-009**: 시스템은 언어, 제목, 구조화 콘텐츠, 렌더링 텍스트, 안전 상태, 면책문 존재 여부, 평가 점수와 보고서 버전을 포함한 최종 및 중간 보고서 레코드를 반드시 저장해야 한다.
- **FR-010**: 시스템은 evidence ID, source type, source name, source URL, collection time, ticker, metric name, metric value, description과 해당되는 경우 요청/결과/보고서 관계를 가진 EvidenceItem 호환 레코드를 반드시 저장해야 한다.
- **FR-011**: 시스템은 의미 및 키워드 인덱스를 재구축할 수 있도록 뉴스, 재무제표 설명, 보고서 텍스트, 로그와 파동 이론 자료의 출처 문서 메타데이터와 청크 출처 추적 정보를 반드시 저장해야 한다.
- **FR-012**: 시스템은 그래프 추론을 사용할 때 기준 파동 이론 규칙, 시나리오, 무효화 조건과 근거 경로 레코드를 반드시 저장하되, 그래프 데이터베이스는 파생 투영으로 유지해야 한다.
- **FR-013**: 시스템은 출처 레코드, 투영 유형, 상태, 마지막 시도 시각, 오류 메시지와 멱등성 키를 포함한 Pinecone, Neo4j와 OpenSearch 대상의 인덱스 투영 상태를 반드시 저장해야 한다.
- **FR-014**: 시스템은 중개 서비스 연결, 주문 제출 또는 보장된 목표 실행을 나타내는 필드 없이 사용자 설정, 알림, 포트폴리오, 포트폴리오 항목과 관심 목록 컨텍스트를 반드시 저장해야 한다.
- **FR-015**: 시스템은 중복된 기준 evidence ID, 동일 요청의 중복 보고서 버전, 사용자별 중복 설정 키와 동일 심볼/시장 쌍의 중복 ticker 레코드를 방지하는 고유성 및 무결성 규칙을 반드시 정의해야 한다.
- **FR-016**: 삭제하면 보고서, 근거 또는 감사 이력이 훼손되는 사용자 대상 레코드에 대해 시스템은 소프트 삭제 또는 보관 상태를 반드시 지원해야 한다.
- **FR-017**: 시스템은 첫 PostgreSQL MVP 테이블 집합에 리서치 워크플로, 출처 문서와 투영 상태 테이블과 함께 사용자, 설정, 알림, 포트폴리오, 포트폴리오 항목, 파동 규칙, 파동 시나리오, 무효화 조건, 근거 경로와 근거 경로 단계를 반드시 포함해야 한다.
- **FR-018**: 이후 계획에서 명시적으로 승격하지 않는 한 시스템은 공급자 배치 동기화, 변경 불가능한 원시 공급자 페이로드 보존, 요청 연결 이력과 상세 알림 라우팅 선호 설정을 보류된 확장 테이블로 유지해야 한다.
- **FR-019**: 시스템은 PII와 사용자 소유 UX 레코드를 삭제하거나 익명화하면서 보고서, 근거, 출처 문서, 워크플로 노드 실행과 감사 레코드를 익명 소유자 아래에 보존하는 방식으로 사용자 삭제를 반드시 지원해야 한다.
- **FR-020**: 시스템은 MVP 알림을 앱 내부 알림 레코드로 반드시 제한하고 이메일, webhook, push 전달, 채널 선호 설정과 전달 재시도 추적을 보류해야 한다.

### 안전성, 근거 및 신뢰성 요구사항 *(리서치 워크플로 변경 시 필수)*

- **SER-001**: 시스템은 직접적인 매수·매도·보유 권유, 거래 지시, 수익 보장 주장, 목표가 보장 주장과 주문 실행 동작을 반드시 피해야 한다.
- **SER-002**: 기능이 리서치 출력에 영향을 주는 경우 시스템은 중요한 수치 또는 사실에 관한 보고서 주장을 `EvidenceItem` 레코드로 반드시 뒷받침해야 한다.
- **SER-003**: 최종 한국어 보고서는 constitution에 정확히 정의된 교육 목적의 비권유 면책문을 반드시 포함해야 한다.
- **SER-004**: 시스템은 누락되거나 성능이 저하된 시장·기업·뉴스 데이터를 조작하여 만들어내지 않고 반드시 공개해야 한다.
- **SER-005**: 워크플로에 영향을 주는 기능은 검증 실패, 공급자 실패, 평가 실패와 재작성 제한에 대해 결정적인 동작을 반드시 정의해야 한다.
- **SER-006**: 런타임에 영향을 주는 기능은 constitution이 요구하는 구조화 로그, 보고서 저장, 상태 확인과 지표를 반드시 보존해야 한다.
- **SER-007**: Redis는 실행 상태, 저장된 리서치 상태에 영향을 주는 세션, 생성된 보고서, 알림 또는 사용자에게 보이는 분석 결과의 유일한 영구 사본이어서는 안 된다.
- **SER-008**: Pinecone, Neo4j와 OpenSearch 투영 실패는 기준 데이터 누락이 아니라 투영 경고 또는 실패한 투영 상태로 반드시 기록해야 한다.

### 핵심 개체 *(기능에 데이터가 포함되는 경우)*

#### MVP 핵심 테이블

- **users**: 로컬/데모와 향후 인증 사용을 위한 선택적 소유자 프로필. 주요 열: `id`, `email`, `display_name`, `role`, `status`, `anonymized_at`, `created_at`, `updated_at`, `deleted_at`. 관계: 분석 요청, 설정, 알림, 포트폴리오를 소유한다. 규칙: email은 값이 있을 때 고유하다. 로컬/데모 시스템 실행에는 사용자가 없을 수 있다. 로그인 자격 증명, 비밀번호 해시, SSO 식별자와 영구 세션 자격 증명은 MVP 필드가 아니다. 삭제된 사용자는 감사 레코드를 유지하면서 익명화할 수 있다.
- **tickers**: 리서치를 위한 기준 증권 식별자. 주요 열: `id`, `symbol`, `market`, `exchange`, `currency`, `name`, `sector`, `industry`, `provider_metadata`, `is_active`, `created_at`, `updated_at`. 관계: 요청, 근거, 출처 문서, 보고서와 포트폴리오가 참조한다. 규칙: `(symbol, market)`은 고유하다.
- **analysis_requests**: 사용자의 단일 리서치/백테스트/최적화 요청. 주요 열: `id`, `user_id`, `ticker_id`, `request_type`, `horizon`, `risk_profile`, `parameters`, `status`, `degraded_reason`, `warning_summary`, `error_summary`, `created_at`, `started_at`, `completed_at`. 관계: 워크플로 이벤트, 결과, 보고서와 근거를 소유한다. 규칙: 상태 값에는 pending, running, success, degraded, insufficient_data, failed, cancelled가 포함된다.
- **workflow_node_runs**: 관측 가능한 노드 수준 실행 레코드. 주요 열: `id`, `request_id`, `run_id`, `node_name`, `attempt_number`, `status`, `started_at`, `ended_at`, `duration_ms`, `error_type`, `error_message`, `evaluation_score`, `metadata`. 관계: 분석 요청에 속하며 분석 결과에 연결될 수 있다. 규칙: 각 노드 시도마다 하나의 레코드를 둔다.
- **analysis_results**: 워크플로 구성 요소의 구조화 결과. 주요 열: `id`, `request_id`, `ticker_id`, `result_type`, `summary`, `metrics`, `warnings`, `missing_data_notes`, `status`, `created_at`. 관계: 근거를 인용하고 보고서에 입력될 수 있다. 규칙: 결과 유형에는 market, fundamental, news, graph_context, backtest, optimization, coordinator_draft, evaluation, rewrite가 포함된다.
- **reports**: 생성된 한국어 리서치 보고서 레코드. 주요 열: `id`, `request_id`, `ticker_id`, `current_version_id`, `title`, `language`, `status`, `safety_status`, `evaluation_score`, `disclaimer_present`, `created_at`, `updated_at`. 관계: 보고서 버전과 보고서-근거 인용을 소유한다. 규칙: 상태 값에는 draft, final, failed_review, archived가 포함된다.
- **report_versions**: 변경 불가능한 보고서 콘텐츠 스냅샷. 주요 열: `id`, `report_id`, `version_number`, `stage`, `report_json`, `report_markdown`, `created_by_node`, `created_at`. 관계: Evaluator와 검색 투영이 인용한다. 규칙: `(report_id, version_number)`는 고유하다.
- **evidence_items**: EvidenceItem 호환 기준 근거. 주요 열: `id`, `evidence_id`, `request_id`, `ticker_id`, `analysis_result_id`, `source_document_id`, `source_type`, `source_name`, `source_url`, `collected_at`, `metric_name`, `metric_value`, `description`, `created_at`. 관계: 보고서와 근거 경로가 인용한다. 규칙: `evidence_id`는 고유하며 보고서에서 사용할 수 있다.
- **report_evidence_citations**: 보고서 버전과 근거 항목 사이의 연결 테이블. 주요 열: `id`, `report_version_id`, `evidence_item_id`, `section_name`, `claim_text`, `created_at`. 관계: Evaluator의 근거성 검사를 지원한다. 규칙: 동일한 보고서 섹션과 근거 항목의 중복 인용을 방지한다.

#### 출처 문서 및 투영 테이블

- **source_documents**: 기준 공급자 또는 생성 문서 메타데이터. 주요 열: `id`, `ticker_id`, `document_type`, `source_name`, `source_url`, `title`, `language`, `published_at`, `collected_at`, `raw_content_ref`, `content_hash`, `revision_group_id`, `supersedes_document_id`, `metadata`, `status`. 관계: 청크를 소유하고 근거를 생성할 수 있으며 같은 개정 그룹의 이전 문서 버전을 대체할 수 있다. 규칙: URL이 있으면 출처 URL과 콘텐츠 해시로 중복 수집을 식별한다. 다시 수집하거나 수정한 문서는 `revision_group_id`와 `supersedes_document_id`를 통해 버전 계보를 보존한다.
- **document_chunks**: 의미 및 키워드 인덱싱을 위한 기준 청크 출처 추적 정보. 주요 열: `id`, `source_document_id`, `chunk_index`, `chunk_text`, `chunk_hash`, `token_count`, `metadata`, `created_at`. 관계: 의미 및 키워드 인덱스로 투영된다. 규칙: `(source_document_id, chunk_index)`는 고유하다.
- **index_projection_status**: 파생 시스템을 위한 투영 원장. 주요 열: `id`, `source_table`, `source_id`, `target_system`, `projection_type`, `projection_key`, `status`, `attempt_count`, `last_attempt_at`, `last_success_at`, `error_message`, `idempotency_key`. 관계: 테이블과 ID를 사용해 출처 레코드를 가리킨다. 규칙: `(target_system, projection_type, idempotency_key)`는 고유하다.
- **keyword_terms**: 검색 지원을 위한 선택적 정규화 키워드 목록. 주요 열: `id`, `term`, `normalized_term`, `language`, `created_at`. 관계: 출처 문서와 보고서에 연결될 수 있다. 규칙: 정규화 용어와 언어 조합은 고유하다.
- **structured_log_events**: 로그를 파일보다 오래 기준 보존해야 할 때 사용하는 영구 운영 이벤트 요약. 주요 열: `id`, `request_id`, `run_id`, `ticker_id`, `node_name`, `event_name`, `status`, `message`, `error_message`, `evaluation_score`, `occurred_at`, `metadata`. 관계: 전문 로그 검색으로 투영된다. 규칙: 대량의 원시 추적이 아니라 감사 요약에 사용한다.

#### MVP 그래프 지식 테이블

- **wave_rules**: 기준 파동 이론 규칙 레코드. 주요 열: `id`, `rule_code`, `name`, `description`, `rule_type`, `status`, `source_document_id`, `created_at`, `updated_at`. 관계: 시나리오와 무효화 조건에서 사용한다. 규칙: `rule_code`는 고유하다.
- **wave_scenarios**: 기준 시나리오 레코드. 주요 열: `id`, `ticker_id`, `name`, `description`, `timeframe`, `status`, `confidence_label`, `created_at`, `updated_at`. 관계: 규칙을 사용하고 무효화 조건이 있으며 근거를 인용한다. 규칙: 시나리오 텍스트는 거래 조언이 아닌 리서치 표현이어야 한다.
- **wave_invalidation_conditions**: 시나리오 무효화 조건. 주요 열: `id`, `scenario_id`, `condition_text`, `metric_name`, `threshold_value`, `direction`, `source_document_id`, `created_at`. 관계: 시나리오에 속하며 근거를 인용할 수 있다. 규칙: 임계값은 보장된 목표가 아니라 시나리오 검토 조건이다.
- **wave_scenario_rules**: 시나리오와 규칙을 연결하는 연결 테이블. 주요 열: `id`, `scenario_id`, `rule_id`, `role`, `created_at`. 관계: 그래프 인덱스로 투영된다. 규칙: 중복된 시나리오-규칙-역할 행을 방지한다.
- **evidence_paths**: 기준 설명 가능성 경로. 주요 열: `id`, `request_id`, `ticker_id`, `path_type`, `path_summary`, `source_node_ref`, `target_node_ref`, `confidence_label`, `created_at`. 관계: 경로 단계를 소유하며 근거를 인용할 수 있다. 규칙: 경로 요약은 기준 근거/규칙/시나리오 레코드로 확인되어야 한다.
- **evidence_path_steps**: 순서가 있는 경로 단계. 주요 열: `id`, `evidence_path_id`, `step_index`, `node_table`, `node_id`, `relationship_type`, `description`, `created_at`. 관계: 그래프 인덱스로 투영된다. 규칙: `(evidence_path_id, step_index)`는 고유하다.

#### MVP 사용자 경험 테이블

- **user_settings**: 사용자 또는 시스템 선호 설정. 주요 열: `id`, `user_id`, `setting_key`, `setting_value`, `scope`, `updated_at`, `deleted_at`. 관계: 선택적으로 사용자에게 속한다. 규칙: `(user_id, setting_key, scope)`는 고유하며 시스템 기본값에는 사용자가 없을 수 있다. 사용자 소유 설정은 사용자 삭제 시 삭제하거나 익명화한다.
- **notifications**: 사용자 대상 앱 내부 메시지 또는 알림 상태. 주요 열: `id`, `user_id`, `ticker_id`, `notification_type`, `title`, `body`, `payload`, `status`, `created_at`, `read_at`, `updated_at`, `deleted_at`. 관계: 페이로드에서 요청/보고서/ticker를 참조할 수 있다. 규칙: 알림은 정보 제공용이며 거래를 지시해서는 안 된다. MVP 알림은 외부 전달 채널 상태를 저장하지 않는다. 사용자 소유 알림은 사용자 삭제 시 삭제하거나 익명화한다.
- **portfolios**: 리서치 전용 포트폴리오 컨텍스트. 주요 열: `id`, `user_id`, `name`, `description`, `base_currency`, `status`, `created_at`, `updated_at`, `deleted_at`. 관계: 포트폴리오 항목을 소유한다. 규칙: 포트폴리오 레코드는 중개 서비스 실행이 아니라 컨텍스트와 모니터링에 사용한다. 사용자 소유 포트폴리오 레코드는 사용자 삭제 시 삭제하거나 익명화한다.
- **portfolio_items**: 포트폴리오 안의 ticker 연결. 주요 열: `id`, `portfolio_id`, `ticker_id`, `label`, `quantity_note`, `cost_basis_note`, `metadata`, `created_at`, `updated_at`, `deleted_at`. 관계: 포트폴리오와 ticker에 속한다. 규칙: 수치는 선택적 리서치 메모이며 주문을 실행해서는 안 된다. 사용자 소유 포트폴리오 항목은 사용자 삭제 시 삭제하거나 익명화한다.

#### 보류된 확장 테이블

- **provider_sync_runs**: yfinance, 뉴스 검색, 공시와 문서 수집을 위한 공급자 수집 배치 메타데이터. 주요 열: `id`, `provider_name`, `sync_type`, `status`, `started_at`, `completed_at`, `error_message`, `metadata`.
- **provider_payloads**: 변경 불가능한 원시 또는 정규화 공급자 페이로드 참조. 주요 열: `id`, `sync_run_id`, `ticker_id`, `source_name`, `source_url`, `payload_ref`, `payload_hash`, `collected_at`, `metadata`.
- **analysis_request_links**: 재작성, 재실행, 비교 또는 최적화 하위 실행 등 요청 간 관계. 주요 열: `id`, `from_request_id`, `to_request_id`, `link_type`, `created_at`.
- **notification_preferences**: 이메일, webhook, push 또는 기타 외부 채널을 위한 향후 알림 라우팅 선호 설정. 주요 열: `id`, `user_id`, `channel`, `is_enabled`, `rules`, `updated_at`.
- **notification_deliveries**: 향후 외부 알림 전달 시도. 주요 열: `id`, `notification_id`, `channel`, `delivery_target_ref`, `status`, `attempt_count`, `last_attempt_at`, `error_message`.
- **auth_identities**: 비밀번호 없는 로그인, OAuth 또는 SSO 공급자를 위한 향후 로그인 식별자 매핑. 주요 열: `id`, `user_id`, `provider`, `provider_subject`, `created_at`, `updated_at`.
- **user_sessions**: 실제 로그인이 있을 때 사용할 향후 영구 인증 세션 레코드. 주요 열: `id`, `user_id`, `session_ref`, `created_at`, `expires_at`, `revoked_at`.

## 성공 기준 *(필수)*

### 측정 가능한 결과

- **SC-001**: 검토자는 계획되지 않은 테이블을 추가하지 않고 MVP 리서치 워크플로 산출물의 100%를 기준 테이블에 매핑할 수 있다.
- **SC-002**: 사용자, 설정, 포트폴리오 컨텍스트, 요청, 노드 이벤트, 시장 결과, 기업 결과, 뉴스 결과, 근거, 보고서 초안, Evaluator 결과, 최종 보고서, 알림과 투영 상태 레코드로 완전한 표본 플랫폼 흐름을 표현할 수 있다.
- **SC-003**: 표본 보고서의 보고서 근거 인용 100%가 출처, 수집 시각, ticker, 지표 이름, 지표 값과 설명을 가진 기준 근거 레코드로 확인된다.
- **SC-004**: 표본 인덱싱 실행의 의미·그래프·키워드 투영 레코드 100%가 기준 출처 레코드로 역추적된다.
- **SC-005**: 스키마 검토에서 중개 서비스 연결, 주문 실행, 수익 보장, 목표가 보장 또는 직접적인 매수·매도·보유 지시 동작을 가능하게 하는 필드가 하나도 없음을 확인한다.
- **SC-006**: 성능이 저하된 공급자 시나리오를 경고와 누락 데이터 메모와 함께 저장하면서 사용자 대상 최종 보고서와 감사 추적을 보존할 수 있다.
- **SC-007**: MVP 테이블 집합은 안정적인 ID로 향후 투영을 계속 지원하면서 보류된 공급자 동기화와 선호 설정 라우팅 테이블과 독립적으로 구현할 수 있다.
- **SC-008**: 사용자 삭제 시나리오는 사용자 식별 데이터를 제거하거나 익명화하면서 과거 리서치 출력을 검토하는 데 필요한 보고서, 근거, 출처 문서, 워크플로 노드와 감사 참조의 100%를 보존한다.
- **SC-009**: 알림 시나리오는 외부 전달 채널 없이 앱 내부 알림을 생성, 목록 조회, 읽음 표시하고 삭제하거나 익명화할 수 있다.
- **SC-010**: 그래프 컨텍스트 표본은 그래프 인덱스 투영이 성공하기 전에 파동 규칙, 시나리오, 무효화 조건, 근거 경로와 순서가 있는 경로 단계를 기준 정보로 저장할 수 있다.

## 가정

- PostgreSQL은 영구 기준 정보원이며 Pinecone, Neo4j와 OpenSearch는 재구축 가능한 투영이고 Redis는 임시 런타임 지원이다.
- MVP 구현은 핵심 리서치 테이블, 출처 문서/청크 출처 추적, 투영 상태, 사용자, 설정, 알림, 포트폴리오, 포트폴리오 항목, 파동 규칙, 파동 시나리오, 무효화 조건, 근거 경로와 근거 경로 단계로 시작한다. 상세 공급자 동기화, 원시 페이로드 보존, 요청 연결 이력과 알림 선호 설정 테이블은 보류할 수 있다.
- 로컬/데모 모드는 실제 사용자 인증 없이 실행할 수 있으며, 첫 MVP 사용자 테이블은 향후 인증 식별자 테이블을 위한 여지를 두고 소유권/프로필 컨텍스트만 저장한다.
- 보고서는 리서치 전용으로 유지하며 최종 한국어 보고서는 constitution이 요구하는 면책문을 유지한다.
- 포트폴리오 레코드는 리서치 컨텍스트와 관심 목록 형태의 구성에만 사용한다. 실제 거래, 중개 서비스 연동과 주문 실행은 계속 범위에서 제외한다.
- MVP 알림은 앱 내부 레코드만 지원하며 외부 전달 채널과 재시도 큐는 향후 확장이다.
- 대용량 원시 콘텐츠의 전체 본문을 인라인으로 저장하기 어렵다면 영구 콘텐츠 참조를 사용할 수 있다.
- 사용자 삭제는 사용자 식별 필드와 사용자 소유 UX 레코드의 개인정보 보호를 우선하면서, 근거 기반 보고서 검토에 필요한 비PII 리서치 감사 레코드를 보존한다.
