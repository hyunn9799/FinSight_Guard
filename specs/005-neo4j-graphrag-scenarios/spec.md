# Feature Specification: Neo4j GraphRAG Scenarios

**Feature Branch**: `[005-neo4j-graphrag-scenarios]`

**Created**: 2026-06-21

**Status**: Draft

**Input**: User description: "내 프로젝트 정의는 이 프로젝트는 LangGraph + PostgreSQL + Neo4j GraphRAG 기반의 AI 금융 시나리오 리서치 Assistant다. 사용자가 종목을 입력하면, 시스템은 PostgreSQL에 저장된 원천/정식 데이터를 바탕으로 Neo4j에서 Company/Ticker 중심 subgraph를 탐색한다. 이 subgraph에는 뉴스 이벤트, 재무 지표, 기술분석 결과, NEoWave rule, WaveAnalysis, Risk, Evidence가 연결되어 있다. LLM은 이 graph context를 근거로 Bull/Base/Bear 시나리오 리포트를 생성하고, 생성된 시나리오가 어떤 뉴스·재무·기술·리스크 근거에 의해 지지됐는지 다시 Neo4j에 저장한다. 이 서비스는 매수/매도 추천이나 자동매매가 아니라, 투자 의사결정을 보조하는 근거 기반 리서치 시스템이다."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate Graph-Grounded Scenario Report (Priority: P1)

A research user enters a ticker and receives a Korean Bull/Base/Bear scenario report grounded in related company, news, financial, technical, wave, risk, and evidence relationships.

**Why this priority**: This is the core product identity. The feature is valuable only if the generated report is based on connected evidence rather than isolated summaries.

**Independent Test**: Can be tested with a deterministic ticker fixture containing company, news, metric, technical, wave, risk, and evidence records; the resulting report must include all three scenario types and cite the graph-derived evidence context.

**Acceptance Scenarios**:

1. **Given** a ticker with available graph context, **When** the user requests scenario research, **Then** the system returns Bull, Base, and Bear scenarios with supporting evidence, confirmation conditions, invalidation conditions, risks, limitations, data date, and the required disclaimer.
2. **Given** a scenario references a factual or numeric claim, **When** the report is inspected, **Then** the claim resolves to one or more evidence records from the retrieved context.

---

### User Story 2 - Trace Scenario Support Relationships (Priority: P2)

An auditor or developer can trace each generated scenario back to the news, financial, technical, wave, risk, and evidence relationships that supported it.

**Why this priority**: Scenario generation must be explainable and reviewable. Post-generation relationship persistence is what turns the report into an auditable GraphRAG workflow.

**Independent Test**: Can be tested by generating a report from deterministic fixture data and then verifying that every scenario has stored support or risk relationships to the evidence categories used in the report.

**Acceptance Scenarios**:

1. **Given** a finalized report, **When** a reviewer traces the Bull scenario, **Then** the reviewer can identify which news events, metrics, technical results, wave analyses, risks, and evidence records support or affect it.
2. **Given** a generated scenario with no valid supporting evidence, **When** persistence is attempted, **Then** the system marks the scenario as insufficiently grounded rather than storing unsupported support relationships.

---

### User Story 3 - Degrade Safely When Graph Context Is Missing (Priority: P3)

A user still receives a safe research response when graph retrieval is incomplete, stale, or unavailable, with clear disclosure of missing context.

**Why this priority**: The assistant must remain trustworthy during partial data or projection failures and must not fabricate graph relationships.

**Independent Test**: Can be tested by simulating missing graph paths and verifying that the response either falls back to canonical evidence or returns a degraded report with explicit warnings.

**Acceptance Scenarios**:

1. **Given** graph context is unavailable but canonical evidence exists, **When** the user requests a report, **Then** the system may produce a degraded report that discloses the graph-context limitation.
2. **Given** neither graph context nor sufficient canonical evidence exists, **When** the user requests a report, **Then** the system returns an insufficient-data response instead of fabricating scenarios.

### Edge Cases

- Ticker validation fails or maps to multiple companies or markets.
- The company graph exists but lacks one or more categories such as news, wave analysis, or risk.
- Graph context contains stale relationships that no longer match canonical evidence status.
- A NEoWave rule check is unknown or requires human review.
- Retrieved evidence conflicts across news, financial, technical, or wave sources.
- The generated report uses unsafe investment recommendation language.
- The evaluator fails the draft report twice after rewrite attempts.
- Relationship persistence succeeds for some scenario paths but fails for others.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a ticker-level research request and resolve it to a Company/Ticker context before scenario generation.
- **FR-002**: System MUST retrieve a bounded Company/Ticker-centered graph context containing relevant NewsEvent, FinancialMetric, TechnicalAnalysis, WaveAnalysis, NEoWaveRule, Risk, Evidence, and prior Report or Scenario relationships when available.
- **FR-003**: System MUST use canonical source identifiers when representing graph context so retrieved relationships can be traced back to durable research records.
- **FR-004**: System MUST generate Korean Bull, Base, and Bear scenario sections from the retrieved context and available canonical evidence.
- **FR-005**: Each generated scenario MUST include supporting evidence, news grounds, financial grounds, technical grounds, wave grounds when available, confirmation conditions, invalidation conditions, key risks, limitations, and data date.
- **FR-006**: System MUST persist finalized Scenario records and their support, affect, risk, contains, or evidence relationships after report generation.
- **FR-007**: System MUST distinguish support relationships by evidence category so a reviewer can identify whether a scenario was supported by news, financial metrics, technical analysis, wave analysis, risk records, or other evidence.
- **FR-008**: System MUST mark WaveAnalysis as candidate-based and MUST represent NEoWave rule checks as passed, failed, unknown, or needs_human_review.
- **FR-009**: System MUST avoid definitive automatic NEoWave counts and MUST disclose uncertainty when wave evidence is partial, conflicting, or rule-incomplete.
- **FR-010**: System MUST fall back to canonical evidence or return an insufficient-data response when graph retrieval is missing, stale, or unavailable.
- **FR-011**: System MUST record graph-context degradation, missing evidence categories, and relationship persistence failures in the research output or audit trail.
- **FR-012**: System MUST evaluate generated reports for grounding, numeric consistency, graph evidence path consistency, unsafe recommendation language, risk disclosure, limitations, data freshness, and disclaimer presence.
- **FR-013**: System MUST route failed evaluations to rewrite and MUST stop after at most two rewrite attempts.
- **FR-014**: System MUST NOT graph every raw OHLCV candle, every news sentence, every financial statement row, or every temporary calculation as part of this feature.
- **FR-015**: System MUST NOT treat relationship retrieval storage as the canonical durable source of truth for reports, evidence, requests, or source documents.

### Safety, Evidence & Reliability Requirements *(mandatory for research workflow changes)*

- **SER-001**: System MUST avoid direct buy/sell/hold recommendations, trading
  instructions, guaranteed return claims, guaranteed target claims, and order
  execution behavior.
- **SER-002**: System MUST back important numeric or factual report claims with
  `EvidenceItem` records when the feature affects research output.
- **SER-003**: Final Korean reports MUST include the required education-only,
  no-recommendation disclaimer exactly as defined in the constitution.
- **SER-004**: System MUST disclose unavailable or degraded market, fundamental,
  news, technical, wave, or graph-context data instead of fabricating missing facts.
- **SER-005**: Workflow-affecting features MUST define deterministic behavior for
  validation failure, provider failure, evaluator failure, graph retrieval failure,
  relationship persistence failure, and rewrite limits.
- **SER-006**: Runtime-affecting features MUST preserve structured logs, report
  storage, health checks, and metrics required by the constitution.

### Key Entities *(include if feature involves data)*

- **Company**: The research subject around which scenario graph context is centered.
- **Ticker**: A tradable symbol or market identifier associated with a company.
- **NewsEvent**: A summarized event or news item that may support, weaken, or contextualize a scenario.
- **FinancialMetric**: A canonical financial data point used as scenario evidence.
- **TechnicalAnalysis**: A technical indicator or market-position result used as scenario evidence.
- **WaveAnalysis**: A candidate NEoWave-derived analysis result with uncertainty and review status.
- **NEoWaveRule**: A rule or reference used to check candidate wave analysis.
- **Risk**: A scenario-relevant risk, uncertainty, or negative factor.
- **Evidence**: A traceable source record supporting factual, numeric, or relationship claims.
- **Scenario**: A Bull, Base, or Bear outcome frame with grounds, conditions, risks, and limitations.
- **Report**: The generated Korean research output containing scenario sections and citations.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a deterministic ticker fixture with complete context, 100% of generated reports contain Bull, Base, and Bear sections.
- **SC-002**: For a deterministic complete-context report, 100% of scenario sections include at least one traceable supporting evidence record.
- **SC-003**: For a deterministic complete-context report, each scenario includes confirmation conditions, invalidation conditions, key risks, limitations, data date, and the required disclaimer.
- **SC-004**: For simulated missing graph context, the system either produces a degraded report with explicit graph-context warnings or returns an insufficient-data response without fabricated relationships.
- **SC-005**: Evaluator rejects 100% of test reports that contain direct buy/sell/hold recommendations, guaranteed return claims, missing disclaimer text, or nonexistent evidence IDs.
- **SC-006**: After a finalized deterministic report, each stored Scenario can be traced to its supporting or affecting news, financial, technical, wave, risk, and evidence relationships when those categories were used in generation.

## Assumptions

- Bull/Base/Bear is the approved scenario vocabulary for this feature.
- PostgreSQL canonical persistence from the storage feature exists or is being implemented in parallel.
- Relationship retrieval records are rebuildable from canonical records and are not the original source of truth.
- NEoWave materials are treated as research references and rule-check evidence, not as a source of guaranteed targets or definitive automated counts.
- Tests use deterministic fixtures and mocked external providers, graph retrieval, and model responses.
- Constitution wording has been updated to standardize Bull/Base/Bear as the MVP scenario vocabulary.
