# Feature Specification: Provider-Agnostic MCP Contracts

**Feature Branch**: `[006-provider-mcp-contracts]`

**Created**: 2026-06-21

**Status**: Draft

**Input**: User description: "현재 constitution은 완료되었다. 이번 feature는 뉴스 MCP, 기업분석 MCP, 시장데이터 MCP를 실제 구현하기 전에, 외부 provider가 바뀌어도 Agent와 GraphRAG 파이프라인이 흔들리지 않도록 provider-agnostic MCP integration contract를 정의하는 것이다. 목표: NewsAgent, FinancialAgent, MarketDataProvider, ScenarioAgent가 특정 MCP 응답 형식에 직접 의존하지 않도록 표준 데이터 계약, 정규화 객체, 저장 구조, GraphRAG 매핑 규칙을 정의한다. 사용자가 종목명 또는 ticker를 입력하면, 시스템은 외부 provider에서 뉴스, 기업 정보, 재무 지표, 시장 데이터를 가져오되, provider별 응답 차이를 내부 표준 객체로 정규화해야 한다. 정규화된 데이터는 PostgreSQL에 canonical data로 저장되고, Neo4j에는 Company/Ticker 중심 GraphRAG retrieval에 필요한 핵심 관계만 저장된다. 이후 ScenarioAgent는 정규화된 데이터와 graph context를 사용해 Bull/Base/Bear 시나리오 리포트를 생성할 수 있어야 한다."

## Scope Ownership And Prior Specs

This feature defines the provider-agnostic integration contract layer. It does
not replace or re-own the canonical database schema from
`specs/004-postgresql-table-schema`, and it does not replace or re-own the
GraphRAG scenario model from `specs/005-neo4j-graphrag-scenarios`.

Ownership boundaries:

- `004-postgresql-table-schema` owns canonical table implementation, persistence
  repositories, migrations, and source-of-truth storage behavior.
- `005-neo4j-graphrag-scenarios` owns the Bull/Base/Bear scenario graph model,
  graph retrieval semantics, and scenario-to-evidence relationship behavior.
- `006-provider-mcp-contracts` owns provider interface contracts,
  provider-response normalization contracts, lineage requirements, and the
  mapping contract that says how normalized records are eligible for 004
  persistence and 005 graph projection.

Because 004 and parts of the persistence layer may already exist, this feature
is a compatibility and contract-alignment feature. If planning finds that the
existing 004 schema cannot represent a required raw or normalized contract,
those changes MUST be treated as explicit schema extension or migration tasks in
the plan, not as silent redefinition by this spec.

TechnicalAnalysisResult and WaveAnalysisResult are internal derived analysis
contracts. They are downstream of normalized market data, evidence, and NEoWave
rule references; they are not direct external provider normalization outputs.

Agent naming alias: this spec uses the conceptual names `ScenarioAgent` and
`MarketDataAgent` to describe contract consumers. In the current implementation,
`ScenarioAgent` responsibilities are carried by `CoordinatorAgent`
(`src/agents/coordinator_agent.py`) and `MarketDataAgent` responsibilities are
carried by `MarketAgent` (`src/agents/market_agent.py`). These may be split into
dedicated agents later; until then the conceptual name and the implementing
agent are equivalent.

Safety enforcement scope: the no-trading-field safety contract (SER-001, SC-006)
is enforced structurally via snake_case token/phrase matching over field NAMES,
and applies ONLY to this feature's MVP provider and scenario contracts
(`CompanyProfile`, `NewsEvent`, `FinancialMetric`, `TechnicalAnalysisResult`,
`WaveAnalysisResult`, `ScenarioReportInput`, and provider-interface outputs).
Future Phase 2/3 simulation contracts (`SignalCandidate`, `StrategyRule`,
`PaperTradingExecution`) live in a separate namespace with their own allowlist
and are intentionally NOT blocked by this contract, so the long-term simulated
paper-trading goal (SER-007) is not foreclosed; those contracts still remain
simulated-only and MUST NOT carry live order-execution fields.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Normalize Provider Data For Agents (Priority: P1)

A research user enters a company name or ticker and the system normalizes provider-specific news, company, financial, and market data into stable internal objects before any agent consumes provider-derived data.

**Why this priority**: This is the core contract boundary. Without normalized objects, every MCP provider change can break NewsAgent, FinancialAgent, MarketDataAgent, ScenarioAgent, and GraphRAG context generation.

**Independent Test**: Can be tested with two deterministic provider fixtures that return different raw shapes for the same ticker; both must produce the same normalized object types and agent-facing fields.

**Acceptance Scenarios**:

1. **Given** two providers return equivalent news data in different response shapes, **When** the system normalizes the responses, **Then** NewsAgent receives stable NewsEvent objects and no provider-specific raw fields.
2. **Given** a provider returns company profile, financial metric, and market data, **When** normalization completes and internal analysis runs, **Then** FinancialAgent, MarketDataAgent, and ScenarioAgent can consume CompanyProfile, FinancialMetric, TechnicalAnalysisResult, and ScenarioReportInput without reading raw provider payloads.

---

### User Story 2 - Trace Raw And Normalized Persistence (Priority: P2)

A developer or auditor can trace every provider-normalized research object back to the raw provider response and provider metadata that produced it, and can trace every internal derived analysis result back to its normalized inputs.

**Why this priority**: Provider-agnostic processing must remain auditable. Raw responses and normalized records must be separated while preserving lineage.

**Independent Test**: Can be tested by ingesting deterministic raw provider fixtures and verifying that each provider-normalized record links back to a raw response and canonical research request, while technical and wave results link to normalized market/evidence inputs.

**Acceptance Scenarios**:

1. **Given** a raw provider response is ingested, **When** provider-normalized records are created, **Then** the raw response remains stored separately and each provider-normalized object can be traced to it.
2. **Given** normalization partially succeeds, **When** reviewers inspect persistence, **Then** successful normalized records and failed normalization reasons are both recoverable.
3. **Given** technical or wave analysis is generated, **When** reviewers inspect lineage, **Then** those derived results trace to normalized market data, Evidence, and rule references rather than pretending to be raw provider records.

---

### User Story 3 - Map Normalized Data To GraphRAG Context (Priority: P3)

ScenarioAgent can receive a ScenarioReportInput built from normalized data and Company/Ticker-centered graph context, without depending on raw provider formats.

**Why this priority**: The GraphRAG pipeline must connect canonical normalized records to retrieval-ready graph relationships before Bull/Base/Bear scenario generation.

**Independent Test**: Can be tested by mapping deterministic normalized objects to graph node/relationship specs and verifying that ScenarioReportInput references the expected graph context and evidence IDs.

**Acceptance Scenarios**:

1. **Given** normalized company, news, financial, technical, wave, risk, and evidence records exist, **When** ScenarioReportInput is built, **Then** it contains only normalized objects, evidence references, and graph context needed for scenario generation.
2. **Given** graph projection is missing or stale, **When** ScenarioReportInput is built, **Then** the system either falls back to canonical normalized data or returns a degraded input with explicit graph-context warnings.

### Edge Cases

- Provider returns a ticker alias, company name variant, or market code that conflicts with the canonical ticker.
- Provider returns duplicate news events or financial metrics.
- Provider returns partial records with missing URL, timestamp, currency, period, unit, or source name.
- Provider response is malformed, empty, rate-limited, or unavailable.
- Raw response is stored but normalization fails.
- Normalization succeeds but graph mapping fails.
- Graph mapping would create too many nodes, such as one node per candle, one node per raw financial row, or one node per news sentence.
- Provider data conflicts across news, company profile, financial metrics, or market data.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST define provider-agnostic output contracts for CompanyProfile, NewsEvent, FinancialMetric, TechnicalAnalysisResult, WaveAnalysisResult, and ScenarioReportInput, while distinguishing provider-normalized contracts from internal derived analysis contracts.
- **FR-002**: System MUST define provider interfaces for NewsProvider, FinancialProvider, and MarketDataProvider that return normalized objects or normalization results rather than raw provider payloads.
- **FR-003**: System MUST prevent NewsAgent, FinancialAgent, MarketDataAgent, ScenarioAgent, and Graph Context Builder from depending on provider-specific raw response fields.
- **FR-004**: System MUST store every external provider response as a raw_provider_response record with provider identity, request metadata, collection time, status, and raw payload reference or body.
- **FR-005**: System MUST store normalized news_events separately from raw provider responses and preserve lineage to the raw source response.
- **FR-006**: System MUST store normalized company_profiles separately from raw provider responses and preserve lineage to the raw source response.
- **FR-007**: System MUST store normalized financial_metrics separately from raw provider responses and preserve lineage to the raw source response.
- **FR-008**: System MUST define technical_analysis_results as internal derived analysis records that preserve lineage to normalized market data, technical indicator inputs, Evidence records, and the research request.
- **FR-009**: System MUST define wave_analysis_results as internal derived analysis records that preserve lineage to normalized market data, NEoWave rule references, Evidence records, and the research request.
- **FR-010**: System MUST define normalization status values that distinguish success, partial_success, degraded, failed, unsupported_field, and insufficient_data outcomes.
- **FR-011**: System MUST preserve canonical request, ticker, company, provider, and evidence identifiers so raw responses, normalized records, graph relationships, and reports can be traced together.
- **FR-012**: System MUST define the mapping contract from normalized and derived records into the graph model owned by `specs/005-neo4j-graphrag-scenarios`, including Company, Ticker, NewsEvent, FinancialMetric, TechnicalAnalysisResult, WaveAnalysisResult, Risk, and Evidence eligibility rules.
- **FR-013**: System MUST limit Neo4j graph projection to retrieval-ready Company/Ticker-centered relationships and MUST NOT graph every raw candle, every news sentence, every financial statement row, or every temporary calculation.
- **FR-014**: System MUST define ScenarioReportInput as the stable ScenarioAgent input built from normalized data, evidence references, graph context, missing-data notes, and degradation status.
- **FR-015**: System MUST define fallback and degradation behavior for provider failure, partial provider success, normalization failure, and graph mapping failure.
- **FR-016**: System MUST allow partial scenario report generation when enough normalized evidence exists and MUST return an insufficient-data state when required evidence categories are unavailable.
- **FR-017**: System MUST record normalization warnings, provider errors, missing fields, and graph mapping warnings so Evaluator and report generation can disclose limitations.
- **FR-018**: System MUST define how vector retrieval references news originals, NEoWave rule explanations, and report chunks while preserving canonical source references.
- **FR-019**: System MUST keep PostgreSQL as the canonical store for raw provider responses, normalized records, derived analysis records, and lineage references, following the ownership and migration rules of `specs/004-postgresql-table-schema`.
- **FR-020**: System MUST keep actual News MCP, company-analysis MCP, market-data MCP, API key integration, and live provider calls outside this feature.
- **FR-021**: System MUST document whether any required persistence field is already covered by 004 or requires an explicit extension or migration task during planning.

### Safety, Evidence & Reliability Requirements *(mandatory for research workflow changes)*

- **SER-001**: System MUST avoid direct buy/sell/hold recommendations, trading
  instructions, guaranteed return claims, guaranteed target claims, and live
  order execution behavior.
- **SER-002**: System MUST back important numeric or factual report claims with
  `EvidenceItem` records or graph evidence paths when the feature affects
  research output.
- **SER-003**: Final Korean reports MUST include the required education-only,
  no-recommendation disclaimer exactly as defined in the constitution.
- **SER-004**: System MUST disclose unavailable or degraded market, fundamental,
  news, technical, wave, or graph-context data instead of fabricating missing
  facts.
- **SER-005**: Workflow-affecting features MUST define deterministic behavior for
  validation failure, provider failure, graph retrieval failure, evaluator
  failure, relationship persistence failure, and rewrite limits.
- **SER-006**: Runtime-affecting features MUST preserve structured logs, report
  storage, scenario/evidence-path traceability, health checks, and metrics
  required by the constitution.
- **SER-007**: Paper trading or mock investment API features MUST remain
  simulated-only and MUST NOT connect report output directly to live orders.

### Key Entities *(include if feature involves data)*

- **RawProviderResponse**: Stored original provider response with provider name, request context, status, collection time, payload reference or body, and error metadata.
- **CompanyProfile**: Normalized company identity and business profile used by financial and scenario analysis.
- **NewsEvent**: Normalized event-level news record with source, time, summary, event type, risk or sentiment tags, and evidence reference.
- **FinancialMetric**: Normalized financial metric with period, currency, unit, value, source, and evidence reference.
- **TechnicalAnalysisResult**: Internal derived market/technical analysis output such as indicator values, trend state, volatility, source input references, and evidence reference.
- **WaveAnalysisResult**: Internal derived NEoWave-related candidate output with rule status, confirmation condition, invalidation condition, uncertainty, source input references, and evidence reference.
- **ScenarioReportInput**: Stable ScenarioAgent input combining normalized records, graph context, evidence references, missing-data notes, and degradation status.
- **ProviderInterface**: Provider-agnostic contract for NewsProvider, FinancialProvider, and MarketDataProvider.
- **MarketDataAgent**: Consumer of normalized market data and producer or coordinator of internal technical analysis results; it is distinct from the MarketDataProvider interface.
- **GraphMappingRule**: Mapping from provider-normalized and internal derived records to retrieval-ready graph nodes and relationships owned by the scenario GraphRAG feature.
- **VectorReference**: Lightweight reference from ScenarioReportInput to semantically retrievable material (news originals, NEoWave rule explanations, report chunks) carrying `source_kind`, a canonical PostgreSQL reference, an optional source URI, and an optional future chunk id; it preserves canonical source references and excludes vector-store internals, scores, and embedding details.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Two deterministic provider fixtures with different raw response shapes produce the same agent-facing normalized object types for the same ticker.
- **SC-002**: 100% of provider-normalized records created from deterministic fixtures can be traced to a raw provider response and canonical ticker or company reference.
- **SC-003**: 100% of ScenarioReportInput fixtures contain only normalized records, evidence references, graph context, missing-data notes, and degradation status; no provider-specific raw fields are exposed.
- **SC-004**: GraphRAG mapping fixtures define how provider-normalized and internal derived records map into the 005 graph model without graphing raw candles, raw news sentences, or raw financial rows.
- **SC-005**: Provider failure, partial success, normalization failure, and graph mapping failure fixtures each produce deterministic degraded or insufficient-data outcomes.
- **SC-006**: Safety review of the contract identifies zero fields or flows that require live order execution, direct buy/sell/hold recommendations, guaranteed returns, or guaranteed targets.
- **SC-007**: Planning review can identify for each contract object whether 006 owns the contract, 004 owns the canonical table implementation, or 005 owns the graph projection behavior.

## Assumptions

- This feature defines contracts and schemas before real MCP implementation.
- MCP provider adapters may be added later, but they must conform to these contracts.
- PostgreSQL canonical persistence from 004 exists or is implemented in parallel; this feature may propose explicit schema extensions but does not silently redefine 004 tables.
- Neo4j projection behavior from 005 exists or is implemented in parallel; this feature defines mapping eligibility and input contract, not the full graph retrieval behavior.
- Vector retrieval may be used for semantic lookup of news originals, NEoWave rule explanations, and report chunks, but canonical source references remain mandatory.
- Tests use deterministic provider fixtures and do not call live MCP servers or external APIs.
