# Research: Provider-Agnostic MCP Contracts

## Decision: 006 owns contracts, not 004/005 storage or graph semantics

**Rationale**: 004 already owns PostgreSQL source-of-truth tables and
repositories. 005 owns the Bull/Base/Bear graph scenario model and retrieval
semantics. 006 should define the provider interface and normalization boundary
that feeds those existing layers.

**Alternatives considered**:
- Let 006 define new canonical tables independently. Rejected because it creates
  conflicting ownership with 004.
- Let 006 redefine Neo4j node/edge semantics. Rejected because it duplicates 005
  and makes graph retrieval ownership unclear.

## Decision: separate provider-normalized objects from internal derived results

**Rationale**: CompanyProfile, NewsEvent, and FinancialMetric are direct
provider-normalized contracts. TechnicalAnalysisResult and WaveAnalysisResult
are internal derived outputs built from normalized market data, Evidence, and
rule references. Their lineage should point to normalized inputs, not directly
to raw provider payloads.

**Alternatives considered**:
- Treat every output as direct provider normalization. Rejected because technical
  and wave analysis are computed by internal engines.
- Exclude technical/wave results from this feature. Rejected because
  ScenarioReportInput needs a stable contract for those downstream outputs.

## Decision: store raw responses separately from normalized records

**Rationale**: Provider behavior changes over time. Raw provider payloads are
needed for audit and re-normalization, while normalized objects are stable
agent-facing records. Separation also prevents agents from depending on raw
provider-specific field names.

**Alternatives considered**:
- Store only normalized data. Rejected because audit and re-normalization would
  lose source context.
- Store raw payloads inside each normalized object. Rejected because it leaks
  provider-specific fields into agent-facing contracts and duplicates storage.

## Decision: define mapping eligibility, not full graph projection behavior

**Rationale**: 005 owns GraphRAG retrieval semantics. 006 only needs to define
which normalized or derived records are eligible to become Company/Ticker-centered
graph nodes or relationships and what canonical references must be carried.

**Alternatives considered**:
- Build full Neo4j projection behavior in 006. Rejected because it duplicates
  005 and expands the feature beyond MCP contract readiness.
- Omit graph mapping from 006. Rejected because provider contracts must preserve
  the identifiers needed by GraphRAG.

## Decision: use deterministic fixtures instead of live MCP calls

**Rationale**: The constitution requires deterministic tests. This feature is
pre-implementation contract work, so fixtures can cover provider shape changes,
partial success, malformed responses, and missing fields without live APIs.

**Alternatives considered**:
- Validate against live MCP servers. Rejected because it would make tests
  nondeterministic and require credentials.
- Skip provider failure tests until MCP implementation. Rejected because failure
  semantics are a core part of the contract.
