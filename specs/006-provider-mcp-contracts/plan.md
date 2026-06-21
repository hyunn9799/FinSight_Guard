# Implementation Plan: Provider-Agnostic MCP Contracts

**Branch**: `[main]` | **Date**: 2026-06-21 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/006-provider-mcp-contracts/spec.md`

## Summary

Define a provider-agnostic contract layer before implementing real News,
Financial, or Market MCP adapters. The feature standardizes provider interfaces,
normalization outputs, raw-response lineage, ScenarioReportInput construction,
and eligibility rules for PostgreSQL canonical persistence and Neo4j GraphRAG
projection.

This is a compatibility and contract-alignment feature. It does not re-own the
PostgreSQL canonical schema from 004 or the scenario graph model from 005. It
defines the stable boundary that future MCP adapters must satisfy so agents and
GraphRAG retrieval do not depend on provider-specific response shapes.

## Technical Context

**Language/Version**: Python 3.12, matching the active project virtual
environment and existing PostgreSQL migration/runtime stack.

**Primary Dependencies**: Existing Pydantic v2 contracts, FastAPI/Streamlit
runtime, LangGraph workflow modules, SQLAlchemy/Alembic/psycopg PostgreSQL
persistence, existing `src/tools/*` provider helpers, existing
`src/graph_rag/*` context builders, pytest.

**Storage**: PostgreSQL remains the canonical store. Existing 004 tables cover
requests, results, reports, evidence, source documents, projection status, and
evidence paths. Planning identifies explicit 004 schema-extension needs for raw
provider responses and normalized provider/derived records. Neo4j remains a
rebuildable GraphRAG projection owned by the 005 scenario feature.

**Testing**: pytest with deterministic provider fixtures and fake provider
adapters. Tests must not call live MCP servers, yfinance, Tavily, Firecrawl,
OpenAI, Neo4j, Pinecone, OpenSearch, broker APIs, or paper-trading APIs.

**Target Platform**: Local portfolio/demo runtime through FastAPI, Streamlit,
Docker Compose, and pytest.

**Project Type**: Single Python web-service plus Streamlit UI and LangGraph
research workflow.

**Performance Goals**: Non-binding guidance only — NOT gated by tests in this
feature. As a rough sanity reference, contract normalization for one
deterministic ticker fixture is expected to complete well within ~1 second and
ScenarioReportInput construction from already-normalized fixture records well
within ~500 ms, all without external calls. These are validation guides, not
production SLAs, and `tasks.md` intentionally defines no deterministic timing
test for them. If timing ever becomes a hard requirement, add an explicit
lightweight deterministic timing task to `tasks.md` at that time.

**Constraints**: No real News MCP, company-analysis MCP, or market-data MCP
implementation in this feature. No API key integration. No live provider calls.
No live trading, order execution, guaranteed returns, or direct buy/sell/hold
recommendations. Do not graph every candle, news sentence, financial row, or
temporary calculation.

**Scale/Scope**: Define contracts for CompanyProfile, NewsEvent,
FinancialMetric, TechnicalAnalysisResult, WaveAnalysisResult,
ScenarioReportInput, NewsProvider, FinancialProvider, and MarketDataProvider.
Define persistence lineage and graph-mapping eligibility. Defer concrete MCP
adapters and full graph retrieval implementation.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Graph grounding: PASS. The plan requires provider-normalized and derived
  objects to carry canonical identifiers, evidence references, or graph mapping
  references before they can feed ScenarioReportInput.
- Scenario safety: PASS. The feature does not generate report text directly, but
  preserves Bull/Base/Bear ScenarioReportInput compatibility and does not
  introduce action-instruction wording.
- Financial safety: PASS. Real trading, live order execution, guaranteed
  returns, guaranteed targets, and direct buy/sell/hold recommendations are out
  of scope.
- LangGraph workflow: PASS. The feature stabilizes agent inputs without changing
  existing workflow routing, Evaluator limits, or Rewrite behavior.
- Deterministic quality: PASS. Validation relies on deterministic provider
  fixtures and mocked provider/graph/vector clients only.
- Observability: PASS. Raw response status, normalization warnings, missing
  fields, graph mapping warnings, and degradation states are included in the
  contracts and data model.

**Constitution Result**: PASS. No exceptions required.

## Project Structure

### Documentation (this feature)

```text
specs/006-provider-mcp-contracts/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── provider-normalization-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── providers/
│   ├── __init__.py
│   ├── enums.py            # status/degradation enums, lineage refs
│   ├── interfaces.py       # NewsProvider/FinancialProvider/MarketDataProvider
│   ├── entities.py         # normalized + derived Pydantic contracts (extra="forbid")
│   ├── safety.py           # FORBIDDEN_TOKENS/PHRASES, SAFETY_CHECKED_CONTRACTS, assert_no_trading_fields
│   ├── normalization.py
│   └── scenario_input.py   # ScenarioReportInput + VectorReference
├── db/
│   ├── models.py
│   ├── repositories/
│   │   └── provider_repository.py
│   └── migrations/
├── graph_rag/
│   └── mapping_contracts.py
└── agents/
    ├── news_agent.py
    ├── fundamental_agent.py
    ├── market_agent.py
    └── coordinator_agent.py

tests/
├── fixtures/
│   └── provider_contracts.py
├── test_provider_contracts.py
├── test_provider_persistence_contracts.py
├── test_provider_graphrag_mapping_contracts.py
├── test_scenario_report_input_contract.py
├── test_provider_observability_contract.py
└── test_provider_safety_contract.py
```

**Structure Decision**: Keep provider contracts in a narrow `src/providers`
boundary so existing agents consume stable objects rather than provider payloads.
Use `src/db` only for explicit 004-compatible schema extensions and repository
work. Use `src/graph_rag` only for mapping eligibility/spec builders, not for
owning the graph model itself.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design & Contracts

- Data model: [data-model.md](./data-model.md)
- Provider contract: [contracts/provider-normalization-contract.md](./contracts/provider-normalization-contract.md)
- Validation guide: [quickstart.md](./quickstart.md)

## Post-Design Constitution Check

- Graph grounding: PASS. Data model and contract require canonical IDs,
  evidence references, raw-response lineage, and graph mapping eligibility.
- Scenario safety: PASS. ScenarioReportInput is evidence/context input only and
  does not create direct investment instructions.
- Financial safety: PASS. Provider contracts exclude live trading and order
  execution.
- LangGraph workflow: PASS. Agent-facing contracts are stable and do not bypass
  Evaluator/Rewrite routing.
- Deterministic quality: PASS. Quickstart validates fixture-based tests only.
- Observability: PASS. Normalization, provider, mapping, and degradation status
  are explicitly modeled.

**Post-Design Constitution Result**: PASS. No exceptions required.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
