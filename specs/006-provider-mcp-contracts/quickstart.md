# Quickstart: Provider-Agnostic MCP Contracts

## Purpose

Validate that provider-specific raw responses are normalized into stable agent
contracts, preserve lineage, and can feed ScenarioReportInput and GraphRAG
mapping without live MCP calls.

## Prerequisites

- Python environment for this repository is installed.
- PostgreSQL test database is available when running persistence tests.
- `TEST_DATABASE_URL` is exported for PostgreSQL-backed tests.
- No live MCP/API keys are required.

## Validation Commands

This repository has no `pyproject.toml`. Use the virtual-environment runner directly:

```bash
# Compile check
.venv/bin/python -m compileall -q src

# Lint
.venv/bin/python -m ruff check src tests

# Contract test suite
.venv/bin/python -m pytest tests/test_provider_contracts.py
.venv/bin/python -m pytest tests/test_provider_persistence_contracts.py
.venv/bin/python -m pytest tests/test_provider_graphrag_mapping_contracts.py
.venv/bin/python -m pytest tests/test_scenario_report_input_contract.py
.venv/bin/python -m pytest tests/test_provider_observability_contract.py
.venv/bin/python -m pytest tests/test_provider_safety_contract.py
```

> **Note:** `tests/test_provider_persistence_contracts.py` requires a live PostgreSQL
> database. Without it the tests are automatically skipped. To enable them, export:
>
> ```bash
> export TEST_DATABASE_URL=postgresql+psycopg://finsight:finsight@localhost:5432/finsight_test
> ```

## Scenario 1: Provider Shape Independence

Fixture:
- Provider A returns news as `{title, content, url}`.
- Provider B returns equivalent news as `{headline, summary_text, source_url}`.

Expected:
- Both normalize to NewsEvent.
- NewsAgent receives no provider-specific raw fields.
- Contract tests pass.

## Scenario 2: Raw And Normalized Lineage

Fixture:
- One raw response produces CompanyProfile, FinancialMetric, and NewsEvent.

Expected:
- RawProviderResponse remains separately traceable.
- Each provider-normalized object links to raw response and canonical ticker.
- TechnicalAnalysisResult and WaveAnalysisResult link to normalized market/evidence
  inputs, not directly to raw provider payload.

## Scenario 3: Graph Mapping Eligibility

Fixture:
- Normalized company, news, financial metric, technical result, wave result, risk,
  and evidence records exist.

Expected:
- Mapping contract produces graph-eligible specs for the 005 graph model.
- RawProviderResponse is not projected directly.
- No raw candle, news sentence, or financial row becomes a graph node.

## Scenario 4: Degraded Provider Behavior

Fixture:
- News provider succeeds.
- Financial provider returns partial data.
- Market provider fails.

Expected:
- Successful normalized records are preserved.
- Failed or missing categories create missing-data notes.
- ScenarioReportInput is either degraded with warnings or insufficient_data.
- No facts or graph relationships are fabricated.

## Validation results

The canonical validation set for feature 006 consists of the following six test modules:

| Test file | Coverage area |
|---|---|
| `tests/test_provider_contracts.py` | Core normalization boundary (CompanyProfile, FinancialMetric, NewsEvent, MarketData) |
| `tests/test_provider_persistence_contracts.py` | ORM round-trip and Alembic migration; requires `TEST_DATABASE_URL` (skipped otherwise) |
| `tests/test_provider_graphrag_mapping_contracts.py` | Graph-mapping eligibility; raw responses excluded from projection |
| `tests/test_scenario_report_input_contract.py` | ScenarioReportInput assembly from normalized contracts |
| `tests/test_provider_observability_contract.py` | Observability fields (provider, fetched_at, latency_ms) present on all contract types |
| `tests/test_provider_safety_contract.py` | Safety invariants: no PII, no fabricated facts, boundary not crossed |

To run the full validation suite against this environment:

```bash
.venv/bin/python -m compileall -q src
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pytest tests/test_provider_contracts.py \
    tests/test_provider_persistence_contracts.py \
    tests/test_provider_graphrag_mapping_contracts.py \
    tests/test_scenario_report_input_contract.py \
    tests/test_provider_observability_contract.py \
    tests/test_provider_safety_contract.py
```

Persistence tests skip automatically when `TEST_DATABASE_URL` is not set; all other
tests run without any external service.
