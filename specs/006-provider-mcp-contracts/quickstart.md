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

```bash
python -m compileall src
pytest tests/test_provider_contracts.py
pytest tests/test_provider_persistence_contracts.py
pytest tests/test_provider_graphrag_mapping_contracts.py
pytest tests/test_scenario_report_input_contract.py
```

If the project uses `uv`:

```bash
uv run python -m compileall src
uv run pytest tests/test_provider_contracts.py
uv run pytest tests/test_provider_persistence_contracts.py
uv run pytest tests/test_provider_graphrag_mapping_contracts.py
uv run pytest tests/test_scenario_report_input_contract.py
```

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
