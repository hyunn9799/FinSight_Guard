# Provider-Agnostic MCP Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the provider-agnostic *contract layer* (Pydantic contracts, provider interfaces, normalization helpers, canonical persistence lineage, GraphRAG mapping eligibility, and `ScenarioReportInput`) that future MCP adapters must satisfy — entirely behind deterministic fixtures, with no live MCP/API/Neo4j/vector calls.

**Architecture:** A narrow `src/providers/` package owns the contracts. It is split by responsibility — `enums.py` (status/degradation/lineage primitives), `interfaces.py` (provider Protocols), `entities.py` (normalized + derived Pydantic contracts), `safety.py` (token-based no-trading-field enforcement), `normalization.py` (raw-fixture → contract helpers), `scenario_input.py` (`ScenarioReportInput` + `VectorReference`). `src/db/` gets explicit 004-compatible *extension* tables + a `ProviderRepository`. `src/graph_rag/mapping_contracts.py` emits 005 graph-eligibility specs. Existing agents get additive typed boundary functions that consume contracts, never raw payloads.

**Tech Stack:** Python 3.12, Pydantic v2, SQLAlchemy 2.0 + Alembic + psycopg (PostgreSQL), pytest, `uv`, `ruff`.

## Global Constraints

These apply to **every** task. Copied verbatim from spec/plan/contract:

- **Python 3.12**, matching the active project virtualenv. Run everything via `uv run …`.
- **Every contract Pydantic model MUST set `model_config = ConfigDict(extra="forbid")`.** Unknown (e.g. smuggled trading) fields must be rejected, not silently absorbed.
- **No live anything.** Tests MUST NOT call live MCP servers, yfinance, Tavily, Firecrawl, OpenAI, Neo4j, Pinecone/OpenSearch, broker APIs, or paper-trading APIs. All tests are deterministic-fixture-based.
- **No trading-instruction fields anywhere in 006 contracts.** No buy/sell/hold, order execution, guaranteed return/target. Enforced structurally (SER-001, SC-006).
- **Safety matching is snake_case TOKEN/PHRASE based, never substring.** `threshold`/`household` must NOT match `hold`. Applies to field **names** only, not values, and ONLY to `SAFETY_CHECKED_CONTRACTS` (the MVP contracts). Future Phase 2/3 simulation contracts (`SignalCandidate`, `StrategyRule`, `PaperTradingExecution`) are a **separate namespace** and MUST NOT be blocked.
- **VectorReference is lightweight:** `source_kind`, `canonical_ref_id`, optional `source_uri`, optional `chunk_id`. **No score/store/embedding fields.**
- **Ownership boundaries are hard:** 004 owns canonical tables/migrations, 005 owns the graph model + retrieval, 006 owns contracts/mapping only. Any 004 schema need is an **explicit extension/migration task** (T023), never silent redefinition.
- **`RawProviderResponse` is never graph-projected directly.** No node per raw candle / news sentence / financial row.
- **TechnicalAnalysisResult / WaveAnalysisResult are internal derived contracts** — they trace to normalized market data + Evidence + rule refs, NOT to raw provider payloads.
- **TDD, test-first.** Within each story: write failing test → confirm it fails → minimal implementation → confirm pass → commit. Contract models before normalizers; models+migrations before repositories; mapping rules before `ScenarioReportInput` integration; agent integration last.
- **Agent alias:** spec's `ScenarioAgent` ≡ `CoordinatorAgent` (`src/agents/coordinator_agent.py`); `MarketDataAgent` ≡ `MarketAgent` (`src/agents/market_agent.py`).

---

## Status / Progress (as of 2026-06-22)

- ✅ Spec artifacts complete and committed (`45a3fdd`): `spec.md`, `plan.md`, `tasks.md`, `data-model.md`, `contracts/provider-normalization-contract.md`, `quickstart.md`, `research.md`.
- ❌ **No implementation code exists.** `src/providers/` is absent; no provider tests; no plan file before this one. Working tree clean, fresh clone.
- ▶️ **Start here:** Phase 1, Task 1. Implementation is at 0%.

---

## Existing-Code Facts (verified, do not re-derive)

- `src/db/models.py` defines `Base(DeclarativeBase)` with `NAMING_CONVENTION`, plus `UUIDMixin` (`id: uuid4 PK`) and `TimestampMixin` (`created_at`/`updated_at`). Use `Mapped[...]` + `mapped_column(...)`. JSONB/UUID come from `sqlalchemy.dialects.postgresql`.
- Existing canonical tables already present: `tickers`, `analysis_requests`, `analysis_results`, `workflow_node_runs`, `structured_log_events`, `index_projection_status`, `wave_scenarios`, `evidence_paths`, `source_documents`, etc. **004 owns these.** 006 only adds new extension tables.
- **Alembic migration head is `56464a69bd55`** (`add_query_indexes`). The chain is `40be5b7bbb1d → d305f4d1ff77 → 07c7492119e4 → 56464a69bd55`. The new 006 migration's `down_revision` MUST be `'56464a69bd55'`.
- Repositories subclass `BaseRepository(session)` (`src/db/repositories/base.py`) and are exported from `src/db/repositories/__init__.py`.
- `tests/conftest.py` provides `db_session` (begins a transaction, rolls back after each test) and `alembic_migrated_db` (skips if `TEST_DATABASE_URL` unset). Persistence tests depend on `db_session`.
- Pydantic v2 idiom in repo: `from pydantic import BaseModel, Field` (see `src/evidence/evidence_schema.py`). We additionally import `ConfigDict`.
- `src/safety/forbidden_phrases.py` holds Korean *value*-phrase matching — **unrelated** to our field-name token matching. No conflict; do not modify it.
- Agents are free-function modules operating on `src/graph/state.GraphState` (e.g. `src/agents/news_agent.py`). Integration tasks add new typed boundary functions; they do not rewrite existing workflow nodes.

---

## File Structure

**Create (new):**
- `src/providers/__init__.py` — package marker + stable public re-exports.
- `src/providers/enums.py` — `ProviderKind`, `NormalizationStatus`, `DegradationStatus`, `RuleStatus`, lineage reference models (`RawResponseRef`, `EvidenceRef`, `MarketDataRef`, `RuleRef`), shared `Warning`/`ProviderError`.
- `src/providers/interfaces.py` — `NewsProvider`, `FinancialProvider`, `MarketDataProvider` Protocols + their typed result models.
- `src/providers/entities.py` — `CompanyProfile`, `NewsEvent`, `FinancialMetric`, `TechnicalAnalysisResult`, `WaveAnalysisResult` (all `extra="forbid"`).
- `src/providers/safety.py` — `FORBIDDEN_TOKENS`, `FORBIDDEN_TOKEN_PHRASES`, `SAFETY_CHECKED_CONTRACTS`, `assert_no_trading_fields(model)`.
- `src/providers/normalization.py` — `RawNewsItem`/`RawCompanyPayload`/`RawFinancialRow`/`RawMarketData` raw fixture shapes + `NormalizationResult` containers + normalizer functions.
- `src/providers/scenario_input.py` — `VectorReference`, `ScenarioReportInput` + builder.
- `src/graph_rag/mapping_contracts.py` — `GraphMappingRule`, `GraphEligibleSpec`, eligibility builders.
- `src/db/repositories/provider_repository.py` — `ProviderRepository`.
- `src/db/migrations/versions/8f1a2b3c4d5e_provider_contract_records.py` — extension migration.
- `tests/fixtures/provider_contracts.py` — deterministic raw + normalized fixture builders.
- `tests/test_provider_contracts.py`, `tests/test_provider_persistence_contracts.py`, `tests/test_provider_graphrag_mapping_contracts.py`, `tests/test_scenario_report_input_contract.py`, `tests/test_provider_observability_contract.py`, `tests/test_provider_safety_contract.py`.

**Modify (existing):**
- `tests/fixtures/__init__.py` — expose provider fixture helpers.
- `src/db/models.py` — add extension ORM models.
- `src/db/repositories/__init__.py` — export `ProviderRepository`.
- `src/db/persistence.py` — add raw/normalized orchestration helpers.
- `src/agents/news_agent.py`, `src/agents/fundamental_agent.py`, `src/agents/market_agent.py`, `src/agents/coordinator_agent.py` — additive typed boundary functions.
- `src/graph_rag/graph_context_builder.py` — degraded/stale graph-context handling at the contract boundary.
- `specs/006-provider-mcp-contracts/quickstart.md`, `PROJECT_PLAN.md` — doc alignment.

---

# Phase 1: Setup (Shared Infrastructure)

### Task 1: Provider package + fixture skeleton (T001, T002, T003)

**Files:**
- Create: `src/providers/__init__.py`
- Create: `tests/fixtures/provider_contracts.py`
- Modify: `tests/fixtures/__init__.py`

**Interfaces:**
- Produces: importable `src.providers` package; `tests.fixtures.provider_contracts` module with placeholder builders that later tasks fill in; `tests.fixtures` re-exporting them.

- [ ] **Step 1 (T001): Create the package marker**

`src/providers/__init__.py`:

```python
"""Provider-agnostic MCP contract layer (006).

Owns provider interface contracts, normalization contracts, lineage
requirements, and graph-mapping eligibility. Does NOT own canonical tables
(004) or the graph model (005). No live MCP/API/Neo4j/vector calls live here.

Stable public exports are populated in T008.
"""
```

- [ ] **Step 2 (T002): Create the fixture module skeleton**

`tests/fixtures/provider_contracts.py`:

```python
"""Deterministic provider-contract fixtures.

No live calls. Every builder returns fixed, hand-authored data so contract
tests are reproducible. Real raw shapes for two providers (A/B) plus
normalized-record and degraded-scenario builders are filled in by later tasks.
"""

from __future__ import annotations

# Builders are added incrementally:
#   T009/T012 -> raw_news_provider_a / raw_news_provider_b
#   T013      -> raw_company_payload / raw_financial_rows
#   T014      -> raw_market_data
#   T018+     -> normalized record builders for persistence
#   T029+     -> scenario_report_input builder
```

- [ ] **Step 3 (T003): Re-export from the fixtures package**

Read `tests/fixtures/__init__.py` first, then append:

```python
from tests.fixtures import provider_contracts  # noqa: F401
```

- [ ] **Step 4: Verify import**

Run: `uv run python -c "import src.providers, tests.fixtures.provider_contracts; print('ok')"`
Expected: prints `ok`, no error.

- [ ] **Step 5: Commit**

```bash
git add src/providers/__init__.py tests/fixtures/provider_contracts.py tests/fixtures/__init__.py
git commit -m "feat(006): scaffold provider contract package and fixture surface (T001-T003)"
```

---

# Phase 2: Foundational (Blocking Prerequisites)

> **CRITICAL:** No user-story work begins until Phase 2 is complete. T004/T005/T006/T048 touch separate files and are parallelizable `[P]`; T007 then T008 follow.

### Task 2: Contract primitives — enums & lineage refs (T004)

**Files:**
- Create: `src/providers/enums.py`
- Test: `tests/test_provider_contracts.py` (a small import/sanity test here; behavioral tests come in US1)

**Interfaces:**
- Produces: `ProviderKind`, `NormalizationStatus`, `DegradationStatus`, `RuleStatus` (str Enums); `RawResponseRef`, `EvidenceRef`, `MarketDataRef`, `RuleRef`, `Warning`, `ProviderError` (Pydantic models, `extra="forbid"`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_provider_contracts.py`:

```python
"""User Story 1 contract tests (deterministic, no live calls)."""

import pytest
from pydantic import ValidationError

from src.providers.enums import (
    DegradationStatus,
    NormalizationStatus,
    ProviderKind,
    RawResponseRef,
    Warning,
)


def test_enums_have_required_members():
    assert ProviderKind.NEWS.value == "news"
    assert ProviderKind.FINANCIAL.value == "financial"
    assert ProviderKind.MARKET_DATA.value == "market_data"
    assert NormalizationStatus.SUCCESS.value == "success"
    assert NormalizationStatus.PARTIAL_SUCCESS.value == "partial_success"
    assert NormalizationStatus.DEGRADED.value == "degraded"
    assert NormalizationStatus.FAILED.value == "failed"
    assert NormalizationStatus.UNSUPPORTED_FIELD.value == "unsupported_field"
    assert NormalizationStatus.INSUFFICIENT_DATA.value == "insufficient_data"
    assert DegradationStatus.COMPLETE.value == "complete"
    assert DegradationStatus.INSUFFICIENT_DATA.value == "insufficient_data"


def test_lineage_ref_forbids_extra_fields():
    RawResponseRef(raw_response_id="r1")
    with pytest.raises(ValidationError):
        RawResponseRef(raw_response_id="r1", bogus="x")


def test_warning_is_structured():
    w = Warning(code="missing_url", message="news event has no source url")
    assert w.code == "missing_url"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.providers.enums'`.

- [ ] **Step 3: Write the implementation**

`src/providers/enums.py`:

```python
"""Status/degradation enums and lineage reference primitives (006).

These are the shared vocabulary for normalization outcomes and the canonical
references that tie raw responses, normalized records, evidence, market data,
and NEoWave rules together. No DB/graph/provider imports here.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProviderKind(str, Enum):
    NEWS = "news"
    FINANCIAL = "financial"
    MARKET_DATA = "market_data"


class NormalizationStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNSUPPORTED_FIELD = "unsupported_field"
    INSUFFICIENT_DATA = "insufficient_data"


class DegradationStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL_PROVIDER_FAILURE = "partial_provider_failure"
    PARTIAL_NORMALIZATION_FAILURE = "partial_normalization_failure"
    GRAPH_MAPPING_DEGRADED = "graph_mapping_degraded"
    INSUFFICIENT_DATA = "insufficient_data"


class RuleStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class _Contract(BaseModel):
    """Base for all 006 contract models: reject unknown fields."""

    model_config = ConfigDict(extra="forbid")


class RawResponseRef(_Contract):
    raw_response_id: str


class EvidenceRef(_Contract):
    evidence_id: str


class MarketDataRef(_Contract):
    market_data_ref_id: str


class RuleRef(_Contract):
    rule_id: str


class Warning(_Contract):
    code: str
    message: str = Field(min_length=1)
    field: str | None = None


class ProviderError(_Contract):
    code: str
    message: str = Field(min_length=1)
    provider_name: str | None = None
```

> Note: `_Contract` is the single source of `extra="forbid"`. Every contract model in `entities.py`, `interfaces.py`, `normalization.py`, and `scenario_input.py` subclasses `_Contract` so the constraint is uniform.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/providers/enums.py tests/test_provider_contracts.py
git commit -m "feat(006): provider status enums and lineage refs (T004)"
```

---

### Task 3: Normalized & derived entity contracts (T006)

> Implement **before** interfaces (T005) because `interfaces.py` result models reference these entities. (tasks.md lists T005/T006 as parallel `[P]`; we order entities first to keep imports clean — both still land before the Phase-2 checkpoint.)

**Files:**
- Create: `src/providers/entities.py`
- Test: `tests/test_provider_contracts.py` (extend)

**Interfaces:**
- Consumes: `_Contract`, `NormalizationStatus`, `RuleStatus`, `RawResponseRef`, `EvidenceRef`, `MarketDataRef`, `RuleRef`, `Warning` from `enums.py`.
- Produces: `CompanyProfile`, `NewsEvent`, `FinancialMetric`, `TechnicalAnalysisResult`, `WaveAnalysisResult` — field names exactly per `data-model.md`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_provider_contracts.py`)

```python
from datetime import UTC, datetime

from src.providers.entities import (
    CompanyProfile,
    FinancialMetric,
    NewsEvent,
    TechnicalAnalysisResult,
    WaveAnalysisResult,
)
from src.providers.enums import NormalizationStatus, RuleStatus


def test_news_event_minimal_and_forbids_extra():
    ev = NewsEvent(
        request_id="req1",
        ticker_id="tk1",
        raw_response_id="raw1",
        title="Acme beats earnings",
        normalization_status=NormalizationStatus.SUCCESS,
    )
    assert ev.title == "Acme beats earnings"
    assert ev.warnings == []
    with pytest.raises(ValidationError):
        NewsEvent(
            request_id="req1",
            ticker_id="tk1",
            raw_response_id="raw1",
            title="x",
            normalization_status=NormalizationStatus.SUCCESS,
            content="RAW PROVIDER FIELD",  # not in contract -> rejected
        )


def test_derived_results_trace_to_market_data_not_raw():
    tar = TechnicalAnalysisResult(
        request_id="req1",
        ticker_id="tk1",
        source_market_data_refs=["md1"],
        indicator_values={"rsi_14": 55.0},
        normalization_or_derivation_status=NormalizationStatus.SUCCESS,
    )
    assert tar.source_market_data_refs == ["md1"]
    # derived contract has no raw_response_id field at all
    assert "raw_response_id" not in TechnicalAnalysisResult.model_fields

    war = WaveAnalysisResult(
        request_id="req1",
        ticker_id="tk1",
        source_market_data_refs=["md1"],
        rule_refs=["rule_neowave_1"],
        rule_statuses={"rule_neowave_1": RuleStatus.NEEDS_HUMAN_REVIEW},
    )
    assert war.rule_statuses["rule_neowave_1"] is RuleStatus.NEEDS_HUMAN_REVIEW


def test_company_and_financial_profiles():
    cp = CompanyProfile(
        request_id="req1", ticker_id="tk1", raw_response_id="raw1",
        company_name="Acme Corp", normalization_status=NormalizationStatus.SUCCESS,
    )
    assert cp.sector is None
    fm = FinancialMetric(
        request_id="req1", ticker_id="tk1", raw_response_id="raw1",
        metric_name="revenue", metric_value=1234.5, period="FY2025",
        normalization_status=NormalizationStatus.SUCCESS,
    )
    assert fm.metric_value == 1234.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.providers.entities'`.

- [ ] **Step 3: Write the implementation**

`src/providers/entities.py`:

```python
"""Normalized provider contracts and internal derived analysis contracts (006).

Field names follow specs/006-provider-mcp-contracts/data-model.md exactly.
Normalized contracts (CompanyProfile/NewsEvent/FinancialMetric) trace to a
RawProviderResponse via raw_response_id. Derived contracts
(TechnicalAnalysisResult/WaveAnalysisResult) trace to normalized market data,
evidence, and rule refs — NOT to raw provider payloads.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from src.providers.enums import NormalizationStatus, RuleStatus, Warning, _Contract


class CompanyProfile(_Contract):
    company_profile_id: str | None = None
    request_id: str
    ticker_id: str
    raw_response_id: str
    company_name: str
    legal_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    exchange: str | None = None
    currency: str | None = None
    description: str | None = None
    normalization_status: NormalizationStatus
    warnings: list[Warning] = Field(default_factory=list)
    evidence_id: str | None = None


class NewsEvent(_Contract):
    news_event_id: str | None = None
    request_id: str
    ticker_id: str
    raw_response_id: str
    title: str
    summary: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    published_at: datetime | None = None
    collected_at: datetime | None = None
    event_type: str | None = None
    sentiment_label: str | None = None
    risk_tags: list[str] = Field(default_factory=list)
    normalization_status: NormalizationStatus
    warnings: list[Warning] = Field(default_factory=list)
    evidence_id: str | None = None


class FinancialMetric(_Contract):
    financial_metric_id: str | None = None
    request_id: str
    ticker_id: str
    raw_response_id: str
    metric_name: str
    metric_value: float | str | None = None
    period: str | None = None
    currency: str | None = None
    unit: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    collected_at: datetime | None = None
    normalization_status: NormalizationStatus
    warnings: list[Warning] = Field(default_factory=list)
    evidence_id: str | None = None


class TechnicalAnalysisResult(_Contract):
    technical_analysis_result_id: str | None = None
    request_id: str
    ticker_id: str
    source_market_data_refs: list[str] = Field(default_factory=list)
    indicator_values: dict[str, float] = Field(default_factory=dict)
    trend_state: str | None = None
    momentum_state: str | None = None
    volatility_state: str | None = None
    normalization_or_derivation_status: NormalizationStatus
    warnings: list[Warning] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class WaveAnalysisResult(_Contract):
    wave_analysis_result_id: str | None = None
    request_id: str
    ticker_id: str
    source_market_data_refs: list[str] = Field(default_factory=list)
    rule_refs: list[str] = Field(default_factory=list)
    candidate_summary: str | None = None
    rule_statuses: dict[str, RuleStatus] = Field(default_factory=dict)
    confirmation_conditions: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    uncertainty_notes: str | None = None
    warnings: list[Warning] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add src/providers/entities.py tests/test_provider_contracts.py
git commit -m "feat(006): normalized and derived entity contracts (T006)"
```

---

### Task 4: Provider interface Protocols + result models (T005)

**Files:**
- Create: `src/providers/interfaces.py`
- Test: `tests/test_provider_contracts.py` (extend)

**Interfaces:**
- Consumes: `_Contract`, `NormalizationStatus`, `Warning`, `ProviderError`, `RawResponseRef`, `MarketDataRef` from `enums.py`; `CompanyProfile`, `NewsEvent`, `FinancialMetric` from `entities.py`.
- Produces: result models `NewsProviderResult`, `FinancialProviderResult`, `MarketDataProviderResult`; Protocols `NewsProvider`, `FinancialProvider`, `MarketDataProvider`; input models `NewsProviderRequest`, `FinancialProviderRequest`, `MarketDataProviderRequest`.

- [ ] **Step 1: Write the failing test** (append)

```python
from typing import get_type_hints

from src.providers.interfaces import (
    FinancialProvider,
    FinancialProviderResult,
    MarketDataProvider,
    MarketDataProviderResult,
    NewsProvider,
    NewsProviderResult,
)


def test_provider_results_carry_normalized_objects_only():
    res = NewsProviderResult(
        raw_response_ref="raw1",
        normalization_status=NormalizationStatus.SUCCESS,
        news_events=[],
    )
    assert res.news_events == []
    assert res.warnings == []
    # result must not allow a raw payload field
    with pytest.raises(ValidationError):
        NewsProviderResult(
            raw_response_ref="raw1",
            normalization_status=NormalizationStatus.SUCCESS,
            news_events=[],
            payload_body={"x": 1},
        )


def test_protocols_are_runtime_checkable():
    class _FakeNews:
        def fetch_news(self, request):  # noqa: ANN001
            return NewsProviderResult(
                raw_response_ref="raw1",
                normalization_status=NormalizationStatus.SUCCESS,
                news_events=[],
            )

    assert isinstance(_FakeNews(), NewsProvider)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.providers.interfaces'`.

- [ ] **Step 3: Write the implementation**

`src/providers/interfaces.py`:

```python
"""Provider-agnostic interfaces (006).

Providers return normalized objects + lineage + status, never raw payloads.
These Protocols define the boundary that future MCP adapters must satisfy.
No adapter is implemented in this feature.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from src.providers.entities import CompanyProfile, FinancialMetric, NewsEvent
from src.providers.enums import (
    NormalizationStatus,
    ProviderError,
    Warning,
    _Contract,
)


class NewsProviderRequest(_Contract):
    ticker: str
    company_hint: str | None = None
    as_of_date: date | None = None
    max_results: int = 20


class FinancialProviderRequest(_Contract):
    ticker: str
    company_hint: str | None = None
    as_of_date: date | None = None
    requested_metrics: list[str] = []


class MarketDataProviderRequest(_Contract):
    ticker: str
    period: str | None = None
    interval: str | None = None
    as_of_date: date | None = None


class NewsProviderResult(_Contract):
    raw_response_ref: str
    normalization_status: NormalizationStatus
    news_events: list[NewsEvent] = []
    warnings: list[Warning] = []
    errors: list[ProviderError] = []


class FinancialProviderResult(_Contract):
    raw_response_ref: str
    normalization_status: NormalizationStatus
    company_profile: CompanyProfile | None = None
    financial_metrics: list[FinancialMetric] = []
    warnings: list[Warning] = []
    errors: list[ProviderError] = []


class MarketDataProviderResult(_Contract):
    raw_response_ref: str
    normalization_status: NormalizationStatus
    normalized_market_data_ref: str | None = None
    warnings: list[Warning] = []
    errors: list[ProviderError] = []


@runtime_checkable
class NewsProvider(Protocol):
    def fetch_news(self, request: NewsProviderRequest) -> NewsProviderResult: ...


@runtime_checkable
class FinancialProvider(Protocol):
    def fetch_financials(
        self, request: FinancialProviderRequest
    ) -> FinancialProviderResult: ...


@runtime_checkable
class MarketDataProvider(Protocol):
    def fetch_market_data(
        self, request: MarketDataProviderRequest
    ) -> MarketDataProviderResult: ...
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/providers/interfaces.py tests/test_provider_contracts.py
git commit -m "feat(006): provider interface protocols and result models (T005)"
```

---

### Task 5: Safety field contract (T048)

**Files:**
- Create: `src/providers/safety.py`
- Test: `tests/test_provider_contracts.py` (extend with focused unit tests; the full structural+instance sweep lands in T046)

**Interfaces:**
- Consumes: nothing from 006 yet (pure helper); imports the MVP contract classes lazily inside `SAFETY_CHECKED_CONTRACTS` assembly.
- Produces: `FORBIDDEN_TOKENS: frozenset[str]`, `FORBIDDEN_TOKEN_PHRASES: frozenset[tuple[str, ...]]`, `SAFETY_CHECKED_CONTRACTS: tuple[type[BaseModel], ...]`, `assert_no_trading_fields(model: type[BaseModel]) -> None`, and `find_trading_fields(model) -> list[str]`.

- [ ] **Step 1: Write the failing test** (append)

```python
from pydantic import BaseModel

from src.providers.safety import (
    FORBIDDEN_TOKENS,
    SAFETY_CHECKED_CONTRACTS,
    assert_no_trading_fields,
    find_trading_fields,
)


def test_token_matching_rejects_trading_field_names():
    class Bad(BaseModel):
        buy_signal: int = 0

    class Bad2(BaseModel):
        target_price: float = 0.0

    class Bad3(BaseModel):
        order_action: str = ""

    for m in (Bad, Bad2, Bad3):
        assert find_trading_fields(m), f"{m.__name__} should be flagged"
        with pytest.raises(ValueError):
            assert_no_trading_fields(m)


def test_substrings_do_not_false_positive():
    class Ok(BaseModel):
        threshold: float = 0.0   # contains "hold" as substring -> must NOT match
        household_count: int = 0  # token "household" != "hold"
        metric_name: str = ""     # value could be "price"; names are clean

    assert find_trading_fields(Ok) == []
    assert_no_trading_fields(Ok)  # no raise


def test_all_mvp_contracts_are_clean():
    assert len(SAFETY_CHECKED_CONTRACTS) >= 5
    for contract in SAFETY_CHECKED_CONTRACTS:
        assert_no_trading_fields(contract)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.providers.safety'`.

- [ ] **Step 3: Write the implementation**

`src/providers/safety.py`:

```python
"""No-trading-field safety contract (006), enforced structurally.

Matching is over snake_case TOKENS of field NAMES, never raw substrings, so
'threshold'/'household' never collide with 'hold'. Applies ONLY to this
feature's MVP contracts (SAFETY_CHECKED_CONTRACTS). Future Phase 2/3 simulation
contracts (SignalCandidate, StrategyRule, PaperTradingExecution) live in a
separate namespace with their own allowlist and are intentionally NOT scanned
here.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.providers.entities import (
    CompanyProfile,
    FinancialMetric,
    NewsEvent,
    TechnicalAnalysisResult,
    WaveAnalysisResult,
)
from src.providers.interfaces import (
    FinancialProviderResult,
    MarketDataProviderResult,
    NewsProviderResult,
)

FORBIDDEN_TOKENS: frozenset[str] = frozenset(
    {
        "buy",
        "sell",
        "hold",
        "order",
        "execute",
        "guaranteed",
        "recommend",
        "recommendation",
    }
)

FORBIDDEN_TOKEN_PHRASES: frozenset[tuple[str, ...]] = frozenset(
    {
        ("target", "price"),
        ("position", "size"),
        ("stop", "loss"),
        ("take", "profit"),
        ("guaranteed", "return"),
    }
)


def _tokens(field_name: str) -> list[str]:
    return [t for t in field_name.split("_") if t]


def _field_violates(field_name: str) -> bool:
    tokens = _tokens(field_name)
    if any(t in FORBIDDEN_TOKENS for t in tokens):
        return True
    for phrase in FORBIDDEN_TOKEN_PHRASES:
        n = len(phrase)
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i : i + n]) == phrase:
                return True
    return False


def find_trading_fields(model: type[BaseModel]) -> list[str]:
    """Return field names on `model` that violate the no-trading-field rule."""
    return [name for name in model.model_fields if _field_violates(name)]


def assert_no_trading_fields(model: type[BaseModel]) -> None:
    """Raise ValueError if `model` declares any trading-instruction field name."""
    violations = find_trading_fields(model)
    if violations:
        raise ValueError(
            f"{model.__name__} declares forbidden trading field(s): {violations}"
        )


SAFETY_CHECKED_CONTRACTS: tuple[type[BaseModel], ...] = (
    CompanyProfile,
    NewsEvent,
    FinancialMetric,
    TechnicalAnalysisResult,
    WaveAnalysisResult,
    NewsProviderResult,
    FinancialProviderResult,
    MarketDataProviderResult,
    # ScenarioReportInput is appended after it exists (T031); T046 asserts coverage.
)
```

> Note: `ScenarioReportInput` is added to `SAFETY_CHECKED_CONTRACTS` in T031's step, once the class exists, to avoid a circular import at Phase 2. T046 verifies the full set including it.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/providers/safety.py tests/test_provider_contracts.py
git commit -m "feat(006): token-based no-trading-field safety contract (T048)"
```

---

### Task 6: Normalization result containers + helper signatures (T007)

**Files:**
- Create: `src/providers/normalization.py`
- Test: `tests/test_provider_contracts.py` (extend with container shape test only; behavior in US1)

**Interfaces:**
- Consumes: `_Contract`, status/warning types from `enums.py`; entity contracts from `entities.py`.
- Produces: raw fixture shapes `RawNewsItem`, `RawCompanyPayload`, `RawFinancialRow`, `RawMarketData`; container `NormalizationResult` (generic-ish wrapper with `status`, `warnings`, `errors`, `records`); helper **signatures** `normalize_news(...)`, `normalize_company(...)`, `normalize_financials(...)`, `normalize_market_data(...)` raising `NotImplementedError` until US1.

- [ ] **Step 1: Write the failing test** (append)

```python
from src.providers.normalization import (
    NormalizationResult,
    RawNewsItem,
    normalize_news,
)


def test_normalization_result_container_shape():
    result = NormalizationResult(
        status=NormalizationStatus.SUCCESS, records=[], warnings=[], errors=[]
    )
    assert result.records == []


def test_normalize_news_signature_exists_but_unimplemented():
    # US1 (T012) implements the body; here we only assert the seam exists.
    with pytest.raises(NotImplementedError):
        normalize_news(
            raw_items=[RawNewsItem(headline="x")],
            request_id="req1",
            ticker_id="tk1",
            raw_response_id="raw1",
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.providers.normalization'`.

- [ ] **Step 3: Write the implementation**

`src/providers/normalization.py`:

```python
"""Raw-fixture shapes, normalization result containers, and helper seams (006).

Raw* models intentionally use LOOSE provider-specific field names (e.g. two
providers spell the same news field differently). Normalizers translate those
into the stable entity contracts. Bodies are filled in US1 (T012-T014).
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from src.providers.entities import (
    CompanyProfile,
    FinancialMetric,
    NewsEvent,
)
from src.providers.enums import (
    NormalizationStatus,
    ProviderError,
    Warning,
    _Contract,
)


# --- Raw provider shapes (deliberately permissive: extra allowed) -----------
class _Raw(_Contract):
    # Raw payloads vary by provider; allow unknown keys so fixtures stay honest.
    model_config = ConfigDict(extra="allow")


class RawNewsItem(_Raw):
    # Provider A: {title, content, url}; Provider B: {headline, summary_text, source_url}
    title: str | None = None
    headline: str | None = None
    content: str | None = None
    summary_text: str | None = None
    url: str | None = None
    source_url: str | None = None
    source: str | None = None
    published: str | None = None


class RawCompanyPayload(_Raw):
    name: str | None = None
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    exchange: str | None = None
    currency: str | None = None
    about: str | None = None


class RawFinancialRow(_Raw):
    name: str | None = None
    metric: str | None = None
    value: Any = None
    period: str | None = None
    currency: str | None = None
    unit: str | None = None


class RawMarketData(_Raw):
    ticker: str | None = None
    candles: list[dict[str, Any]] = Field(default_factory=list)


# --- Normalization result container -----------------------------------------
class NormalizationResult(_Contract):
    status: NormalizationStatus
    records: list[Any] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)
    errors: list[ProviderError] = Field(default_factory=list)
    normalized_market_data_ref: str | None = None


# --- Helper seams (implemented in US1) --------------------------------------
def normalize_news(
    *, raw_items: list[RawNewsItem], request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    raise NotImplementedError("Implemented in US1 / T012")


def normalize_company(
    *, raw: RawCompanyPayload, request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    raise NotImplementedError("Implemented in US1 / T013")


def normalize_financials(
    *, raw_rows: list[RawFinancialRow], request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    raise NotImplementedError("Implemented in US1 / T013")


def normalize_market_data(
    *, raw: RawMarketData, request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    raise NotImplementedError("Implemented in US1 / T014")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/providers/normalization.py tests/test_provider_contracts.py
git commit -m "feat(006): normalization result containers and helper seams (T007)"
```

---

### Task 7: Stable public exports (T008)

**Files:**
- Modify: `src/providers/__init__.py`
- Test: `tests/test_provider_contracts.py` (extend)

**Interfaces:**
- Produces: top-level re-exports so consumers do `from src.providers import NewsEvent, normalize_news, assert_no_trading_fields, ...`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_public_exports_are_stable():
    import src.providers as p

    for name in (
        "CompanyProfile", "NewsEvent", "FinancialMetric",
        "TechnicalAnalysisResult", "WaveAnalysisResult",
        "NewsProvider", "FinancialProvider", "MarketDataProvider",
        "NormalizationStatus", "DegradationStatus",
        "normalize_news", "normalize_company", "normalize_financials",
        "normalize_market_data",
        "assert_no_trading_fields", "SAFETY_CHECKED_CONTRACTS",
    ):
        assert hasattr(p, name), f"missing export: {name}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_provider_contracts.py::test_public_exports_are_stable -q`
Expected: FAIL — `AssertionError: missing export: CompanyProfile`.

- [ ] **Step 3: Write the implementation** — replace `src/providers/__init__.py` body with the docstring (kept) plus:

```python
from src.providers.entities import (
    CompanyProfile,
    FinancialMetric,
    NewsEvent,
    TechnicalAnalysisResult,
    WaveAnalysisResult,
)
from src.providers.enums import (
    DegradationStatus,
    NormalizationStatus,
    ProviderError,
    ProviderKind,
    RuleStatus,
    Warning,
)
from src.providers.interfaces import (
    FinancialProvider,
    FinancialProviderResult,
    MarketDataProvider,
    MarketDataProviderResult,
    NewsProvider,
    NewsProviderResult,
)
from src.providers.normalization import (
    NormalizationResult,
    normalize_company,
    normalize_financials,
    normalize_market_data,
    normalize_news,
)
from src.providers.safety import (
    FORBIDDEN_TOKEN_PHRASES,
    FORBIDDEN_TOKENS,
    SAFETY_CHECKED_CONTRACTS,
    assert_no_trading_fields,
    find_trading_fields,
)

__all__ = [
    "CompanyProfile", "FinancialMetric", "NewsEvent",
    "TechnicalAnalysisResult", "WaveAnalysisResult",
    "DegradationStatus", "NormalizationStatus", "ProviderError",
    "ProviderKind", "RuleStatus", "Warning",
    "FinancialProvider", "FinancialProviderResult",
    "MarketDataProvider", "MarketDataProviderResult",
    "NewsProvider", "NewsProviderResult",
    "NormalizationResult", "normalize_company", "normalize_financials",
    "normalize_market_data", "normalize_news",
    "FORBIDDEN_TOKEN_PHRASES", "FORBIDDEN_TOKENS",
    "SAFETY_CHECKED_CONTRACTS", "assert_no_trading_fields", "find_trading_fields",
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_provider_contracts.py -q`
Expected: PASS.

- [ ] **Step 5 (Phase-2 checkpoint): Verify the package imports with zero DB/graph/provider side effects**

Run: `uv run python -c "import src.providers as p; print(sorted(p.__all__)[:3])"`
Expected: prints a list, no DB/graph import errors.
Run: `uv run python -m compileall src/providers`
Expected: compiles clean.

- [ ] **Step 6: Commit**

```bash
git add src/providers/__init__.py tests/test_provider_contracts.py
git commit -m "feat(006): stable public provider contract exports (T008)"
```

**Checkpoint:** Provider contract primitives exist and import without touching DB, graph, or live providers. ✅

---

# Phase 3: User Story 1 — Normalize Provider Data For Agents (P1, MVP)

**Independent test:** two raw fixtures with different shapes for the same ticker → identical normalized object types, no provider-specific raw fields exposed.

> Test tasks T009/T010/T011 share `tests/test_provider_contracts.py` and are sequenced by one implementer. Implementation T012/T013/T014 share `src/providers/normalization.py` (sequence them). Agent integration T015/T016/T017 touch separate agent files.

### Task 8: Fixtures for two raw news shapes + normalization tests (T009)

**Files:**
- Modify: `tests/fixtures/provider_contracts.py`
- Test: `tests/test_provider_contracts.py`

**Interfaces:**
- Produces (fixtures): `raw_news_provider_a() -> list[RawNewsItem]`, `raw_news_provider_b() -> list[RawNewsItem]` — equivalent content, different field spellings.

- [ ] **Step 1: Add the fixtures** to `tests/fixtures/provider_contracts.py`:

```python
from src.providers.normalization import RawNewsItem


def raw_news_provider_a() -> list[RawNewsItem]:
    """Provider A spelling: {title, content, url}."""
    return [
        RawNewsItem(
            title="Acme beats Q2 earnings",
            content="Acme reported revenue above consensus.",
            url="https://news.example.com/acme-q2",
            source="Example Wire",
            published="2026-06-01T13:00:00Z",
        )
    ]


def raw_news_provider_b() -> list[RawNewsItem]:
    """Provider B spelling: {headline, summary_text, source_url}."""
    return [
        RawNewsItem(
            headline="Acme beats Q2 earnings",
            summary_text="Acme reported revenue above consensus.",
            source_url="https://news.example.com/acme-q2",
            source="Example Wire",
            published="2026-06-01T13:00:00Z",
        )
    ]
```

- [ ] **Step 2: Write the failing test** (append to `tests/test_provider_contracts.py`)

```python
from src.providers.normalization import normalize_company, normalize_financials, normalize_market_data
from tests.fixtures.provider_contracts import raw_news_provider_a, raw_news_provider_b


def test_equivalent_news_shapes_normalize_identically():
    common = dict(request_id="req1", ticker_id="tk1", raw_response_id="raw1")
    res_a = normalize_news(raw_items=raw_news_provider_a(), **common)
    res_b = normalize_news(raw_items=raw_news_provider_b(), **common)

    assert res_a.status == NormalizationStatus.SUCCESS
    assert res_b.status == NormalizationStatus.SUCCESS
    assert [type(r) for r in res_a.records] == [NewsEvent]
    assert [type(r) for r in res_b.records] == [NewsEvent]

    a, b = res_a.records[0], res_b.records[0]
    assert a.title == b.title == "Acme beats Q2 earnings"
    assert a.summary == b.summary
    assert a.source_url == b.source_url
    # no provider-specific raw field leaks onto the contract:
    assert "content" not in NewsEvent.model_fields
    assert "headline" not in NewsEvent.model_fields
```

- [ ] **Step 3: Run to verify it fails** — Run: `uv run pytest tests/test_provider_contracts.py::test_equivalent_news_shapes_normalize_identically -q` → FAIL with `NotImplementedError` (from T007 seam).

- [ ] **Step 4: (implementation is T012 below — proceed there, then re-run)**

- [ ] **Step 5: Commit fixtures + test** (red state intentionally — implementation lands next task; commit together with T012 if you prefer a green-only history). Recommended: commit fixtures now, leave test for the T012 commit.

```bash
git add tests/fixtures/provider_contracts.py
git commit -m "test(006): add two-shape raw news fixtures (T009)"
```

### Task 9: Implement news normalization (T012)

**Files:** Modify: `src/providers/normalization.py`

- [ ] **Step 1: Implement `normalize_news`** — replace the `NotImplementedError` body:

```python
from datetime import datetime


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_news(
    *, raw_items: list[RawNewsItem], request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    records: list[NewsEvent] = []
    warnings: list[Warning] = []
    for item in raw_items:
        title = item.title or item.headline
        summary = item.content or item.summary_text
        source_url = item.url or item.source_url
        status = NormalizationStatus.SUCCESS
        if title is None:
            # cannot build a meaningful event without a title
            warnings.append(Warning(code="missing_title", message="news item missing title"))
            status = NormalizationStatus.PARTIAL_SUCCESS
            continue
        if source_url is None:
            warnings.append(Warning(code="missing_url", message="news item missing source url", field="source_url"))
            status = NormalizationStatus.PARTIAL_SUCCESS
        records.append(
            NewsEvent(
                request_id=request_id,
                ticker_id=ticker_id,
                raw_response_id=raw_response_id,
                title=title,
                summary=summary,
                source_name=item.source,
                source_url=source_url,
                published_at=_parse_dt(item.published),
                normalization_status=status,
                warnings=[w for w in warnings if w.field == "source_url"],
            )
        )
    overall = NormalizationStatus.SUCCESS if not warnings else NormalizationStatus.PARTIAL_SUCCESS
    if not records:
        overall = NormalizationStatus.INSUFFICIENT_DATA
    return NormalizationResult(status=overall, records=records, warnings=warnings)
```

- [ ] **Step 2: Run T009's test** — Run: `uv run pytest tests/test_provider_contracts.py::test_equivalent_news_shapes_normalize_identically -q` → PASS.

- [ ] **Step 3: Commit**

```bash
git add src/providers/normalization.py tests/test_provider_contracts.py
git commit -m "feat(006): implement raw news -> NewsEvent normalization (T012)"
```

### Task 10: Company/financial/market fixtures + boundary tests (T010) and implementations (T013, T014)

**Files:** Modify `tests/fixtures/provider_contracts.py`, `tests/test_provider_contracts.py`, `src/providers/normalization.py`.

- [ ] **Step 1: Add fixtures** to `tests/fixtures/provider_contracts.py`:

```python
from src.providers.normalization import RawCompanyPayload, RawFinancialRow, RawMarketData


def raw_company_payload() -> RawCompanyPayload:
    return RawCompanyPayload(
        name="Acme Corp", sector="Technology", industry="Software",
        country="US", exchange="NASDAQ", currency="USD", about="Maker of widgets.",
    )


def raw_financial_rows() -> list[RawFinancialRow]:
    return [
        RawFinancialRow(name="revenue", value=1234.5, period="FY2025", currency="USD", unit="millions"),
        RawFinancialRow(metric="net_income", value=210.0, period="FY2025", currency="USD", unit="millions"),
    ]


def raw_market_data() -> RawMarketData:
    return RawMarketData(
        ticker="ACME",
        candles=[{"t": "2026-06-01", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 1000}],
    )
```

- [ ] **Step 2: Write the failing test** (append):

```python
from src.providers.entities import CompanyProfile, FinancialMetric
from tests.fixtures.provider_contracts import (
    raw_company_payload, raw_financial_rows, raw_market_data,
)


def test_company_and_financial_boundary_preserved():
    common = dict(request_id="req1", ticker_id="tk1", raw_response_id="raw1")
    cres = normalize_company(raw=raw_company_payload(), **common)
    fres = normalize_financials(raw_rows=raw_financial_rows(), **common)

    assert [type(r) for r in cres.records] == [CompanyProfile]
    assert cres.records[0].company_name == "Acme Corp"
    assert all(type(r) is FinancialMetric for r in fres.records)
    assert {r.metric_name for r in fres.records} == {"revenue", "net_income"}
    # all normalized records trace to the raw response:
    assert cres.records[0].raw_response_id == "raw1"
    assert all(r.raw_response_id == "raw1" for r in fres.records)


def test_market_data_normalizes_to_reference_not_derived_results():
    res = normalize_market_data(
        raw=raw_market_data(), request_id="req1", ticker_id="tk1", raw_response_id="raw1",
    )
    assert res.status == NormalizationStatus.SUCCESS
    assert res.normalized_market_data_ref is not None
    # market normalization MUST NOT emit technical/wave results
    assert res.records == []
```

- [ ] **Step 3: Run to verify it fails** — `NotImplementedError`.

- [ ] **Step 4: Implement `normalize_company`, `normalize_financials`, `normalize_market_data`** in `src/providers/normalization.py`:

```python
def normalize_company(
    *, raw: RawCompanyPayload, request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    name = raw.company_name or raw.name
    if not name:
        return NormalizationResult(
            status=NormalizationStatus.INSUFFICIENT_DATA,
            warnings=[Warning(code="missing_company_name", message="no company name in payload")],
        )
    profile = CompanyProfile(
        request_id=request_id, ticker_id=ticker_id, raw_response_id=raw_response_id,
        company_name=name, sector=raw.sector, industry=raw.industry,
        country=raw.country, exchange=raw.exchange, currency=raw.currency,
        description=raw.about, normalization_status=NormalizationStatus.SUCCESS,
    )
    return NormalizationResult(status=NormalizationStatus.SUCCESS, records=[profile])


def normalize_financials(
    *, raw_rows: list[RawFinancialRow], request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    records: list[FinancialMetric] = []
    warnings: list[Warning] = []
    for row in raw_rows:
        metric_name = row.metric or row.name
        if not metric_name:
            warnings.append(Warning(code="missing_metric_name", message="financial row missing metric name"))
            continue
        value = row.value
        records.append(
            FinancialMetric(
                request_id=request_id, ticker_id=ticker_id, raw_response_id=raw_response_id,
                metric_name=metric_name,
                metric_value=value if isinstance(value, (int, float, str)) or value is None else str(value),
                period=row.period, currency=row.currency, unit=row.unit,
                normalization_status=NormalizationStatus.SUCCESS,
            )
        )
    status = NormalizationStatus.SUCCESS if records and not warnings else (
        NormalizationStatus.PARTIAL_SUCCESS if records else NormalizationStatus.INSUFFICIENT_DATA
    )
    return NormalizationResult(status=status, records=records, warnings=warnings)


def normalize_market_data(
    *, raw: RawMarketData, request_id: str, ticker_id: str, raw_response_id: str
) -> NormalizationResult:
    if not raw.candles:
        return NormalizationResult(
            status=NormalizationStatus.INSUFFICIENT_DATA,
            warnings=[Warning(code="no_candles", message="market data has no candles")],
        )
    # A deterministic, content-addressable reference. Real market-data storage
    # is 004/future work; 006 only emits a normalized reference handle.
    ref = f"md::{request_id}::{ticker_id}::{raw_response_id}"
    return NormalizationResult(
        status=NormalizationStatus.SUCCESS, records=[], normalized_market_data_ref=ref,
    )
```

- [ ] **Step 5: Run to verify it passes** — `uv run pytest tests/test_provider_contracts.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/provider_contracts.py tests/test_provider_contracts.py src/providers/normalization.py
git commit -m "feat(006): company/financial/market normalization preserving provider boundary (T010,T013,T014)"
```

### Task 11: Degradation/status tests (T011)

**Files:** Test: `tests/test_provider_contracts.py`

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_failed_news_yields_insufficient_data():
    res = normalize_news(
        raw_items=[RawNewsItem(content="no title here")],
        request_id="req1", ticker_id="tk1", raw_response_id="raw1",
    )
    assert res.status == NormalizationStatus.INSUFFICIENT_DATA
    assert res.records == []
    assert any(w.code == "missing_title" for w in res.warnings)


def test_partial_news_missing_url_warns():
    res = normalize_news(
        raw_items=[RawNewsItem(title="Acme update")],
        request_id="req1", ticker_id="tk1", raw_response_id="raw1",
    )
    assert res.status == NormalizationStatus.PARTIAL_SUCCESS
    assert res.records[0].normalization_status == NormalizationStatus.PARTIAL_SUCCESS
    assert any(w.code == "missing_url" for w in res.warnings)


def test_empty_market_data_insufficient():
    res = normalize_market_data(
        raw=RawMarketData(candles=[]), request_id="req1", ticker_id="tk1", raw_response_id="raw1",
    )
    assert res.status == NormalizationStatus.INSUFFICIENT_DATA
```

- [ ] **Step 2: Run** — `uv run pytest tests/test_provider_contracts.py -q`. If any fails, adjust normalizer status logic (these assertions are the spec for FR-010/FR-015). Expected after green: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_provider_contracts.py
git commit -m "test(006): provider failure/partial/insufficient status coverage (T011)"
```

### Task 12: Agent boundary functions (T015, T016, T017)

> **Additive, not destructive.** Real MCP wiring is out of scope; existing workflow nodes stay intact. Each agent module gets a typed boundary function that accepts normalized contracts and exposes them to the agent's analysis without referencing raw provider fields.

**Files:** Modify `src/agents/news_agent.py`, `src/agents/fundamental_agent.py`, `src/agents/market_agent.py`; Test: `tests/test_provider_contracts.py`.

- [ ] **Step 1: Write the failing test** (append):

```python
from src.agents.news_agent import news_events_to_agent_input
from src.agents.fundamental_agent import fundamentals_to_agent_input
from src.agents.market_agent import market_inputs_to_agent_input
from src.providers.entities import TechnicalAnalysisResult


def test_news_agent_boundary_consumes_contracts_only():
    res = normalize_news(
        raw_items=raw_news_provider_a(), request_id="req1", ticker_id="tk1", raw_response_id="raw1",
    )
    agent_input = news_events_to_agent_input(res.records)
    assert agent_input["count"] == 1
    assert agent_input["titles"] == ["Acme beats Q2 earnings"]
    # no raw provider keys present
    assert "content" not in agent_input and "headline" not in agent_input


def test_fundamental_agent_boundary():
    common = dict(request_id="req1", ticker_id="tk1", raw_response_id="raw1")
    cp = normalize_company(raw=raw_company_payload(), **common).records[0]
    metrics = normalize_financials(raw_rows=raw_financial_rows(), **common).records
    agent_input = fundamentals_to_agent_input(cp, metrics)
    assert agent_input["company_name"] == "Acme Corp"
    assert agent_input["metrics"]["revenue"] == 1234.5


def test_market_agent_keeps_provider_data_separate_from_derived():
    md_ref = "md::req1::tk1::raw1"
    derived = [TechnicalAnalysisResult(
        request_id="req1", ticker_id="tk1", source_market_data_refs=[md_ref],
        indicator_values={"rsi_14": 55.0},
        normalization_or_derivation_status=NormalizationStatus.SUCCESS,
    )]
    agent_input = market_inputs_to_agent_input(market_data_ref=md_ref, technical_results=derived)
    assert agent_input["market_data_ref"] == md_ref
    assert agent_input["technical"][0]["indicator_values"]["rsi_14"] == 55.0
```

- [ ] **Step 2: Run to verify it fails** — `ImportError` for the new functions.

- [ ] **Step 3: Implement boundary functions.** Append to `src/agents/news_agent.py`:

```python
from src.providers.entities import NewsEvent as ContractNewsEvent


def news_events_to_agent_input(events: list[ContractNewsEvent]) -> dict:
    """Adapt normalized NewsEvent contracts into NewsAgent's analysis input.

    Reads only contract fields — never provider-specific raw payload keys.
    """
    return {
        "count": len(events),
        "titles": [e.title for e in events],
        "summaries": [e.summary for e in events if e.summary],
        "risk_tags": sorted({t for e in events for t in e.risk_tags}),
    }
```

Append to `src/agents/fundamental_agent.py`:

```python
from src.providers.entities import CompanyProfile, FinancialMetric


def fundamentals_to_agent_input(
    profile: CompanyProfile, metrics: list[FinancialMetric]
) -> dict:
    """Adapt normalized company/financial contracts into FundamentalAgent input."""
    return {
        "company_name": profile.company_name,
        "sector": profile.sector,
        "industry": profile.industry,
        "metrics": {m.metric_name: m.metric_value for m in metrics},
    }
```

Append to `src/agents/market_agent.py`:

```python
from src.providers.entities import TechnicalAnalysisResult


def market_inputs_to_agent_input(
    *, market_data_ref: str | None, technical_results: list[TechnicalAnalysisResult]
) -> dict:
    """Keep provider market-data reference separate from derived technical results."""
    return {
        "market_data_ref": market_data_ref,
        "technical": [
            {
                "indicator_values": t.indicator_values,
                "trend_state": t.trend_state,
                "momentum_state": t.momentum_state,
                "volatility_state": t.volatility_state,
            }
            for t in technical_results
        ],
    }
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_provider_contracts.py -q` → PASS. Also run the existing agent suites to confirm no regression:
Run: `uv run pytest tests/test_workflow_e2e.py tests/test_coordinator_agent.py -q`
Expected: PASS (the additions are new functions; existing nodes untouched).

- [ ] **Step 5: Commit**

```bash
git add src/agents/news_agent.py src/agents/fundamental_agent.py src/agents/market_agent.py tests/test_provider_contracts.py
git commit -m "feat(006): agent boundary functions consuming normalized contracts (T015-T017)"
```

**US1 Checkpoint:** Run `uv run pytest tests/test_provider_contracts.py -q` → all pass, no live calls. ✅

---

# Phase 4: User Story 2 — Trace Raw And Normalized Persistence (P2)

> 004 owns canonical tables. These are **explicit 004-compatible extension** tables for raw/normalized/derived provider records (FR-004…FR-009, FR-021). Tests require `TEST_DATABASE_URL`; without it they skip (conftest behavior).

### Task 13: Persistence lineage tests (T018, T019, T020)

**Files:** Create `tests/test_provider_persistence_contracts.py`.

**Interfaces:**
- Consumes (from T021-T026): `ProviderRepository(session)` with methods `create_raw_response(...) -> RawProviderResponseRow`, `create_company_profile(...)`, `create_news_event(...)`, `create_financial_metric(...)`, `create_technical_result(...)`, `create_wave_result(...)`, and read/lineage methods `get_normalized_for_raw(raw_response_id)`.

- [ ] **Step 1: Write the failing tests:**

```python
"""US2 persistence lineage tests. Skips without TEST_DATABASE_URL (conftest)."""

import pytest

from src.db.repositories.provider_repository import ProviderRepository
from src.providers.enums import NormalizationStatus


@pytest.fixture()
def repo(db_session):
    return ProviderRepository(db_session)


@pytest.fixture()
def request_and_ticker(db_session):
    from src.db.repositories.analysis_repository import AnalysisRepository
    ar = AnalysisRepository(db_session)
    ticker = ar.upsert_ticker("ACME", market="US")
    db_session.flush()
    req = ar.create_request(ticker_id=ticker.id, request_type="research", status="pending")
    db_session.flush()
    return req.id, ticker.id


def test_raw_response_persists_and_normalized_records_trace_back(repo, request_and_ticker):
    request_id, ticker_id = request_and_ticker
    raw = repo.create_raw_response(
        request_id=request_id, ticker_id=ticker_id, provider_name="provider_a",
        provider_kind="news", status="success", payload_body={"items": []},
    )
    repo.session.flush()
    ev = repo.create_news_event(
        request_id=request_id, ticker_id=ticker_id, raw_response_id=raw.id,
        title="Acme beats earnings", normalization_status=NormalizationStatus.SUCCESS.value,
    )
    repo.session.flush()
    assert ev.raw_response_id == raw.id
    linked = repo.get_normalized_for_raw(raw.id)
    assert ev.id in {n.id for n in linked["news_events"]}


def test_derived_results_trace_to_market_and_evidence_not_raw(repo, request_and_ticker):
    request_id, ticker_id = request_and_ticker
    tar = repo.create_technical_result(
        request_id=request_id, ticker_id=ticker_id,
        source_market_data_refs=["md::1"], indicator_values={"rsi_14": 55.0},
        status=NormalizationStatus.SUCCESS.value, evidence_ids=["ev1"],
    )
    repo.session.flush()
    assert tar.source_market_data_refs == ["md::1"]
    # derived row has no raw_response_id column
    assert not hasattr(tar, "raw_response_id")


def test_partial_and_failed_normalization_recoverable(repo, request_and_ticker):
    request_id, ticker_id = request_and_ticker
    raw = repo.create_raw_response(
        request_id=request_id, ticker_id=ticker_id, provider_name="provider_x",
        provider_kind="financial", status="failed", error_message="rate limited",
    )
    repo.session.flush()
    assert raw.status == "failed"
    assert raw.error_message == "rate limited"
    # partial normalized record still persists with warnings
    fm = repo.create_financial_metric(
        request_id=request_id, ticker_id=ticker_id, raw_response_id=raw.id,
        metric_name="revenue", normalization_status=NormalizationStatus.PARTIAL_SUCCESS.value,
        warnings=[{"code": "missing_period", "message": "no period"}],
    )
    repo.session.flush()
    assert fm.normalization_status == "partial_success"
    assert fm.warnings[0]["code"] == "missing_period"
```

- [ ] **Step 2: Run to verify failure** — Run: `uv run pytest tests/test_provider_persistence_contracts.py -q`
Expected: collection error / `ModuleNotFoundError: provider_repository` (or skip if `TEST_DATABASE_URL` unset — set it first: `export TEST_DATABASE_URL=postgresql+psycopg://...`). To force a real failure locally, ensure the DB URL is set.

- [ ] **Step 3: Commit (red test, paired with next tasks)** — or hold and commit with T024. Recommended hold.

### Task 14: Extension ORM models (T021, T022)

**Files:** Modify `src/db/models.py`.

**Interfaces:**
- Produces ORM models: `RawProviderResponse`, `ProviderCompanyProfile`, `ProviderNewsEvent`, `ProviderFinancialMetric`, `ProviderTechnicalAnalysisResult`, `ProviderWaveAnalysisResult`. Prefix `Provider*` on table names that risk colliding with 004/005 tables (e.g. `wave_scenarios` already exists → use `provider_wave_analysis_results`).

- [ ] **Step 1: Add models** to `src/db/models.py` (after existing models, reuse `UUIDMixin`/`TimestampMixin`/`Base`):

```python
class RawProviderResponse(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "raw_provider_responses"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    provider_name: Mapped[str] = mapped_column(String, nullable=False)
    provider_kind: Mapped[str] = mapped_column(String, nullable=False)  # news/financial/market_data
    provider_request: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    collected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    payload_body: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    payload_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class ProviderCompanyProfile(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "provider_company_profiles"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    raw_response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("raw_provider_responses.id"), nullable=False)
    company_name: Mapped[str] = mapped_column(String, nullable=False)
    legal_name: Mapped[str | None] = mapped_column(String, nullable=True)
    sector: Mapped[str | None] = mapped_column(String, nullable=True)
    industry: Mapped[str | None] = mapped_column(String, nullable=True)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    exchange: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalization_status: Mapped[str] = mapped_column(String, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    evidence_id: Mapped[str | None] = mapped_column(String, nullable=True)


class ProviderNewsEvent(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "provider_news_events"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    raw_response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("raw_provider_responses.id"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_name: Mapped[str | None] = mapped_column(String, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    event_type: Mapped[str | None] = mapped_column(String, nullable=True)
    sentiment_label: Mapped[str | None] = mapped_column(String, nullable=True)
    risk_tags: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    normalization_status: Mapped[str] = mapped_column(String, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    evidence_id: Mapped[str | None] = mapped_column(String, nullable=True)


class ProviderFinancialMetric(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "provider_financial_metrics"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    raw_response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("raw_provider_responses.id"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String, nullable=False)
    metric_value: Mapped[str | None] = mapped_column(String, nullable=True)  # stringified; numeric kept in JSONB metadata if needed
    period: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str | None] = mapped_column(String, nullable=True)
    unit: Mapped[str | None] = mapped_column(String, nullable=True)
    source_name: Mapped[str | None] = mapped_column(String, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    normalization_status: Mapped[str] = mapped_column(String, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    evidence_id: Mapped[str | None] = mapped_column(String, nullable=True)


class ProviderTechnicalAnalysisResult(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "provider_technical_analysis_results"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    source_market_data_refs: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    indicator_values: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    trend_state: Mapped[str | None] = mapped_column(String, nullable=True)
    momentum_state: Mapped[str | None] = mapped_column(String, nullable=True)
    volatility_state: Mapped[str | None] = mapped_column(String, nullable=True)
    normalization_status: Mapped[str] = mapped_column(String, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    evidence_ids: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)


class ProviderWaveAnalysisResult(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "provider_wave_analysis_results"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    source_market_data_refs: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    rule_refs: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    candidate_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    rule_statuses: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    confirmation_conditions: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    invalidation_conditions: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    uncertainty_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    evidence_ids: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
```

- [ ] **Step 2: Verify models import** — Run: `uv run python -c "from src.db import models; print(models.RawProviderResponse.__tablename__)"` → prints `raw_provider_responses`.

- [ ] **Step 3: Commit**

```bash
git add src/db/models.py
git commit -m "feat(006): extension ORM models for provider contract records (T021,T022)"
```

### Task 15: Alembic migration (T023)

**Files:** Create `src/db/migrations/versions/8f1a2b3c4d5e_provider_contract_records.py`.

- [ ] **Step 1: Generate autogenerate migration** (preferred — keeps it in sync with models):

Run: `uv run alembic revision --autogenerate -m "provider contract records" --rev-id 8f1a2b3c4d5e`
Then **rename/verify** the produced file is `8f1a2b3c4d5e_provider_contract_records.py` and that its header reads:

```python
revision = '8f1a2b3c4d5e'
down_revision = '56464a69bd55'
branch_labels = None
depends_on = None
```

> If `TEST_DATABASE_URL`/dev DB is unavailable for autogenerate, hand-write the migration creating the six tables above with `op.create_table(...)` using `op.f('pk_...')`/`op.f('fk_...')` naming (mirror `d305f4d1ff77`'s style), and a matching `downgrade()` that drops them in reverse FK order (children before `raw_provider_responses`).

- [ ] **Step 2: Verify the chain has a single head**

Run: `uv run alembic heads`
Expected: exactly one head, `8f1a2b3c4d5e`.

- [ ] **Step 3: Apply against the test DB**

Run: `export TEST_DATABASE_URL=postgresql+psycopg://...; uv run alembic upgrade head`
Expected: creates the six tables, no error. (Confirm downgrade too: `uv run alembic downgrade -1` then `upgrade head`.)

- [ ] **Step 4: Commit**

```bash
git add src/db/migrations/versions/8f1a2b3c4d5e_provider_contract_records.py
git commit -m "feat(006): alembic migration for provider contract records (T023)"
```

### Task 16: ProviderRepository + export (T024, T025)

**Files:** Create `src/db/repositories/provider_repository.py`; Modify `src/db/repositories/__init__.py`.

- [ ] **Step 1: Implement the repository:**

```python
"""Repository for provider raw/normalized/derived contract records (006)."""

from __future__ import annotations

from src.db.models import (
    ProviderCompanyProfile,
    ProviderFinancialMetric,
    ProviderNewsEvent,
    ProviderTechnicalAnalysisResult,
    ProviderWaveAnalysisResult,
    RawProviderResponse,
)
from src.db.repositories.base import BaseRepository


class ProviderRepository(BaseRepository):
    def create_raw_response(self, **fields) -> RawProviderResponse:
        row = RawProviderResponse(**fields)
        self.session.add(row)
        return row

    def create_company_profile(self, **fields) -> ProviderCompanyProfile:
        row = ProviderCompanyProfile(**fields)
        self.session.add(row)
        return row

    def create_news_event(self, **fields) -> ProviderNewsEvent:
        row = ProviderNewsEvent(**fields)
        self.session.add(row)
        return row

    def create_financial_metric(self, **fields) -> ProviderFinancialMetric:
        row = ProviderFinancialMetric(**fields)
        self.session.add(row)
        return row

    def create_technical_result(self, **fields) -> ProviderTechnicalAnalysisResult:
        row = ProviderTechnicalAnalysisResult(**fields)
        self.session.add(row)
        return row

    def create_wave_result(self, **fields) -> ProviderWaveAnalysisResult:
        row = ProviderWaveAnalysisResult(**fields)
        self.session.add(row)
        return row

    def get_normalized_for_raw(self, raw_response_id) -> dict:
        """Return all normalized records that trace to a raw response."""
        q = self.session.query
        return {
            "company_profiles": q(ProviderCompanyProfile)
            .filter(ProviderCompanyProfile.raw_response_id == raw_response_id).all(),
            "news_events": q(ProviderNewsEvent)
            .filter(ProviderNewsEvent.raw_response_id == raw_response_id).all(),
            "financial_metrics": q(ProviderFinancialMetric)
            .filter(ProviderFinancialMetric.raw_response_id == raw_response_id).all(),
        }
```

- [ ] **Step 2: Export it.** Read `src/db/repositories/__init__.py`, then add:

```python
from src.db.repositories.provider_repository import ProviderRepository  # noqa: F401
```
(and add `"ProviderRepository"` to `__all__` if the file defines one.)

- [ ] **Step 3: Run the US2 persistence tests** (now unskipped with `TEST_DATABASE_URL`):

Run: `export TEST_DATABASE_URL=postgresql+psycopg://...; uv run pytest tests/test_provider_persistence_contracts.py -q`
Expected: PASS (3 tests). Without the env var they skip cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/db/repositories/provider_repository.py src/db/repositories/__init__.py tests/test_provider_persistence_contracts.py
git commit -m "feat(006): ProviderRepository lineage create/read + export (T024,T025)"
```

### Task 17: Persistence orchestration helper (T026)

**Files:** Modify `src/db/persistence.py`; Test: extend `tests/test_provider_persistence_contracts.py`.

**Interfaces:**
- Produces: `persist_normalization(session, *, request_id, ticker_id, raw_kwargs, normalization_result) -> dict` that writes the raw response then the normalized records from a `NormalizationResult`, preserving 004 ownership (only writes 006 extension tables; never mutates 004 tables).

- [ ] **Step 1: Write the failing test** (append):

```python
def test_persist_normalization_writes_raw_then_normalized(db_session, request_and_ticker):
    from src.db.persistence import persist_normalization
    from src.providers.normalization import normalize_news
    from tests.fixtures.provider_contracts import raw_news_provider_a

    request_id, ticker_id = request_and_ticker
    res = normalize_news(
        raw_items=raw_news_provider_a(), request_id=str(request_id),
        ticker_id=str(ticker_id), raw_response_id="placeholder",
    )
    out = persist_normalization(
        db_session, request_id=request_id, ticker_id=ticker_id,
        raw_kwargs=dict(provider_name="provider_a", provider_kind="news", status="success"),
        normalization_result=res,
    )
    db_session.flush()
    assert out["raw_response_id"] is not None
    assert len(out["news_events"]) == 1
    assert out["news_events"][0].raw_response_id == out["raw_response_id"]
```

- [ ] **Step 2: Run to verify failure** — `ImportError: cannot import name 'persist_normalization'`.

- [ ] **Step 3: Implement** in `src/db/persistence.py` (append):

```python
from src.db.repositories.provider_repository import ProviderRepository
from src.providers.entities import (
    CompanyProfile, FinancialMetric, NewsEvent,
)
from src.providers.normalization import NormalizationResult


def persist_normalization(
    session, *, request_id, ticker_id, raw_kwargs: dict, normalization_result: NormalizationResult
) -> dict:
    """Persist a raw response and its normalized records (006 extension tables only).

    Does NOT mutate any 004-owned table; only writes provider_* / raw_provider_responses.
    """
    repo = ProviderRepository(session)
    raw = repo.create_raw_response(request_id=request_id, ticker_id=ticker_id, **raw_kwargs)
    session.flush()
    out: dict = {"raw_response_id": raw.id, "news_events": [], "company_profiles": [], "financial_metrics": []}
    for rec in normalization_result.records:
        warnings = [w.model_dump() for w in rec.warnings]
        if isinstance(rec, NewsEvent):
            out["news_events"].append(repo.create_news_event(
                request_id=request_id, ticker_id=ticker_id, raw_response_id=raw.id,
                title=rec.title, summary=rec.summary, source_name=rec.source_name,
                source_url=rec.source_url, published_at=rec.published_at,
                event_type=rec.event_type, sentiment_label=rec.sentiment_label,
                risk_tags=rec.risk_tags, normalization_status=rec.normalization_status.value,
                warnings=warnings, evidence_id=rec.evidence_id,
            ))
        elif isinstance(rec, CompanyProfile):
            out["company_profiles"].append(repo.create_company_profile(
                request_id=request_id, ticker_id=ticker_id, raw_response_id=raw.id,
                company_name=rec.company_name, legal_name=rec.legal_name, sector=rec.sector,
                industry=rec.industry, country=rec.country, exchange=rec.exchange,
                currency=rec.currency, description=rec.description,
                normalization_status=rec.normalization_status.value,
                warnings=warnings, evidence_id=rec.evidence_id,
            ))
        elif isinstance(rec, FinancialMetric):
            out["financial_metrics"].append(repo.create_financial_metric(
                request_id=request_id, ticker_id=ticker_id, raw_response_id=raw.id,
                metric_name=rec.metric_name,
                metric_value=None if rec.metric_value is None else str(rec.metric_value),
                period=rec.period, currency=rec.currency, unit=rec.unit,
                source_name=rec.source_name, source_url=rec.source_url,
                normalization_status=rec.normalization_status.value,
                warnings=warnings, evidence_id=rec.evidence_id,
            ))
    return out
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_provider_persistence_contracts.py -q` (with DB URL) → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/db/persistence.py tests/test_provider_persistence_contracts.py
git commit -m "feat(006): raw+normalized persistence orchestration preserving 004 boundaries (T026)"
```

**US2 Checkpoint:** `uv run pytest tests/test_provider_persistence_contracts.py -q` passes against the test PostgreSQL DB. ✅

---

# Phase 5: User Story 3 — Map Normalized Data To GraphRAG Context (P3)

> 006 emits **eligibility specs only**; 005 owns graph semantics. `ScenarioReportInput` exposes no raw payloads and degrades explicitly.

### Task 18: Graph mapping eligibility (T030) + tests (T027, T028)

**Files:** Create `src/graph_rag/mapping_contracts.py`; Create `tests/test_provider_graphrag_mapping_contracts.py`.

**Interfaces:**
- Produces: `GraphMappingRule` (Pydantic `_Contract`), `GraphEligibleSpec`, `build_eligible_specs(records) -> list[GraphEligibleSpec]`, and `is_graph_eligible(record) -> bool`. Eligible node types per contract: Company, Ticker, NewsEvent, FinancialMetric, TechnicalAnalysisResult, WaveAnalysisResult, Risk, Evidence. `RawProviderResponse`/`VectorReference` never eligible.

- [ ] **Step 1: Write the failing tests** in `tests/test_provider_graphrag_mapping_contracts.py`:

```python
"""US3 graph mapping eligibility tests (no live graph)."""

from src.graph_rag.mapping_contracts import (
    GraphEligibleSpec, build_eligible_specs, is_graph_eligible,
)
from src.providers.entities import CompanyProfile, NewsEvent
from src.providers.enums import NormalizationStatus


def _news():
    return NewsEvent(
        request_id="req1", ticker_id="tk1", raw_response_id="raw1",
        title="Acme beats earnings", source_url="https://x/y", evidence_id="ev1",
        normalization_status=NormalizationStatus.SUCCESS,
    )


def test_normalized_records_are_graph_eligible():
    specs = build_eligible_specs([_news()])
    assert all(isinstance(s, GraphEligibleSpec) for s in specs)
    assert specs[0].graph_node_type == "NewsEvent"
    assert specs[0].source_canonical_ref  # must resolve to canonical reference


def test_raw_and_row_level_data_not_projected():
    # raw payloads / candles / sentences / financial rows must be ineligible
    assert not is_graph_eligible({"raw_response_id": "raw1", "payload_body": {}})
    assert not is_graph_eligible({"candle": {"o": 1}})
    specs = build_eligible_specs([{"raw_candle": True}])
    assert specs == []
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError: mapping_contracts`.

- [ ] **Step 3: Implement** `src/graph_rag/mapping_contracts.py`:

```python
"""006 graph-mapping eligibility contract. 005 owns the graph model itself."""

from __future__ import annotations

from src.providers.entities import (
    CompanyProfile, FinancialMetric, NewsEvent,
    TechnicalAnalysisResult, WaveAnalysisResult,
)
from src.providers.enums import _Contract

_NODE_TYPE_BY_CLASS = {
    CompanyProfile: "Company",
    NewsEvent: "NewsEvent",
    FinancialMetric: "FinancialMetric",
    TechnicalAnalysisResult: "TechnicalAnalysisResult",
    WaveAnalysisResult: "WaveAnalysisResult",
}

ELIGIBLE_NODE_TYPES = frozenset(
    {"Company", "Ticker", "NewsEvent", "FinancialMetric",
     "TechnicalAnalysisResult", "WaveAnalysisResult", "Risk", "Evidence"}
)


class GraphMappingRule(_Contract):
    source_contract_type: str
    graph_node_type: str
    relationship_type: str | None = None
    required_fields: list[str] = []
    evidence_id_required: bool = False
    projection_scope: str = "company_ticker_centered"


class GraphEligibleSpec(_Contract):
    source_contract_type: str
    source_canonical_ref: str
    graph_node_type: str
    required_fields: list[str] = []
    evidence_ref: str | None = None


def is_graph_eligible(record) -> bool:
    return type(record) in _NODE_TYPE_BY_CLASS


def _canonical_ref(record) -> str:
    # Prefer the record's own canonical id field; fall back to request/ticker.
    for attr in ("company_profile_id", "news_event_id", "financial_metric_id",
                 "technical_analysis_result_id", "wave_analysis_result_id"):
        val = getattr(record, attr, None)
        if val:
            return val
    return f"{record.request_id}:{record.ticker_id}"


def build_eligible_specs(records: list) -> list[GraphEligibleSpec]:
    specs: list[GraphEligibleSpec] = []
    for rec in records:
        node_type = _NODE_TYPE_BY_CLASS.get(type(rec))
        if node_type is None:
            continue  # raw payloads, candles, rows, dicts -> never projected
        evidence_ref = getattr(rec, "evidence_id", None)
        if evidence_ref is None:
            ev_ids = getattr(rec, "evidence_ids", []) or []
            evidence_ref = ev_ids[0] if ev_ids else None
        specs.append(GraphEligibleSpec(
            source_contract_type=type(rec).__name__,
            source_canonical_ref=_canonical_ref(rec),
            graph_node_type=node_type,
            evidence_ref=evidence_ref,
        ))
    return specs
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_provider_graphrag_mapping_contracts.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_rag/mapping_contracts.py tests/test_provider_graphrag_mapping_contracts.py
git commit -m "feat(006): graph mapping eligibility specs + ineligibility for raw/rows (T027,T028,T030)"
```

### Task 19: VectorReference + ScenarioReportInput schema (T031, T044) and tests (T029, T042)

**Files:** Create `src/providers/scenario_input.py`; Modify `src/providers/safety.py` (append `ScenarioReportInput` to `SAFETY_CHECKED_CONTRACTS`); Create `tests/test_scenario_report_input_contract.py`; Modify `src/providers/__init__.py` (export the new names).

**Interfaces:**
- Produces: `VectorReference` (`source_kind` ∈ `news_original`/`neowave_rule_explanation`/`report_chunk`, `canonical_ref_id` required, optional `source_uri`/`chunk_id`; **no score/store/embedding**); `ScenarioReportInput` (`_Contract`, fields exactly per data-model.md); validator requiring `canonical_ref_id` non-empty.

- [ ] **Step 1: Write the failing tests** in `tests/test_scenario_report_input_contract.py`:

```python
"""US3 ScenarioReportInput + VectorReference contract tests."""

import pytest
from pydantic import ValidationError

from src.providers.entities import CompanyProfile, NewsEvent
from src.providers.enums import DegradationStatus, NormalizationStatus
from src.providers.scenario_input import ScenarioReportInput, VectorReference


def test_vector_reference_is_lightweight_canonical_only():
    vr = VectorReference(source_kind="news_original", canonical_ref_id="src_doc_1")
    assert vr.source_uri is None
    # no score/store/embedding fields exist on the contract
    for forbidden in ("score", "embedding", "vector_store", "store"):
        assert forbidden not in VectorReference.model_fields
    # dangling reference (empty canonical id) is invalid
    with pytest.raises(ValidationError):
        VectorReference(source_kind="news_original", canonical_ref_id="")
    # unknown source_kind rejected
    with pytest.raises(ValidationError):
        VectorReference(source_kind="raw_payload", canonical_ref_id="x")


def test_scenario_report_input_excludes_raw_and_carries_required_fields():
    sri = ScenarioReportInput(
        request_id="req1", ticker="ACME",
        company_profile=None, news_events=[], financial_metrics=[],
        technical_analysis_results=[], wave_analysis_results=[],
        graph_context={}, evidence_ids=["ev1"],
        vector_references=[VectorReference(source_kind="report_chunk", canonical_ref_id="ev1")],
        missing_data_notes=["no market data"],
        degradation_status=DegradationStatus.PARTIAL_PROVIDER_FAILURE,
    )
    assert sri.degradation_status == DegradationStatus.PARTIAL_PROVIDER_FAILURE
    # raw payload fields cannot be attached
    with pytest.raises(ValidationError):
        ScenarioReportInput(
            request_id="req1", ticker="ACME", degradation_status=DegradationStatus.COMPLETE,
            payload_body={"x": 1},
        )
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: scenario_input`.

- [ ] **Step 3: Implement** `src/providers/scenario_input.py`:

```python
"""ScenarioReportInput + VectorReference (006). No raw payloads, no trading fields."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from src.providers.entities import (
    CompanyProfile, FinancialMetric, NewsEvent,
    TechnicalAnalysisResult, WaveAnalysisResult,
)
from src.providers.enums import DegradationStatus, Warning, _Contract


class VectorReference(_Contract):
    source_kind: Literal["news_original", "neowave_rule_explanation", "report_chunk"]
    canonical_ref_id: str
    source_uri: str | None = None
    chunk_id: str | None = None

    @field_validator("canonical_ref_id")
    @classmethod
    def _ref_must_resolve(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("VectorReference.canonical_ref_id must resolve to a canonical PG reference")
        return v


class ScenarioReportInput(_Contract):
    request_id: str
    ticker: str
    company_profile: CompanyProfile | None = None
    news_events: list[NewsEvent] = Field(default_factory=list)
    financial_metrics: list[FinancialMetric] = Field(default_factory=list)
    technical_analysis_results: list[TechnicalAnalysisResult] = Field(default_factory=list)
    wave_analysis_results: list[WaveAnalysisResult] = Field(default_factory=list)
    graph_context: dict = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    vector_references: list[VectorReference] = Field(default_factory=list)
    missing_data_notes: list[str] = Field(default_factory=list)
    degradation_status: DegradationStatus
    warnings: list[Warning] = Field(default_factory=list)
```

- [ ] **Step 4: Add `ScenarioReportInput` to the safety set.** In `src/providers/safety.py`, import it and append to `SAFETY_CHECKED_CONTRACTS`:

```python
from src.providers.scenario_input import ScenarioReportInput  # add near other imports
# ...append to the tuple:
SAFETY_CHECKED_CONTRACTS = SAFETY_CHECKED_CONTRACTS + (ScenarioReportInput,)
```
(Or rebuild the tuple literal to include it — keep it a module-level constant.)

- [ ] **Step 5: Export** `VectorReference`, `ScenarioReportInput` from `src/providers/__init__.py` (`__all__` + import).

- [ ] **Step 6: Run** — `uv run pytest tests/test_scenario_report_input_contract.py tests/test_provider_contracts.py -q` → PASS (including the earlier `test_all_mvp_contracts_are_clean`, now covering ScenarioReportInput).

- [ ] **Step 7: Commit**

```bash
git add src/providers/scenario_input.py src/providers/safety.py src/providers/__init__.py tests/test_scenario_report_input_contract.py
git commit -m "feat(006): VectorReference + ScenarioReportInput contracts, safety coverage (T031,T044,T029,T042)"
```

### Task 20: ScenarioReportInput builder + degradation (T032, T045) and observability test (T043)

**Files:** Modify `src/providers/scenario_input.py`; Create `tests/test_provider_observability_contract.py`; Modify `tests/fixtures/provider_contracts.py` (scenario builder fixture).

**Interfaces:**
- Produces: `build_scenario_report_input(*, request_id, ticker, company_profile, news_events, financial_metrics, technical_analysis_results, wave_analysis_results, graph_context, vector_references) -> ScenarioReportInput` that derives `degradation_status`, `missing_data_notes`, aggregated `warnings`, and `evidence_ids` from inputs.

- [ ] **Step 1: Add a scenario fixture** to `tests/fixtures/provider_contracts.py`:

```python
def scenario_inputs_complete():
    """Normalized records for a fully-populated scenario (deterministic)."""
    common = dict(request_id="req1", ticker_id="tk1", raw_response_id="raw1")
    from src.providers.normalization import (
        normalize_company, normalize_financials, normalize_news,
    )
    cp = normalize_company(raw=raw_company_payload(), **common).records[0]
    news = normalize_news(raw_items=raw_news_provider_a(), **common).records
    metrics = normalize_financials(raw_rows=raw_financial_rows(), **common).records
    return cp, news, metrics
```

- [ ] **Step 2: Write the failing observability tests** in `tests/test_provider_observability_contract.py`:

```python
"""US3 observability contract tests (FR-017, SER-006). No live services."""

from src.providers.entities import TechnicalAnalysisResult
from src.providers.enums import DegradationStatus, NormalizationStatus
from src.providers.scenario_input import build_scenario_report_input
from tests.fixtures.provider_contracts import scenario_inputs_complete


def test_complete_inputs_produce_complete_status():
    cp, news, metrics = scenario_inputs_complete()
    sri = build_scenario_report_input(
        request_id="req1", ticker="ACME", company_profile=cp,
        news_events=news, financial_metrics=metrics,
        technical_analysis_results=[TechnicalAnalysisResult(
            request_id="req1", ticker_id="tk1", source_market_data_refs=["md1"],
            normalization_or_derivation_status=NormalizationStatus.SUCCESS,
        )],
        wave_analysis_results=[], graph_context={"company": "ACME"}, vector_references=[],
    )
    assert sri.degradation_status == DegradationStatus.COMPLETE
    assert sri.missing_data_notes == []


def test_missing_categories_surface_as_structured_warnings():
    sri = build_scenario_report_input(
        request_id="req1", ticker="ACME", company_profile=None,
        news_events=[], financial_metrics=[], technical_analysis_results=[],
        wave_analysis_results=[], graph_context={}, vector_references=[],
    )
    assert sri.degradation_status in (
        DegradationStatus.INSUFFICIENT_DATA, DegradationStatus.PARTIAL_PROVIDER_FAILURE,
    )
    assert sri.missing_data_notes  # explicit, non-empty
    assert any(w.code for w in sri.warnings)


def test_missing_graph_context_marks_degraded():
    cp, news, metrics = scenario_inputs_complete()
    sri = build_scenario_report_input(
        request_id="req1", ticker="ACME", company_profile=cp, news_events=news,
        financial_metrics=metrics, technical_analysis_results=[], wave_analysis_results=[],
        graph_context={}, vector_references=[],  # empty graph context
    )
    assert any("graph" in note.lower() for note in sri.missing_data_notes)
```

- [ ] **Step 3: Run to verify failure** — `ImportError: build_scenario_report_input`.

- [ ] **Step 4: Implement the builder** in `src/providers/scenario_input.py` (append):

```python
def build_scenario_report_input(
    *, request_id, ticker, company_profile, news_events, financial_metrics,
    technical_analysis_results, wave_analysis_results, graph_context, vector_references,
):
    notes: list[str] = []
    warnings: list[Warning] = []

    def _check(present: bool, category: str) -> None:
        if not present:
            notes.append(f"missing {category}")
            warnings.append(Warning(code=f"missing_{category}", message=f"no {category} available"))

    _check(company_profile is not None, "company_profile")
    _check(bool(news_events), "news_events")
    _check(bool(financial_metrics), "financial_metrics")
    _check(bool(technical_analysis_results), "technical_analysis")
    if not graph_context:
        notes.append("missing graph_context")
        warnings.append(Warning(code="missing_graph_context", message="graph context missing or stale"))

    have_any = any([company_profile, news_events, financial_metrics, technical_analysis_results])
    required_core = company_profile is not None and bool(financial_metrics)
    if not have_any:
        status = DegradationStatus.INSUFFICIENT_DATA
    elif not required_core:
        status = DegradationStatus.PARTIAL_PROVIDER_FAILURE
    elif not graph_context:
        status = DegradationStatus.GRAPH_MAPPING_DEGRADED
    elif notes:
        status = DegradationStatus.PARTIAL_PROVIDER_FAILURE
    else:
        status = DegradationStatus.COMPLETE

    evidence_ids: list[str] = []
    if company_profile and company_profile.evidence_id:
        evidence_ids.append(company_profile.evidence_id)
    for e in news_events:
        if e.evidence_id:
            evidence_ids.append(e.evidence_id)
    for m in financial_metrics:
        if m.evidence_id:
            evidence_ids.append(m.evidence_id)

    return ScenarioReportInput(
        request_id=request_id, ticker=ticker, company_profile=company_profile,
        news_events=list(news_events), financial_metrics=list(financial_metrics),
        technical_analysis_results=list(technical_analysis_results),
        wave_analysis_results=list(wave_analysis_results),
        graph_context=graph_context, evidence_ids=evidence_ids,
        vector_references=list(vector_references), missing_data_notes=notes,
        degradation_status=status, warnings=warnings,
    )
```

> Note: in `test_complete_inputs_produce_complete_status` graph_context is non-empty and core categories present → COMPLETE. Adjust the precedence above if your story needs technical-analysis to be optional for COMPLETE; the test fixture provides one, so COMPLETE holds.

- [ ] **Step 5: Export** `build_scenario_report_input` from `src/providers/__init__.py`.

- [ ] **Step 6: Run** — `uv run pytest tests/test_provider_observability_contract.py tests/test_scenario_report_input_contract.py -q` → PASS.

- [ ] **Step 7: Commit**

```bash
git add src/providers/scenario_input.py src/providers/__init__.py tests/fixtures/provider_contracts.py tests/test_provider_observability_contract.py
git commit -m "feat(006): ScenarioReportInput builder with degradation + observability fields (T032,T045,T043)"
```

### Task 21: Graph-context boundary handling (T033)

**Files:** Modify `src/graph_rag/graph_context_builder.py`; Test: extend `tests/test_provider_graphrag_mapping_contracts.py`.

**Interfaces:**
- Produces: `build_contract_graph_context(specs, *, projection_status="ready") -> dict` returning `{"nodes": [...], "degraded": bool, "warnings": [...]}` that marks degraded/stale when projection is missing — without calling Neo4j.

- [ ] **Step 1: Write the failing test** (append to `tests/test_provider_graphrag_mapping_contracts.py`):

```python
from src.graph_rag.graph_context_builder import build_contract_graph_context


def test_graph_context_degrades_when_projection_missing():
    specs = build_eligible_specs([_news()])
    ok = build_contract_graph_context(specs, projection_status="ready")
    assert ok["degraded"] is False and ok["nodes"]

    stale = build_contract_graph_context(specs, projection_status="stale")
    assert stale["degraded"] is True
    assert any("stale" in w.lower() or "missing" in w.lower() for w in stale["warnings"])
```

- [ ] **Step 2: Run to verify failure** — `ImportError: build_contract_graph_context`.

- [ ] **Step 3: Implement** (append to `src/graph_rag/graph_context_builder.py`):

```python
def build_contract_graph_context(specs, *, projection_status: str = "ready") -> dict:
    """Build a 006-boundary graph context from eligibility specs.

    Does not call Neo4j; reflects projection status into degraded/warnings so
    ScenarioReportInput can disclose missing/stale graph context (FR-015, SER-006).
    """
    degraded = projection_status != "ready"
    warnings: list[str] = []
    if degraded:
        warnings.append(f"graph projection {projection_status}: context missing or stale")
    nodes = [
        {"node_type": s.graph_node_type, "canonical_ref": s.source_canonical_ref,
         "evidence_ref": s.evidence_ref}
        for s in specs
    ]
    return {"nodes": nodes, "degraded": degraded, "warnings": warnings}
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_provider_graphrag_mapping_contracts.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_rag/graph_context_builder.py tests/test_provider_graphrag_mapping_contracts.py
git commit -m "feat(006): graph-context degradation/stale handling at contract boundary (T033)"
```

### Task 22: CoordinatorAgent scenario boundary (T034)

**Files:** Modify `src/agents/coordinator_agent.py`; Test: extend `tests/test_scenario_report_input_contract.py`.

**Interfaces:**
- Produces: `scenario_report_input_to_agent_input(sri: ScenarioReportInput) -> dict` that exposes scenario fields to CoordinatorAgent (ScenarioAgent role) with no raw provider payloads.

- [ ] **Step 1: Write the failing test** (append):

```python
from src.agents.coordinator_agent import scenario_report_input_to_agent_input
from src.providers.scenario_input import build_scenario_report_input
from tests.fixtures.provider_contracts import scenario_inputs_complete


def test_coordinator_consumes_scenario_input_without_raw_fields():
    cp, news, metrics = scenario_inputs_complete()
    sri = build_scenario_report_input(
        request_id="req1", ticker="ACME", company_profile=cp, news_events=news,
        financial_metrics=metrics, technical_analysis_results=[], wave_analysis_results=[],
        graph_context={"company": "ACME"}, vector_references=[],
    )
    agent_input = scenario_report_input_to_agent_input(sri)
    assert agent_input["ticker"] == "ACME"
    assert agent_input["news_count"] == len(news)
    assert "payload_body" not in agent_input and "content" not in agent_input
    assert agent_input["degradation_status"] == sri.degradation_status.value
```

- [ ] **Step 2: Run to verify failure** — `ImportError`.

- [ ] **Step 3: Implement** (append to `src/agents/coordinator_agent.py`):

```python
from src.providers.scenario_input import ScenarioReportInput


def scenario_report_input_to_agent_input(sri: ScenarioReportInput) -> dict:
    """Adapt ScenarioReportInput for the Coordinator (ScenarioAgent role).

    Exposes only normalized contract fields; never raw provider payloads.
    """
    return {
        "request_id": sri.request_id,
        "ticker": sri.ticker,
        "company_name": sri.company_profile.company_name if sri.company_profile else None,
        "news_count": len(sri.news_events),
        "metric_names": [m.metric_name for m in sri.financial_metrics],
        "evidence_ids": sri.evidence_ids,
        "missing_data_notes": sri.missing_data_notes,
        "degradation_status": sri.degradation_status.value,
        "warning_count": len(sri.warnings),
    }
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_scenario_report_input_contract.py -q` plus the existing coordinator suite:
Run: `uv run pytest tests/test_coordinator_agent.py -q`
Expected: PASS (additive function; no node changes).

- [ ] **Step 5: Commit**

```bash
git add src/agents/coordinator_agent.py tests/test_scenario_report_input_contract.py
git commit -m "feat(006): coordinator scenario-input boundary function (T034)"
```

**US3 Checkpoint:** `uv run pytest tests/test_provider_graphrag_mapping_contracts.py tests/test_scenario_report_input_contract.py tests/test_provider_observability_contract.py -q` all pass without live graph/vector services. ✅

---

# Phase 6: Polish & Cross-Cutting

### Task 23: Full safety contract test (T046)

**Files:** Create `tests/test_provider_safety_contract.py`.

- [ ] **Step 1: Write the test:**

```python
"""Full structural + instance safety sweep over SAFETY_CHECKED_CONTRACTS (SC-006, SER-001)."""

from src.providers.enums import DegradationStatus
from src.providers.safety import (
    SAFETY_CHECKED_CONTRACTS, assert_no_trading_fields, find_trading_fields,
)
from src.providers.scenario_input import ScenarioReportInput, VectorReference


def test_structural_no_contract_declares_trading_fields():
    assert ScenarioReportInput in SAFETY_CHECKED_CONTRACTS
    for contract in SAFETY_CHECKED_CONTRACTS:
        assert find_trading_fields(contract) == [], contract.__name__
        assert_no_trading_fields(contract)


def test_token_matching_examples():
    from pydantic import BaseModel

    class Buy(BaseModel):
        buy_signal: int = 0

    class Tgt(BaseModel):
        target_price: float = 0.0

    assert find_trading_fields(Buy) == ["buy_signal"]
    assert find_trading_fields(Tgt) == ["target_price"]

    class Clean(BaseModel):
        threshold: float = 0.0
        household_segment: str = ""

    assert find_trading_fields(Clean) == []


def test_instance_scenario_report_input_exposes_no_trading_fields():
    sri = ScenarioReportInput(
        request_id="req1", ticker="ACME", degradation_status=DegradationStatus.COMPLETE,
        vector_references=[VectorReference(source_kind="report_chunk", canonical_ref_id="ev1")],
    )
    dumped = sri.model_dump()
    from src.providers.safety import _field_violates
    assert not any(_field_violates(k) for k in dumped)
```

- [ ] **Step 2: Run** — `uv run pytest tests/test_provider_safety_contract.py -q` → PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_provider_safety_contract.py
git commit -m "test(006): full structural+instance safety contract sweep (T046)"
```

### Task 24: Docs alignment (T035, T036)

**Files:** Modify `specs/006-provider-mcp-contracts/quickstart.md`, `PROJECT_PLAN.md`.

- [ ] **Step 1:** Append a "Validation results" note to `quickstart.md` listing the six test files and the `compileall`/`ruff` commands as the canonical validation set, and note that persistence tests skip without `TEST_DATABASE_URL`.
- [ ] **Step 2:** Add a short "006 provider contract boundary" subsection to `PROJECT_PLAN.md` (read it first to match its structure) stating: 006 owns contracts/mapping; 004 owns canonical tables (extended via migration `8f1a2b3c4d5e`); 005 owns graph semantics.
- [ ] **Step 3: Commit**

```bash
git add specs/006-provider-mcp-contracts/quickstart.md PROJECT_PLAN.md
git commit -m "docs(006): align quickstart and project plan with provider contract boundary (T035,T036)"
```

### Task 25: Final validation gate (T037, T038, T039, T040, T047, T041)

- [ ] **Step 1 (T037):** `uv run python -m compileall src` → no errors.
- [ ] **Step 2 (T038):** `uv run pytest tests/test_provider_contracts.py -q` → PASS.
- [ ] **Step 3 (T039):** `export TEST_DATABASE_URL=postgresql+psycopg://...; uv run pytest tests/test_provider_persistence_contracts.py -q` → PASS (or skip if no DB).
- [ ] **Step 4 (T040):** `uv run pytest tests/test_provider_graphrag_mapping_contracts.py tests/test_scenario_report_input_contract.py -q` → PASS.
- [ ] **Step 5 (T047):** `uv run pytest tests/test_provider_observability_contract.py tests/test_provider_safety_contract.py -q` → PASS.
- [ ] **Step 6 (T041):** `uv run ruff check src tests` → clean (fix any lint, re-run).
- [ ] **Step 7: Full suite regression** — `uv run pytest -q` → no new failures vs. baseline.
- [ ] **Step 8: Commit any lint fixes**

```bash
git add -A
git commit -m "chore(006): final compile/test/ruff validation gate (T037-T041,T047)"
```

---

## Task ID → Plan Task map

| Plan Task | tasks.md IDs | Phase |
|---|---|---|
| 1 | T001, T002, T003 | Setup |
| 2 | T004 | Foundational |
| 3 | T006 | Foundational |
| 4 | T005 | Foundational |
| 5 | T048 | Foundational |
| 6 | T007 | Foundational |
| 7 | T008 | Foundational |
| 8 | T009 | US1 |
| 9 | T012 | US1 |
| 10 | T010, T013, T014 | US1 |
| 11 | T011 | US1 |
| 12 | T015, T016, T017 | US1 |
| 13 | T018, T019, T020 | US2 |
| 14 | T021, T022 | US2 |
| 15 | T023 | US2 |
| 16 | T024, T025 | US2 |
| 17 | T026 | US2 |
| 18 | T027, T028, T030 | US3 |
| 19 | T029, T031, T042, T044 | US3 |
| 20 | T032, T043, T045 | US3 |
| 21 | T033 | US3 |
| 22 | T034 | US3 |
| 23 | T046 | Polish |
| 24 | T035, T036 | Polish |
| 25 | T037, T038, T039, T040, T041, T047 | Polish |

All T001–T048 are covered.

---

## Self-Review

- **Spec coverage:** FR-001…FR-021 map to Tasks 2–22; SER-001/SC-006 → Tasks 5 & 23; FR-017/SER-006 observability → Task 20; FR-018 vector refs → Task 19; FR-021 (004 extension decision) → Task 14/15 (explicit migration `8f1a2b3c4d5e`, `down_revision=56464a69bd55`). SC-001 → Task 8; SC-002 → Tasks 13/16/17; SC-003 → Task 19; SC-004 → Task 18; SC-005 → Tasks 11/20; SC-007 → Ownership Map honored throughout.
- **Decision compliance:** (1) `src/providers/` split into the six files ✅; (2) every contract via `_Contract` → `extra="forbid"` ✅; (3) token/phrase matching + `SAFETY_CHECKED_CONTRACTS`, simulation namespace untouched ✅; (4) `VectorReference` carries only canonical refs ✅; (5) 004 changes are an explicit migration task ✅; (6) Setup→Foundational→US1→US2→US3→Polish, test-first, T-ID mapped ✅.
- **Type consistency:** `normalize_*` signatures match their call sites; `NormalizationResult.records` typed `list[Any]` holds heterogeneous entity contracts; `ProviderRepository` method names match the US2 tests; `build_scenario_report_input`/`scenario_report_input_to_agent_input` names consistent across Tasks 20 & 22.
- **No live calls:** every test uses fixtures; persistence tests skip without `TEST_DATABASE_URL`; no MCP/Neo4j/vector client is imported.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-22-provider-mcp-contracts.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
