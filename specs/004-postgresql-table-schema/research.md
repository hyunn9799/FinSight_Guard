# Research: PostgreSQL Source-of-Truth Table Schema

## Decision: Use SQLAlchemy 2.x ORM/Core With Alembic Migrations

**Rationale**: The project is Python/FastAPI/Pydantic based and currently has no database layer. SQLAlchemy plus Alembic is the most common maintainable path for schema definition, migrations, deterministic tests, and repository boundaries without tying the agents directly to SQL strings.

**Alternatives considered**:

- Raw psycopg only: simpler dependency surface but high migration and relationship-management burden.
- SQLModel: convenient with Pydantic, but less explicit for a large schema and migration design.
- Django ORM: too large a framework shift for the current FastAPI/Streamlit project.

## Decision: PostgreSQL Is Canonical, Derived Stores Are Projection Targets

**Rationale**: The feature explicitly defines PostgreSQL as the ledger. Pinecone, Neo4j, and OpenSearch records must be rebuildable from source documents, chunks, evidence, reports, graph records, and projection status rows. Redis data must remain ephemeral.

**Alternatives considered**:

- Let each specialized store own its data: creates split-brain recovery and audit problems.
- PostgreSQL-only forever: simpler, but rejects the selected semantic, graph, and keyword search architecture.

## Decision: Keep Repository Boundary Under `src/db`

**Rationale**: Agents and LangGraph nodes should continue to reason in terms of state, analysis results, evidence, and reports. A repository boundary lets persistence evolve without pushing SQL concerns into workflow code.

**Alternatives considered**:

- Direct DB writes inside agents: fast initially but hard to test and risks coupling safety logic to storage details.
- A separate service process: unnecessary for a portfolio MVP and adds operational overhead.

## Decision: Local/Demo User Ownership Without MVP Login

**Rationale**: The clarified scope includes `users`, settings, notifications, and portfolios but defers login, password, SSO, and durable session credentials. This supports ownership and demo UX without expanding into auth/security implementation.

**Alternatives considered**:

- Full email login now: broadens scope into credential handling and session security.
- No user table: conflicts with clarified full platform MVP scope.

## Decision: Anonymize User-Owned Data While Preserving Research Audit Records

**Rationale**: Evidence-grounded research requires historical report/evidence auditability. User deletion should remove or anonymize identifying user and UX data while keeping non-PII reports, evidence, source documents, node runs, and audit references intact.

**Alternatives considered**:

- Soft delete everything: simplest but leaves more PII around than needed.
- Hard delete the full user tree: breaks report/evidence audit trails and derived-index rebuilds.

## Decision: In-App Notifications Only For MVP

**Rationale**: In-app records meet the user-facing persistence requirement without introducing email addresses, webhooks, push services, delivery retries, or channel consent.

**Alternatives considered**:

- Email-ready notifications: still creates delivery-target and privacy complexity.
- Multi-channel notification model: premature for the current portfolio MVP.

## Decision: Include Graph Knowledge Tables In First PostgreSQL MVP

**Rationale**: Clarification selected graph tables now. Canonical wave rules, scenarios, invalidation conditions, and evidence paths should exist before Neo4j projection so graph traversal is auditable and rebuildable.

**Alternatives considered**:

- Defer graph tables: lower immediate table count but conflicts with clarified scope.
- Evidence paths only: would not support canonical wave-rule and scenario ownership.

## Decision: Preserve Existing File Exports Temporarily

**Rationale**: The current workflow already saves JSON/Markdown reports locally. Keeping these as exports during migration reduces implementation risk while PostgreSQL becomes the source of truth.

**Alternatives considered**:

- Remove file stores immediately: unnecessary risk and breaks existing demo behavior.
- Keep files as canonical: conflicts with source-of-truth requirement.
