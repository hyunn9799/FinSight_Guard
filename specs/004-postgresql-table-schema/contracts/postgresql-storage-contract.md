# Contract: PostgreSQL Storage Boundary

## Purpose

This contract defines the observable behavior expected from the PostgreSQL source-of-truth layer. It is intentionally expressed as repository and persistence contracts rather than executable code.

## General Rules

- PostgreSQL owns canonical IDs and durable lifecycle state.
- Repository methods must be deterministic in tests and must not call live market, news, LLM, vector, graph, search, or cache services.
- Derived index clients may consume canonical records, but their failures must update projection status rather than deleting or mutating source records.
- All timestamps are timezone-aware UTC timestamps.
- Repository operations either complete atomically or return a clear failure without partially persisted workflow state.

## Core Repository Contracts

### Ticker Repository

**Must support**

- Upsert ticker by `(symbol, market)`.
- Fetch ticker by ID.
- Fetch ticker by `(symbol, market)`.
- Mark ticker active/inactive.

**Acceptance**

- Repeated upsert of the same `(symbol, market)` returns the same canonical ticker.
- Symbols are normalized consistently.

### Analysis Repository

**Must support**

- Create analysis request.
- Transition request status.
- Append workflow node attempt.
- Save analysis result.
- Fetch complete request bundle including node runs, results, reports, evidence, and warnings.

**Acceptance**

- Degraded and failed requests preserve warning/error summaries.
- Node attempts preserve attempt number, timing, status, and error message.

### Evidence Repository

**Must support**

- Insert EvidenceItem-compatible record.
- Fetch by `evidence_id`.
- Link evidence to report version claim.
- List evidence by request, ticker, source document, or report version.

**Acceptance**

- Duplicate `evidence_id` is rejected.
- Report citations always resolve to existing evidence items.

### Report Repository

**Must support**

- Create report container.
- Append immutable report version.
- Mark current version.
- Set safety status and evaluation score.
- Archive report.

**Acceptance**

- Report version numbers are unique per report.
- Final Korean reports can be checked for required disclaimer presence.

### Source Document Repository

**Must support**

- Upsert source document by source URL/content hash.
- Create corrected or recrawled source document versions with `revision_group_id` and `supersedes_document_id`.
- Insert ordered document chunks.
- Fetch chunks by source document.
- Resolve chunks for projection.

**Acceptance**

- Duplicate source crawls do not create ambiguous chunk ownership.
- Chunk indexes are stable for a source document version.
- Corrected or recrawled source document versions preserve lineage and do not overwrite prior canonical chunks.

### Projection Repository

**Must support**

- Upsert projection status by target system, projection type, and idempotency key.
- Mark projection success.
- Mark projection failure with error message.
- List pending/stale/failed projections.

**Acceptance**

- Projection failures never mutate canonical source records.
- Projection records resolve back to `source_table` and `source_id`.

### User Repository

**Must support**

- Create local/demo user.
- Update profile fields.
- Anonymize user.
- Soft-delete user.

**Acceptance**

- Email may be null.
- Auth credentials, password hashes, SSO IDs, and durable session credentials are not required for MVP.
- User anonymization clears or replaces user-identifying fields while preserving research audit references.

### Notification Repository

**Must support**

- Create in-app notification.
- List notifications by user.
- Mark notification read.
- Archive/delete notification.

**Acceptance**

- No external delivery target is required.
- Notification text must not contain trading instructions or guaranteed outcome language.

### Portfolio Repository

**Must support**

- Create research-only portfolio.
- Add/update/remove portfolio item.
- List portfolio items by user or portfolio.
- Soft-delete/anonymize user-owned portfolio records.

**Acceptance**

- No brokerage account, order, execution, or guaranteed target fields exist.
- Portfolio items reference canonical tickers.

### Graph Repository

**Must support**

- Create/update wave rule.
- Create/update wave scenario.
- Link scenario to rules.
- Add invalidation condition.
- Create evidence path and ordered path steps.
- Fetch path with all canonical node references.

**Acceptance**

- Wave rules have unique rule codes.
- Evidence path steps are ordered and resolve to canonical table/ID references.
- Scenario and invalidation text remains research framing, not advice.

## API Behavior Expectations

Existing FastAPI endpoints may continue to return current response shapes, but once PostgreSQL persistence is implemented:

- `/analyze` responses must be backed by an `analysis_requests` row.
- `/backtest` responses must be backed by an `analysis_requests` row.
- `/backtest/optimize` responses must be backed by an `analysis_requests` row and optimization analysis result.
- `/metrics` must remain available and must not require derived index availability.
- `/health` must remain available; future DB health detail may be added without breaking the existing `{"status": "ok"}` contract.

## Failure Contracts

- If PostgreSQL is unavailable during a request that requires persistence, the workflow must return a user-facing persistence error or degraded persistence warning rather than pretending the report was saved.
- If a derived projection fails, the source workflow remains complete and projection status records the failure.
- If user anonymization fails partway through, the operation must be retried or rolled back so PII is not partially cleared in an inconsistent state.
