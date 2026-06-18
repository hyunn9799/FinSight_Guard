# Feature Specification: Deferred Platform Storage Architecture

**Feature Branch**: `[003-platform-storage-architecture]`

**Created**: 2026-06-18

**Status**: Draft / Deferred

**Input**: Deferred scope moved out of `specs/002-walk-forward-optimization` to keep the walk-forward MVP small and implementation-ready.

## Scope Boundary

This draft captures future platform storage work only. It must not block the walk-forward optimization MVP.

## Future Platform Direction

- PostgreSQL may become the canonical source of truth for users, tickers, analysis requests, reports, analysis results, settings, notifications, portfolios, evidence metadata, and robust optimization outputs.
- Pinecone may store semantic document chunk indexes for news, financial-statement explanations, and wave-theory materials.
- Neo4j may store wave-theory rules, scenarios, invalidation conditions, graph evidence paths, and explainable relationship traversal.
- OpenSearch may store news full text, report full text, logs, keyword indexes, and operational search views.
- Redis may support cache, queue, rate limiting, session state, and short-lived workflow coordination.

## Non-Goals

- This spec does not implement trading, brokerage integration, order execution, guaranteed return claims, or financial advice.
- This spec does not change the MVP local report-store requirement in `002-walk-forward-optimization`.

## Notes for Later Planning

- Any future implementation plan must re-run the constitution check.
- External infrastructure must be optional or locally reproducible for development.
- Tests must use fakes, monkeypatches, or isolated local services and must not depend on live paid providers.
- GitHub issues for this work should be grouped by platform milestone and written in natural Korean.
