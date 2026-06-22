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
