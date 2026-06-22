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
from src.providers.scenario_input import ScenarioReportInput

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
    ScenarioReportInput,
)
