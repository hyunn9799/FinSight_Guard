"""Provider-agnostic MCP contract layer (006).

Owns provider interface contracts, normalization contracts, lineage
requirements, and graph-mapping eligibility. Does NOT own canonical tables
(004) or the graph model (005). No live MCP/API/Neo4j/vector calls live here.

Stable public exports are populated in T008.
"""

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
from src.providers.scenario_input import ScenarioReportInput, VectorReference

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
    "VectorReference", "ScenarioReportInput",
]
