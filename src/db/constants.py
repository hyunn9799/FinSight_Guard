"""Shared status vocabularies for canonical persistence."""

REQUEST_STATUSES = frozenset(
    {"pending", "running", "success", "degraded", "insufficient_data", "failed", "cancelled"}
)
RESULT_STATUSES = frozenset({"success", "degraded", "insufficient_data", "failed"})
REPORT_STATUSES = frozenset({"draft", "final", "failed_review", "archived"})
SAFETY_STATUSES = frozenset({"pass", "fail", "not_evaluated"})
REPORT_VERSION_STAGES = frozenset({"draft", "rewrite", "final", "failed"})
RESULT_TYPES = frozenset(
    {
        "market", "fundamental", "news", "graph_context", "backtest",
        "optimization", "coordinator_draft", "evaluation", "rewrite",
    }
)
REQUEST_TYPES = frozenset({"research", "backtest", "robust_optimization", "graph_context"})
