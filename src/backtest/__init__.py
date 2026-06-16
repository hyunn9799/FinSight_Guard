"""Strategy backtesting module.

Ported from the standalone kernel-regression backtesting project and adapted
to FinSight Guard's evidence-grounded, no-recommendation positioning. The
backtest produces *historical simulation* results used as technical reference
evidence; it never emits live buy/sell advice.
"""

from src.backtest.strategy import BacktestParams, BacktestResult, run_backtest

__all__ = ["BacktestParams", "BacktestResult", "run_backtest"]
