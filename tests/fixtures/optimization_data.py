"""Deterministic synthetic fixtures for robust optimization tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(n: int = 500, start: str = "2022-01-03", seed: int = 42) -> pd.DataFrame:
    """Deterministic OHLC price series with a sine wave trend."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    t = np.arange(n)
    close = 100.0 + 20.0 * np.sin(t / 30.0) + t * 0.05 + rng.normal(0, 0.5, n).cumsum()
    close = np.maximum(close, 10.0)
    return pd.DataFrame(
        {
            "Open": close - 0.3,
            "High": close + 0.8,
            "Low": close - 0.8,
            "Close": close,
            "Volume": np.full(n, 10_000),
        },
        index=idx,
    )


def synthetic_trades(n_sells: int = 35, initial_balance: float = 10_000.0, seed: int = 42) -> pd.DataFrame:
    """Deterministic paired buy/sell trades DataFrame matching BacktestResult.trades schema."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    balance = initial_balance
    dates = pd.bdate_range("2022-01-03", periods=n_sells * 2 + 10)
    date_idx = 0
    for _ in range(n_sells):
        buy_price = float(rng.uniform(80, 120))
        qty = balance * 0.999 / buy_price
        rows.append({"date": dates[date_idx], "type": "Buy", "price": buy_price,
                     "quantity": qty, "balance": 0.0, "profit": 0.0})
        date_idx += 5
        sell_price = buy_price * float(rng.uniform(0.92, 1.12))
        profit = qty * (sell_price - buy_price) * 0.999
        balance = qty * sell_price * 0.999
        rows.append({"date": dates[date_idx], "type": "Sell", "price": sell_price,
                     "quantity": qty, "balance": balance, "profit": profit})
        date_idx += 5
    return pd.DataFrame(rows)


def synthetic_trades_few(n_sells: int = 10) -> pd.DataFrame:
    """Fewer than 30 completed trades — should fail robust_label guardrail."""
    return synthetic_trades(n_sells=n_sells, seed=7)


def synthetic_trades_high_mdd(initial_balance: float = 10_000.0) -> pd.DataFrame:
    """Trades that produce >25% drawdown."""
    rng = np.random.default_rng(99)
    rows: list[dict] = []
    balance = initial_balance
    dates = pd.bdate_range("2022-01-03", periods=200)
    date_idx = 0
    for i in range(35):
        buy_price = 100.0
        qty = balance * 0.999 / buy_price
        rows.append({"date": dates[date_idx], "type": "Buy", "price": buy_price,
                     "quantity": qty, "balance": 0.0, "profit": 0.0})
        date_idx += 2
        sell_price = buy_price * (0.90 if i < 4 else float(rng.uniform(1.0, 1.05)))
        profit = qty * (sell_price - buy_price) * 0.999
        balance = qty * sell_price * 0.999
        rows.append({"date": dates[date_idx], "type": "Sell", "price": sell_price,
                     "quantity": qty, "balance": balance, "profit": profit})
        date_idx += 2
    return pd.DataFrame(rows)
