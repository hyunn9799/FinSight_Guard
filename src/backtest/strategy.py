"""Kernel-regression + RSI-divergence + Bollinger backtest strategy.

Ported from the standalone ``backtest_core.py`` with the Streamlit dependency
removed and a typed parameter/result surface added. The trading logic is kept
behaviourally identical to the original so historical results remain
reproducible.

Positioning note: signals here describe a *historical simulation* of a fixed
rule set. They are technical reference material, not investment advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from statsmodels.nonparametric.kernel_regression import KernelReg


@dataclass(frozen=True)
class BacktestParams:
    """Strategy hyper-parameters for a single backtest run."""

    rsi_period: int = 14
    kr_window: int = 30
    kr_bandwidth: float = 5.0
    bb_k: float = 2.0
    extrema_order: int = 5
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    @classmethod
    def from_dict(cls, params: dict) -> "BacktestParams":
        """Build params from a plain dict (e.g. Streamlit widget state)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in params.items() if key in known})


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    profit_pct: float
    final_value: float
    trades: pd.DataFrame
    enriched: pd.DataFrame
    divergences: list[tuple] = field(default_factory=list)

    @property
    def trade_count(self) -> int:
        return 0 if self.trades is None or self.trades.empty else len(self.trades)


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(period).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def run_backtest(
    df_input: pd.DataFrame,
    params: BacktestParams | dict,
    initial_balance: float,
    fee: float = 0.001,
) -> BacktestResult:
    """Run the composite-signal backtest over a price history DataFrame.

    Args:
        df_input: OHLC price history with a ``Close`` column and DatetimeIndex.
        params: Strategy hyper-parameters (``BacktestParams`` or a dict).
        initial_balance: Starting capital for the simulation.
        fee: Per-trade proportional fee (default 0.1%).

    Returns:
        A ``BacktestResult``. If the kernel-regression window is larger than the
        available history, returns an empty result with ``profit_pct == -100``.
    """
    if isinstance(params, dict):
        params = BacktestParams.from_dict(params)

    df_temp = df_input.copy()
    df_temp["RSI"] = _compute_rsi(df_temp["Close"].squeeze(), params.rsi_period)

    x = np.arange(len(df_temp))
    y = df_temp["Close"].to_numpy().ravel()
    y_pred = np.full(len(y), np.nan)
    window = int(params.kr_window)

    if window >= len(y):
        return BacktestResult(
            profit_pct=-100.0,
            final_value=0.0,
            trades=pd.DataFrame(),
            enriched=pd.DataFrame(),
            divergences=[],
        )

    for i in range(window, len(y)):
        x_train = x[i - window:i]
        y_train = y[i - window:i]
        try:
            kr = KernelReg(
                endog=y_train,
                exog=x_train,
                var_type="c",
                bw=[params.kr_bandwidth],
            )
            y_pred[i] = kr.fit([x[i]])[0][0]
        except Exception:
            y_pred[i] = np.nan

    df_temp["y_pred"] = y_pred
    vol = pd.Series(y).rolling(20).std()
    df_temp["band"] = (params.bb_k * vol).values

    order = int(params.extrema_order)
    local_max_price = argrelextrema(df_temp["Close"].values, np.greater_equal, order=order)[0]
    local_min_price = argrelextrema(df_temp["Close"].values, np.less_equal, order=order)[0]
    divergences: list[tuple] = []

    for i in range(1, len(local_min_price)):
        p1_idx, p2_idx = int(local_min_price[i - 1]), int(local_min_price[i])
        if df_temp["Close"].iloc[p2_idx].item() < df_temp["Close"].iloc[p1_idx].item() and \
           df_temp["RSI"].iloc[p2_idx].item() > df_temp["RSI"].iloc[p1_idx].item():
            if df_temp["RSI"].iloc[p2_idx].item() <= params.rsi_oversold:
                divergences.append((df_temp.index[p1_idx], df_temp.index[p2_idx], "bullish"))

    for i in range(1, len(local_max_price)):
        p1_idx, p2_idx = int(local_max_price[i - 1]), int(local_max_price[i])
        if df_temp["Close"].iloc[p2_idx].item() > df_temp["Close"].iloc[p1_idx].item() and \
           df_temp["RSI"].iloc[p2_idx].item() < df_temp["RSI"].iloc[p1_idx].item():
            if df_temp["RSI"].iloc[p2_idx].item() >= params.rsi_overbought:
                divergences.append((df_temp.index[p1_idx], df_temp.index[p2_idx], "bearish"))

    signal = np.zeros(len(y))
    for i in range(len(y)):
        if np.isnan(y_pred[i]) or np.isnan(df_temp["band"].iloc[i]):
            continue
        if y[i] > y_pred[i] + df_temp["band"].iloc[i]:
            signal[i] = -1
        elif y[i] < y_pred[i] - df_temp["band"].iloc[i]:
            signal[i] = 1
        else:
            signal[i] = 0

    for _, date_p2, div_type in divergences:
        if date_p2 in df_temp.index:
            idx = df_temp.index.get_loc(date_p2)
            if div_type == "bullish":
                signal[idx] = 1
            elif div_type == "bearish":
                signal[idx] = -1

    filtered_signal = np.zeros(len(signal))
    last_signal = 0
    for i in range(len(signal)):
        if signal[i] != 0 and signal[i] != last_signal:
            filtered_signal[i] = signal[i]
            last_signal = signal[i]
    df_temp["signal"] = filtered_signal

    capital = initial_balance
    balance = capital
    position = 0.0
    buy_price = 0.0
    trades: list[dict] = []
    dates = df_temp.index.to_list()
    prices = df_temp["Close"].to_numpy()
    signals = df_temp["signal"].to_numpy()

    for i in range(len(df_temp) - 1):
        price = prices[i + 1].item()
        trade_date = dates[i + 1]

        if signals[i] == 1 and position == 0:
            position = balance * (1 - fee) / price
            buy_price = price
            balance = 0
            trades.append({"date": trade_date, "type": "Buy", "price": round(price, 3),
                           "quantity": round(position, 6), "balance": round(balance, 3), "profit": 0.0})
        elif signals[i] == -1 and position > 0:
            trade_profit = round(position * (price - buy_price) * (1 - fee), 3)
            balance = position * price * (1 - fee)
            trades.append({"date": trade_date, "type": "Sell", "price": round(price, 3),
                           "quantity": round(position, 6), "balance": round(balance, 3), "profit": trade_profit})
            position = 0.0
            buy_price = 0.0

    if position > 0:
        price = float(prices[-1].item())
        trade_date = dates[-1]
        trade_profit = round(position * (price - buy_price) * (1 - fee), 3)
        balance = position * price * (1 - fee)
        trades.append({"date": trade_date, "type": "Sell", "price": round(price, 3),
                       "quantity": round(position, 6), "balance": round(balance, 3), "profit": trade_profit})

    final_value = balance if balance > 0 else position * prices[-1].item()
    profit_pct = -100.0 if final_value == 0 else (final_value - capital) / capital * 100

    return BacktestResult(
        profit_pct=float(profit_pct),
        final_value=float(final_value),
        trades=pd.DataFrame(trades),
        enriched=df_temp,
        divergences=divergences,
    )
