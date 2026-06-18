"""Market regime classification and per-regime performance summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.robust import CandidateMetrics, CostAssumptions

BULL_THRESHOLD = 0.05
BEAR_THRESHOLD = -0.05
HIGH_VOL_PERCENTILE = 70
LOW_VOL_PERCENTILE = 30


def classify_regime_periods(
    price_series: pd.Series,
    lookback_days: int = 60,
) -> pd.Series:
    """Classify each trading day by trend/volatility regime."""
    rolling_return = price_series.pct_change(periods=lookback_days)
    regimes = pd.Series(index=price_series.index, dtype=object)
    regimes[rolling_return >= BULL_THRESHOLD] = "bull"
    regimes[rolling_return <= BEAR_THRESHOLD] = "bear"
    mask_sideways = (rolling_return > BEAR_THRESHOLD) & (rolling_return < BULL_THRESHOLD)
    regimes[mask_sideways] = "sideways"

    rolling_vol = price_series.pct_change().rolling(lookback_days).std()
    high_vol_threshold = rolling_vol.quantile(HIGH_VOL_PERCENTILE / 100.0)
    low_vol_threshold = rolling_vol.quantile(LOW_VOL_PERCENTILE / 100.0)

    null_trend = regimes.isna()
    regimes[null_trend & (rolling_vol >= high_vol_threshold)] = "high_volatility"
    regimes[null_trend & (rolling_vol <= low_vol_threshold)] = "low_volatility"
    regimes = regimes.fillna("sideways")

    return regimes


def compute_regime_performance(
    trades_df: pd.DataFrame,
    regime_labels: pd.Series,
    initial_balance: float,
    cost: CostAssumptions,
) -> list[dict]:
    """Summarize trade performance per regime segment."""
    if trades_df.empty or regime_labels.empty:
        return []

    sell_trades = trades_df[trades_df["type"] == "Sell"].copy()
    if sell_trades.empty:
        return []

    sell_trades["date_ts"] = pd.to_datetime(sell_trades["date"])
    if sell_trades["date_ts"].dt.tz is not None:
        sell_trades["date_ts"] = sell_trades["date_ts"].dt.tz_localize(None)
    sell_trades = sell_trades.set_index("date_ts")

    # Normalize regime labels to tz-naive so .isin() matches tz-naive trade dates.
    # (yfinance price indexes are often tz-aware; trade dates are tz-naive.)
    regime_labels = regime_labels.copy()
    if isinstance(regime_labels.index, pd.DatetimeIndex) and regime_labels.index.tz is not None:
        regime_labels.index = regime_labels.index.tz_localize(None)

    summaries: list[dict] = []
    for regime in ["bull", "bear", "sideways", "high_volatility", "low_volatility"]:
        regime_dates = regime_labels[regime_labels == regime].index
        if len(regime_dates) == 0:
            continue

        regime_trades = sell_trades[sell_trades.index.isin(regime_dates)]
        completed_trades = len(regime_trades)
        trading_days = len(regime_dates)

        confidence = "low" if (completed_trades < 10 or trading_days < 60) else "normal"
        low_confidence_reason: str | None = None
        if confidence == "low":
            reasons = []
            if completed_trades < 10:
                reasons.append(f"only {completed_trades} completed trades (minimum 10)")
            if trading_days < 60:
                reasons.append(f"only {trading_days} trading days (minimum 60)")
            low_confidence_reason = "; ".join(reasons)

        if completed_trades > 0:
            profits = regime_trades["profit"].to_numpy(dtype=float)
            wins = float((profits > 0).mean() * 100.0)
            gross_profit = float(profits[profits > 0].sum())
            gross_loss = float(abs(profits[profits < 0].sum()))
            pf = gross_profit / gross_loss if gross_loss > 0 else None
            ret_pct = float(profits.sum() / initial_balance * 100.0)
            metrics = CandidateMetrics(
                cost_adjusted_return_pct=ret_pct,
                total_return_pct=ret_pct,
                completed_trades=completed_trades,
                win_rate_pct=wins,
                profit_factor=pf,
            )
        else:
            metrics = CandidateMetrics()

        summaries.append({
            "regime": regime,
            "start": str(regime_dates[0].date()),
            "end": str(regime_dates[-1].date()),
            "trading_days": trading_days,
            "completed_trades": completed_trades,
            "metrics": metrics.model_dump(),
            "confidence": confidence,
            "low_confidence_reason": low_confidence_reason,
        })

    return summaries
