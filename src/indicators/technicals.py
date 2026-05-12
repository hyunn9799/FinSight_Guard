"""Technical indicator calculations."""

import numpy as np
import pandas as pd


def _empty_series(name: str, index: pd.Index) -> pd.Series:
    """Return a float series aligned to the input frame index."""
    return pd.Series(dtype="float64", index=index, name=name)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Market dataframe is missing required columns: {missing}")


def calculate_sma(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate a simple moving average over the Close column."""
    market_data = df.copy()
    name = f"MA{window}"
    if market_data.empty:
        return _empty_series(name, market_data.index)

    _require_columns(market_data, ["Close"])
    return (
        market_data["Close"]
        .astype(float)
        .rolling(window=window, min_periods=window)
        .mean()
        .rename(name)
    )


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate the relative strength index over the Close column."""
    market_data = df.copy()
    if market_data.empty:
        return _empty_series("RSI", market_data.index)

    _require_columns(market_data, ["Close"])
    close = market_data["Close"].astype(float)
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.rolling(window=window, min_periods=window).mean()
    average_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)
    return rsi.rename("RSI")


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Calculate MACD, signal, and histogram columns over the Close column."""
    market_data = df.copy()
    columns = ["MACD", "MACD_signal", "MACD_hist"]
    if market_data.empty:
        return pd.DataFrame({column: _empty_series(column, market_data.index) for column in columns})

    _require_columns(market_data, ["Close"])
    close = market_data["Close"].astype(float)
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=1).mean()
    histogram = macd - signal_line
    return pd.DataFrame(
        {
            "MACD": macd,
            "MACD_signal": signal_line,
            "MACD_hist": histogram,
        },
        index=market_data.index,
    )


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate average true range from High, Low, and Close columns."""
    market_data = df.copy()
    if market_data.empty:
        return _empty_series("ATR", market_data.index)

    _require_columns(market_data, ["High", "Low", "Close"])
    high = market_data["High"].astype(float)
    low = market_data["Low"].astype(float)
    close = market_data["Close"].astype(float)
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean().rename("ATR")


def enrich_market_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of market data with core technical indicators attached."""
    enriched = df.copy()
    for column in ["MA20", "MA60", "MA120", "RSI", "MACD", "MACD_signal", "MACD_hist", "ATR"]:
        if column not in enriched.columns:
            enriched[column] = pd.Series(dtype="float64", index=enriched.index)

    if enriched.empty:
        return enriched

    enriched["MA20"] = calculate_sma(enriched, 20)
    enriched["MA60"] = calculate_sma(enriched, 60)
    enriched["MA120"] = calculate_sma(enriched, 120)
    enriched["RSI"] = calculate_rsi(enriched)
    macd = calculate_macd(enriched)
    enriched["MACD"] = macd["MACD"]
    enriched["MACD_signal"] = macd["MACD_signal"]
    enriched["MACD_hist"] = macd["MACD_hist"]
    enriched["ATR"] = calculate_atr(enriched)
    return enriched
