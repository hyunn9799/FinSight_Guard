"""Tests for technical indicators."""

import pandas as pd
import pytest

from src.indicators.technicals import (
    calculate_atr,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    enrich_market_indicators,
)


def _sample_price_frame(rows: int = 130) -> pd.DataFrame:
    close = pd.Series(range(1, rows + 1), dtype="float64")
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=rows),
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": [1000] * rows,
        }
    )


def test_calculate_sma_returns_expected_values_without_mutating_input() -> None:
    df = _sample_price_frame(25)
    original_columns = list(df.columns)

    sma = calculate_sma(df, 20)

    assert sma.name == "MA20"
    assert pd.isna(sma.iloc[18])
    assert sma.iloc[19] == pytest.approx(10.5)
    assert list(df.columns) == original_columns


def test_calculate_rsi_identifies_positive_momentum() -> None:
    df = _sample_price_frame(30)

    rsi = calculate_rsi(df)

    assert pd.isna(rsi.iloc[13])
    assert rsi.iloc[-1] == pytest.approx(100.0)


def test_calculate_macd_returns_signal_and_histogram() -> None:
    df = _sample_price_frame(60)

    macd = calculate_macd(df)

    assert list(macd.columns) == ["MACD", "MACD_signal", "MACD_hist"]
    assert macd.iloc[0]["MACD"] == pytest.approx(0.0)
    assert macd.iloc[-1]["MACD"] > 0
    assert macd.iloc[-1]["MACD_hist"] >= 0


def test_calculate_atr_returns_expected_average_true_range() -> None:
    df = _sample_price_frame(20)

    atr = calculate_atr(df)

    assert pd.isna(atr.iloc[12])
    assert atr.iloc[13] == pytest.approx(2.0)
    assert atr.iloc[-1] == pytest.approx(2.0)


def test_enrich_market_indicators_adds_expected_columns_without_mutating_input() -> None:
    df = _sample_price_frame()
    original_columns = list(df.columns)

    enriched = enrich_market_indicators(df)

    assert list(df.columns) == original_columns
    for column in ["MA20", "MA60", "MA120", "RSI", "MACD", "MACD_signal", "MACD_hist", "ATR"]:
        assert column in enriched.columns
    assert enriched.iloc[-1]["MA20"] == pytest.approx(120.5)
    assert enriched.iloc[-1]["MA60"] == pytest.approx(100.5)
    assert enriched.iloc[-1]["MA120"] == pytest.approx(70.5)


def test_indicators_handle_empty_dataframe() -> None:
    empty = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    assert calculate_sma(empty, 20).empty
    assert calculate_rsi(empty).empty
    assert calculate_macd(empty).empty
    assert calculate_atr(empty).empty

    enriched = enrich_market_indicators(empty)

    assert enriched.empty
    for column in ["MA20", "MA60", "MA120", "RSI", "MACD", "MACD_signal", "MACD_hist", "ATR"]:
        assert column in enriched.columns
