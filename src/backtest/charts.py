"""Matplotlib chart rendering for backtest results (Streamlit-independent).

Ported from the standalone project's inline plotting code. The original hardcoded
a Windows-only Korean font (``C:\\Windows\\Fonts\\malgun.ttf``); this version
auto-detects an available CJK font across macOS, Linux, and Windows and degrades
gracefully when none is installed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; Streamlit renders Agg figures fine

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Candidate Korean font families in preference order (macOS, Linux, Windows).
_KOREAN_FONT_CANDIDATES = (
    "AppleSDGothicNeo",
    "Apple SD Gothic Neo",
    "AppleGothic",
    "NanumGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Malgun Gothic",
    "Source Han Sans KR",
)
# Known font file locations to register if family lookup misses them.
_KOREAN_FONT_PATHS = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    r"C:\Windows\Fonts\malgun.ttf",
)

_configured_font: str | None = None


def configure_korean_font() -> str | None:
    """Set a Korean-capable matplotlib font if one is available.

    Returns the resolved font family name, or ``None`` when no CJK font is found
    (in which case charts still render but Korean labels may show as boxes).
    Idempotent: the lookup runs only once per process.
    """
    global _configured_font
    if _configured_font is not None:
        return _configured_font or None

    import os

    for path in _KOREAN_FONT_PATHS:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
            except Exception:  # noqa: BLE001 - registration is best-effort
                continue

    available = {font.name for font in fm.fontManager.ttflist}
    for name in _KOREAN_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            _configured_font = name
            return name

    # No CJK font found: keep minus signs sane and leave the default family.
    plt.rcParams["axes.unicode_minus"] = False
    _configured_font = ""
    return None


def _plot_divergences(
    ax: Axes, enriched: pd.DataFrame, divergences: list[tuple], column: str
) -> None:
    bullish_labeled, bearish_labeled = False, False
    for start_div, end_div, div_type in divergences:
        if start_div not in enriched.index or end_div not in enriched.index:
            continue
        y1 = enriched.loc[start_div, column]
        y2 = enriched.loc[end_div, column]
        if div_type == "bullish":
            label = None if bullish_labeled else "강세 다이버전스"
            ax.plot([start_div, end_div], [y1, y2], color="lime", linestyle="--", linewidth=2, label=label)
            bullish_labeled = True
        elif div_type == "bearish":
            label = None if bearish_labeled else "약세 다이버전스"
            ax.plot([start_div, end_div], [y1, y2], color="magenta", linestyle="--", linewidth=2, label=label)
            bearish_labeled = True


def build_backtest_figure(
    enriched: pd.DataFrame,
    divergences: list[tuple],
    *,
    ticker: str,
    rsi_oversold: float,
    rsi_overbought: float,
) -> Figure:
    """Build a 2-panel (price + RSI) backtest figure. Returns a matplotlib Figure.

    ``enriched`` is the ``BacktestResult.enriched`` frame and must contain
    ``Close`` and ``RSI``; ``y_pred``, ``band``, and ``signal`` are optional.
    """
    configure_korean_font()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(enriched.index, enriched["Close"], color="#1f77b4", label=f"{ticker} 종가")
    if "y_pred" in enriched and "band" in enriched:
        ax1.plot(enriched.index, enriched["y_pred"], color="orange", linestyle="--", label="커널 회귀 예측")
        ax1.fill_between(
            enriched.index,
            enriched["y_pred"] - enriched["band"],
            enriched["y_pred"] + enriched["band"],
            color="gray",
            alpha=0.2,
            label="변동성 밴드",
        )
    if "signal" in enriched:
        signal = enriched["signal"].to_numpy()
        buy_idx = np.where(signal == 1)[0]
        sell_idx = np.where(signal == -1)[0]
        ax1.scatter(
            enriched.index[buy_idx], enriched["Close"].iloc[buy_idx],
            marker="^", color="green", s=90, alpha=0.85, label="매수 신호 (시뮬레이션)",
        )
        ax1.scatter(
            enriched.index[sell_idx], enriched["Close"].iloc[sell_idx],
            marker="v", color="red", s=90, alpha=0.85, label="매도 신호 (시뮬레이션)",
        )
    _plot_divergences(ax1, enriched, divergences, "Close")
    ax1.set_title(f"{ticker} 가격 및 시뮬레이션 신호 (과거 데이터)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=True)

    ax2.plot(enriched.index, enriched["RSI"], color="#1f77b4", label="RSI")
    ax2.axhline(rsi_overbought, color="red", linestyle="--", alpha=0.5, label="과매수")
    ax2.axhline(rsi_oversold, color="green", linestyle="--", alpha=0.5, label="과매도")
    _plot_divergences(ax2, enriched, divergences, "RSI")
    ax2.set_title("RSI")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
