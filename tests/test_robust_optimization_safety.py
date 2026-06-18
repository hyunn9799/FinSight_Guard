"""Tests: optimization output text avoids advice-like wording."""
import pytest
from src.backtest.robust import FORBIDDEN_OPTIMIZATION_PHRASES, check_optimization_text_safety


SAFE_TEXT = (
    "이 파라미터 세트는 과거 시뮬레이션에서 MDD 18.2%를 기록했습니다. "
    "결과는 연구 목적이며 투자 권유가 아닙니다."
)

UNSAFE_TEXTS = [
    "이 파라미터로 매수하세요.",
    "guaranteed return of 15% using these settings.",
    "이 설정을 사용하면 반드시 수익을 낼 수 있습니다.",
    "recommended trading parameters for immediate use.",
    "자동 주문을 실행하세요.",
]

UNSAFE_WHEN_GUARDRAIL_FAILED = [
    "이 파라미터는 우수한 성과를 보였습니다.",
    "탁월한 전략 파라미터입니다.",
    "이 파라미터 세트는 강력히 권장됩니다.",
]


def test_safe_text_passes_safety_check():
    assert check_optimization_text_safety(SAFE_TEXT, robust_label_allowed=True) is True


def test_forbidden_phrases_fail_safety_check():
    for text in UNSAFE_TEXTS:
        assert check_optimization_text_safety(text, robust_label_allowed=True) is False, (
            f"Expected FAIL for: {text}"
        )


def test_strong_positive_fails_when_robust_label_not_allowed():
    for text in UNSAFE_WHEN_GUARDRAIL_FAILED:
        assert check_optimization_text_safety(text, robust_label_allowed=False) is False, (
            f"Expected FAIL when guardrail failed: {text}"
        )


def test_strong_positive_passes_when_robust_label_allowed():
    text = "이 파라미터는 OOS 폴드에서 우수한 MDD 관리를 보였습니다."
    assert check_optimization_text_safety(text, robust_label_allowed=True) is True


def test_forbidden_phrases_list_not_empty():
    assert len(FORBIDDEN_OPTIMIZATION_PHRASES) >= 5
