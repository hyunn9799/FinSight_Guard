"""Tests for safety checker behavior."""

from datetime import date

import pytest

from src.graph.state import ResearchReport
from src.safety.forbidden_phrases import FORBIDDEN_PHRASES, REQUIRED_DISCLAIMER
from src.safety.safety_checker import (
    find_forbidden_phrases,
    has_limitations,
    has_required_disclaimer,
    has_risk_disclosure,
)


def _report(
    *,
    risk_factors: str = "가격 변동성, 실적 발표, 거시 환경 변화 리스크가 있습니다.",
    limitations: str = "본 분석은 공개 데이터와 제한된 뉴스 검색 결과에 기반합니다.",
    disclaimer: str = REQUIRED_DISCLAIMER,
) -> ResearchReport:
    return ResearchReport(
        title="AAPL Research Report",
        ticker="AAPL",
        data_date=date.today(),
        executive_summary="교육 목적의 리서치 요약입니다.",
        market_section="시장 지표 요약",
        fundamental_section="재무 지표 요약",
        news_section="뉴스 요약",
        scenario_analysis="관망 시나리오, 분할 접근 시나리오, 리스크 회피 시나리오",
        risk_factors=risk_factors,
        limitations=limitations,
        evidence_summary="근거 요약",
        disclaimer=disclaimer,
    )


def test_find_forbidden_phrases_returns_matches() -> None:
    text = "이 보고서는 무조건 매수 또는 수익 보장을 말하면 안 됩니다."

    matches = find_forbidden_phrases(text)

    assert "무조건 매수" in matches
    assert "수익 보장" in matches


@pytest.mark.parametrize("phrase", FORBIDDEN_PHRASES)
def test_each_configured_forbidden_phrase_is_detected(phrase: str) -> None:
    matches = find_forbidden_phrases(f"검토 문장: {phrase}")

    assert phrase in matches


def test_find_forbidden_phrases_returns_empty_list_for_safe_text() -> None:
    assert find_forbidden_phrases("관망 시나리오와 리스크 회피 시나리오를 비교합니다.") == []


def test_required_disclaimer_detection() -> None:
    assert has_required_disclaimer(_report()) is True
    assert has_required_disclaimer(_report(disclaimer="")) is False


def test_required_disclaimer_detection_accepts_dict_report() -> None:
    assert has_required_disclaimer({"disclaimer": REQUIRED_DISCLAIMER}) is True
    assert has_required_disclaimer({"disclaimer": "교육 목적 고지"}) is False


def test_risk_disclosure_detection() -> None:
    assert has_risk_disclosure(_report()) is True
    assert has_risk_disclosure(_report(risk_factors="")) is False


def test_limitations_detection() -> None:
    assert has_limitations(_report()) is True
    assert has_limitations(_report(limitations="")) is False
