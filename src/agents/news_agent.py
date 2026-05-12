"""News analysis agent."""

from datetime import UTC, datetime
from typing import Any

from src.evidence.evidence_builder import build_news_evidence
from src.evidence.evidence_schema import EvidenceItem
from src.graph.state import GraphState, NewsAnalysis, WorkflowError
from src.tools.news_search import search_recent_news


NEWS_NODE = "news_agent"

POSITIVE_KEYWORDS = [
    "growth",
    "beat",
    "record",
    "surge",
    "upgrade",
    "profit",
    "strong",
    "상승",
    "성장",
    "호조",
    "개선",
    "기대",
    "수혜",
]
NEGATIVE_KEYWORDS = [
    "miss",
    "decline",
    "downgrade",
    "loss",
    "weak",
    "lawsuit",
    "하락",
    "부진",
    "악화",
    "손실",
    "소송",
    "우려",
]
RISK_KEYWORDS = [
    "risk",
    "volatility",
    "regulation",
    "investigation",
    "earnings",
    "macro",
    "리스크",
    "위험",
    "변동성",
    "규제",
    "조사",
    "실적",
    "거시",
]


def _ticker_from_state(state: GraphState) -> str:
    ticker = state.get("ticker")
    if ticker:
        return ticker.strip().upper()

    user_input = state.get("user_input")
    if user_input is not None:
        return user_input.ticker.strip().upper()

    return ""


def _company_name_from_state(state: GraphState) -> str | None:
    fundamental = state.get("fundamental_analysis")
    if fundamental is None:
        return None

    for item in fundamental.evidence:
        if item.metric_name == "longName" and item.metric_value:
            return str(item.metric_value)
    return None


def _text_for_news(item: dict[str, Any]) -> str:
    return f"{item.get('title') or ''} {item.get('summary') or ''}".lower()


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def _news_label(item: dict[str, Any]) -> str:
    title = item.get("title") or "제목 없는 뉴스"
    source = item.get("source") or "unknown"
    published_at = item.get("published_at")
    if published_at:
        return f"{title} ({source}, {published_at})"
    return f"{title} ({source})"


def _classify_news_item(item: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    text = _text_for_news(item)
    label = _news_label(item)
    positives: list[str] = []
    negatives: list[str] = []
    risks: list[str] = []

    if _contains_any(text, POSITIVE_KEYWORDS):
        positives.append(label)
    if _contains_any(text, NEGATIVE_KEYWORDS):
        negatives.append(label)
    if _contains_any(text, RISK_KEYWORDS):
        risks.append(label)

    if not positives and not negatives and not risks:
        risks.append(f"{label} - 방향성 판단 전 추가 확인 필요")

    return positives, negatives, risks


def _parse_published_at(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
    return None


def _build_evidence(ticker: str, news_items: list[dict[str, Any]]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for index, item in enumerate(news_items, start=1):
        title = item.get("title") or "제목 없는 뉴스"
        summary = item.get("summary") or ""
        source_name = str(item.get("source") or "news")
        evidence.append(
            build_news_evidence(
                ticker=ticker,
                source_name=source_name,
                source_url=item.get("url"),
                metric_name="news_item",
                metric_value=title,
                description=f"{ticker} 뉴스 {index}: {title}. {summary}".strip(),
                collected_at=_parse_published_at(item.get("published_at")),
            )
        )
    return evidence


def run_news_agent(state: GraphState) -> dict:
    """Search recent news and return a simple keyword-classified analysis."""
    ticker = _ticker_from_state(state)
    if not ticker:
        error = WorkflowError(
            node=NEWS_NODE,
            message="Ticker is missing for news analysis.",
            recoverable=False,
        )
        return {"status": "failed", "errors": [*state.get("errors", []), error]}

    try:
        news_items = search_recent_news(
            ticker,
            company_name=_company_name_from_state(state),
            max_results=5,
        )
    except Exception as exc:
        note = f"뉴스 검색을 수행하지 못했습니다: {exc}"
        analysis = NewsAnalysis(
            ticker=ticker,
            summary="뉴스 데이터 수집 실패로 이벤트 분석을 제한적으로만 제공합니다.",
            missing_data_notes=[note],
        )
        error = WorkflowError(
            node=NEWS_NODE,
            message=note,
            error_type="news_search_error",
            recoverable=True,
        )
        return {
            "status": "degraded",
            "news_analysis": analysis,
            "errors": [*state.get("errors", []), error],
            "warnings": [*state.get("warnings", []), note],
        }

    if not news_items:
        note = "최근 뉴스 검색 결과가 없어 뉴스 기반 요인을 비워 둡니다."
        analysis = NewsAnalysis(
            ticker=ticker,
            summary="확인 가능한 최근 뉴스가 없어 가격/재무 데이터 중심으로 해석해야 합니다.",
            missing_data_notes=[note],
        )
        return {
            "status": "degraded",
            "news_analysis": analysis,
            "warnings": [*state.get("warnings", []), note],
        }

    positive_factors: list[str] = []
    negative_factors: list[str] = []
    event_risks: list[str] = []
    for item in news_items:
        positives, negatives, risks = _classify_news_item(item)
        positive_factors.extend(positives)
        negative_factors.extend(negatives)
        event_risks.extend(risks)

    evidence = _build_evidence(ticker, news_items)
    missing_data_notes = [
        f"{_news_label(item)} 항목에 원문 URL이 없습니다."
        for item in news_items
        if not item.get("url")
    ]
    sources = sorted({str(item.get("source") or "unknown") for item in news_items})
    analysis = NewsAnalysis(
        ticker=ticker,
        summary=(
            f"최근 뉴스 {len(news_items)}건을 단순 키워드 규칙으로 분류했습니다. "
            f"확인된 출처: {', '.join(sources)}. 이 분류는 투자 행동 지시가 아닌 이벤트 점검용입니다."
        ),
        positive_factors=positive_factors,
        negative_factors=negative_factors,
        event_risks=event_risks,
        evidence=evidence,
        missing_data_notes=missing_data_notes,
    )
    return {
        "status": "success",
        "news_analysis": analysis,
        "evidence": [*state.get("evidence", []), *evidence],
    }
