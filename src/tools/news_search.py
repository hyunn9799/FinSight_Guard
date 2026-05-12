"""News search provider tools."""

from datetime import UTC, datetime
import json
import logging
import os
from typing import Any
import urllib.error
import urllib.request


logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
REQUEST_TIMEOUT_SECONDS = 20


def mock_news_provider(
    ticker: str,
    company_name: str | None = None,
) -> list[dict[str, str | None]]:
    """Return deterministic placeholder news when live providers are unavailable."""
    clean_ticker = ticker.strip().upper()
    display_name = company_name or clean_ticker
    published_at = datetime.now(UTC).date().isoformat()
    return [
        {
            "title": f"{display_name} 최근 시장 관심 요인 점검",
            "summary": (
                f"{clean_ticker} 관련 최근 뉴스 API가 설정되지 않아 mock 뉴스로 "
                "대체되었습니다."
            ),
            "url": None,
            "published_at": published_at,
            "source": "mock",
        },
        {
            "title": f"{display_name} 리스크 요인 모니터링 필요",
            "summary": "실제 뉴스 검색 전까지 가격 변동성, 실적 발표, 거시 이벤트를 확인해야 합니다.",
            "url": None,
            "published_at": published_at,
            "source": "mock",
        },
    ]


def _news_query(ticker: str, company_name: str | None) -> str:
    """Build a compact query for company-related news search."""
    subject = company_name.strip() if company_name and company_name.strip() else ticker
    return f"{subject} {ticker} latest company news"


def _post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON to an authenticated provider endpoint and return parsed JSON."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "FinSight-Guard/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Provider returned HTTP {exc.code}: {message[:300]}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Provider returned invalid JSON.") from exc


def _clean_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _search_with_tavily(
    ticker: str,
    company_name: str | None,
    max_results: int,
) -> list[dict[str, str | None]]:
    """Search recent company news with Tavily and normalize the response."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")

    payload = {
        "query": _news_query(ticker, company_name),
        "topic": "news",
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    data = _post_json(TAVILY_SEARCH_URL, api_key, payload)
    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Tavily response did not include a results list.")

    normalized: list[dict[str, str | None]] = []
    for item in results[:max_results]:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title"), fallback=f"{ticker} news")
        summary = _clean_text(item.get("content") or item.get("raw_content"))
        normalized.append(
            {
                "title": title,
                "summary": summary,
                "url": _clean_text(item.get("url")) or None,
                "published_at": _clean_text(item.get("published_date")) or None,
                "source": "tavily",
            }
        )
    return normalized


def _search_with_firecrawl(
    ticker: str,
    company_name: str | None,
    max_results: int,
) -> list[dict[str, str | None]]:
    """Search recent company news with Firecrawl and normalize the response."""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError("FIRECRAWL_API_KEY is not configured.")

    payload = {
        "query": _news_query(ticker, company_name),
        "limit": max_results,
        "sources": ["web"],
    }
    data = _post_json(FIRECRAWL_SEARCH_URL, api_key, payload)
    if data.get("success") is False:
        raise RuntimeError("Firecrawl response reported success=false.")

    response_data = data.get("data")
    if isinstance(response_data, dict):
        results = response_data.get("web", [])
    else:
        results = response_data or []
    if not isinstance(results, list):
        raise RuntimeError("Firecrawl response did not include a result list.")

    normalized: list[dict[str, str | None]] = []
    for item in results[:max_results]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        title = _clean_text(
            item.get("title") or metadata.get("title"),
            fallback=f"{ticker} news",
        )
        summary = _clean_text(
            item.get("description")
            or metadata.get("description")
            or item.get("markdown")
            or item.get("html")
        )
        normalized.append(
            {
                "title": title,
                "summary": summary,
                "url": _clean_text(item.get("url") or metadata.get("sourceURL")) or None,
                "published_at": _clean_text(
                    item.get("publishedDate")
                    or item.get("published_at")
                    or metadata.get("publishedTime")
                )
                or None,
                "source": "firecrawl",
            }
        )
    return normalized


def search_recent_news(
    ticker: str,
    company_name: str | None = None,
    max_results: int = 5,
) -> list[dict[str, str | None]]:
    """Search recent news or fall back to deterministic mock news."""
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        raise ValueError("Ticker must not be empty.")
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    tavily_key = os.getenv("TAVILY_API_KEY")
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")

    if tavily_key:
        try:
            return _search_with_tavily(clean_ticker, company_name, max_results)[:max_results]
        except Exception as exc:
            logger.warning(
                "tavily_news_search_failed",
                extra={"ticker": clean_ticker, "error": str(exc)},
            )

    if firecrawl_key:
        try:
            return _search_with_firecrawl(clean_ticker, company_name, max_results)[:max_results]
        except Exception as exc:
            logger.warning(
                "firecrawl_news_search_failed",
                extra={"ticker": clean_ticker, "error": str(exc)},
            )

    return mock_news_provider(clean_ticker, company_name)[:max_results]
