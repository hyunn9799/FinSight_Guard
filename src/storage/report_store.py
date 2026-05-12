"""Report persistence helpers."""

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

from src.config import REPORT_DIR


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _extract_ticker(result: Any) -> str:
    if hasattr(result, "ticker"):
        return str(result.ticker)
    if isinstance(result, dict):
        if result.get("ticker"):
            return str(result["ticker"])
        report = result.get("final_report") or result.get("draft_report") or result.get("report")
        if hasattr(report, "ticker"):
            return str(report.ticker)
        if isinstance(report, dict) and report.get("ticker"):
            return str(report["ticker"])
    return "UNKNOWN"


def _safe_filename_part(value: str) -> str:
    clean_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().upper())
    return clean_value or "UNKNOWN"


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _report_to_markdown(report: Any) -> str:
    data = _to_jsonable(report)
    if not isinstance(data, dict):
        return str(data)

    title = data.get("title", "Research Report")
    ordered_fields = [
        "executive_summary",
        "market_section",
        "fundamental_section",
        "news_section",
        "scenario_analysis",
        "risk_factors",
        "limitations",
        "evidence_summary",
        "disclaimer",
    ]
    headings = {
        "executive_summary": "요약",
        "market_section": "시장 분석",
        "fundamental_section": "펀더멘털 분석",
        "news_section": "뉴스 분석",
        "scenario_analysis": "시나리오 분석",
        "risk_factors": "리스크",
        "limitations": "한계",
        "evidence_summary": "근거 요약",
        "disclaimer": "고지문",
    }
    lines = [f"# {title}", "", f"- Ticker: {data.get('ticker', 'UNKNOWN')}", f"- Data Date: {data.get('data_date', '')}", ""]
    for field in ordered_fields:
        value = data.get(field)
        if value:
            lines.extend([f"## {headings[field]}", str(value), ""])
    return "\n".join(lines).strip() + "\n"


def save_report_json(run_id: str, result: Any) -> str:
    """Save workflow result as JSON under REPORT_DIR and return path."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ticker = _safe_filename_part(_extract_ticker(result))
    path = REPORT_DIR / f"{run_id}_{ticker}_{_timestamp()}.json"
    path.write_text(
        json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def save_report_markdown(run_id: str, report: Any) -> str:
    """Save a ResearchReport-like object as Markdown under REPORT_DIR."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ticker = _safe_filename_part(_extract_ticker(report))
    path = REPORT_DIR / f"{run_id}_{ticker}_{_timestamp()}.md"
    path.write_text(_report_to_markdown(report), encoding="utf-8")
    return str(path)


def load_report_json(path: str | Path) -> dict:
    """Load a saved report JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
