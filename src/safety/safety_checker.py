"""Safety checks for generated reports."""

from typing import Any

from src.safety.forbidden_phrases import FORBIDDEN_PHRASES, REQUIRED_DISCLAIMER


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        data = value.model_dump()
        return "\n".join(_value_to_text(item) for item in data.values())
    if isinstance(value, dict):
        return "\n".join(_value_to_text(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return "\n".join(_value_to_text(item) for item in value)
    return str(value)


def _field_text(report: Any, field_name: str) -> str:
    if isinstance(report, dict):
        return _value_to_text(report.get(field_name))
    return _value_to_text(getattr(report, field_name, ""))


def find_forbidden_phrases(text: str) -> list[str]:
    """Return forbidden phrases found in text, preserving configured order."""
    normalized_text = text or ""
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in normalized_text]


def has_required_disclaimer(report: Any) -> bool:
    """Check whether the report includes the exact required disclaimer."""
    disclaimer_text = _field_text(report, "disclaimer")
    full_text = _value_to_text(report)
    return REQUIRED_DISCLAIMER in disclaimer_text or REQUIRED_DISCLAIMER in full_text


def has_risk_disclosure(report: Any) -> bool:
    """Check whether risk disclosure content is present."""
    risk_text = _field_text(report, "risk_factors").strip()
    if isinstance(report, dict) and "risk_factors" in report:
        return bool(risk_text)
    if hasattr(report, "risk_factors"):
        return bool(risk_text)
    if risk_text:
        return True

    full_text = _value_to_text(report)
    risk_keywords = ["리스크", "위험", "변동성", "손실", "risk"]
    return any(keyword.lower() in full_text.lower() for keyword in risk_keywords)


def has_limitations(report: Any) -> bool:
    """Check whether limitations or caveats are present."""
    limitations_text = _field_text(report, "limitations").strip()
    if isinstance(report, dict) and "limitations" in report:
        return bool(limitations_text)
    if hasattr(report, "limitations"):
        return bool(limitations_text)
    if limitations_text:
        return True

    full_text = _value_to_text(report)
    limitation_keywords = ["한계", "제한", "불확실", "limitation", "limited"]
    return any(keyword.lower() in full_text.lower() for keyword in limitation_keywords)
