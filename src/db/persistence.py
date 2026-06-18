"""Persist a full research workflow run to the PostgreSQL source of truth."""

from src.db.postgres import session_scope
from src.db.repositories.analysis_repository import AnalysisRepository
from src.db.repositories.evidence_repository import EvidenceRepository
from src.db.repositories.report_repository import ReportRepository

_RESULT_STATUS_FOR = {
    "success": "success",
    "degraded": "degraded",
    "insufficient_data": "insufficient_data",
    "failed": "failed",
}


def _read_value(value, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _report_json(report) -> dict:
    if isinstance(report, dict):
        return report
    return report.model_dump(mode="json")


def _evaluation_summary(evaluation) -> tuple[float | None, str]:
    if evaluation is None:
        return None, "not_evaluated"
    eval_score = _read_value(evaluation, "source_grounding_score")
    overall_pass = bool(_read_value(evaluation, "overall_pass", False))
    return eval_score, "pass" if overall_pass else "fail"


def persist_research_run(
    *,
    run_id: str,
    ticker: str,
    status: str,
    report,
    evidence,
    evaluation,
    node_runs: list[dict] | None = None,
    missing_data_notes: list[str] | None = None,
) -> dict:
    node_runs = node_runs or []
    missing_data_notes = missing_data_notes or []
    request_status = _RESULT_STATUS_FOR.get(status, "failed")

    with session_scope() as session:
        analysis = AnalysisRepository(session)
        evidence_repo = EvidenceRepository(session)
        reports = ReportRepository(session)

        ticker_row = analysis.upsert_ticker(ticker)
        notes_text = "; ".join(missing_data_notes) or None
        request = analysis.create_request(
            ticker_row.id,
            "research",
            status=request_status,
            degraded_reason=notes_text if request_status == "degraded" else None,
            error_summary=notes_text if request_status == "failed" else None,
        )

        for index, run in enumerate(node_runs, start=1):
            analysis.record_node_run(
                request.id,
                run_id,
                run["node_name"],
                run.get("status", "success"),
                attempt_number=run.get("attempt_number", index),
                duration_ms=run.get("duration_ms"),
                error_type=run.get("error_type"),
                error_message=run.get("error_message"),
            )

        result = {"request_id": request.id, "report_id": None, "report_version_id": None}
        if report is None:
            return result

        stage = "final" if status == "success" else "draft"
        report_status = "final" if status == "success" else "draft"
        report_title = _read_value(report, "title", "")
        report_row = reports.create_report(
            request.id, ticker_row.id, title=report_title, status=report_status
        )
        version = reports.add_version(
            report_row.id,
            1,
            stage,
            _report_json(report),
            "",
            created_by_node="coordinator_node",
        )
        disclaimer = _read_value(report, "disclaimer", "")
        disclaimer_present = bool(disclaimer.strip()) if isinstance(disclaimer, str) else False
        eval_score, safety = _evaluation_summary(evaluation)
        reports.set_status(
            report_row.id,
            safety_status=safety,
            evaluation_score=eval_score,
            disclaimer_present=disclaimer_present,
        )

        for item in evidence:
            row = evidence_repo.add_evidence(item, request_id=request.id, ticker_id=ticker_row.id)
            evidence_repo.add_citation(
                version.id, row.id, section_name="evidence_summary", claim_text=item.description
            )

        result["report_id"] = report_row.id
        result["report_version_id"] = version.id
        return result
