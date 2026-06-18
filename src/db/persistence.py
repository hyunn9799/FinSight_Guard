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
        report_row = reports.create_report(
            request.id, ticker_row.id, title=getattr(report, "title", ""), status=report_status
        )
        version = reports.add_version(
            report_row.id,
            1,
            stage,
            report.model_dump(mode="json"),
            "",
            created_by_node="coordinator_node",
        )
        disclaimer_present = bool(getattr(report, "disclaimer", "").strip())
        eval_score = evaluation.source_grounding_score if evaluation is not None else None
        safety = "pass" if (evaluation is not None and evaluation.overall_pass) else "not_evaluated"
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
