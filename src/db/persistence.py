"""Persist a full research workflow run to the PostgreSQL source of truth."""

from src.db.postgres import session_scope
from src.db.repositories.analysis_repository import AnalysisRepository
from src.db.repositories.evidence_repository import EvidenceRepository
from src.db.repositories.graph_repository import GraphRepository
from src.db.repositories.projection_repository import ProjectionRepository
from src.db.repositories.provider_repository import ProviderRepository
from src.db.repositories.report_repository import ReportRepository
from src.graph_rag.graph_context_builder import build_evidence_path_spec
from src.providers.entities import (
    CompanyProfile,
    FinancialMetric,
    NewsEvent,
)
from src.providers.normalization import NormalizationResult

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
    graph_context=None,
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

        result = {
            "request_id": request.id,
            "report_id": None,
            "report_version_id": None,
            "evidence_path_id": None,
        }
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

        evidence_id_to_uuid: dict = {}
        for item in evidence:
            row = evidence_repo.add_evidence(item, request_id=request.id, ticker_id=ticker_row.id)
            evidence_repo.add_citation(
                version.id, row.id, section_name="evidence_summary", claim_text=item.description
            )
            if getattr(item, "evidence_id", None):
                evidence_id_to_uuid[item.evidence_id] = row.id

        evidence_path = GraphRepository(session).persist_evidence_path_from_spec(
            build_evidence_path_spec(graph_context) if graph_context is not None else None,
            evidence_id_to_uuid=evidence_id_to_uuid,
            request_id=request.id,
            ticker_id=ticker_row.id,
        )

        if evidence_path is not None:
            ProjectionRepository(session).upsert_status(
                source_table="evidence_paths",
                source_id=evidence_path.id,
                target_system="neo4j",
                projection_type="graph_evidence_path",
                projection_key=f"evidence_path:{evidence_path.id}",
                idempotency_key=f"evidence_paths:{evidence_path.id}:neo4j:graph_evidence_path",
                status="pending",
            )

        result["report_id"] = report_row.id
        result["report_version_id"] = version.id
        result["evidence_path_id"] = evidence_path.id if evidence_path is not None else None
        return result


def persist_normalization(
    session, *, request_id, ticker_id, raw_kwargs: dict, normalization_result: NormalizationResult
) -> dict:
    """Persist a raw response and its normalized records (006 extension tables only).

    Does NOT mutate any 004-owned table; only writes provider_* / raw_provider_responses.
    """
    repo = ProviderRepository(session)
    raw = repo.create_raw_response(request_id=request_id, ticker_id=ticker_id, **raw_kwargs)
    session.flush()
    out: dict = {"raw_response_id": raw.id, "news_events": [], "company_profiles": [], "financial_metrics": []}
    for rec in normalization_result.records:
        warnings = [w.model_dump() for w in rec.warnings]
        if isinstance(rec, NewsEvent):
            out["news_events"].append(repo.create_news_event(
                request_id=request_id, ticker_id=ticker_id, raw_response_id=raw.id,
                title=rec.title, summary=rec.summary, source_name=rec.source_name,
                source_url=rec.source_url, published_at=rec.published_at,
                event_type=rec.event_type, sentiment_label=rec.sentiment_label,
                risk_tags=rec.risk_tags, normalization_status=rec.normalization_status.value,
                warnings=warnings, evidence_id=rec.evidence_id,
            ))
        elif isinstance(rec, CompanyProfile):
            out["company_profiles"].append(repo.create_company_profile(
                request_id=request_id, ticker_id=ticker_id, raw_response_id=raw.id,
                company_name=rec.company_name, legal_name=rec.legal_name, sector=rec.sector,
                industry=rec.industry, country=rec.country, exchange=rec.exchange,
                currency=rec.currency, description=rec.description,
                normalization_status=rec.normalization_status.value,
                warnings=warnings, evidence_id=rec.evidence_id,
            ))
        elif isinstance(rec, FinancialMetric):
            out["financial_metrics"].append(repo.create_financial_metric(
                request_id=request_id, ticker_id=ticker_id, raw_response_id=raw.id,
                metric_name=rec.metric_name,
                metric_value=None if rec.metric_value is None else str(rec.metric_value),
                period=rec.period, currency=rec.currency, unit=rec.unit,
                source_name=rec.source_name, source_url=rec.source_url,
                normalization_status=rec.normalization_status.value,
                warnings=warnings, evidence_id=rec.evidence_id,
            ))
        else:
            raise TypeError(f"persist_normalization received unsupported record type: {type(rec).__name__}")
    return out
