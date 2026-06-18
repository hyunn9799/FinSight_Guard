"""Repository for evidence items and report-evidence citations."""

from src.db.models import EvidenceItemRecord, ReportEvidenceCitation
from src.db.repositories.base import BaseRepository
from src.evidence.evidence_schema import EvidenceItem


class EvidenceRepository(BaseRepository):
    def add_evidence(
        self,
        item: EvidenceItem,
        *,
        request_id=None,
        ticker_id=None,
        analysis_result_id=None,
        source_document_id=None,
    ) -> EvidenceItemRecord:
        metric_value = None if item.metric_value is None else {"value": item.metric_value}
        record = EvidenceItemRecord(
            evidence_id=item.evidence_id,
            request_id=request_id,
            ticker_id=ticker_id,
            analysis_result_id=analysis_result_id,
            source_document_id=source_document_id,
            source_type=item.source_type,
            source_name=item.source_name,
            source_url=item.source_url,
            collected_at=item.collected_at,
            metric_name=item.metric_name,
            metric_value=metric_value,
            description=item.description,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def list_for_request(self, request_id) -> list[EvidenceItemRecord]:
        return (
            self.session.query(EvidenceItemRecord)
            .filter(EvidenceItemRecord.request_id == request_id)
            .order_by(EvidenceItemRecord.created_at)
            .all()
        )

    def add_citation(
        self,
        report_version_id,
        evidence_item_id,
        *,
        section_name: str = "",
        claim_text: str = "",
    ) -> ReportEvidenceCitation:
        citation = ReportEvidenceCitation(
            report_version_id=report_version_id,
            evidence_item_id=evidence_item_id,
            section_name=section_name,
            claim_text=claim_text,
        )
        self.session.add(citation)
        self.session.flush()
        return citation
