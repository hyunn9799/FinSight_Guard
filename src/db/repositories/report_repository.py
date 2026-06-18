"""Repository for reports and immutable report versions."""

from sqlalchemy import func

from src.db.models import Report, ReportVersion
from src.db.repositories.base import BaseRepository


class ReportRepository(BaseRepository):
    def create_report(
        self,
        request_id,
        ticker_id,
        *,
        title: str = "",
        language: str = "ko",
        status: str = "draft",
    ) -> Report:
        report = Report(
            request_id=request_id,
            ticker_id=ticker_id,
            title=title,
            language=language,
            status=status,
        )
        self.session.add(report)
        self.session.flush()
        return report

    def next_version_number(self, report_id) -> int:
        current_max = (
            self.session.query(func.max(ReportVersion.version_number))
            .filter(ReportVersion.report_id == report_id)
            .scalar()
        )
        return (current_max or 0) + 1

    def add_version(
        self,
        report_id,
        version_number: int,
        stage: str,
        report_json: dict,
        report_markdown: str,
        *,
        created_by_node: str | None = None,
    ) -> ReportVersion:
        version = ReportVersion(
            report_id=report_id,
            version_number=version_number,
            stage=stage,
            report_json=report_json,
            report_markdown=report_markdown,
            created_by_node=created_by_node,
        )
        self.session.add(version)
        self.session.flush()
        report = self.session.get(Report, report_id)
        report.current_version_id = version.id
        self.session.flush()
        return version

    def set_status(
        self,
        report_id,
        *,
        status: str | None = None,
        safety_status: str | None = None,
        evaluation_score: float | None = None,
        disclaimer_present: bool | None = None,
    ) -> Report:
        report = self.session.get(Report, report_id)
        if status is not None:
            report.status = status
        if safety_status is not None:
            report.safety_status = safety_status
        if evaluation_score is not None:
            report.evaluation_score = evaluation_score
        if disclaimer_present is not None:
            report.disclaimer_present = disclaimer_present
        self.session.flush()
        return report
