"""Repository for tickers, requests, node runs, results, and log events."""

from datetime import datetime

from sqlalchemy.exc import IntegrityError

from src.db.models import (
    AnalysisRequest,
    AnalysisResult,
    StructuredLogEvent,
    Ticker,
    WorkflowNodeRun,
)
from src.db.repositories.base import BaseRepository


class AnalysisRepository(BaseRepository):
    def upsert_ticker(self, symbol: str, market: str | None = None, **fields) -> Ticker:
        symbol = symbol.strip().upper()
        existing = (
            self.session.query(Ticker)
            .filter(Ticker.symbol == symbol, Ticker.market == market)
            .one_or_none()
        )
        if existing is not None:
            return existing
        try:
            with self.session.begin_nested():
                ticker = Ticker(symbol=symbol, market=market, **fields)
                self.session.add(ticker)
                self.session.flush()
                return ticker
        except IntegrityError:
            return (
                self.session.query(Ticker)
                .filter(Ticker.symbol == symbol, Ticker.market == market)
                .one()
            )

    def create_request(
        self,
        ticker_id,
        request_type: str,
        *,
        user_id=None,
        parameters: dict | None = None,
        status: str = "pending",
        **fields,
    ) -> AnalysisRequest:
        request = AnalysisRequest(
            ticker_id=ticker_id,
            user_id=user_id,
            request_type=request_type,
            parameters=parameters or {},
            status=status,
            **fields,
        )
        self.session.add(request)
        self.session.flush()
        return request

    def update_request_status(
        self,
        request_id,
        status: str,
        *,
        degraded_reason: str | None = None,
        warning_summary: str | None = None,
        error_summary: str | None = None,
        completed_at: datetime | None = None,
    ) -> AnalysisRequest:
        request = self.session.get(AnalysisRequest, request_id)
        request.status = status
        if degraded_reason is not None:
            request.degraded_reason = degraded_reason
        if warning_summary is not None:
            request.warning_summary = warning_summary
        if error_summary is not None:
            request.error_summary = error_summary
        if completed_at is not None:
            request.completed_at = completed_at
        self.session.flush()
        return request

    def record_node_run(
        self,
        request_id,
        run_id: str,
        node_name: str,
        status: str,
        *,
        attempt_number: int = 1,
        duration_ms: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        evaluation_score: float | None = None,
        metadata: dict | None = None,
    ) -> WorkflowNodeRun:
        node = WorkflowNodeRun(
            request_id=request_id,
            run_id=run_id,
            node_name=node_name,
            status=status,
            attempt_number=attempt_number,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=error_message,
            evaluation_score=evaluation_score,
            node_metadata=metadata or {},
        )
        self.session.add(node)
        self.session.flush()
        return node

    def add_result(
        self,
        request_id,
        result_type: str,
        *,
        ticker_id=None,
        summary: str = "",
        metrics: dict | None = None,
        warnings: list | None = None,
        missing_data_notes: list | None = None,
        status: str = "success",
    ) -> AnalysisResult:
        result = AnalysisResult(
            request_id=request_id,
            ticker_id=ticker_id,
            result_type=result_type,
            summary=summary,
            metrics=metrics or {},
            warnings=warnings or [],
            missing_data_notes=missing_data_notes or [],
            status=status,
        )
        self.session.add(result)
        self.session.flush()
        return result

    def add_log_event(
        self,
        *,
        event_name: str,
        status: str,
        occurred_at: datetime,
        request_id=None,
        run_id: str | None = None,
        ticker_id=None,
        node_name: str | None = None,
        message: str | None = None,
        error_message: str | None = None,
        evaluation_score: float | None = None,
    ) -> StructuredLogEvent:
        event = StructuredLogEvent(
            event_name=event_name,
            status=status,
            occurred_at=occurred_at,
            request_id=request_id,
            run_id=run_id,
            ticker_id=ticker_id,
            node_name=node_name,
            message=message,
            error_message=error_message,
            evaluation_score=evaluation_score,
        )
        self.session.add(event)
        self.session.flush()
        return event
