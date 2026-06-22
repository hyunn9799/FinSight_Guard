"""SQLAlchemy ORM models for the PostgreSQL source of truth (US1)."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, MetaData, Numeric, String, Text, UniqueConstraint, Boolean, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class UUIDMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class User(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "users"
    email: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    role: Mapped[str] = mapped_column(String, default="demo", nullable=False)
    status: Mapped[str] = mapped_column(String, default="active", nullable=False)
    anonymized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Ticker(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "tickers"
    __table_args__ = (UniqueConstraint("symbol", "market", postgresql_nulls_not_distinct=True),)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    market: Mapped[str | None] = mapped_column(String, nullable=True)
    exchange: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str | None] = mapped_column(String, nullable=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    sector: Mapped[str | None] = mapped_column(String, nullable=True)
    industry: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class AnalysisRequest(UUIDMixin, Base):
    __tablename__ = "analysis_requests"
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    request_type: Mapped[str] = mapped_column(String, nullable=False)
    horizon: Mapped[str | None] = mapped_column(String, nullable=True)
    risk_profile: Mapped[str | None] = mapped_column(String, nullable=True)
    parameters: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    degraded_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    warning_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WorkflowNodeRun(UUIDMixin, Base):
    __tablename__ = "workflow_node_runs"
    __table_args__ = (UniqueConstraint("request_id", "node_name", "attempt_number"),)
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    node_name: Mapped[str] = mapped_column(String, nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluation_score: Mapped[float | None] = mapped_column(Numeric(asdecimal=False), nullable=True)
    node_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)


class AnalysisResult(UUIDMixin, Base):
    __tablename__ = "analysis_results"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    result_type: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    missing_data_notes: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Report(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "reports"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("report_versions.id", use_alter=True, name="fk_reports_current_version"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String, default="", nullable=False)
    language: Mapped[str] = mapped_column(String, default="ko", nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    safety_status: Mapped[str] = mapped_column(String, default="not_evaluated", nullable=False)
    evaluation_score: Mapped[float | None] = mapped_column(Numeric(asdecimal=False), nullable=True)
    disclaimer_present: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class ReportVersion(UUIDMixin, Base):
    __tablename__ = "report_versions"
    __table_args__ = (UniqueConstraint("report_id", "version_number"),)
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reports.id"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    report_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    report_markdown: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_by_node: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvidenceItemRecord(UUIDMixin, Base):
    __tablename__ = "evidence_items"
    evidence_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    request_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("analysis_requests.id"), index=True, nullable=True
    )
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    analysis_result_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("analysis_results.id"), nullable=True)
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("source_documents.id"), nullable=True)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_name: Mapped[str] = mapped_column(String, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metric_name: Mapped[str] = mapped_column(String, nullable=False)
    metric_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ReportEvidenceCitation(UUIDMixin, Base):
    __tablename__ = "report_evidence_citations"
    __table_args__ = (
        UniqueConstraint("report_version_id", "evidence_item_id", "section_name", "claim_text"),
    )
    report_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("report_versions.id"), nullable=False)
    evidence_item_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("evidence_items.id"), nullable=False)
    section_name: Mapped[str] = mapped_column(String, default="", nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SourceDocument(UUIDMixin, Base):
    __tablename__ = "source_documents"
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    document_type: Mapped[str] = mapped_column(String, nullable=False)
    source_name: Mapped[str] = mapped_column(String, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_content_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    revision_group_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    supersedes_document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("source_documents.id"), nullable=True)
    doc_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DocumentChunk(UUIDMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("source_document_id", "chunk_index"),)
    source_document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("source_documents.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class StructuredLogEvent(UUIDMixin, Base):
    __tablename__ = "structured_log_events"
    request_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("analysis_requests.id"), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    node_name: Mapped[str | None] = mapped_column(String, nullable=True)
    event_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluation_score: Mapped[float | None] = mapped_column(Numeric(asdecimal=False), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)


# ---------------------------------------------------------------------------
# US2: Index-projection status, keyword terms, wave-theory graph records,
#      and explainable evidence paths
# ---------------------------------------------------------------------------


class IndexProjectionStatus(UUIDMixin, Base):
    __tablename__ = "index_projection_status"
    __table_args__ = (
        UniqueConstraint("target_system", "projection_type", "idempotency_key"),
        Index("ix_index_projection_status_source", "source_table", "source_id"),
    )
    source_table: Mapped[str] = mapped_column(String, nullable=False)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    target_system: Mapped[str] = mapped_column(String, nullable=False)
    projection_type: Mapped[str] = mapped_column(String, nullable=False)
    projection_key: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending", index=True, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String, nullable=False)


class KeywordTerm(UUIDMixin, Base):
    __tablename__ = "keyword_terms"
    __table_args__ = (
        UniqueConstraint("normalized_term", "language", postgresql_nulls_not_distinct=True),
    )
    term: Mapped[str] = mapped_column(String, nullable=False)
    normalized_term: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class WaveRule(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "wave_rules"
    rule_code: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    rule_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="active", nullable=False)
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("source_documents.id"), nullable=True
    )


class WaveScenario(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "wave_scenarios"
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    timeframe: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active", nullable=False)
    confidence_label: Mapped[str | None] = mapped_column(String, nullable=True)


class WaveInvalidationCondition(UUIDMixin, Base):
    __tablename__ = "wave_invalidation_conditions"
    scenario_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("wave_scenarios.id"), nullable=False)
    condition_text: Mapped[str] = mapped_column(Text, nullable=False)
    metric_name: Mapped[str | None] = mapped_column(String, nullable=True)
    threshold_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    direction: Mapped[str | None] = mapped_column(String, nullable=True)
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("source_documents.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class WaveScenarioRule(UUIDMixin, Base):
    __tablename__ = "wave_scenario_rules"
    __table_args__ = (UniqueConstraint("scenario_id", "rule_id", "role"),)
    scenario_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("wave_scenarios.id"), nullable=False)
    rule_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("wave_rules.id"), nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvidencePath(UUIDMixin, Base):
    __tablename__ = "evidence_paths"
    request_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("analysis_requests.id"), nullable=True
    )
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    path_type: Mapped[str] = mapped_column(String, nullable=False)
    path_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source_node_ref: Mapped[str] = mapped_column(String, nullable=False)
    target_node_ref: Mapped[str] = mapped_column(String, nullable=False)
    confidence_label: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvidencePathStep(UUIDMixin, Base):
    __tablename__ = "evidence_path_steps"
    __table_args__ = (UniqueConstraint("evidence_path_id", "step_index"),)
    evidence_path_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evidence_paths.id"), nullable=False
    )
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    node_table: Mapped[str] = mapped_column(String, nullable=False)
    node_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    relationship_type: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# US3: MVP user-experience tables
# ---------------------------------------------------------------------------


class UserSettings(UUIDMixin, Base):
    __tablename__ = "user_settings"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "setting_key", "scope", postgresql_nulls_not_distinct=True
        ),
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    setting_key: Mapped[str] = mapped_column(String, nullable=False)
    setting_value: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    scope: Mapped[str] = mapped_column(String, default="user", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Notification(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "notifications"
    __table_args__ = (
        Index("ix_notifications_user_id_created_at", "user_id", "created_at"),
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    ticker_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickers.id"), nullable=True)
    notification_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, default="", nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, default="unread", nullable=False)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Portfolio(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "portfolios"
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_currency: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active", nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PortfolioItem(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "portfolio_items"
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("portfolios.id"), index=True, nullable=False
    )
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    quantity_note: Mapped[str | None] = mapped_column(String, nullable=True)
    cost_basis_note: Mapped[str | None] = mapped_column(String, nullable=True)
    item_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# 006: Provider-agnostic MCP contract records (US2 extension — additive only)
# ---------------------------------------------------------------------------


class RawProviderResponse(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "raw_provider_responses"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    provider_name: Mapped[str] = mapped_column(String, nullable=False)
    provider_kind: Mapped[str] = mapped_column(String, nullable=False)  # news/financial/market_data
    provider_request: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    collected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    payload_body: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    payload_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class ProviderCompanyProfile(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "provider_company_profiles"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    raw_response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("raw_provider_responses.id"), nullable=False)
    company_name: Mapped[str] = mapped_column(String, nullable=False)
    legal_name: Mapped[str | None] = mapped_column(String, nullable=True)
    sector: Mapped[str | None] = mapped_column(String, nullable=True)
    industry: Mapped[str | None] = mapped_column(String, nullable=True)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    exchange: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalization_status: Mapped[str] = mapped_column(String, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    evidence_id: Mapped[str | None] = mapped_column(String, nullable=True)


class ProviderNewsEvent(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "provider_news_events"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    raw_response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("raw_provider_responses.id"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_name: Mapped[str | None] = mapped_column(String, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    event_type: Mapped[str | None] = mapped_column(String, nullable=True)
    sentiment_label: Mapped[str | None] = mapped_column(String, nullable=True)
    risk_tags: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    normalization_status: Mapped[str] = mapped_column(String, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    evidence_id: Mapped[str | None] = mapped_column(String, nullable=True)


class ProviderFinancialMetric(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "provider_financial_metrics"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    raw_response_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("raw_provider_responses.id"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String, nullable=False)
    metric_value: Mapped[str | None] = mapped_column(String, nullable=True)  # stringified; numeric kept in JSONB metadata if needed
    period: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str | None] = mapped_column(String, nullable=True)
    unit: Mapped[str | None] = mapped_column(String, nullable=True)
    source_name: Mapped[str | None] = mapped_column(String, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    normalization_status: Mapped[str] = mapped_column(String, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    evidence_id: Mapped[str | None] = mapped_column(String, nullable=True)


class ProviderTechnicalAnalysisResult(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "provider_technical_analysis_results"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    source_market_data_refs: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    indicator_values: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    trend_state: Mapped[str | None] = mapped_column(String, nullable=True)
    momentum_state: Mapped[str | None] = mapped_column(String, nullable=True)
    volatility_state: Mapped[str | None] = mapped_column(String, nullable=True)
    normalization_status: Mapped[str] = mapped_column(String, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    evidence_ids: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)


class ProviderWaveAnalysisResult(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "provider_wave_analysis_results"
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False)
    ticker_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickers.id"), nullable=False)
    source_market_data_refs: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    rule_refs: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    candidate_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    rule_statuses: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    confirmation_conditions: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    invalidation_conditions: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    uncertainty_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    evidence_ids: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
