"""Repository for provider raw/normalized/derived contract records (006)."""

from __future__ import annotations

from src.db.models import (
    ProviderCompanyProfile,
    ProviderFinancialMetric,
    ProviderNewsEvent,
    ProviderTechnicalAnalysisResult,
    ProviderWaveAnalysisResult,
    RawProviderResponse,
)
from src.db.repositories.base import BaseRepository


class ProviderRepository(BaseRepository):
    def create_raw_response(self, **fields) -> RawProviderResponse:
        row = RawProviderResponse(**fields)
        self.session.add(row)
        return row

    def create_company_profile(self, **fields) -> ProviderCompanyProfile:
        row = ProviderCompanyProfile(**fields)
        self.session.add(row)
        return row

    def create_news_event(self, **fields) -> ProviderNewsEvent:
        row = ProviderNewsEvent(**fields)
        self.session.add(row)
        return row

    def create_financial_metric(self, **fields) -> ProviderFinancialMetric:
        row = ProviderFinancialMetric(**fields)
        self.session.add(row)
        return row

    def create_technical_result(self, **fields) -> ProviderTechnicalAnalysisResult:
        row = ProviderTechnicalAnalysisResult(**fields)
        self.session.add(row)
        return row

    def create_wave_result(self, **fields) -> ProviderWaveAnalysisResult:
        row = ProviderWaveAnalysisResult(**fields)
        self.session.add(row)
        return row

    def get_normalized_for_raw(self, raw_response_id) -> dict:
        """Return all normalized records that trace to a raw response."""
        q = self.session.query
        return {
            "company_profiles": q(ProviderCompanyProfile)
            .filter(ProviderCompanyProfile.raw_response_id == raw_response_id).all(),
            "news_events": q(ProviderNewsEvent)
            .filter(ProviderNewsEvent.raw_response_id == raw_response_id).all(),
            "financial_metrics": q(ProviderFinancialMetric)
            .filter(ProviderFinancialMetric.raw_response_id == raw_response_id).all(),
        }
