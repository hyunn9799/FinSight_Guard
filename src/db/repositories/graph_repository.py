"""Repository for canonical wave-theory graph records and evidence paths (US2)."""

from src.db.models import (
    EvidencePath,
    EvidencePathStep,
    WaveInvalidationCondition,
    WaveRule,
    WaveScenario,
    WaveScenarioRule,
)
from src.db.repositories.base import BaseRepository


class GraphRepository(BaseRepository):
    def add_rule(
        self,
        *,
        rule_code: str,
        name: str,
        rule_type: str,
        description: str = "",
        status: str = "active",
        source_document_id=None,
    ) -> WaveRule:
        rule = WaveRule(
            rule_code=rule_code,
            name=name,
            rule_type=rule_type,
            description=description,
            status=status,
            source_document_id=source_document_id,
        )
        self.session.add(rule)
        self.session.flush()
        return rule

    def add_scenario(
        self,
        *,
        name: str,
        ticker_id=None,
        description: str = "",
        timeframe: str | None = None,
        status: str = "active",
        confidence_label: str | None = None,
    ) -> WaveScenario:
        scenario = WaveScenario(
            name=name,
            ticker_id=ticker_id,
            description=description,
            timeframe=timeframe,
            status=status,
            confidence_label=confidence_label,
        )
        self.session.add(scenario)
        self.session.flush()
        return scenario

    def add_invalidation_condition(
        self,
        scenario_id,
        *,
        condition_text: str,
        metric_name: str | None = None,
        threshold_value=None,
        direction: str | None = None,
        source_document_id=None,
    ) -> WaveInvalidationCondition:
        if threshold_value is not None and not isinstance(threshold_value, dict):
            threshold_value = {"value": threshold_value}
        condition = WaveInvalidationCondition(
            scenario_id=scenario_id,
            condition_text=condition_text,
            metric_name=metric_name,
            threshold_value=threshold_value,
            direction=direction,
            source_document_id=source_document_id,
        )
        self.session.add(condition)
        self.session.flush()
        return condition

    def link_scenario_rule(self, scenario_id, rule_id, *, role: str) -> WaveScenarioRule:
        existing = (
            self.session.query(WaveScenarioRule)
            .filter(
                WaveScenarioRule.scenario_id == scenario_id,
                WaveScenarioRule.rule_id == rule_id,
                WaveScenarioRule.role == role,
            )
            .one_or_none()
        )
        if existing is not None:
            return existing
        link = WaveScenarioRule(scenario_id=scenario_id, rule_id=rule_id, role=role)
        self.session.add(link)
        self.session.flush()
        return link

    def list_rules_for_scenario(self, scenario_id) -> list[WaveRule]:
        return (
            self.session.query(WaveRule)
            .join(WaveScenarioRule, WaveScenarioRule.rule_id == WaveRule.id)
            .filter(WaveScenarioRule.scenario_id == scenario_id)
            .order_by(WaveRule.rule_code)
            .all()
        )
