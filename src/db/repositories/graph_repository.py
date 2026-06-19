"""Repository for canonical wave-theory graph records and evidence paths (US2)."""

from sqlalchemy.dialects.postgresql import insert as pg_insert

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
        stmt = (
            pg_insert(WaveScenarioRule)
            .values(scenario_id=scenario_id, rule_id=rule_id, role=role)
            .on_conflict_do_update(
                index_elements=["scenario_id", "rule_id", "role"],
                set_={"scenario_id": WaveScenarioRule.scenario_id},
            )
            .returning(WaveScenarioRule)
        )
        return self.session.scalars(stmt).one()

    def list_rules_for_scenario(self, scenario_id) -> list[WaveRule]:
        return (
            self.session.query(WaveRule)
            .join(WaveScenarioRule, WaveScenarioRule.rule_id == WaveRule.id)
            .filter(WaveScenarioRule.scenario_id == scenario_id)
            .order_by(WaveRule.rule_code)
            .all()
        )

    def add_evidence_path(
        self,
        *,
        path_type: str,
        path_summary: str,
        source_node_ref: str,
        target_node_ref: str,
        request_id=None,
        ticker_id=None,
        confidence_label: str | None = None,
    ) -> EvidencePath:
        path = EvidencePath(
            path_type=path_type,
            path_summary=path_summary,
            source_node_ref=source_node_ref,
            target_node_ref=target_node_ref,
            request_id=request_id,
            ticker_id=ticker_id,
            confidence_label=confidence_label,
        )
        self.session.add(path)
        self.session.flush()
        return path

    def add_path_step(
        self,
        evidence_path_id,
        *,
        step_index: int,
        node_table: str,
        node_id,
        relationship_type: str,
        description: str = "",
    ) -> EvidencePathStep:
        step = EvidencePathStep(
            evidence_path_id=evidence_path_id,
            step_index=step_index,
            node_table=node_table,
            node_id=node_id,
            relationship_type=relationship_type,
            description=description,
        )
        self.session.add(step)
        self.session.flush()
        return step

    def list_steps(self, evidence_path_id) -> list[EvidencePathStep]:
        return (
            self.session.query(EvidencePathStep)
            .filter(EvidencePathStep.evidence_path_id == evidence_path_id)
            .order_by(EvidencePathStep.step_index)
            .all()
        )

    def persist_evidence_path_from_spec(
        self,
        spec: dict | None,
        *,
        evidence_id_to_uuid: dict,
        request_id=None,
        ticker_id=None,
    ) -> EvidencePath | None:
        # build_evidence_path_spec returns None when no evidence-backed edges
        # exist; accept that directly so the two functions compose cleanly.
        if not spec:
            return None
        steps = [
            step
            for step in spec.get("steps", [])
            if step.get("evidence_id") in evidence_id_to_uuid
        ]
        if not steps:
            return None
        path = self.add_evidence_path(
            path_type=spec["path_type"],
            path_summary=spec["path_summary"],
            source_node_ref=spec["source_node_ref"],
            target_node_ref=spec["target_node_ref"],
            request_id=request_id,
            ticker_id=ticker_id,
            confidence_label=spec.get("confidence_label"),
        )
        for index, step in enumerate(steps):
            self.add_path_step(
                path.id,
                step_index=index,
                node_table="evidence_items",
                node_id=evidence_id_to_uuid[step["evidence_id"]],
                relationship_type=step["relationship_type"],
                description=step.get("description", ""),
            )
        return path
