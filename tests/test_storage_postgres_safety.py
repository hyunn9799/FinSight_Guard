FORBIDDEN_TOKENS = (
    "broker", "brokerage", "order", "execute", "execution", "trade_exec",
    "buy_signal", "sell_signal", "guaranteed", "guarantee_target", "api_secret",
    "password", "credential", "session_token",
)

US2_TABLES = {
    "index_projection_status",
    "keyword_terms",
    "wave_rules",
    "wave_scenarios",
    "wave_invalidation_conditions",
    "wave_scenario_rules",
    "evidence_paths",
    "evidence_path_steps",
}


def test_no_forbidden_columns_in_schema():
    from src.db.models import Base

    offenders = []
    for table in Base.metadata.tables.values():
        for column in table.columns:
            lowered = column.name.lower()
            for token in FORBIDDEN_TOKENS:
                if token in lowered:
                    offenders.append(f"{table.name}.{column.name}")
    assert offenders == [], f"forbidden columns present: {offenders}"


def test_us2_tables_present_in_metadata():
    from src.db.models import Base

    assert US2_TABLES.issubset(set(Base.metadata.tables.keys()))
