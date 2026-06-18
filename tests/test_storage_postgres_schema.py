import importlib


def test_config_exposes_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    import src.config as config
    importlib.reload(config)
    assert config.DATABASE_URL == "postgresql+psycopg://u:p@localhost:5432/db"
