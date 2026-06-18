import os
import pytest

REQUIRES_DB = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set"
)


@REQUIRES_DB
def test_db_session_fixture_roundtrips_a_ticker(db_session):
    from src.db.models import Ticker
    db_session.add(Ticker(symbol="AAPL", market="NASDAQ"))
    db_session.flush()
    found = db_session.query(Ticker).filter_by(symbol="AAPL").one()
    assert found.market == "NASDAQ"
    assert found.id is not None
