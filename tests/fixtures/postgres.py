"""Deterministic sample-record builders for PostgreSQL tests."""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.db.models import AnalysisRequest, Ticker, User


def make_user(
    session: Session,
    *,
    email: str | None = None,
    display_name: str | None = None,
    role: str = "demo",
    status: str = "active",
) -> User:
    user = User(email=email, display_name=display_name, role=role, status=status)
    session.add(user)
    session.flush()
    return user


def make_ticker(session: Session, symbol: str = "AAPL", market: str = "NASDAQ", **kw) -> Ticker:
    ticker = Ticker(symbol=symbol, market=market, **kw)
    session.add(ticker)
    session.flush()
    return ticker


def make_request(session: Session, ticker: Ticker, request_type: str = "research", **kw) -> AnalysisRequest:
    request = AnalysisRequest(
        ticker_id=ticker.id,
        request_type=request_type,
        status=kw.pop("status", "pending"),
        created_at=kw.pop("created_at", datetime.now(UTC)),
        **kw,
    )
    session.add(request)
    session.flush()
    return request
