"""Repository for research-only portfolios and items (US3). No order execution."""

from datetime import UTC, datetime

from src.db.models import Portfolio, PortfolioItem
from src.db.repositories.base import BaseRepository


class PortfolioRepository(BaseRepository):
    def create_portfolio(
        self,
        *,
        name: str,
        description: str | None = None,
        base_currency: str | None = None,
        status: str = "active",
        user_id=None,
    ) -> Portfolio:
        portfolio = Portfolio(
            user_id=user_id,
            name=name,
            description=description,
            base_currency=base_currency,
            status=status,
        )
        self.session.add(portfolio)
        self.session.flush()
        return portfolio

    def add_item(
        self,
        portfolio_id,
        ticker_id,
        *,
        label: str | None = None,
        quantity_note: str | None = None,
        cost_basis_note: str | None = None,
        metadata: dict | None = None,
    ) -> PortfolioItem:
        item = PortfolioItem(
            portfolio_id=portfolio_id,
            ticker_id=ticker_id,
            label=label,
            quantity_note=quantity_note,
            cost_basis_note=cost_basis_note,
            item_metadata=metadata if metadata is not None else {},
        )
        self.session.add(item)
        self.session.flush()
        return item

    def list_items(self, portfolio_id) -> list[PortfolioItem]:
        return (
            self.session.query(PortfolioItem)
            .filter(
                PortfolioItem.portfolio_id == portfolio_id,
                PortfolioItem.deleted_at.is_(None),
            )
            .order_by(PortfolioItem.created_at)
            .all()
        )

    def list_for_user(self, user_id) -> list[Portfolio]:
        return (
            self.session.query(Portfolio)
            .filter(
                Portfolio.user_id == user_id,
                Portfolio.deleted_at.is_(None),
            )
            .order_by(Portfolio.created_at)
            .all()
        )

    def soft_delete_portfolio(self, record: Portfolio) -> Portfolio:
        record.deleted_at = datetime.now(UTC)
        self.session.flush()
        return record
