"""
ORM model for the `shareholding_patterns` table.

One row per company per quarter. Sourced from manually-verified metadata.json
values — NOT from PDF extraction. Kept as a dedicated table (not folded into
extracted_features) to preserve the source separation.

Pledge pct convention:
  - Use 0.0 when the source shows "-" (genuinely nil pledge)
  - Use NULL only if the value is truly unknown / not reported
  - Never use 0.0 as a stand-in for a missing promoter_holding_pct value
"""

from datetime import datetime

from sqlalchemy import Date, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class ShareholdingPattern(Base):
    __tablename__ = "shareholding_patterns"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"))
    fiscal_year: Mapped[str | None] = mapped_column(String(10))    # e.g. 'FY2018'
    fiscal_quarter: Mapped[str | None] = mapped_column(String(10)) # 'Q1', 'Q2', 'Q3', 'Q4'
    period_end: Mapped[datetime | None] = mapped_column(Date)
    promoter_holding_pct: Mapped[float | None] = mapped_column(Numeric(5, 2))
    # ^ Always fill actual value; never use 0.0 unless genuinely zero
    pledge_pct: Mapped[float | None] = mapped_column(Numeric(5, 2))
    # ^ Use 0.0 when source shows "-"; NULL only if truly unknown
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Relationships
    company: Mapped["Company"] = relationship(  # noqa: F821
        "Company", back_populates="shareholding_patterns"
    )

    def __repr__(self) -> str:
        return (
            f"<ShareholdingPattern(id={self.id}, company_id={self.company_id}, "
            f"fiscal_year={self.fiscal_year!r}, fiscal_quarter={self.fiscal_quarter!r})>"
        )
