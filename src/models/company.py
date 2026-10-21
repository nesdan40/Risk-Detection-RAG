"""
ORM model for the `companies` table.

One row per company. No distress/healthy labels — label-leakage prevention
is a hard project constraint. Ground truth lives exclusively in
ground_truth.json on the filesystem, loaded only during Semester 8 evaluation.
"""

from datetime import datetime

from sqlalchemy import Date, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(100))
    listing_type: Mapped[str | None] = mapped_column(String(20))
    # ^ 'equity' or 'debt_only'
    collection_window_start: Mapped[datetime | None] = mapped_column(Date)
    collection_window_end: Mapped[datetime | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Relationships
    documents: Mapped[list["Document"]] = relationship(  # noqa: F821
        "Document", back_populates="company"
    )
    shareholding_patterns: Mapped[list["ShareholdingPattern"]] = relationship(  # noqa: F821
        "ShareholdingPattern", back_populates="company"
    )
    extracted_features: Mapped[list["ExtractedFeature"]] = relationship(  # noqa: F821
        "ExtractedFeature", back_populates="company"
    )
    predictions: Mapped[list["Prediction"]] = relationship(  # noqa: F821
        "Prediction", back_populates="company"
    )

    def __repr__(self) -> str:
        return f"<Company(id={self.id}, ticker={self.ticker!r}, name={self.name!r})>"
