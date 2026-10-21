"""
ORM model for the `extracted_features` table.

Primary feature store for both the rule-based engine (Sem 7) and ML (Sem 8).
Uses typed columns for known signals so downstream ML can read feature rows
directly without transformation, plus a JSONB overflow column for anything
discovered later.

NULL semantics: missing values are stored as NULL, not defaulted to zero —
except where explicitly defined (e.g., pledge_pct = 0.0 when source shows "-").
"""

from datetime import datetime

from sqlalchemy import Boolean, Date, ForeignKey, Index, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class ExtractedFeature(Base):
    __tablename__ = "extracted_features"

    # ── Partial unique indexes for upsert conflict targets ────────────────────
    # These mirror the indexes in migration 0002 and are used by feature_writer
    # for ON CONFLICT DO UPDATE resolution.
    __table_args__ = (
        # Document-level financial rows: (company_id, document_id, feature_type)
        Index(
            "ix_ef_financial_upsert",
            "company_id",
            "document_id",
            "feature_type",
            unique=True,
            postgresql_where="document_id IS NOT NULL AND feature_type = 'financial'",
        ),
        # Document-level credit rows: (company_id, document_id, feature_type)
        Index(
            "ix_ef_credit_upsert",
            "company_id",
            "document_id",
            "feature_type",
            unique=True,
            postgresql_where="document_id IS NOT NULL AND feature_type = 'credit'",
        ),
        # Governance period aggregates: document_id is always NULL
        Index(
            "ix_ef_governance_upsert",
            "company_id",
            "fiscal_year",
            "fiscal_quarter",
            "feature_type",
            unique=True,
            postgresql_where="document_id IS NULL AND feature_type = 'governance'",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"))
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"))
    # ^ Nullable — some features are quarter-level aggregates, not tied to one doc
    fiscal_year: Mapped[str | None] = mapped_column(String(10))
    fiscal_quarter: Mapped[str | None] = mapped_column(String(10))  # NULL for annual-level
    period_end: Mapped[datetime | None] = mapped_column(Date)
    feature_type: Mapped[str | None] = mapped_column(String(50))
    # ^ 'financial', 'governance', 'credit', 'shareholding'

    # ── Financial signals (nullable — not every doc has all) ──────────────────
    revenue: Mapped[float | None] = mapped_column(Numeric(20, 2))
    revenue_growth_pct: Mapped[float | None] = mapped_column(Numeric(8, 2))
    pat: Mapped[float | None] = mapped_column(Numeric(20, 2))
    ebitda_margin_pct: Mapped[float | None] = mapped_column(Numeric(8, 2))
    debt_to_equity: Mapped[float | None] = mapped_column(Numeric(8, 4))
    interest_coverage: Mapped[float | None] = mapped_column(Numeric(8, 4))
    current_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    cfo: Mapped[float | None] = mapped_column(Numeric(20, 2))

    # ── Governance signals (counts per period) ────────────────────────────────
    board_meeting_count: Mapped[int | None] = mapped_column(Integer)
    director_change_count: Mapped[int | None] = mapped_column(Integer)
    appointment_count: Mapped[int | None] = mapped_column(Integer)
    resignation_count: Mapped[int | None] = mapped_column(Integer)
    agm_delay_flag: Mapped[bool | None] = mapped_column(Boolean)
    clarification_count: Mapped[int | None] = mapped_column(Integer)
    # ^ Press Release + News Clarification + News Verification
    press_release_count: Mapped[int | None] = mapped_column(Integer)

    # ── Credit signals ────────────────────────────────────────────────────────
    credit_rating: Mapped[str | None] = mapped_column(String(20))
    # ^ 'AAA', 'AA+', 'D', etc.
    rating_outlook: Mapped[str | None] = mapped_column(String(20))
    # ^ 'Stable', 'Negative', 'Watch'
    rating_change_flag: Mapped[bool | None] = mapped_column(Boolean)
    # ^ TRUE if rating changed this period

    # ── Overflow ──────────────────────────────────────────────────────────────
    additional_signals: Mapped[dict | None] = mapped_column(JSONB)
    # ^ Anything that doesn't fit the typed columns above

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Relationships
    company: Mapped["Company"] = relationship(  # noqa: F821
        "Company", back_populates="extracted_features"
    )
    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document", back_populates="extracted_features"
    )

    def __repr__(self) -> str:
        return (
            f"<ExtractedFeature(id={self.id}, company_id={self.company_id}, "
            f"feature_type={self.feature_type!r}, fiscal_year={self.fiscal_year!r}, "
            f"fiscal_quarter={self.fiscal_quarter!r})>"
        )
