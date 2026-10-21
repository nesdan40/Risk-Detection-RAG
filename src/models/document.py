"""
ORM model for the `documents` table.

One row per ingested PDF file. Tracks identity, origin, and processing state.
The three boolean flags (text_extracted, chunked, embedded) represent current
state; full event history is in the processing_log table.
"""

from datetime import datetime

from sqlalchemy import Boolean, Date, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"))
    file_name: Mapped[str | None] = mapped_column(String(255))
    file_path: Mapped[str | None] = mapped_column(Text)
    document_type: Mapped[str | None] = mapped_column(String(50))
    # ^ 'annual_report', 'quarterly_result', 'corporate_announcement', 'credit_rating'
    document_category: Mapped[str | None] = mapped_column(String(100))
    # ^ For announcements: 'Board Meeting', 'Director Change', 'Appointment',
    #   'Resignation', 'Press Release', 'News Clarification', 'News Verification',
    #   'AGM / Annual General Meeting Updates', 'Financial Results'
    fiscal_year: Mapped[str | None] = mapped_column(String(10))    # e.g. 'FY2017'
    fiscal_quarter: Mapped[str | None] = mapped_column(String(10)) # e.g. 'Q1'; NULL for annual
    period_start: Mapped[datetime | None] = mapped_column(Date)
    period_end: Mapped[datetime | None] = mapped_column(Date)
    source: Mapped[str | None] = mapped_column(String(50))
    # ^ 'BSE', 'NSE', 'company_ir', 'MCA'
    text_extracted: Mapped[bool] = mapped_column(Boolean, server_default="false")
    chunked: Mapped[bool] = mapped_column(Boolean, server_default="false")
    embedded: Mapped[bool] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Relationships
    company: Mapped["Company"] = relationship(  # noqa: F821
        "Company", back_populates="documents"
    )
    processing_logs: Mapped[list["ProcessingLog"]] = relationship(  # noqa: F821
        "ProcessingLog", back_populates="document"
    )
    extracted_features: Mapped[list["ExtractedFeature"]] = relationship(  # noqa: F821
        "ExtractedFeature", back_populates="document"
    )

    def __repr__(self) -> str:
        return (
            f"<Document(id={self.id}, company_id={self.company_id}, "
            f"file_name={self.file_name!r}, document_type={self.document_type!r})>"
        )
