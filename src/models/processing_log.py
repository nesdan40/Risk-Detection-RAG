"""
ORM model for the `processing_log` table.

Event log — one row per processing event (not per document).
Provides the full audit trail for debugging extraction failures.
The boolean flags on `documents` show current state only;
this table shows the complete history.
"""

from datetime import datetime

from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class ProcessingLog(Base):
    __tablename__ = "processing_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"))
    stage: Mapped[str | None] = mapped_column(String(50))
    # ^ 'text_extraction', 'chunking', 'embedding', 'feature_extraction'
    status: Mapped[str | None] = mapped_column(String(20))
    # ^ 'started', 'success', 'failed'
    message: Mapped[str | None] = mapped_column(Text)  # error detail if failed
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Relationships
    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document", back_populates="processing_logs"
    )

    def __repr__(self) -> str:
        return (
            f"<ProcessingLog(id={self.id}, document_id={self.document_id}, "
            f"stage={self.stage!r}, status={self.status!r})>"
        )
