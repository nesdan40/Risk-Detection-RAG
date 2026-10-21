"""
ORM model for the `predictions` table.

Stores output of both the rule-based engine (Semester 7) and ML models
(Semester 8) in the same table, distinguished by model_version.

model_version examples:
  - 'rule_based_v1'  — rule-based risk engine output
  - 'xgboost_v1'     — XGBoost classifier output (Sem 8)

risk_score is in [0.0, 1.0].
risk_flags and evidence_chunks are JSONB to allow flexible structured output
without schema changes as the rule engine evolves.
"""

from datetime import datetime

from sqlalchemy import ForeignKey, Index, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Prediction(Base):
    __tablename__ = "predictions"
    # Mirrors migration 0003; conflict target for the prediction writer's upsert.
    __table_args__ = (
        Index(
            "ix_predictions_upsert",
            "company_id", "fiscal_year", "fiscal_quarter", "model_version",
            unique=True,
            postgresql_nulls_not_distinct=True,  # migration 0004
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"))
    fiscal_year: Mapped[str | None] = mapped_column(String(10))
    fiscal_quarter: Mapped[str | None] = mapped_column(String(10))
    risk_score: Mapped[float | None] = mapped_column(Numeric(5, 4))
    # ^ 0.0 to 1.0
    risk_level: Mapped[str | None] = mapped_column(String(20))
    # ^ 'Low', 'Medium', 'High', 'Critical'
    risk_flags: Mapped[dict | None] = mapped_column(JSONB)
    # ^ Which rules fired and why
    evidence_chunks: Mapped[dict | None] = mapped_column(JSONB)
    # ^ Qdrant chunk references supporting each flag
    model_version: Mapped[str | None] = mapped_column(String(50))
    # ^ 'rule_based_v1', 'xgboost_v1', etc.
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Relationships
    company: Mapped["Company"] = relationship(  # noqa: F821
        "Company", back_populates="predictions"
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, company_id={self.company_id}, "
            f"risk_level={self.risk_level!r}, model_version={self.model_version!r})>"
        )
