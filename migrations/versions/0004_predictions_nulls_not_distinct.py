"""Make the predictions upsert index treat NULL fiscal_quarter as a value.

With the default NULLS DISTINCT semantics, annual-level predictions
(fiscal_quarter IS NULL) never conflict, so every re-run of the risk engine
inserted duplicate rows. PostgreSQL 15+ supports NULLS NOT DISTINCT.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_predictions_upsert", table_name="predictions")
    op.create_index(
        "ix_predictions_upsert",
        "predictions",
        ["company_id", "fiscal_year", "fiscal_quarter", "model_version"],
        unique=True,
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_upsert", table_name="predictions")
    op.create_index(
        "ix_predictions_upsert",
        "predictions",
        ["company_id", "fiscal_year", "fiscal_quarter", "model_version"],
        unique=True,
    )
