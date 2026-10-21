"""Add unique index on predictions for upsert conflict target.

A unique index on (company_id, fiscal_year, fiscal_quarter, model_version)
is required to support ON CONFLICT DO UPDATE in the prediction writer so
that re-running the risk engine for the same period overwrites the existing
row rather than creating a duplicate.

Revision ID: 0003
Revises: 0002
Create Date: 2025-01-01 00:00:00.000000
"""

from alembic import op

# revision identifiers
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_predictions_upsert",
        "predictions",
        ["company_id", "fiscal_year", "fiscal_quarter", "model_version"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_upsert", table_name="predictions")
