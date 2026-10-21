"""Add unique indexes on extracted_features for upsert conflict targets.

Three partial unique indexes are needed to support ON CONFLICT DO UPDATE:

  1. Document-level financial features:
     UNIQUE (company_id, document_id, feature_type)
     WHERE document_id IS NOT NULL AND feature_type = 'financial'

  2. Document-level credit features:
     UNIQUE (company_id, document_id, feature_type)
     WHERE document_id IS NOT NULL AND feature_type = 'credit'

  3. Governance period aggregates (document_id is always NULL):
     UNIQUE (company_id, fiscal_year, fiscal_quarter, feature_type)
     WHERE document_id IS NULL AND feature_type = 'governance'

Using partial indexes keeps the index size small and makes the conflict
semantics explicit — no accidental cross-type collisions.

Revision ID: 0002
Revises: 0001
Create Date: 2025-01-01 00:00:00.000000
"""

from alembic import op

# revision identifiers
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Financial: unique on (company_id, document_id, feature_type) ─────────
    op.create_index(
        "ix_ef_financial_upsert",
        "extracted_features",
        ["company_id", "document_id", "feature_type"],
        unique=True,
        postgresql_where="document_id IS NOT NULL AND feature_type = 'financial'",
    )

    # ── Credit: unique on (company_id, document_id, feature_type) ────────────
    op.create_index(
        "ix_ef_credit_upsert",
        "extracted_features",
        ["company_id", "document_id", "feature_type"],
        unique=True,
        postgresql_where="document_id IS NOT NULL AND feature_type = 'credit'",
    )

    # ── Governance: unique on (company_id, fiscal_year, fiscal_quarter, feature_type) ──
    op.create_index(
        "ix_ef_governance_upsert",
        "extracted_features",
        ["company_id", "fiscal_year", "fiscal_quarter", "feature_type"],
        unique=True,
        postgresql_where="document_id IS NULL AND feature_type = 'governance'",
    )


def downgrade() -> None:
    op.drop_index("ix_ef_governance_upsert", table_name="extracted_features")
    op.drop_index("ix_ef_credit_upsert", table_name="extracted_features")
    op.drop_index("ix_ef_financial_upsert", table_name="extracted_features")
