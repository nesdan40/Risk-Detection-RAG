"""Initial schema — all six tables for the Risk Detection RAG project.

Tables created:
  - companies
  - documents
  - processing_log
  - shareholding_patterns
  - extracted_features
  - predictions

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── companies ────────────────────────────────────────────────────────────
    op.create_table(
        "companies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("sector", sa.String(100), nullable=True),
        sa.Column("listing_type", sa.String(20), nullable=True),
        sa.Column("collection_window_start", sa.Date(), nullable=True),
        sa.Column("collection_window_end", sa.Date(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker"),
    )

    # ── documents ────────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column("file_name", sa.String(255), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("document_type", sa.String(50), nullable=True),
        sa.Column("document_category", sa.String(100), nullable=True),
        sa.Column("fiscal_year", sa.String(10), nullable=True),
        sa.Column("fiscal_quarter", sa.String(10), nullable=True),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column("source", sa.String(50), nullable=True),
        sa.Column(
            "text_extracted",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "chunked",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "embedded",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── processing_log ───────────────────────────────────────────────────────
    op.create_table(
        "processing_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=True),
        sa.Column("stage", sa.String(50), nullable=True),
        sa.Column("status", sa.String(20), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── shareholding_patterns ─────────────────────────────────────────────────
    op.create_table(
        "shareholding_patterns",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column("fiscal_year", sa.String(10), nullable=True),
        sa.Column("fiscal_quarter", sa.String(10), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column("promoter_holding_pct", sa.Numeric(5, 2), nullable=True),
        sa.Column("pledge_pct", sa.Numeric(5, 2), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── extracted_features ────────────────────────────────────────────────────
    op.create_table(
        "extracted_features",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column("document_id", sa.Integer(), nullable=True),
        sa.Column("fiscal_year", sa.String(10), nullable=True),
        sa.Column("fiscal_quarter", sa.String(10), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column("feature_type", sa.String(50), nullable=True),
        # Financial signals
        sa.Column("revenue", sa.Numeric(20, 2), nullable=True),
        sa.Column("revenue_growth_pct", sa.Numeric(8, 2), nullable=True),
        sa.Column("pat", sa.Numeric(20, 2), nullable=True),
        sa.Column("ebitda_margin_pct", sa.Numeric(8, 2), nullable=True),
        sa.Column("debt_to_equity", sa.Numeric(8, 4), nullable=True),
        sa.Column("interest_coverage", sa.Numeric(8, 4), nullable=True),
        sa.Column("current_ratio", sa.Numeric(8, 4), nullable=True),
        sa.Column("cfo", sa.Numeric(20, 2), nullable=True),
        # Governance signals
        sa.Column("board_meeting_count", sa.Integer(), nullable=True),
        sa.Column("director_change_count", sa.Integer(), nullable=True),
        sa.Column("appointment_count", sa.Integer(), nullable=True),
        sa.Column("resignation_count", sa.Integer(), nullable=True),
        sa.Column("agm_delay_flag", sa.Boolean(), nullable=True),
        sa.Column("clarification_count", sa.Integer(), nullable=True),
        sa.Column("press_release_count", sa.Integer(), nullable=True),
        # Credit signals
        sa.Column("credit_rating", sa.String(20), nullable=True),
        sa.Column("rating_outlook", sa.String(20), nullable=True),
        sa.Column("rating_change_flag", sa.Boolean(), nullable=True),
        # Overflow
        sa.Column("additional_signals", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"]),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── predictions ───────────────────────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column("fiscal_year", sa.String(10), nullable=True),
        sa.Column("fiscal_quarter", sa.String(10), nullable=True),
        sa.Column("risk_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("risk_level", sa.String(20), nullable=True),
        sa.Column("risk_flags", JSONB(), nullable=True),
        sa.Column("evidence_chunks", JSONB(), nullable=True),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    # Drop in reverse FK-dependency order
    op.drop_table("predictions")
    op.drop_table("extracted_features")
    op.drop_table("shareholding_patterns")
    op.drop_table("processing_log")
    op.drop_table("documents")
    op.drop_table("companies")
