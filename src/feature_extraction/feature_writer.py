"""
Feature writer — upserts rows into ``extracted_features``.

All three feature types (financial, governance, credit) are written here using
PostgreSQL's ``INSERT ... ON CONFLICT DO UPDATE`` (via
``sqlalchemy.dialects.postgresql.insert``) so each write is atomic — no
separate SELECT + INSERT/UPDATE pattern that could race.

Conflict targets
----------------
- **financial** and **credit** rows use the partial unique index
  ``ix_ef_financial_upsert`` / ``ix_ef_credit_upsert``:
  ``(company_id, document_id, feature_type)``
  ``WHERE document_id IS NOT NULL``

- **governance** rows use the partial unique index ``ix_ef_governance_upsert``:
  ``(company_id, fiscal_year, fiscal_quarter, feature_type)``
  ``WHERE document_id IS NULL``

Savepoint pattern (Req 6.5)
---------------------------
Each upsert is executed inside ``session.begin_nested()`` so the
``extracted_features`` write and the surrounding ``processing_log`` success
row (written by the caller) commit together or neither does.

Label guard (Req 6.4)
---------------------
None of the upsert functions ever reference columns named ``group``,
``distressed``, ``healthy``, ``label``, or ``ground_truth``.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 8.2
"""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from models.feature import ExtractedFeature

if TYPE_CHECKING:
    from feature_extraction.financial_extractor import FinancialResult
    from feature_extraction.governance_extractor import GovernanceResult
    from feature_extraction.credit_extractor import CreditResult

# ---------------------------------------------------------------------------
# Label guard — these column names must NEVER appear in upsert payloads
# ---------------------------------------------------------------------------

_FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {"group", "distressed", "healthy", "label", "ground_truth"}
)


def _assert_no_forbidden_columns(payload: dict) -> None:
    """Raise AssertionError if any forbidden column name is present in payload."""
    violations = _FORBIDDEN_COLUMNS & payload.keys()
    if violations:
        raise AssertionError(
            f"Label guard violation: forbidden columns in upsert payload: "
            f"{sorted(violations)}"
        )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def upsert_financial(
    session: Session,
    company_id: int,
    document_id: int,
    fiscal_year: str,
    fiscal_quarter: str | None,
    period_end: date,
    result: "FinancialResult",
) -> ExtractedFeature:
    """
    Upsert a financial ``extracted_features`` row.

    Conflict key: ``(company_id, document_id, feature_type='financial')``
    via partial unique index ``ix_ef_financial_upsert``.

    On conflict all signal columns are overwritten with the freshly extracted
    values — including columns that are ``None`` — so a re-run after a fix
    always reflects the latest extraction.

    ``additional_signals`` receives any NBFC-specific overflow ratios from
    ``result.additional_signals`` (may be empty dict).

    The upsert runs inside ``session.begin_nested()`` (savepoint) so a log
    failure on the caller side never rolls back this write (Req 6.5).

    Args:
        session:        Active SQLAlchemy session (must be within an open
                        transaction).
        company_id:     FK to ``companies.id``.
        document_id:    FK to ``documents.id`` — must not be None for financial
                        rows.
        fiscal_year:    e.g. ``'FY2019'``.
        fiscal_quarter: e.g. ``'Q2'``, or ``None`` for annual reports.
        period_end:     Last calendar date of the document's fiscal period.
        result:         :class:`FinancialResult` from the financial extractor.

    Returns:
        The refreshed :class:`ExtractedFeature` ORM object for the row (loaded
        via the row's primary key after upsert).
    """
    additional: dict | None = result.additional_signals if result.additional_signals else None

    payload: dict = {
        "company_id": company_id,
        "document_id": document_id,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "period_end": period_end,
        "feature_type": "financial",
        # Financial signals — all nullable; Decimal values stored as-is
        "revenue": result.revenue,
        "revenue_growth_pct": result.revenue_growth_pct,
        "pat": result.pat,
        "ebitda_margin_pct": result.ebitda_margin_pct,
        "debt_to_equity": result.debt_to_equity,
        "interest_coverage": result.interest_coverage,
        "current_ratio": result.current_ratio,
        "cfo": result.cfo,
        # Governance / credit fields — always NULL for financial rows
        "board_meeting_count": None,
        "director_change_count": None,
        "appointment_count": None,
        "resignation_count": None,
        "agm_delay_flag": None,
        "clarification_count": None,
        "press_release_count": None,
        "credit_rating": None,
        "rating_outlook": None,
        "rating_change_flag": None,
        # Overflow
        "additional_signals": additional,
    }

    _assert_no_forbidden_columns(payload)

    # Columns that form the conflict key — excluded from the SET clause
    _conflict_columns = {"company_id", "document_id", "feature_type", "created_at"}

    set_clause = {k: v for k, v in payload.items() if k not in _conflict_columns}

    stmt = (
        pg_insert(ExtractedFeature)
        .values(**payload)
        .on_conflict_do_update(
            index_elements=["company_id", "document_id", "feature_type"],
            index_where=text(
                "document_id IS NOT NULL AND feature_type = 'financial'"
            ),
            set_=set_clause,
        )
        .returning(ExtractedFeature.id)
    )

    with session.begin_nested():
        result_row = session.execute(stmt)
        row_id = result_row.scalar_one()

    return session.get(ExtractedFeature, row_id)


def upsert_governance(
    session: Session,
    company_id: int,
    fiscal_year: str,
    fiscal_quarter: str,
    period_end: date,
    result: "GovernanceResult",
) -> ExtractedFeature:
    """
    Upsert a governance ``extracted_features`` row.

    Conflict key: ``(company_id, fiscal_year, fiscal_quarter, feature_type='governance')``
    via partial unique index ``ix_ef_governance_upsert``.

    ``document_id`` is always ``NULL`` for governance rows — governance features
    are period-level aggregates, not tied to any single document.

    AGM delay flag is stored on the quarter in which the AGM took place
    (NULL when no AGM was held in that quarter).

    ``corporate_actions`` from ``result.corporate_actions`` is stored in
    ``additional_signals['corporate_actions']`` as a list of dicts.

    ``clarification_count`` intentionally includes all ``press_release_count``
    documents as a superset (per schema design note).

    Args:
        session:        Active SQLAlchemy session.
        company_id:     FK to ``companies.id``.
        fiscal_year:    e.g. ``'FY2019'``.
        fiscal_quarter: One of ``'Q1'``, ``'Q2'``, ``'Q3'``, ``'Q4'``.
        period_end:     Last calendar date of the fiscal quarter.
        result:         :class:`GovernanceResult` from the governance extractor.

    Returns:
        The refreshed :class:`ExtractedFeature` ORM object for the row.
    """
    # AGM delay flag: set on the quarter the AGM occurred in
    agm_flag = result.agm_delay_flag

    # Build additional_signals with corporate_actions (if any)
    additional: dict | None = None
    if result.corporate_actions:
        additional = {"corporate_actions": result.corporate_actions}

    payload: dict = {
        "company_id": company_id,
        "document_id": None,  # governance rows are never doc-level
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "period_end": period_end,
        "feature_type": "governance",
        # Governance signals
        "board_meeting_count": result.board_meeting_count,
        "director_change_count": result.director_change_count,
        "appointment_count": result.appointment_count,
        "resignation_count": result.resignation_count,
        "press_release_count": result.press_release_count,
        "clarification_count": result.clarification_count,
        # clarification_count is intentionally a superset of press_release_count
        "agm_delay_flag": agm_flag,
        # Financial / credit fields — always NULL for governance rows
        "revenue": None,
        "revenue_growth_pct": None,
        "pat": None,
        "ebitda_margin_pct": None,
        "debt_to_equity": None,
        "interest_coverage": None,
        "current_ratio": None,
        "cfo": None,
        "credit_rating": None,
        "rating_outlook": None,
        "rating_change_flag": None,
        # Overflow
        "additional_signals": additional,
    }

    _assert_no_forbidden_columns(payload)

    _conflict_columns = {
        "company_id",
        "fiscal_year",
        "fiscal_quarter",
        "feature_type",
        "document_id",
        "created_at",
    }

    set_clause = {k: v for k, v in payload.items() if k not in _conflict_columns}

    stmt = (
        pg_insert(ExtractedFeature)
        .values(**payload)
        .on_conflict_do_update(
            index_elements=[
                "company_id",
                "fiscal_year",
                "fiscal_quarter",
                "feature_type",
            ],
            index_where=text(
                "document_id IS NULL AND feature_type = 'governance'"
            ),
            set_=set_clause,
        )
        .returning(ExtractedFeature.id)
    )

    with session.begin_nested():
        result_row = session.execute(stmt)
        row_id = result_row.scalar_one()

    return session.get(ExtractedFeature, row_id)


def upsert_credit(
    session: Session,
    company_id: int,
    document_id: int,
    fiscal_year: str | None,
    fiscal_quarter: str | None,
    period_end: date | None,
    result: "CreditResult",
) -> ExtractedFeature:
    """
    Upsert a credit ``extracted_features`` row.

    Conflict key: ``(company_id, document_id, feature_type='credit')``
    via partial unique index ``ix_ef_credit_upsert``.

    ``instrument_ratings`` from ``result.instrument_ratings`` is stored in
    ``additional_signals['instrument_ratings']`` as a list of dicts.

    Args:
        session:        Active SQLAlchemy session.
        company_id:     FK to ``companies.id``.
        document_id:    FK to ``documents.id``.
        fiscal_year:    e.g. ``'FY2019'``, or ``None`` if the credit rating
                        document has no fiscal period association.
        fiscal_quarter: e.g. ``'Q2'``, or ``None``.
        period_end:     Document's ``period_end`` from the ``documents`` table,
                        or ``None``.
        result:         :class:`CreditResult` from the credit extractor.

    Returns:
        The refreshed :class:`ExtractedFeature` ORM object for the row.
    """
    # Build additional_signals: agency/extras plus every rating mention (if any)
    additional_map: dict = dict(getattr(result, "extras", None) or {})
    if result.instrument_ratings:
        additional_map["instrument_ratings"] = result.instrument_ratings
    additional: dict | None = additional_map or None

    payload: dict = {
        "company_id": company_id,
        "document_id": document_id,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "period_end": period_end,
        "feature_type": "credit",
        # Credit signals
        "credit_rating": result.credit_rating,
        "rating_outlook": result.rating_outlook,
        "rating_change_flag": result.rating_change_flag,
        # Financial / governance fields — always NULL for credit rows
        "revenue": None,
        "revenue_growth_pct": None,
        "pat": None,
        "ebitda_margin_pct": None,
        "debt_to_equity": None,
        "interest_coverage": None,
        "current_ratio": None,
        "cfo": None,
        "board_meeting_count": None,
        "director_change_count": None,
        "appointment_count": None,
        "resignation_count": None,
        "agm_delay_flag": None,
        "clarification_count": None,
        "press_release_count": None,
        # Overflow
        "additional_signals": additional,
    }

    _assert_no_forbidden_columns(payload)

    _conflict_columns = {"company_id", "document_id", "feature_type", "created_at"}

    set_clause = {k: v for k, v in payload.items() if k not in _conflict_columns}

    stmt = (
        pg_insert(ExtractedFeature)
        .values(**payload)
        .on_conflict_do_update(
            index_elements=["company_id", "document_id", "feature_type"],
            index_where=text(
                "document_id IS NOT NULL AND feature_type = 'credit'"
            ),
            set_=set_clause,
        )
        .returning(ExtractedFeature.id)
    )

    with session.begin_nested():
        result_row = session.execute(stmt)
        row_id = result_row.scalar_one()

    return session.get(ExtractedFeature, row_id)


def delete_company_features(
    session: Session,
    company_id: int,
) -> int:
    """
    Delete all ``extracted_features`` rows for a company.

    Used by the ``--force`` flag before re-extraction to allow a full refresh.
    Deletion is scoped strictly to ``extracted_features`` — ``processing_log``
    rows are never touched (Req 8.2 hard constraint).

    The delete is performed directly on the table (not ORM-level bulk delete)
    to avoid loading all rows into memory.

    Args:
        session:    Active SQLAlchemy session.
        company_id: PK of the company whose features should be deleted.

    Returns:
        Count of deleted rows.
    """
    stmt = (
        delete(ExtractedFeature)
        .where(ExtractedFeature.company_id == company_id)
    )
    result = session.execute(stmt)
    return result.rowcount
