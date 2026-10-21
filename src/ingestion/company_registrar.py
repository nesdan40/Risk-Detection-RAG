"""
Company registrar for the document ingestion pipeline.

Responsible for upserting rows into the `companies` table.  Uses ticker as
the unique key: if a row already exists it is updated; otherwise a new row is
inserted.  All writable fields are always overwritten from metadata (absent
fields become NULL), which is required for idempotency (Req 3.1).

Label-leakage note
------------------
`listing_type` is derived exclusively from the `exchange` field in
metadata.json.  No distress/group/label information is ever read or stored
here (Req 2.4).
"""

from __future__ import annotations

import sys

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ingestion.metadata_reader import CompanyMetadata
from models.company import Company


# ---------------------------------------------------------------------------
# Listing type mapping
# ---------------------------------------------------------------------------

_EQUITY_EXCHANGES: frozenset[str] = frozenset({"BSE", "NSE", "BSE+NSE"})
_DEBT_ONLY_EXCHANGES: frozenset[str] = frozenset({"MCA"})


def map_listing_type(exchange: str) -> str | None:
    """
    Map the ``exchange`` field from metadata.json to the ``listing_type``
    column value.

    Mapping:
        'BSE', 'NSE', 'BSE+NSE'  →  'equity'
        'MCA'                     →  'debt_only'
        anything else             →  None  (+ stderr warning)

    Args:
        exchange: Raw exchange string from metadata.json (may be empty).

    Returns:
        'equity', 'debt_only', or None.
    """
    normalized = (exchange or "").strip()

    if normalized in _EQUITY_EXCHANGES:
        return "equity"
    if normalized in _DEBT_ONLY_EXCHANGES:
        return "debt_only"

    # Unknown / empty — warn and return None (Req 2.6)
    print(
        f"WARNING: Unrecognised exchange value {exchange!r}. "
        "listing_type will be set to NULL.",
        file=sys.stderr,
    )
    return None


# ---------------------------------------------------------------------------
# Company registration
# ---------------------------------------------------------------------------


def register_company(metadata: CompanyMetadata, session: Session) -> int | None:
    """
    Upsert a company row based on ``ticker`` (unique key).

    If a row with the given ticker already exists, ALL writable fields are
    overwritten with the values from *metadata* (absent fields become NULL,
    not left unchanged).  This ensures idempotency across repeated runs
    (Req 3.1).

    If no matching row exists, a new row is inserted (Req 3.2).

    The caller is responsible for committing the session.  This function only
    adds/modifies the ORM object and flushes so that the PK is available for
    the return value.

    Args:
        metadata: Parsed company metadata (never contains label fields).
        session:  Active SQLAlchemy session.

    Returns:
        The ``company.id`` (int) on success, or None on error.
    """
    listing_type = map_listing_type(metadata.exchange)

    try:
        company = (
            session.query(Company).filter_by(ticker=metadata.ticker).first()
        )

        if company is None:
            # INSERT path (Req 3.2)
            company = Company(
                ticker=metadata.ticker,
                name=metadata.company_name,
                sector=metadata.sector,
                listing_type=listing_type,
                collection_window_start=metadata.collection_window_start,
                collection_window_end=metadata.collection_window_end,
            )
            session.add(company)
        else:
            # UPDATE path — overwrite ALL writable fields (Req 3.1)
            # Absent fields in metadata arrive as None and become NULL,
            # which is the required behaviour for idempotency.
            company.name = metadata.company_name
            company.sector = metadata.sector
            company.listing_type = listing_type
            company.collection_window_start = metadata.collection_window_start
            company.collection_window_end = metadata.collection_window_end

        # Flush so the PK is populated without committing the transaction.
        session.flush()
        return company.id

    except SQLAlchemyError as exc:
        # Rollback the session to a clean state, log the error, and return
        # None so the caller can continue with other companies (Req 3.6).
        session.rollback()
        print(
            f"ERROR: Failed to register company {metadata.ticker!r}: {exc}",
            file=sys.stderr,
        )
        return None
