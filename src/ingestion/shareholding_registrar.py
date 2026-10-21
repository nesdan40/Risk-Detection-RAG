"""
Shareholding registrar for the document ingestion pipeline.

Responsible for upserting rows into the `shareholding_patterns` table from
the `documents_collected.shareholding_patterns.data` block in metadata.json.

Source data is manually verified (not PDF-derived), so this module reads
structured values directly — no text extraction involved.

Key conventions
---------------
* ``pledge_pct``: source value ``"-"`` or ``0.0`` → stored as ``0.0``
  (genuinely nil pledge).  ``None`` / absent means truly unknown (Req 6.4).
* ``promoter_holding_pct``: ``None`` / absent / ``""`` → stored as ``NULL``
  on insert; on upsert the existing DB value is *preserved* (not overwritten
  with NULL) if the incoming value is absent (Req 6.3, 6.5).
* Composite unique key: ``(company_id, fiscal_year, fiscal_quarter)`` (Req 6.3).
* Malformed quarter keys are skipped with a stderr warning; processing
  continues for remaining valid keys (Req 6.7).

Label-leakage note
------------------
This module reads only shareholding percentages — no distress/group/label
information is ever read or stored here.
"""

from __future__ import annotations

import re
import sys
from datetime import date

from sqlalchemy.orm import Session

from models.shareholding_pattern import ShareholdingPattern

# ---------------------------------------------------------------------------
# Quarter-key pattern
# ---------------------------------------------------------------------------

# Matches e.g. "Q1FY2017", "Q4FY2023"
_QUARTER_KEY_RE = re.compile(r"^Q([1-4])FY(\d{4})$")

# Last day of each fiscal quarter for the Indian fiscal calendar
# (fiscal year starts April 1; Q1=Apr–Jun, Q2=Jul–Sep, Q3=Oct–Dec, Q4=Jan–Mar)
_QUARTER_PERIOD_END: dict[int, tuple[int, int]] = {
    1: (6, 30),   # June 30  (calendar year YYYY-1)
    2: (9, 30),   # Sep 30   (calendar year YYYY-1)
    3: (12, 31),  # Dec 31   (calendar year YYYY-1)
    4: (3, 31),   # Mar 31   (calendar year YYYY  )
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_quarter_key(quarter_key: str) -> tuple[str, str, date] | None:
    """
    Parse a quarter key string into fiscal components.

    Args:
        quarter_key: String of the form ``Q{1-4}FY{YYYY}`` (e.g. ``Q1FY2017``).

    Returns:
        ``(fiscal_year, fiscal_quarter, period_end)`` on success, e.g.
        ``('FY2017', 'Q1', date(2016, 6, 30))``.
        ``None`` if the key does not match the expected pattern (with a
        stderr warning emitted — Req 6.7).

    Indian fiscal calendar:
        Q1FY{YYYY} → period_end = June 30,    year YYYY-1
        Q2FY{YYYY} → period_end = Sep 30,     year YYYY-1
        Q3FY{YYYY} → period_end = Dec 31,     year YYYY-1
        Q4FY{YYYY} → period_end = March 31,   year YYYY
    """
    match = _QUARTER_KEY_RE.match(quarter_key.strip())
    if match is None:
        print(
            f"WARNING: Malformed quarter key {quarter_key!r} — expected pattern "
            "Q{{1-4}}FY{{YYYY}} (e.g. Q1FY2017). Skipping.",
            file=sys.stderr,
        )
        return None

    quarter_num = int(match.group(1))   # 1-4
    fiscal_year_num = int(match.group(2))  # e.g. 2017

    fiscal_year = f"FY{fiscal_year_num}"
    fiscal_quarter = f"Q{quarter_num}"

    month, day = _QUARTER_PERIOD_END[quarter_num]
    # Q1-Q3 fall in the calendar year *before* the fiscal year number;
    # Q4 falls in the same calendar year as the fiscal year number.
    cal_year = fiscal_year_num if quarter_num == 4 else fiscal_year_num - 1
    period_end = date(cal_year, month, day)

    return fiscal_year, fiscal_quarter, period_end


def upsert_shareholding_pattern(
    company_id: int,
    fiscal_year: str,
    fiscal_quarter: str,
    period_end: date,
    promoter_holding_pct: float | None,
    pledge_pct: float | None,
    session: Session,
) -> None:
    """
    Upsert a single shareholding pattern row.

    Composite unique key: ``(company_id, fiscal_year, fiscal_quarter)``.

    Update rules (Req 6.3):
    * If the row already exists:
        - Update ``promoter_holding_pct`` **only** if *promoter_holding_pct*
          is not ``None``; otherwise preserve the existing DB value.
        - Always update ``pledge_pct`` with the incoming value.
    * If the row does not exist:
        - Insert a new row with all provided values.

    Args:
        company_id:           FK to ``companies.id``.
        fiscal_year:          e.g. ``'FY2017'``.
        fiscal_quarter:       e.g. ``'Q1'``.
        period_end:           Last day of the fiscal quarter.
        promoter_holding_pct: Promoter holding % or ``None`` if absent.
        pledge_pct:           Pledge % (already normalised: ``"-"`` → ``0.0``).
        session:              Active SQLAlchemy session.
    """
    existing: ShareholdingPattern | None = (
        session.query(ShareholdingPattern)
        .filter_by(
            company_id=company_id,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )
        .first()
    )

    if existing is None:
        # INSERT path
        row = ShareholdingPattern(
            company_id=company_id,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            period_end=period_end,
            promoter_holding_pct=promoter_holding_pct,
            pledge_pct=pledge_pct,
        )
        session.add(row)
    else:
        # UPDATE path
        # promoter_holding_pct: only overwrite if incoming has a real value
        if promoter_holding_pct is not None:
            existing.promoter_holding_pct = promoter_holding_pct
        # pledge_pct: always update
        existing.pledge_pct = pledge_pct
        # period_end: keep in sync (harmless to update)
        existing.period_end = period_end

    session.flush()


def register_shareholding_patterns(
    company_id: int,
    shareholding_data: dict[str, dict[str, float | None]],
    session: Session,
) -> int:
    """
    Upsert all shareholding pattern rows for a company.

    Iterates over every quarter key in *shareholding_data*, parses it,
    normalises ``pledge_pct``, and calls :func:`upsert_shareholding_pattern`.

    Skips absent/empty *shareholding_data* without raising an error (Req 6.6).
    Skips malformed quarter keys with a stderr warning (Req 6.7).

    ``pledge_pct`` normalisation (Req 6.4):
        * Source value ``"-"`` (string sentinel)  →  stored as ``0.0``
        * Source value ``0.0`` (numeric zero)      →  stored as ``0.0``
        * Any other numeric value                  →  stored as-is
        * ``None`` / absent                        →  stored as ``NULL``

    ``promoter_holding_pct`` normalisation (Req 6.5):
        * ``None``, absent, or empty string in source  →  ``None`` here
          (on insert this becomes NULL; on update the existing DB value
           is preserved by :func:`upsert_shareholding_pattern`)
        * Any numeric value                            →  stored as-is

    Args:
        company_id:        FK to ``companies.id``.
        shareholding_data: Quarter key → ``{promoter_holding_pct, pledge_pct}``.
                           May be an empty dict (no-op).
        session:           Active SQLAlchemy session.

    Returns:
        The number of rows successfully upserted.
    """
    if not shareholding_data:
        return 0

    upserted = 0

    for quarter_key, values in shareholding_data.items():
        parsed = parse_quarter_key(quarter_key)
        if parsed is None:
            # Warning already emitted inside parse_quarter_key; skip.
            continue

        fiscal_year, fiscal_quarter, period_end = parsed

        # ---- promoter_holding_pct normalisation (Req 6.5) ----------------
        raw_promoter = values.get("promoter_holding_pct")
        promoter_holding_pct: float | None
        if raw_promoter is None or raw_promoter == "":
            promoter_holding_pct = None
        else:
            promoter_holding_pct = float(raw_promoter)

        # ---- pledge_pct normalisation (Req 6.4) ---------------------------
        raw_pledge = values.get("pledge_pct")
        pledge_pct: float | None
        if raw_pledge == "-" or raw_pledge == 0.0:
            pledge_pct = 0.0
        elif raw_pledge is None:
            pledge_pct = None
        else:
            pledge_pct = float(raw_pledge)

        upsert_shareholding_pattern(
            company_id=company_id,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            period_end=period_end,
            promoter_holding_pct=promoter_holding_pct,
            pledge_pct=pledge_pct,
            session=session,
        )

        upserted += 1

    return upserted
