"""
Filename parser for the document ingestion pipeline.

Derives document metadata (type, category, fiscal periods, source) from PDF
filenames and their containing subdirectory. All parsing is done purely from
the filename — no file I/O is performed.

Supported filename patterns:
  Annual Report:          {TICKER}_AnnualReport_FY{YYYY}.pdf
  Quarterly Result:       {TICKER}_Q{n}FY{YYYY}[_SCANNED].pdf
  Corporate Announcement: {TICKER}_Ann_{YYYYMMDD}_{Category}[_(N)][_SCANNED].pdf
  Credit Rating:          {TICKER}_Ann_{YYYYMMDD}_Credit Rating[_(N)][_SCANNED].pdf
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ParsedFilename:
    """Metadata extracted from a PDF filename."""

    document_type: str
    document_category: str | None
    fiscal_year: str | None
    fiscal_quarter: str | None
    period_start: date | None
    period_end: date | None
    source: str


# ---------------------------------------------------------------------------
# Source derivation
# ---------------------------------------------------------------------------

def derive_source(exchange: str) -> str:
    """
    Derive document source from the exchange field in metadata.json.

    When the exchange value contains a '+' separator (e.g. 'BSE+NSE') the
    first token is returned.  Otherwise the value is returned as-is.

    Examples
    --------
    >>> derive_source('BSE+NSE')
    'BSE'
    >>> derive_source('NSE')
    'NSE'
    >>> derive_source('MCA')
    'MCA'
    """
    return exchange.split("+")[0]


# ---------------------------------------------------------------------------
# Fiscal period calculation
# ---------------------------------------------------------------------------

# Quarter → (start month, start day, end month, end day, year offset for end)
# All quarters are relative to the FY year:
#   Q1: Apr–Jun of (FY_YEAR - 1)
#   Q2: Jul–Sep of (FY_YEAR - 1)
#   Q3: Oct–Dec of (FY_YEAR - 1)
#   Q4: Jan–Mar of FY_YEAR
_QUARTER_PERIODS: dict[str, tuple[int, int, int, int, int, int]] = {
    #       start_mon  start_day  end_mon  end_day  start_yr_offset  end_yr_offset
    "Q1": (4,          1,         6,       30,      -1,              -1),
    "Q2": (7,          1,         9,       30,      -1,              -1),
    "Q3": (10,         1,         12,      31,      -1,              -1),
    "Q4": (1,          1,         3,       31,       0,               0),
}


def calculate_fiscal_periods(
    fiscal_year: str, fiscal_quarter: str | None
) -> tuple[date, date]:
    """
    Calculate period_start and period_end for the Indian fiscal calendar.

    The fiscal year string is expected in the form 'FY{YYYY}' (e.g. 'FY2017').
    The Indian fiscal year runs April 1 – March 31, so FY2017 covers
    2016-04-01 to 2017-03-31.

    Parameters
    ----------
    fiscal_year:
        String like 'FY2017'.
    fiscal_quarter:
        One of 'Q1', 'Q2', 'Q3', 'Q4', or None for the full annual period.

    Returns
    -------
    (period_start, period_end)

    Examples
    --------
    >>> calculate_fiscal_periods('FY2017', None)
    (date(2016, 4, 1), date(2017, 3, 31))
    >>> calculate_fiscal_periods('FY2017', 'Q1')
    (date(2016, 4, 1), date(2016, 6, 30))
    >>> calculate_fiscal_periods('FY2017', 'Q4')
    (date(2017, 1, 1), date(2017, 3, 31))
    """
    fy_year = int(fiscal_year[2:])  # 'FY2017' -> 2017

    if fiscal_quarter is None:
        # Annual: April 1 of preceding calendar year to March 31 of FY year
        return date(fy_year - 1, 4, 1), date(fy_year, 3, 31)

    params = _QUARTER_PERIODS[fiscal_quarter]
    start_mon, start_day, end_mon, end_day, start_yr_off, end_yr_off = params
    period_start = date(fy_year + start_yr_off, start_mon, start_day)
    period_end = date(fy_year + end_yr_off, end_mon, end_day)
    return period_start, period_end


def derive_fiscal_period(d: date) -> tuple[str, str]:
    """Indian fiscal year/quarter of a calendar date, e.g. 2016-05-04 -> ('FY2017', 'Q1')."""
    fy = d.year + 1 if d.month >= 4 else d.year
    quarter = {4: "Q1", 5: "Q1", 6: "Q1", 7: "Q2", 8: "Q2", 9: "Q2",
               10: "Q3", 11: "Q3", 12: "Q3", 1: "Q4", 2: "Q4", 3: "Q4"}[d.month]
    return f"FY{fy}", quarter


# ---------------------------------------------------------------------------
# Category mapping and normalization
# ---------------------------------------------------------------------------

# Maps lowercase raw category tokens (from filenames) to canonical DB values.
# Requirements 5.7 and 5.11 define the full canonical set.
CATEGORY_MAPPING: dict[str, str] = {
    "outcome of board meeting": "Board Meeting",
    "financial results updates": "Financial Results",
    "financial result updates": "Financial Results",
    "annual general meeting updates": "AGM / Annual General Meeting Updates",
    "agm / annual general meeting updates": "AGM / Annual General Meeting Updates",
    "press release": "Press Release",
    "news clarification": "News Clarification",
    "news verification": "News Verification",
    "appointment": "Appointment",
    "resignation": "Resignation",
    "director change": "Director Change",
}

# Regex to strip trailing duplicate-copy suffix.
# Handles both forms that appear in real filenames:
#   underscore-prefixed:  _(2), _(3)
#   space-prefixed:       (2), (3)
# e.g. "Financial Results Updates (2)" -> "Financial Results Updates"
_DUPLICATE_SUFFIX_RE = re.compile(r"[\s_]*\(\d+\)\s*$")


def normalize_category(raw_category: str) -> str | None:
    """
    Normalize a raw category token from a filename to a canonical DB value.

    Steps
    -----
    1. Strip any trailing ``_(N)`` duplicate-copy suffix (e.g. ``_(2)``).
    2. Strip surrounding whitespace.
    3. Lowercase.
    4. Look up in :data:`CATEGORY_MAPPING`.
    5. If not found, log a warning to stderr and return ``None``.

    Parameters
    ----------
    raw_category:
        The raw category segment extracted from the filename, e.g.
        ``'Financial Result Updates'`` or ``'Outcome of Board Meeting (2)'``.

    Returns
    -------
    Canonical category string, or ``None`` if unrecognised.
    """
    # Strip trailing duplicate suffix then whitespace
    cleaned = _DUPLICATE_SUFFIX_RE.sub("", raw_category).strip()
    key = cleaned.lower()
    canonical = CATEGORY_MAPPING.get(key)
    if canonical is None:
        # Single source of truth: the feature-extraction normalizer (aliases + fuzzy).
        # Imported lazily because feature_extraction.fiscal_utils imports this module.
        from feature_extraction.category_normalizer import normalize

        label, layer = normalize(cleaned, "", None)
        if label != "unclassified" and layer in ("exact", "fuzzy"):
            return label
    if canonical is None:
        print(
            f"WARNING: Unrecognised document category '{raw_category}' "
            "(no canonical mapping found).",
            file=sys.stderr,
        )
    return canonical


# ---------------------------------------------------------------------------
# Filename parsing patterns
# ---------------------------------------------------------------------------

# Annual report: TICKER_AnnualReport_FY{YYYY}.pdf
_ANNUAL_REPORT_RE = re.compile(
    r"^.+_AnnualReport_FY(?P<year>\d{4})\.pdf$",
    re.IGNORECASE,
)

# Quarterly result: TICKER_Q{n}FY{YYYY}[_SCANNED].pdf
_QUARTERLY_RE = re.compile(
    r"^.+_Q(?P<quarter>[1-4])FY(?P<year>\d{4})(?:_SCANNED)?\.pdf$",
    re.IGNORECASE,
)

# Corporate announcement / credit rating:
# TICKER_Ann_{YYYYMMDD}_{Category}[_(N)][_SCANNED].pdf
# The category segment is everything after the date field, with optional
# trailing _(N) and _SCANNED stripped at parse time.
_ANNOUNCEMENT_RE = re.compile(
    r"^.+_Ann_(?P<date>\d{8})_(?P<category>.+?)(?:_SCANNED)?\.pdf$",
    re.IGNORECASE,
)

# Matches the trailing duplicate-copy suffix (both _(N) and space-(N) forms)
# plus optional _SCANNED from the category capture group.
# e.g. "Financial Results Updates (2)_SCANNED" -> "Financial Results Updates"
# e.g. "Outcome of Board Meeting_(2)_SCANNED"  -> "Outcome of Board Meeting"
_TRAILING_DUPE_AND_SCANNED_RE = re.compile(
    r"(?:[\s_]*\(\d+\))?(?:_SCANNED)?$", re.IGNORECASE
)


def _parse_annual_report(
    stem: str, exchange: str, file_path: Path
) -> ParsedFilename | None:
    """Return ParsedFilename for an annual-report filename, or None on no match."""
    m = _ANNUAL_REPORT_RE.match(file_path.name)
    if not m:
        return None
    fy = f"FY{m.group('year')}"
    period_start, period_end = calculate_fiscal_periods(fy, None)
    return ParsedFilename(
        document_type="annual_report",
        document_category=None,
        fiscal_year=fy,
        fiscal_quarter=None,
        period_start=period_start,
        period_end=period_end,
        source=derive_source(exchange),
    )


def _parse_quarterly_result(
    stem: str, exchange: str, file_path: Path
) -> ParsedFilename | None:
    """Return ParsedFilename for a quarterly-result filename, or None on no match."""
    m = _QUARTERLY_RE.match(file_path.name)
    if not m:
        return None
    fq = f"Q{m.group('quarter')}"
    fy = f"FY{m.group('year')}"
    period_start, period_end = calculate_fiscal_periods(fy, fq)
    return ParsedFilename(
        document_type="quarterly_result",
        document_category=None,
        fiscal_year=fy,
        fiscal_quarter=fq,
        period_start=period_start,
        period_end=period_end,
        source=derive_source(exchange),
    )


def _parse_announcement_or_rating(
    stem: str, exchange: str, file_path: Path, document_type: str
) -> ParsedFilename | None:
    """
    Return ParsedFilename for a corporate-announcement or credit-rating filename.

    Credit-rating files follow the same ``_Ann_`` naming convention as
    announcements, but document_type is already set to ``'credit_rating'`` by
    the caller (determined from the containing subdirectory).
    """
    m = _ANNOUNCEMENT_RE.match(file_path.name)
    if not m:
        return None

    # Parse date field
    raw_date = m.group("date")  # YYYYMMDD
    try:
        ann_date = date(int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:]))
    except ValueError:
        return None

    # Strip trailing _(N) and _SCANNED from the captured category segment
    raw_cat = _TRAILING_DUPE_AND_SCANNED_RE.sub("", m.group("category")).strip()

    # Indian fiscal period of the announcement date (needed for period-scoped
    # retrieval and for ordering rating changes in time).
    fy, fq = derive_fiscal_period(ann_date)

    if document_type == "credit_rating":
        # Credit rating: no category; fiscal period comes from the rating date
        return ParsedFilename(
            document_type="credit_rating",
            document_category=None,
            fiscal_year=fy,
            fiscal_quarter=fq,
            period_start=ann_date,
            period_end=ann_date,
            source=derive_source(exchange),
        )

    # Corporate announcement: normalise category
    category = normalize_category(raw_cat)
    return ParsedFilename(
        document_type="corporate_announcement",
        document_category=category,
        fiscal_year=fy,
        fiscal_quarter=fq,
        period_start=ann_date,
        period_end=ann_date,
        source=derive_source(exchange),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_filename(
    file_path: Path,
    document_type: str,
    exchange: str,
) -> ParsedFilename:
    """
    Parse a PDF filename and return structured document metadata.

    The ``document_type`` is already known from the containing subdirectory
    (determined by the Document Registrar) and is passed in rather than
    re-derived here.  The parser uses it to choose the right sub-parser.

    For unrecognised filename patterns all unresolvable fields are set to
    ``None`` and a warning is printed to ``stderr``.  The ``document_type``
    and ``source`` fields are always populated (from the caller-supplied
    arguments).

    Parameters
    ----------
    file_path:
        Absolute or relative path to the PDF file.  Only the filename
        component is used for parsing.
    document_type:
        One of ``'annual_report'``, ``'quarterly_result'``,
        ``'corporate_announcement'``, ``'credit_rating'``.
    exchange:
        Exchange value from ``metadata.json`` (e.g. ``'BSE'``, ``'BSE+NSE'``).

    Returns
    -------
    :class:`ParsedFilename` with all extractable fields populated.
    """
    stem = file_path.stem  # filename without extension, kept for sub-parsers

    result: ParsedFilename | None = None

    if document_type == "annual_report":
        result = _parse_annual_report(stem, exchange, file_path)

    elif document_type == "quarterly_result":
        result = _parse_quarterly_result(stem, exchange, file_path)

    elif document_type in ("corporate_announcement", "credit_rating"):
        result = _parse_announcement_or_rating(stem, exchange, file_path, document_type)

    if result is not None:
        return result

    # Unrecognised pattern — return minimal ParsedFilename with NULLs and warn
    print(
        f"WARNING: Unrecognised filename pattern: '{file_path.name}'. "
        "Unresolvable fields set to None.",
        file=sys.stderr,
    )
    return ParsedFilename(
        document_type=document_type,
        document_category=None,
        fiscal_year=None,
        fiscal_quarter=None,
        period_start=None,
        period_end=None,
        source=derive_source(exchange),
    )
