"""
Fiscal calendar utilities for the feature extraction pipeline.

Provides three pure helpers — no I/O, no DB access — that map between Python
dates and Indian fiscal year / quarter labels, and between those labels and
concrete calendar period boundaries.

The period-boundary logic delegates to
``src/ingestion/filename_parser.calculate_fiscal_periods`` so that period
boundaries stay consistent across the ingestion and feature-extraction stages.
"""

from __future__ import annotations

from datetime import date

from ingestion.filename_parser import calculate_fiscal_periods


def derive_fiscal_period(announcement_date: date) -> tuple[str, str]:
    """
    Map an announcement date to its Indian fiscal (fiscal_year, fiscal_quarter).

    Indian fiscal year runs April 1 – March 31.
    Quarter boundaries (calendar dates):

    +---------+-------------------+------------------------------------------+
    | Quarter | Calendar months   | Fiscal year label                        |
    +=========+===================+==========================================+
    | Q1      | Apr 1 – Jun 30    | FY{calendar_year + 1}                    |
    +---------+-------------------+------------------------------------------+
    | Q2      | Jul 1 – Sep 30    | FY{calendar_year + 1}                    |
    +---------+-------------------+------------------------------------------+
    | Q3      | Oct 1 – Dec 31    | FY{calendar_year + 1}                    |
    +---------+-------------------+------------------------------------------+
    | Q4      | Jan 1 – Mar 31    | FY{calendar_year}                        |
    +---------+-------------------+------------------------------------------+

    Examples
    --------
    >>> derive_fiscal_period(date(2018, 5, 15))
    ('FY2019', 'Q1')
    >>> derive_fiscal_period(date(2018, 10, 1))
    ('FY2019', 'Q3')
    >>> derive_fiscal_period(date(2019, 2, 28))
    ('FY2019', 'Q4')
    >>> derive_fiscal_period(date(2019, 4, 1))
    ('FY2020', 'Q1')

    Parameters
    ----------
    announcement_date:
        Any Python ``date`` object.

    Returns
    -------
    (fiscal_year, fiscal_quarter)
        e.g. ``('FY2019', 'Q2')``
    """
    month = announcement_date.month
    year = announcement_date.year

    if month in (4, 5, 6):
        return f"FY{year + 1}", "Q1"
    elif month in (7, 8, 9):
        return f"FY{year + 1}", "Q2"
    elif month in (10, 11, 12):
        return f"FY{year + 1}", "Q3"
    else:
        # January, February, March → Q4 of the current calendar year's FY
        return f"FY{year}", "Q4"


def calculate_period_end(fiscal_year: str, fiscal_quarter: str | None) -> date:
    """
    Return the last calendar date of the given Indian fiscal period.

    Delegates to :func:`ingestion.filename_parser.calculate_fiscal_periods`
    so period boundaries are identical to those used during document ingestion.

    Examples
    --------
    >>> calculate_period_end('FY2019', 'Q1')
    datetime.date(2018, 6, 30)
    >>> calculate_period_end('FY2019', 'Q4')
    datetime.date(2019, 3, 31)
    >>> calculate_period_end('FY2019', None)
    datetime.date(2019, 3, 31)

    Parameters
    ----------
    fiscal_year:
        ``'FY{YYYY}'`` string, e.g. ``'FY2019'``.
    fiscal_quarter:
        ``'Q1'`` – ``'Q4'``, or ``None`` for the full annual period.

    Returns
    -------
    Last ``date`` of the period.
    """
    _period_start, period_end = calculate_fiscal_periods(fiscal_year, fiscal_quarter)
    return period_end


def calculate_period_start(fiscal_year: str, fiscal_quarter: str | None) -> date:
    """
    Return the first calendar date of the given Indian fiscal period.

    Delegates to :func:`ingestion.filename_parser.calculate_fiscal_periods`
    so period boundaries are identical to those used during document ingestion.

    Examples
    --------
    >>> calculate_period_start('FY2019', 'Q1')
    datetime.date(2018, 4, 1)
    >>> calculate_period_start('FY2019', 'Q4')
    datetime.date(2019, 1, 1)
    >>> calculate_period_start('FY2019', None)
    datetime.date(2018, 4, 1)

    Parameters
    ----------
    fiscal_year:
        ``'FY{YYYY}'`` string, e.g. ``'FY2019'``.
    fiscal_quarter:
        ``'Q1'`` – ``'Q4'``, or ``None`` for the full annual period.

    Returns
    -------
    First ``date`` of the period.
    """
    period_start, _period_end = calculate_fiscal_periods(fiscal_year, fiscal_quarter)
    return period_start
