"""
Example-based tests for src/feature_extraction/fiscal_utils.py.

Covers derive_fiscal_period(), calculate_period_end(), and
calculate_period_start() against known dates and fiscal period values.

Requirements: 10.4, 10.5
"""

from datetime import date

import pytest

from feature_extraction.fiscal_utils import (
    calculate_period_end,
    calculate_period_start,
    derive_fiscal_period,
)


# ---------------------------------------------------------------------------
# derive_fiscal_period
# ---------------------------------------------------------------------------

class TestDeriveFiscalPeriod:
    """Maps a calendar date to (fiscal_year, fiscal_quarter)."""

    # ---- core quarter mapping ----

    def test_q1_mid_month(self):
        assert derive_fiscal_period(date(2018, 5, 15)) == ("FY2019", "Q1")

    def test_q3_exact_start(self):
        assert derive_fiscal_period(date(2018, 10, 1)) == ("FY2019", "Q3")

    def test_q4_feb(self):
        assert derive_fiscal_period(date(2019, 2, 28)) == ("FY2019", "Q4")

    def test_q1_april_first(self):
        """April 1 is the first day of Q1 and a new fiscal year."""
        assert derive_fiscal_period(date(2019, 4, 1)) == ("FY2020", "Q1")

    # ---- boundary dates ----

    def test_boundary_march_31_is_q4(self):
        """March 31 is the last day of Q4 / the fiscal year."""
        assert derive_fiscal_period(date(2019, 3, 31)) == ("FY2019", "Q4")

    def test_boundary_april_1_starts_q1(self):
        """April 1 crosses into the next fiscal year's Q1."""
        assert derive_fiscal_period(date(2019, 4, 1)) == ("FY2020", "Q1")

    def test_boundary_sep_30_is_q2(self):
        """September 30 is the last day of Q2."""
        assert derive_fiscal_period(date(2018, 9, 30)) == ("FY2019", "Q2")

    def test_boundary_oct_1_starts_q3(self):
        """October 1 is the first day of Q3."""
        assert derive_fiscal_period(date(2018, 10, 1)) == ("FY2019", "Q3")

    def test_boundary_dec_31_is_q3(self):
        """December 31 is the last day of Q3."""
        assert derive_fiscal_period(date(2018, 12, 31)) == ("FY2019", "Q3")

    def test_boundary_jan_1_starts_q4(self):
        """January 1 is the first day of Q4."""
        assert derive_fiscal_period(date(2019, 1, 1)) == ("FY2019", "Q4")

    # ---- fiscal year label correctness ----

    def test_fy_label_format(self):
        """Fiscal year label must be 'FY' followed by a 4-digit year."""
        fy, _ = derive_fiscal_period(date(2020, 7, 15))
        assert fy.startswith("FY")
        assert len(fy) == 6
        assert fy[2:].isdigit()

    def test_q4_uses_calendar_year_for_fy_label(self):
        """Q4 (Jan–Mar) belongs to the fiscal year of the same calendar year."""
        assert derive_fiscal_period(date(2017, 1, 10)) == ("FY2017", "Q4")
        assert derive_fiscal_period(date(2017, 3, 31)) == ("FY2017", "Q4")

    def test_q1_to_q3_use_next_calendar_year_for_fy_label(self):
        """Apr–Dec dates belong to the fiscal year starting the following April."""
        assert derive_fiscal_period(date(2017, 4, 1))[0] == "FY2018"
        assert derive_fiscal_period(date(2017, 12, 31))[0] == "FY2018"


# ---------------------------------------------------------------------------
# calculate_period_end
# ---------------------------------------------------------------------------

class TestCalculatePeriodEnd:
    """Returns the last calendar date of the given fiscal period."""

    def test_fy2019_q1(self):
        assert calculate_period_end("FY2019", "Q1") == date(2018, 6, 30)

    def test_fy2019_q2(self):
        assert calculate_period_end("FY2019", "Q2") == date(2018, 9, 30)

    def test_fy2019_q3(self):
        assert calculate_period_end("FY2019", "Q3") == date(2018, 12, 31)

    def test_fy2019_q4(self):
        assert calculate_period_end("FY2019", "Q4") == date(2019, 3, 31)

    def test_fy2019_annual(self):
        """Annual period (fiscal_quarter=None) ends March 31 of the FY year."""
        assert calculate_period_end("FY2019", None) == date(2019, 3, 31)

    def test_fy2020_q1(self):
        assert calculate_period_end("FY2020", "Q1") == date(2019, 6, 30)

    def test_period_end_is_date_type(self):
        result = calculate_period_end("FY2019", "Q2")
        assert isinstance(result, date)


# ---------------------------------------------------------------------------
# calculate_period_start
# ---------------------------------------------------------------------------

class TestCalculatePeriodStart:
    """Returns the first calendar date of the given fiscal period."""

    def test_fy2019_q1(self):
        assert calculate_period_start("FY2019", "Q1") == date(2018, 4, 1)

    def test_fy2019_q2(self):
        assert calculate_period_start("FY2019", "Q2") == date(2018, 7, 1)

    def test_fy2019_q3(self):
        assert calculate_period_start("FY2019", "Q3") == date(2018, 10, 1)

    def test_fy2019_q4(self):
        assert calculate_period_start("FY2019", "Q4") == date(2019, 1, 1)

    def test_fy2019_annual(self):
        """Annual period starts April 1 of the year before the FY label year."""
        assert calculate_period_start("FY2019", None) == date(2018, 4, 1)

    def test_period_start_is_date_type(self):
        result = calculate_period_start("FY2019", "Q1")
        assert isinstance(result, date)

    # ---- start <= end consistency ----

    @pytest.mark.parametrize("fy,fq", [
        ("FY2019", "Q1"),
        ("FY2019", "Q2"),
        ("FY2019", "Q3"),
        ("FY2019", "Q4"),
        ("FY2019", None),
        ("FY2020", "Q1"),
    ])
    def test_start_before_end(self, fy, fq):
        start = calculate_period_start(fy, fq)
        end = calculate_period_end(fy, fq)
        assert start <= end
