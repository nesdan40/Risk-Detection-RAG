"""
Tests for src/feature_extraction/statement_parser.py.

Inputs are modelled on rows seen in real extracted pages (DHFL and Jet Airways
quarterly results, Infosys and DHFL annual reports) so regressions against real
layouts are caught.
"""

from decimal import Decimal

import pytest

from feature_extraction.statement_parser import (
    find_unit,
    parse_number,
    parse_statements,
    row_numbers,
)

D = Decimal


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("1,234", D(1234)), ("2,36,551", D(236551)), ("(1,671)", D(-1671)),
    ("-45", D(-45)), ("12.5", D("12.5")), ("(12.5)", D("-12.5")),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_formula_hints_are_not_numbers():
    """Jet: 'Net Profit after Tax (3-4) 4,963 ...' -> the (3-4) is a column formula."""
    assert row_numbers(" (3-4) 4,963 5,350 54,882") == [D(4963), D(5350), D(54882)]
    assert row_numbers("(X+XI) 1,025  9,973") == [D(1025), D(9973)]


def test_dates_note_refs_and_fiscal_tags_are_not_numbers():
    assert row_numbers(" 31.12.2016 2,36,551 2016-17 (Note 5) 7") == [D(236551), D(7)]


def test_negative_numbers_in_brackets_survive():
    assert row_numbers(" (1,671) 2,3 (12.5) -45") == [D(-1671), D(23), D(-12.5), D(-45)]


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("header,word,factor", [
    ("(₹ in lacs)", "lacs", D("0.01")),
    ("STATEMENT ... (@ in lacs)", "lacs", D("0.01")),
    ("Amount (7 in lakhs)", "lakhs", D("0.01")),
    ("in ` crore, except per equity share data", "crore", D(1)),
    ("(Rs. in Crores)", "crores", D(1)),
    ("(₹ in million)", "million", D("0.1")),
])
def test_find_unit(header, word, factor):
    assert find_unit([header, "Revenue 1 2 3"], 1) == (word, factor)


def test_find_unit_none_when_absent():
    assert find_unit(["Revenue 1 2 3"], 0) == (None, None)


# ---------------------------------------------------------------------------
# Quarterly results (modelled on DHFL / Jet)
# ---------------------------------------------------------------------------

DHFL_Q = """UNAUDITED STANDALONE FINANCIAL RESULTS
STATEMENT OF UNAUDITED FINANCIAL RESULTS FOR THE QUARTER & NINE MONTHS ENDED DECEMBER 31, 2016 (@ in lacs)
PARTICULARS 31.12.2016 30.09.2016 31.12.2015 31.12.2016 31.12.2015 31.03.2016
1. | Revenue from Operations 2,36,551 2,16,665 1,88,437 6,49,044 5,34,734 7,31,183
5. | Profit Before Tax 37,171 35,222 28,046 1,02,683 81,914 1,10,217
7. | Net Profit After Tax 24,477 23,261 18,591 67,878 53,954 72,920
"""

JET_Q = """STATEMENT OF STANDALONE UNAUDITED FINANCIAL RESULTS
Amount (7 in lakhs)
Total Income 575,818 595,392 577,279 1,169,539 1,111,405
d. Finance Cost 18,929 19,540 22,751 38,469 43,427
e. Depreciation and Amortisation 15,416 14,422 18,803 29,838 36,623
3. |Profit before Tax (1-2) 4,963 5,350 54,882 10,313 57,470
5. |Net Profit after Tax (3-4) 4,963 5,350 54,902 10,313 57,490
"""


def test_lacs_quarterly_table_is_converted_to_crore():
    r = parse_statements([DHFL_Q], "quarterly_result", nbfc=True)
    assert r.revenue == D("2365.51")
    assert r.pat == D("244.77")
    assert r.sources["revenue"]["unit"] == "lacs"
    assert r.sources["revenue"]["scope"] == "standalone"


def test_lender_gets_no_ebitda_margin_or_current_ratio():
    r = parse_statements([DHFL_Q], "quarterly_result", nbfc=True)
    assert r.ebitda_margin_pct is None and r.current_ratio is None


def test_formula_column_does_not_corrupt_pat():
    r = parse_statements([JET_Q], "quarterly_result", nbfc=False)
    assert r.pat == D("49.63")  # not -0.03


def test_ebitda_margin_computed_for_operating_company():
    r = parse_statements([JET_Q], "quarterly_result", nbfc=False)
    # (PBT 49.63 + finance 189.29 + depreciation 154.16) / revenue 5758.18
    assert r.ebitda_margin_pct == D("6.83")
    assert "computed" in r.sources["ebitda_margin_pct"]["how"]


def test_missing_unit_means_null_not_a_guess():
    page = "Revenue from Operations 2,36,551 2,16,665 1,88,437\nNet Profit After Tax 24,477 23,261 18,591\n"
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.revenue is None and r.pat is None


def test_prose_mention_without_columns_is_ignored():
    page = "(₹ in crore)\nThe company's revenue from operations grew strongly this year.\n"
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.revenue is None


def test_loss_is_negative():
    page = "(₹ in crore)\nTotal Income 100 90 80\nNet Profit/(Loss) after Tax (12.5) 3 4\n"
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.pat == D("-12.5")


def test_loss_first_net_profit_label_is_pat():
    """Jet Q4 FY2018: 'Net (Loss)/ Profit (5+6)' was missed and an Indian GAAP
    reconciliation row ('Net profit as per previous Indian GAAP 3 680') was read instead."""
    page = (
        "Amount (₹ in Lakhs)\n"
        "Total Income  605,515  611,617  607,282  2,395,837\n"
        "3. (Loss)/ Profit before Tax (1-2)  (103,600)  16,525  60,242  (76,762)\n"
        "7. Net (Loss)/ Profit (5+6)  (103,600)  16,525  60,242  (76,762)\n"
        "Net profit as per previous Indian GAAP  3 680  39 043\n"
    )
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.pat == D("-1036.00")
    assert r.sources["pat"]["label"].startswith("Net (Loss)/ Profit")


def test_loss_first_net_profit_after_tax_with_ocr_serial():
    """Jet Q3 FY2019: OCR serial 's.' pushed 'Profit after Tax' past the label window."""
    page = (
        "Amount (₹ in Lakhs)\n"
        "Total Income  612,421  623,669  634,934\n"
        "s. Net (Loss)/ Profit after Tax (3-4)  (58,777)  (129,746)  16,525\n"
    )
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.pat == D("-587.77")


# ---------------------------------------------------------------------------
# Annual reports
# ---------------------------------------------------------------------------

INFY_BOARD = """Board's report
in ` crore, except per equity share data
Particulars  Standalone  Consolidated
For the year ended  For the year ended
2018  2017  2018  2017
Revenue from operations  61,941  59,289  70,522  68,484
Profit after tax(1)(2)  16,155  13,818  16,029  14,353
"""


def test_side_by_side_standalone_consolidated_prefers_consolidated():
    r = parse_statements([INFY_BOARD], "annual_report", nbfc=False)
    assert r.revenue == D(70522)
    assert r.sources["revenue"]["scope"] == "consolidated"
    assert r.revenue_prev == D(68484)  # previous-year column of the same scope


def test_stacked_footnote_markers_do_not_become_numbers():
    r = parse_statements([INFY_BOARD], "annual_report", nbfc=False)
    assert r.pat == D(16029)


STATEMENT_PAGE = """Dewan Housing Finance Corporation Limited
Consolidated Statement of Profit and Loss
for the year ended March 31, 2018
(₹ in lakh)
Revenue from Operations  1,051,474  890,672
PROFIT FOR THE YEAR  115,320  279,635
"""
TEN_YEAR_SUMMARY = """Ten year summary (₹ in crore)
Profit After Tax  91.8  150.7  265.1  306.4  451.9  529.0  621.3  729.2  2,896.5  1,172.1
"""


def test_statement_page_beats_ten_year_summary():
    """The summary lists the oldest year first (91.8); the statement holds the real figure."""
    r = parse_statements([TEN_YEAR_SUMMARY, STATEMENT_PAGE], "annual_report", nbfc=True)
    assert r.pat == D("1153.20")
    assert r.sources["pat"]["scope"] == "consolidated"


def test_note_reference_column_is_skipped():
    page = ("Statement of Profit and Loss\n(₹ in crore)\n"
            "Revenue from operations  25  12,345  10,000\n")
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.revenue == D(12345)
    assert r.revenue_prev == D(10000)


def test_consolidated_beats_standalone_regardless_of_page_order():
    standalone = "Standalone Statement of Profit and Loss\n(₹ in crore)\nRevenue from operations  500  400\n"
    consolidated = "Consolidated Statement of Profit and Loss\n(₹ in crore)\nRevenue from operations  900  800\n"
    r = parse_statements([standalone, consolidated], "annual_report", nbfc=False)
    assert r.revenue == D(900)


def test_cash_flow_from_operating_activities():
    page = ("Consolidated Cash Flow Statement\n(₹ in lakh)\n"
            "Net Cash Flow (used in) Operating Activities  (A)  (2,915,399)  (1,559,319)\n")
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.cfo == D("-29153.99")
    # lender (this row is DHFL's): reference only, not the CFO_NEGATIVE input
    r = parse_statements([page], "annual_report", nbfc=True)
    assert r.cfo is None and r.additional["cfo_lender"] == "-29153.99"


def test_current_ratio_computed_from_balance_sheet():
    page = ("Consolidated Balance Sheet\n(₹ in crore)\n"
            "Total current assets  800  700\nTotal current liabilities  200  250\n")
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.current_ratio == D("4.0000")
    assert "computed" in r.sources["current_ratio"]["how"]


def test_reported_ratio_is_preferred_and_range_checked():
    page = "Key ratios\nDebt equity ratio  0.45  0.52\nCurrent ratio  1.8  1.6\n"
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.debt_to_equity == D("0.45")
    assert r.current_ratio == D("1.8")
    assert r.sources["debt_to_equity"]["how"] == "reported ratio"


def test_dotted_note_after_label_words_is_not_pat():
    """HDFC: 'PROFIT AFTER TAX ATTRIBUTABLE TO THE CORPORATION  5.1  11,051.12 ...' -> 11,051.12, not 5.1."""
    page = ("Consolidated Statement of Profit and Loss for the year ended March 31, 2017\n"
            "INCOME  ` in Crore  ` in Crore\n"
            "Revenue from Operations  26  35,005.26  31,873.34\n"
            "PROFIT AFTER TAX ATTRIBUTABLE TO THE CORPORATION  5.1  11,051.12  10,190.26\n")
    assert parse_statements([page], "annual_report", nbfc=True).pat == D("11051.12")


def test_ocr_space_beside_thousands_comma():
    """HDFC Q2 FY2019: '11 ,245.61' is one figure, not 11 and 245.61."""
    assert row_numbers("  11 ,245.61  9,947.35  9,003.21") == [D("11245.61"), D("9947.35"), D("9003.21")]
    assert row_numbers("  17,580.51  13, 111.21") == [D("17580.51"), D("13111.21")]


def test_space_as_thousands_separator_in_wide_column_rows():
    """Reliance Capital: 'Total revenue from operations  20 447  18 718' -> 20447, not a note ref 20 and 447."""
    assert row_numbers("  20 447  18 718") == [D(20447), D(18718)]
    assert row_numbers("  (1 454)  (4 556)") == [D(-1454), D(-4556)]
    assert row_numbers("  28  7 677  7 263") == [D(28), D(7677), D(7263)]
    # Reliance Capital Q4 FY2020: a space inside the loss bracket
    assert row_numbers(" (VIII-IX)  ( 2 179)  ( 145)  ( 2 304)") == [D(-2179), D(-145), D(-2304)]
    page = ("Consolidated statement of profit and loss for the year ended March 31, 2019\n(` in crore)\n"
            "Total revenue from operations  20 447  18 718\nProfit for the year after tax  (1 454)  (4 556)\n")
    r = parse_statements([page], "annual_report", nbfc=True)
    assert (r.revenue, r.pat) == (D(20447), D(-1454))


def test_standalone_segment_table_does_not_outrank_the_statement():
    """Jet Q3 FY2019: the segment table's unbracketed 'Profit after Tax  58 777' was scoped
    'unknown', which outranked the standalone statement's '(58,777)'."""
    page = ("STATEMENT OF STANDALONE UNAUDITED FINANCIAL RESULTS FOR THE QUARTER ENDED 31ST DECEMBER, 2018\n"
            "Amount (₹ in Lakhs)\n"
            "Total Income  612,421  623,669  634,934\n"
            "s. Net (Loss)/ Profit after Tax (3-4)  (58,777)  (129,746)  16,525\n"
            "UNAUDITED STANDALONE SEGMENTWISE REVENUE, PROFIT FOR THE QUARTER ENDED 31ST DECEMBER, 2018\n"
            "Amount (₹ in Lakhs)\n"
            "Profit after Tax  58 777  129 746  16,525\n")
    assert parse_statements([page], "quarterly_result", nbfc=False).pat == D("-587.77")


def test_consolidated_segment_table_does_not_outrank_the_statement():
    """Jet Q4 FY2018: 'Loss Profit after Tax  103 600' in a STANDALONE AND CONSOLIDATED
    SEGMENTWISE table beat the statement's 'Net (Loss)/ Profit (5+6)  (103,600)'."""
    stmt = ("STATEMENT OF CONSOLIDATED AUDITED FINANCIAL RESULTS FOR THE YEAR ENDED 31ST MARCH, 2018\n"
            "Amount (₹ in Lakhs)\n"
            "Total Income  605,515  611,617  607,282\n"
            "7. Net (Loss)/ Profit (5+6)  (103,600)  16,525  60,242\n")
    seg = ("AUDITED STANDALONE AND CONSOLIDATED SEGMENTWISE REVENUE, RESULTS FOR THE YEAR ENDED 31ST MARCH, 2018\n"
           "Amount (₹ in Lakhs)\n"
           "Loss Profit after Tax  103 600  16 525  60 242\n")
    assert parse_statements([stmt, seg], "quarterly_result", nbfc=False).pat == D("-1036.00")


def test_single_space_ocr_columns_are_not_joined():
    """OCR rows separate columns by one space; '145 226' are two columns."""
    assert row_numbers(" 162 19 145 226 205") == [D(162), D(19), D(145), D(226), D(205)]
    # JPINFRATEC: an OCR row with one stray double space is still not space-grouped
    assert row_numbers(" 692 672 300 2,045.  1,826") == [D(692), D(672), D(300), D(2045), D(1826)]


def test_ocr_mangled_minus_in_formula_is_not_a_number():
    """Jet Q3 FY2018: 'Profit before Tax (1·2)  16,525' read as PBT -1."""
    assert row_numbers(" (1·2)  16,525  4,963  30,520") == [D(16525), D(4963), D(30520)]


def test_reported_ratio_as_proportion():
    """PowerGrid: 'Debt/Equity Ratio (#)  71:29  70:30' is 71/29, not 29."""
    page = ("Physical and financial performance\n"
            "Debt/Equity Ratio (#)  71:29  70:30  72:28  69:31  65:35\n"
            "Current Ratio  0.36:1  0.46:1  0.41:1  0.56:1  0.92:1\n")
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.debt_to_equity == D("2.4483")
    assert r.current_ratio == D("0.36")


def test_note_and_page_reference_columns_are_skipped():
    """Tata Steel FY2016: 'Revenue from operations  22  253  1,21,618.42  1,44,298.36' -> not 253."""
    page = ("CONSOLIDATED STATEMENT OF PROFIT AND LOSS\nfor the year ended 31st March, 2016\n(` crore)\n"
            "(a) Revenue from operations  22  253  1,21,618.42  1,44,298.36\n"
            "(3) Profit before Exceptional Items and Tax  2,300.83  2,540.58\n"
            "(f) Finance costs  26  254  4,128.64  4,847.75\n")
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.revenue == D("121618.42")
    assert r.interest_coverage == D("1.5573")  # (2,300.83 + 4,128.64) / 4,128.64; page 254 is not the cost


def test_continuing_operations_subtotal_is_not_pat():
    """Tata Steel FY2017: the year's loss is (4,168.57), not continuing operations' (304.38)."""
    page = ("CONSOLIDATED STATEMENT OF PROFIT AND LOSS\n(` crore)\n"
            "X  Profit/(loss) after tax from continuing operations  (304.38)  2,042.79\n"
            "Profit/(loss) after tax from discontinued operations  (3,864.19)  (2,539.88)\n"
            "XII Profit/(loss) for the year (A)  (4,168.57)  (497.09)\n")
    assert parse_statements([page], "annual_report", nbfc=False).pat == D("-4168.57")


def test_percentage_gearing_is_not_a_debt_to_equity_ratio():
    """Unitech FY2017: 'Net debt to equity ratio  29.15%  26.19%' is net debt / (capital + debt)."""
    page = "Capital management\nNet debt to equity ratio  29.15%  26.19%  25.18%\n"
    assert parse_statements([page], "annual_report", nbfc=False).debt_to_equity is None
    # key-ratio tables add a "% change" column after the ratios
    page = "Key financial ratios\nCurrent ratio  0.25  0.28  (10.71%)\n"
    assert parse_statements([page], "annual_report", nbfc=False).current_ratio == D("0.25")


def test_interest_earned_is_not_revenue_for_non_lenders():
    """PowerGrid FY2015: a cash-flow 'Interest earned on Deposits' row was taken as revenue."""
    page = ("Cash Flow Statement\n(` in Crore)\n"
            "Interest earned on Deposits, Bonds and loans to State Govts.  386.03  192.89\n")
    assert parse_statements([page], "annual_report", nbfc=False).revenue is None
    assert parse_statements([page], "annual_report", nbfc=True).revenue == D("386.03")


def test_empty_pages_return_all_none():
    r = parse_statements(["", "   "], "annual_report", nbfc=False)
    assert r.revenue is None and r.pat is None and r.cfo is None and r.sources == {}


# ---------------------------------------------------------------------------
# OCR / unit sanity (Jaypee Infratech)
# ---------------------------------------------------------------------------


def test_rupee_figures_under_a_crore_header_are_not_read_as_crore():
    page = ("(Rs. in crores)\nSTATEMENT OF PROFIT AND LOSS\n"
            "Net Profit after Tax (`)  (8,763,853,100)  (3,570,015,994)\n")
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.pat == D("-876.39")


def test_negative_revenue_from_ocr_is_rejected():
    page = ("(in Lacs)\n(a) Net sales / Income from operations (4,075) 34,753 1,19,638 96,214\n"
            "Total Income from Operations (Net) 64,071 31,763 1,19,636 96,214\n")
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.revenue == D("640.71")


# ---------------------------------------------------------------------------
# Interest coverage on quarterly results; debt-to-equity from the balance sheet
# ---------------------------------------------------------------------------


def test_interest_coverage_computed_for_quarterly_results():
    r = parse_statements([JET_Q], "quarterly_result", nbfc=False)
    # (PBT 49.63 + finance 189.29) / finance 189.29
    assert r.interest_coverage == D("1.2622")
    assert "computed" in r.sources["interest_coverage"]["how"]


def test_pre_ind_as_pbt_label_is_recognised():
    """Jaypee FY2015 Q1: 'Profit from ordinary activities before tax (7+8)'."""
    page = ("(Rs. in Lacs)\n"
            "1. |Net Sales/Income from Operations 61,120 57,354 70,412 2,59,012\n"
            "7. |Finance costs 21,175 21,483 23,148 89,402\n"
            "9. {Profit from ordinary activities before tax (7+8) 5815 3864 10250 40045\n")
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.interest_coverage == D("1.2746")


def test_finance_cost_sign_is_presentation_only():
    """Jaypee FY2017 Q4 '27,164)' (stray OCR bracket) and SpiceJet '(6,027.20)' (bracketed
    expense) are both a positive finance cost."""
    page = ("(Rs. in Lacs)\nTotal Income from Operations 64,071 31,763 1,19,636\n"
            "6. [Finance costs 27,164) 16,862) 31,786\n"
            "9. |Profit/(Loss) from ordinary activities before tax (7+8) (04,049) (6,428) (19,739)\n")
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.interest_coverage == D("0.8509")  # (-4,049 + 27,164) / 27,164


def test_bracketed_finance_cost_and_loss_before_tax():
    """SpiceJet FY2021 consolidated P&L."""
    page = ("Consolidated Statement of Profit and Loss\n(₹ in millions)\n"
            "Revenue from operations  51,714.50  1,23,571.14\n"
            "Finance costs  37  (6,027.20)  (5,455.29)\n"
            "Loss before tax  (10,298.86)  (9,365.68)\n")
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.interest_coverage == D("-0.7087")  # (-1,029.89 + 602.72) / 602.72


JAYPEE_BS_SCHEDULE_VI = """CONSOLIDATED BALANCE SHEET AS AT 31st MARCH, 2015
Particulars  Consolidated  As at  As at
I. EQUITY AND LIABILITIES
(1) Shareholders' Funds
(a) Share Capital  2  13,889,334,970  13,889,334,970
(b) Reserves and Surplus  3  49,575,959,226  46,643,623,998
(a) Long-term borrowings - Secured  4  67,687,906,156  77,918,923,991
(a) Short-term borrowings  –  –
Total  209,330,056,776  206,359,013,097
"""

INFY_BS_NO_DEBT = """Consolidated Balance Sheet
in ` crore
EQUITY AND LIABILITIES
Equity
Equity share capital  1,144  1,144  572
Other equity  67,838  60,600  54,198
Total equity  68,982  61,744  54,770
Total non-current liabilities  360  367  206
Total current liabilities  14,013  13,239  11,383
Total equity and liabilities  83,355  75,350  66,359
"""

DIRECTORS_REPORT_SUMMARY = """DIRECTORS' REPORT
The balance sheet of the Company is summarised below.
Total Equity  18,201.01  18,600.00
"""


def test_debt_to_equity_from_schedule_vi_components():
    """No total-equity row: equity = share capital + reserves and surplus."""
    r = parse_statements([JAYPEE_BS_SCHEDULE_VI], "annual_report", nbfc=False)
    assert r.debt_to_equity == D("1.0665")
    assert r.sources["debt_to_equity"]["how"] == "computed borrowings / share capital + reserves"


def test_debt_free_balance_sheet_gives_zero():
    r = parse_statements([INFY_BS_NO_DEBT], "annual_report", nbfc=False)
    assert r.debt_to_equity == D("0")
    assert r.sources["debt_to_equity"]["how"] == "balance sheet shows no borrowings"


def test_equity_outside_the_balance_sheet_is_not_paired():
    """A Directors' Report summary mentions 'balance sheet' but is not the liabilities side."""
    r = parse_statements([DIRECTORS_REPORT_SUMMARY, JAYPEE_BS_SCHEDULE_VI], "annual_report", nbfc=False)
    assert r.sources["debt_to_equity"]["page"] == 2


def test_no_liabilities_total_means_no_zero_guess():
    """Equity found but the liabilities section is not on the page: unknown, not zero."""
    page = "Consolidated Balance Sheet\nin ` crore\nEQUITY AND LIABILITIES\nTotal equity  68,982  61,744\n"
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.debt_to_equity is None


def test_highlights_summary_never_yields_a_zero():
    """Infosys 'Year at a glance': equity and liabilities totals, no borrowings rows, no statement title."""
    page = ("The year at a glance\nin ` crore\nNon-current liabilities  861  360\n"
            "Total equity  64,924  68,982\nTotal equity and liabilities  79,890  83,355\n")
    r = parse_statements([page, INFY_BS_NO_DEBT], "annual_report", nbfc=False)
    assert r.debt_to_equity == D("0")
    assert r.sources["debt_to_equity"]["page"] == 2
    assert parse_statements([page], "annual_report", nbfc=False).debt_to_equity is None


def test_liabilities_continued_on_next_page():
    """Infosys FY2019: equity ends page 1; borrowings (none) and liability totals are overleaf."""
    equity_page = ("Consolidated Balance Sheet\nin ` crore\nEquity and Liabilities\nEquity\n"
                   "Equity share capital  2.11  2,170  1,088\nOther equity  62,778  63,835\n"
                   "Total equity  65,006  64,924\n")
    with_debt = "Non-current liabilities\nBorrowings  6,500  6,000\nTotal non-current liabilities  7,000  6,500\n"
    without_debt = "Non-current liabilities\nTotal non-current liabilities  1,094  861\n"
    assert parse_statements([equity_page, with_debt], "annual_report", nbfc=False).debt_to_equity == D("0.1000")
    assert parse_statements([equity_page, without_debt], "annual_report", nbfc=False).debt_to_equity == D("0")


def test_dotted_note_ref_does_not_become_share_capital():
    """No total row and a dotted note ref: the note column is skipped, components summed."""
    page = ("Balance Sheet\nin ` crore\nEquity and Liabilities\n"
            "Equity share capital  2.11  2,170  1,088\nOther equity  62,778  63,835\n"
            "Borrowings  6,500  6,000\nTotal current liabilities  100  90\n")
    # 6,500 / (2,170 + 62,778)
    assert parse_statements([page], "annual_report", nbfc=False).debt_to_equity == D("0.1001")


def test_mismatched_component_columns_are_not_summed():
    page = ("Balance Sheet\nin ` crore\nEquity and Liabilities\n"
            "Equity share capital  2,170  1,088  572\nOther equity  62,778  63,835\n"
            "Borrowings  6,500  6,000\nTotal current liabilities  100  90\n")
    assert parse_statements([page], "annual_report", nbfc=False).debt_to_equity is None


def test_dotted_note_reference_column_is_skipped():
    """Infosys FY2017 consolidated P&L: 'Revenue from operations  2.18  68,484  62,441'."""
    page = ("Consolidated Statement of Profit and Loss\nin ` crore\n"
            "Revenue from operations  2.18  68,484  62,441\n")
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.revenue == D(68484)
    assert r.revenue_prev == D(62441)


def test_small_decimal_figures_are_not_taken_for_note_refs():
    page = "Statement of Profit and Loss\n(₹ in crore)\nRevenue from operations  12.50  10.20  8.40\n"
    assert parse_statements([page], "annual_report", nbfc=False).revenue == D("12.50")


# ---------------------------------------------------------------------------
# Jaypee absolute-rupee statements: CFO label, rupee-column unit, current-ratio subtotals
# ---------------------------------------------------------------------------

JAYPEE_CF = """CONSOLIDATED CASH FLOW STATEMENT FOR THE YEAR ENDED 31ST MARCH, 2017
For the Year ended For the Year ended
`  `
Cash Generated from Operations  11,058,565,547  12,833,069,232
(a) Taxes Paid  353,443,232  684,215,704
CASH INFLOW / (OUTFLOW) FROM OPERATING ACTIVITIES  10,705,122,315  12,148,853,528
"""


def test_cash_inflow_outflow_label_beats_cash_generated():
    r = parse_statements([JAYPEE_CF], "annual_report", nbfc=False)
    assert r.cfo == D("1070.51")
    assert r.sources["cfo"]["unit"] == "rupees"


def test_rupee_symbol_column_header_means_rupees():
    assert find_unit(["Particulars  As at", "`  `  `", "Inventories  7  55,963,894,595"], 2)[0] == "rupees"


JAYPEE_BS_SUBTOTALS = """CONSOLIDATED BALANCE SHEET AS AT 31st MARCH, 2015
I. EQUITY AND LIABILITIES
(4) Current Liabilities
(a) Short-term borrowings  –  –
(b) Trade payables  7  13,352,390,511  11,425,816,613
(c) Other current liabilities  8  49,250,723,550  39,143,144,658
73,701,184,736  61,809,217,372
Total  209,330,056,776  206,359,013,097
II. ASSETS
(2) Current assets
(b) Inventories  13  63,430,793,556  67,258,163,316
(f) Other current assets  17  17,970,580,070  13,208,006,851
99,281,229,489  95,946,280,276
Total  209,330,056,776  206,359,013,097
"""


def test_current_ratio_from_section_subtotals():
    r = parse_statements([JAYPEE_BS_SUBTOTALS], "annual_report", nbfc=False)
    assert r.current_ratio == D("1.3471")  # 99,281,229,489 / 73,701,184,736
    assert "subtotals" in r.sources["current_ratio"]["how"]


def test_section_without_subtotal_gives_no_current_ratio():
    page = ("Balance Sheet\nEQUITY AND LIABILITIES\nCurrent Liabilities\nTrade payables  100  90\n"
            "Total  500  450\nCurrent assets\nInventories  200  180\nTotal  500  450\n")
    assert parse_statements([page], "annual_report", nbfc=False).current_ratio is None


# ---------------------------------------------------------------------------
# Results: same quarter previous year located by header dates
# ---------------------------------------------------------------------------


def test_yoy_column_from_full_header_dates():
    """Jaypee Q1 FY2017: [30.06.2016, 31.03.2016, 30.06.2015, 31.03.2016] -> third column."""
    page = ("(Rs. in Lacs)\n30.06.2016 [34.03.2016 —*| 30.06.2015 34.03.2016\n"
            "(a) Net sales / Income from operations 109869| 119636 51,344 279978\n")
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.revenue == D("1098.69") and r.revenue_prev == D("513.44")


def test_yoy_column_found_when_not_third():
    page = ("(Rs. in Lacs)\n31.12.2016 31.12.2015 30.09.2016\n"
            "Revenue from operations 1,000 800 900\n")
    assert parse_statements([page], "quarterly_result", nbfc=False).revenue_prev == D("8.00")


def test_yoy_column_from_year_only_header():
    """Infosys: '2016  2016  2015  2016' -> third column when its year is one earlier."""
    page = "in ` crore\n2016  2016  2015  2016\nNet revenue from operations  16,782  16,550  14,354  62,441\n"
    assert parse_statements([page], "quarterly_result", nbfc=False).revenue_prev == D(14354)


def test_no_yoy_column_without_a_matching_header():
    page = "in ` crore\n2017  2016  2017\nRevenue from operations  17,120  17,273  16,550\n"
    assert parse_statements([page], "quarterly_result", nbfc=False).revenue_prev is None


def test_half_yearly_statement_of_assets_and_liabilities_in_results():
    """Jaypee Q2 FY2016 results: D/E from the half-yearly balance sheet."""
    page = ("(Rs. in Lacs)\nSTATEMENT OF ASSETS AND LIABILITIES\nA EQUITY AND LIABILITIES\n"
            "1. |Shareholders' Funds:\n_ (a) Share Capital 1,38,893 1,38,893\n"
            "(b) Reserves & Surplus 4,82,004 4,96,342\nSub-Total - Shareholders’ Funds\n"
            "(a) Long-term borrowings 8,76,854 6,44,541\n")
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.debt_to_equity == D("1.4122")  # 8,76,854 / (1,38,893 + 4,82,004)


def test_results_without_balance_sheet_get_no_balance_sheet_ratios():
    r = parse_statements([JET_Q], "quarterly_result", nbfc=False)
    assert r.debt_to_equity is None and r.current_ratio is None


def test_lender_debt_to_equity_is_reference_only():
    """Lenders run at 6-10x by design: the figure is kept, but not as the rule input."""
    page = ("Consolidated Balance Sheet\nin ` crore\nEQUITY AND LIABILITIES\nTotal equity  1,000  900\n"
            "Borrowings  8,000  7,000\nTotal current liabilities  9,500  8,400\n")
    r = parse_statements([page], "annual_report", nbfc=True)
    assert r.debt_to_equity is None
    assert r.additional["debt_to_equity_lender"] == "8.0000"
    assert parse_statements([page], "annual_report", nbfc=False).debt_to_equity == D("8.0000")


# ---------------------------------------------------------------------------
# SpiceJet FY2020-22: segment table, negative equity, covenant prose
# ---------------------------------------------------------------------------


def test_wide_row_second_column_is_not_previous_year():
    page = ("Consolidated Statement of Profit and Loss\n(₹ in millions)\n"
            "Revenue from operations  90,720.72  411.82  150.12  91,282.66  (67.91)  91,214.75\n")
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.revenue == D("9072.07")
    assert r.revenue_prev is None


def test_negative_equity_gives_no_debt_to_equity():
    """'TOTAL EQUITY AND LIABLITIES' (sic) must not be read as equity."""
    page = ("Consolidated Balance Sheet\nEQUITY AND LIABILITIES\nTotal equity  (15,803.98)  (3,499.64)\n"
            "(i) Borrowings  19  4,593.03  5,566.28\nTotal current liabilities  100,000.00  30,000.00\n"
            "TOTAL EQUITY AND LIABLITIES  129,555.21  48,046.49\n")
    assert parse_statements([page], "annual_report", nbfc=False).debt_to_equity is None


def test_covenant_prose_is_not_a_reported_ratio():
    page = ("debt service coverage ratio of 1.25. The Company have not complied with this covenant\n"
            "interest coverage ratio of 2.5 was required\n")
    assert parse_statements([page], "annual_report", nbfc=False).interest_coverage is None


def test_reported_interest_coverage_table_row():
    page = "Key ratios\nInterest Coverage Ratio  (0.72)  (0.92)\nCurrent Ratio  0.33  0.38\n"
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.interest_coverage == D("-0.72") and r.current_ratio == D("0.33")


# ---------------------------------------------------------------------------
# TCS FY2019 Q4 (OCR): mangled formula hint, point-for-comma, "before finance costs"
# ---------------------------------------------------------------------------


def test_ocr_mangled_formula_hint_is_not_a_number():
    assert row_numbers(" 17-8]  8,152  8,121  6,925") == [D(8152), D(8121), D(6925)]
    assert row_numbers(" (7-Bl  7,811  7,486") == [D(7811), D(7486)]


def test_point_read_for_thousands_comma():
    assert row_numbers(" 8.152  8,121  6.925  31,562") == [D(8152), D(8121), D(6925), D(31562)]
    assert row_numbers(" 0.333  1.250") == [D("0.333"), D("1.250")]  # no comma-grouped cell: untouched


def test_profit_before_finance_costs_is_not_finance_cost():
    page = ("(₹ crore)\nRevenue from operations  38,010  37,338  32,075\n"
            "5 PROFIT BEFORE FINANCE COSTS AND TAX (3-4)  10,730  10,727  9,141\n"
            "6 Finance costs  28  16  12\n7 PROFIT BEFORE TAX (5-6)  10,702  10,711  9,129\n"
            "9 NET PROFIT FOR THE PERIOD 17-8]  8.152  8,121  6.925\n")
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.pat == D(8152)
    assert r.interest_coverage == D("383.2143")  # (10,702 + 28) / 28


# ---------------------------------------------------------------------------
# NSE website printouts of results (TCS / TATASTEEL / UNITECH / HDFCBANK / IDEA)
# ---------------------------------------------------------------------------

NSE_HEADER = ("15/07/2026, 11:57  nsearchives.nseindia.com/archives/financial_results/financial_res_TCS_1027727.html\n"
              "Description  Amount(Rs. in  Segment Reporting (Rs.lakhs)\nlakhs)\n")


def test_nse_printout_single_value_rows():
    page = (NSE_HEADER
            + "(a) Net sales/income from operations (Net of excise duty)  2930500.00  Others  430100.00\n"
            + "Finance costs  1200.00\n"
            + "Profit / (Loss) from ordinary activities before tax  831000.00\n"
            + "Tax expense  199200.00  Technology\n"
            + "Net Profit / (Loss) for the period  631800.00  Manufacturing  296700.00\n")
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.revenue == D("29305.00") and r.pat == D("6318.00")
    assert r.interest_coverage == D("693.5000")  # (8,310 + 12) / 12


def test_nse_printout_finance_cost_equal_to_tax_is_rejected():
    """NSE's FY2018 template repeats the tax expense in the finance-cost row."""
    page = (NSE_HEADER
            + "(a) Net sales/income from operations (Inclusive of excise duty)  2958400.00\n"
            + "(f) Finance costs  189600.00\n"
            + "Profit / (Loss) from ordinary activities before tax  784600.00\n"
            + "Tax expense  189600.00  Other Un-allocable Expenditure net off\n"
            + "Net Profit / (Loss) for the period  595000.00\n")
    r = parse_statements([page], "quarterly_result", nbfc=False)
    assert r.interest_coverage is None
    assert "finance_cost_rejected" in r.sources


def test_single_value_rows_ignored_outside_printouts():
    page = "(Rs. in lakhs)\nNet Profit / (Loss) for the period  595000.00\n"
    assert parse_statements([page], "quarterly_result", nbfc=False).pat is None


def test_lender_reported_coverage_and_current_ratio_are_reference_only():
    """DHFL FY2019 key-ratios table: reported by the lender, kept, but not rule inputs."""
    page = "Key Financial Ratios\nInterest Coverage ratio  0.88  1.19\nCurrent ratio  0.97  1.05\n"
    r = parse_statements([page], "annual_report", nbfc=True)
    assert r.interest_coverage is None and r.current_ratio is None
    assert r.additional["interest_coverage_lender"] == "0.88"
    assert r.additional["current_ratio_lender"] == "0.97"
    r = parse_statements([page], "annual_report", nbfc=False)
    assert r.interest_coverage == D("0.88") and r.current_ratio == D("0.97")


# ---------------------------------------------------------------------------
# HDFC Bank: chained formula hints, '000 unit with curly quote, AOC-1 subsidiaries page
# ---------------------------------------------------------------------------


def test_chained_bracket_formula_hint_is_not_a_number():
    assert row_numbers(" (12)-(13)  460144  479928  389384") == [D(460144), D(479928), D(389384)]
    assert row_numbers(" (7)-(8)-(9)  701838  729456") == [D(701838), D(729456)]
    assert row_numbers(" (3)·{6)  864775  883566") == [D(864775), D(883566)]
    assert row_numbers("  (12)  (13)") == [D(-12), D(-13)]  # bracketed values, no operator: kept


def test_thousands_unit_with_curly_quote():
    assert find_unit(["Statement of Profit and Loss", "` in ‘000"], 1)[0] == "thousand"


def test_large_bank_figure_in_thousands_is_not_read_as_rupees():
    page = ("Statement of Profit and Loss\n` in ‘000\nInterest earned  13  1,051,617,740  852,878,437\n")
    assert parse_statements([page], "annual_report", nbfc=True).revenue == D("105161.77")


def test_aoc1_subsidiary_page_is_ignored():
    aoc1 = ("Form AOC - 1: Pursuant to section 129\n(` crore)\n12. Profit after taxation  347.95  1,151.09\n")
    own = ("Statement of Profit and Loss\n(` crore)\nNet profit for the year  22,445.61  17,486.73\n")
    assert parse_statements([aoc1, own], "annual_report", nbfc=True).pat == D("22445.61")


def test_ocr_misread_lacs_unit():
    assert find_unit(["FINANCIAL RESULTS FOR THE QUARTER", "in Jacs)", "Net Profit  479928  464260"], 2)[0] == "lacs"
