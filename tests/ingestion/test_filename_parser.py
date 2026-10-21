"""Tests for ingestion filename parsing, the rename tool, and the pledge sentinel."""

import contextlib
import io
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from ingestion.filename_parser import derive_fiscal_period, parse_filename
from ingestion.metadata_reader import read_metadata

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "tools"))
import normalize_filenames as nf  # noqa: E402


def _parse(name, doc_type, exchange="BSE+NSE"):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        parsed = parse_filename(Path(name), doc_type, exchange)
    return parsed, err.getvalue()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d,fy,fq", [
    (date(2016, 4, 1), "FY2017", "Q1"), (date(2016, 6, 30), "FY2017", "Q1"),
    (date(2016, 12, 28), "FY2017", "Q3"), (date(2017, 1, 1), "FY2017", "Q4"),
    (date(2017, 3, 31), "FY2017", "Q4"),
])
def test_indian_fiscal_period(d, fy, fq):
    assert derive_fiscal_period(d) == (fy, fq)


@pytest.mark.parametrize("name,category", [
    ("DHFL_Ann_20160504_Book Closure_SCANNED.pdf", "Corporate Action - Other"),
    ("INFY_Ann_20170814_Buyback_SCANNED.pdf", "Corporate Action - Other"),
    ("POWERGRID_Ann_20140512_Acquisition_SCANNED.pdf", "Corporate Action - Other"),
    ("INFY_Ann_20170208_News Clarification_SCANNED.pdf", "News Clarification"),
    ("JPINFRATEC_Ann_20141001_Director Change_SCANNED.pdf", "Director Change"),
    ("INFY_Ann_20160519_Annual General Meeting Updates.pdf", "AGM / Annual General Meeting Updates"),
    ("DHFL_Ann_20190129_Press Release (2)_SCANNED.pdf", "Press Release"),
    ("HDFC_Ann_20180430_Outcome Of Board Meeting (2).pdf", "Board Meeting"),
])
def test_announcement_categories_resolve(name, category):
    parsed, warnings = _parse(name, "corporate_announcement")
    assert parsed.document_category == category
    assert warnings == ""


def test_announcements_and_credit_docs_get_fiscal_periods():
    parsed, _ = _parse("DHFL_Ann_20160504_Resignation_SCANNED.pdf", "corporate_announcement")
    assert (parsed.fiscal_year, parsed.fiscal_quarter) == ("FY2017", "Q1")
    assert parsed.period_start == parsed.period_end == date(2016, 5, 4)
    parsed, _ = _parse("DHFL_Ann_20160708_Credit Rating_SCANNED.pdf", "credit_rating")
    assert (parsed.fiscal_year, parsed.fiscal_quarter) == ("FY2017", "Q2")
    assert parsed.document_category is None


def test_annual_and_quarterly():
    a, _ = _parse("DHFL_AnnualReport_FY2018.pdf", "annual_report")
    assert (a.fiscal_year, a.fiscal_quarter, a.period_end) == ("FY2018", None, date(2018, 3, 31))
    q, _ = _parse("DHFL_Q3FY2017_SCANNED.pdf", "quarterly_result")
    assert (q.fiscal_year, q.fiscal_quarter, q.period_end) == ("FY2017", "Q3", date(2016, 12, 31))


@pytest.mark.skipif(not RAW.exists(), reason="data/raw not present")
def test_every_real_filename_parses_cleanly():
    """Guards against silently getting NULL metadata from a misnamed file."""
    kinds = {"annual_reports": "annual_report", "quarterly_results": "quarterly_result",
             "corporate_announcements": "corporate_announcement", "credit_ratings": "credit_rating"}
    bad = []
    for cdir in RAW.iterdir():
        if not cdir.is_dir():
            continue
        for sub, dtype in kinds.items():
            for pdf in (cdir / sub).glob("*.pdf"):
                parsed, warnings = _parse(pdf.name, dtype)
                if warnings and pdf.stem.lower().endswith("announcement"):
                    continue  # a known generic title, left unclassified on purpose
                if warnings or parsed.period_end is None:
                    bad.append(f"{cdir.name}/{pdf.name}: {warnings.strip()[:60]}")
    assert bad == []


# ---------------------------------------------------------------------------
# Rename tool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("old,kind,new", [
    ("JAYPEE_Resignation_09122016_SCANNED.pdf", "ann", "JPINFRATEC_Ann_20161209_Resignation_SCANNED.pdf"),
    ("HDFCBANK_20170421_Financial Results Updates_SCANNED.pdf", "ann", "JPINFRATEC_Ann_20170421_Financial Results Updates_SCANNED.pdf"),
    ("PGCI_OBM_14052014_SCANNED.pdf", "ann", "JPINFRATEC_Ann_20140514_Outcome Of Board Meeting_SCANNED.pdf"),
    ("JSW_creditrating_15072016.pdf", "credit", "JPINFRATEC_Ann_20160715_Credit Rating.pdf"),
    ("IDEA_20190716_Credit Ratins_BWR_SCANNED.pdf", "credit", "JPINFRATEC_Ann_20190716_Credit Rating_BWR_SCANNED.pdf"),
    ("JAYPEE_Resignation_02012017 (2)_SCANNED.pdf", "ann", "JPINFRATEC_Ann_20170102_Resignation (2)_SCANNED.pdf"),
    ("IDEA_AnnualRepost_FY2018.pdf", "annual", "JPINFRATEC_AnnualReport_FY2018.pdf"),
    ("RELCAPITAL_Q1FY19_SCANNED.pdf", "quarterly", "JPINFRATEC_Q1FY2019_SCANNED.pdf"),
    ("JAYPEE_Q3_FY2016_SCANNED.pdf", "quarterly", "JPINFRATEC_Q3FY2016_SCANNED.pdf"),
])
def test_rename_rules(old, kind, new):
    assert nf.canonical(old, kind, "JPINFRATEC") == new


@pytest.mark.parametrize("old", [
    "JSW_Aquisition_311014_SCANNED.pdf",          # 6-digit date
    "TATASTEEL_201500513_Press Release_SCANNED.pdf",  # 9 digits
    "TCS_2018012_Appointment_SCANNED.pdf",         # 7 digits
    "TCS_2018102424_Press Release_SCANNED.pdf",    # 10 digits
])
def test_ambiguous_dates_are_not_guessed(old):
    assert nf.canonical(old, "ann", "X") is None


def test_rename_is_idempotent():
    name = "INFY_Ann_20170208_News Clarification_SCANNED.pdf"
    assert nf.canonical(name, "ann", "INFY") == name


# ---------------------------------------------------------------------------
# Shareholding sentinel
# ---------------------------------------------------------------------------


def test_dash_pledge_becomes_zero_not_null(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "ticker": "T", "company_name": "T Ltd", "exchange": "BSE",
        "documents_collected": {"shareholding_patterns": {"data": {
            "Q1FY2018": {"promoter_holding_pct": 50.0, "pledge_pct": "-"},
            "Q2FY2018": {"promoter_holding_pct": 51.0, "pledge_pct": 4.5},
        }}},
    }), encoding="utf-8")
    meta = read_metadata(tmp_path)
    assert meta.shareholding_data["Q1FY2018"]["pledge_pct"] == 0.0
    assert meta.shareholding_data["Q2FY2018"]["pledge_pct"] == 4.5


def test_label_fields_in_metadata_are_dropped(tmp_path, capsys):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "ticker": "T", "company_name": "T Ltd", "group": "distressed",
    }), encoding="utf-8")
    meta = read_metadata(tmp_path)
    assert not hasattr(meta, "group")
    assert "Prohibited fields" in capsys.readouterr().err
