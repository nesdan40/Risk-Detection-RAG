"""Tests for src/feature_extraction/financial_extractor.py (page-file based extraction)."""

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from feature_extraction import financial_extractor as fe

D = Decimal

ANNUAL = """Consolidated Statement of Profit and Loss
(₹ in crore)
Revenue from operations  12,000  10,000
Profit after tax  1,500  1,200
"""
QUARTERLY = """STATEMENT OF UNAUDITED FINANCIAL RESULTS (₹ in lacs)
Revenue from Operations 2,36,551 2,16,665 1,88,437 6,49,044 5,34,734 7,31,183
Net Profit After Tax 24,477 23,261 18,591 67,878 53,954 72,920
"""


def _write_pages(tmp_path, pages):
    p = tmp_path / "1_pages.json"
    p.write_text(json.dumps({"pages": pages}), encoding="utf-8")
    return p


def _doc(doc_type="annual_report", fy="FY2018", fq=None):
    return SimpleNamespace(id=1, document_type=doc_type, fiscal_year=fy, fiscal_quarter=fq)


def _company(sector="IT Services", listing="equity"):
    return SimpleNamespace(id=7, sector=sector, listing_type=listing)


# ---------------------------------------------------------------------------
# Company classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sector,listing,expected", [
    ("NBFC / Housing Finance", "equity", True),
    ("Housing Finance", "equity", True),
    ("Banking / Financial Services", "equity", True),
    ("NBFC / Diversified Finance", "equity", True),
    ("Telecom", "debt_only", True),
    ("IT Services", "equity", False),
    (None, "equity", False),
])
def test_is_nbfc(sector, listing, expected):
    assert fe.is_nbfc(_company(sector, listing)) is expected


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


def test_missing_pages_file_returns_none(tmp_path):
    assert fe.extract(_doc(), _company(), tmp_path / "nope.json", MagicMock()) is None


def test_blank_pages_return_none(tmp_path):
    assert fe.extract(_doc(), _company(), _write_pages(tmp_path, ["", "  "]), MagicMock()) is None


def test_annual_extraction_populates_signals_and_growth(tmp_path):
    r = fe.extract(_doc(), _company(), _write_pages(tmp_path, [ANNUAL]), MagicMock())
    assert r.revenue == D("12000.00")
    assert r.pat == D("1500.00")
    # growth comes from the previous-year column: (12000-10000)/10000
    assert r.revenue_growth_pct == D("20.00")
    assert r.signals_populated >= 3


def test_provenance_is_recorded(tmp_path):
    r = fe.extract(_doc(), _company(), _write_pages(tmp_path, [ANNUAL]), MagicMock())
    src = r.additional_signals["sources"]
    assert src["revenue"]["page"] == 1
    assert src["revenue"]["scope"] == "consolidated"
    assert src["revenue"]["unit"] == "crore"


def test_quarterly_growth_uses_prior_year_from_db(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "_query_prior_revenue", lambda *a, **k: D("1884.37"))
    r = fe.extract(_doc("quarterly_result", "FY2017", "Q3"), _company("NBFC", "equity"),
                   _write_pages(tmp_path, [QUARTERLY]), MagicMock())
    assert r.revenue == D("2365.51")
    assert r.revenue_growth_pct == D("25.53")


def test_quarterly_without_prior_leaves_growth_null(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "_query_prior_revenue", lambda *a, **k: None)
    r = fe.extract(_doc("quarterly_result", "FY2017", "Q3"), _company("NBFC", "equity"),
                   _write_pages(tmp_path, [QUARTERLY]), MagicMock())
    assert r.revenue_growth_pct is None


def test_document_with_no_statements_yields_result_with_all_none(tmp_path):
    r = fe.extract(_doc(), _company(), _write_pages(tmp_path, ["Just a letter to shareholders."]), MagicMock())
    assert r is not None and r.signals_populated == 0


def test_ratios_are_capped_to_column_precision(tmp_path):
    page = "Key ratios\nInterest coverage ratio  900  800\n"
    r = fe.extract(_doc(), _company(), _write_pages(tmp_path, [page]), MagicMock())
    assert abs(r.interest_coverage) <= fe._RATIO_CAP


def test_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "parse_statements", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert fe.extract(_doc(), _company(), _write_pages(tmp_path, [ANNUAL]), MagicMock()) is None


# ---------------------------------------------------------------------------
# helpers kept for the orchestrator
# ---------------------------------------------------------------------------


def test_convert_to_crores():
    assert fe.convert_to_crores(D(100), "lakh") == D(1)
    assert fe.convert_to_crores(D(5), "crore") == D(5)
    assert fe.convert_to_crores(D(5), "usd") is None


def test_load_chunks_roundtrip(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"chunk_text": "a"}\n\n{"chunk_text": "b"}\n', encoding="utf-8")
    assert fe.concatenate_chunk_text(fe.load_chunks(p)) == "a b"
    assert fe.load_chunks(tmp_path / "missing.jsonl") == []


def test_load_chunks_malformed_raises(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fe.load_chunks(p)
