"""Tests for src/feature_extraction/credit_extractor.py."""

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from feature_extraction import credit_extractor as ce


# ---------------------------------------------------------------------------
# Rating detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("assigned ratings of CARE AAA (Triple AAA) and BWR AAA (Pronounced as BWR Triple A)", ("AAA", None)),
    ("[ICRA]AA+ (Stable) assigned for the NCD programme", ("AA+", "Stable")),
    ("CRISIL D (Issuer not cooperating) rating of the company", ("D", None)),
    ("CARE AA- has been reaffirmed. The outlook is Negative on account of liquidity", ("AA-", "Negative")),
])
def test_agency_qualified_ratings_are_found(text, expected):
    assert ce.extract_rating_and_outlook(text) == expected


def test_ordinary_text_is_never_a_rating():
    """The old pattern matched list markers and words ('a', '(b)', 'C', 'D')."""
    text = "a (b) c D. The Company is Not rated. A copy of the same. Class C shares. Section A"
    assert ce.find_ratings(text) == []
    assert ce.extract_rating_and_outlook(text) == (None, None)


def test_revision_language_picks_the_new_rating():
    text = ("India Ratings (Ind-Ra) has downgraded its rating. "
            "Existing Rating / Outlook IND BBB-/ RWN  Revised Rating IND B/RWN")
    rating, outlook = ce.extract_rating_and_outlook(text)
    assert rating == "B"
    assert outlook == "Watch"


def test_without_revision_the_most_conservative_rating_wins():
    assert ce.extract_rating_and_outlook("CARE AAA and ICRA AA+ and CRISIL AA-")[0] == "AA-"


def test_short_term_ratings_never_become_the_long_term_rating():
    assert ce.extract_rating_and_outlook("[ICRA]A1+ for commercial paper") == (None, None)
    assert [m.group("st") for m in ce.SHORT_RE.finditer("ICRA A1+ and CARE A4")] == ["A1+", "A4"]


@pytest.mark.parametrize("raw,expected", [("RWN", "Watch"), ("CreditWatch", "Watch"), ("stable", "Stable"), (None, None)])
def test_outlook_normalisation(raw, expected):
    assert ce._norm_outlook(raw) == expected


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


def _jsonl(tmp_path, text):
    p = tmp_path / "1_chunks.jsonl"
    p.write_text(json.dumps({"chunk_text": text}) + "\n", encoding="utf-8")
    return p


def test_extract_returns_result_with_agency_and_extras(tmp_path):
    p = _jsonl(tmp_path, "CRISIL has downgraded the NCDs to CRISIL BBB (Negative); ICRA A3 for CP")
    r = ce.extract(SimpleNamespace(id=1), SimpleNamespace(id=1), p, MagicMock())
    assert (r.credit_rating, r.rating_outlook, r.agency) == ("BBB", "Negative", "CRISIL")
    assert r.rating_change_flag is None  # decided later, in date order
    assert r.extras["revision_language"] is True
    assert r.extras["stated_direction"] == "downgrade"
    assert r.extras["short_term_ratings"] == ["A3"]


def test_extract_none_when_no_rating_or_no_file(tmp_path):
    assert ce.extract(SimpleNamespace(id=1), None, _jsonl(tmp_path, "nothing to see"), MagicMock()) is None
    assert ce.extract(SimpleNamespace(id=1), None, tmp_path / "missing.jsonl", MagicMock()) is None


# ---------------------------------------------------------------------------
# recompute_rating_changes (date-ordered, per agency)
# ---------------------------------------------------------------------------


def _row(doc_id, d, rating, agency="CARE"):
    return SimpleNamespace(document_id=doc_id, period_end=d, credit_rating=rating,
                           additional_signals={"agency": agency}, rating_change_flag=None)


def _session(rows):
    s = MagicMock()
    s.execute.return_value.scalars.return_value.all.return_value = rows
    return s


def test_first_rating_has_no_flag_and_downgrade_is_detected():
    rows = [_row(1, date(2016, 1, 1), "AAA"), _row(2, date(2017, 1, 1), "AA+"), _row(3, date(2018, 1, 1), "AA+")]
    ce.recompute_rating_changes(_session(rows), 1)
    assert [r.rating_change_flag for r in rows] == [None, True, False]
    assert rows[1].additional_signals["rating_direction"] == "downgrade"
    assert rows[2].additional_signals["rating_direction"] is None


def test_upgrade_is_not_a_downgrade():
    rows = [_row(1, date(2016, 1, 1), "BBB"), _row(2, date(2017, 1, 1), "A")]
    ce.recompute_rating_changes(_session(rows), 1)
    assert rows[1].additional_signals["rating_direction"] == "upgrade"


def test_comparison_is_per_agency():
    """CARE AAA then ICRA AA+ is two agencies, not a downgrade."""
    rows = [_row(1, date(2016, 1, 1), "AAA", "CARE"), _row(2, date(2016, 6, 1), "AA+", "ICRA"),
            _row(3, date(2017, 1, 1), "AAA", "CARE")]
    ce.recompute_rating_changes(_session(rows), 1)
    assert rows[2].rating_change_flag is False  # CARE AAA -> CARE AAA


# ---------------------------------------------------------------------------
# Annual-report disclosure fallback
# ---------------------------------------------------------------------------

from feature_extraction.credit_extractor import extract_from_annual_report  # noqa: E402


def test_annual_report_own_rating_is_read():
    """Jaypee FY2015 Directors' Report."""
    page = ("During the year the credit rating assigned by CARE Ratings Agency viz. "
            "Long Term Bank Facilities CARE BBB-. CORPORATE GOVERNANCE The Company is committed")
    r = extract_from_annual_report([page])
    assert r.credit_rating == "BBB-" and r.agency == "CARE"
    assert r.extras["source"] == "annual_report_disclosure"


def test_annual_report_investment_ratings_are_ignored():
    """Infosys-style: ratings of instruments the company invests in are not its own rating."""
    page = ("Investments made in non-convertible debentures with high credit rating given by "
            "domestic credit rating agencies such as CRISIL AAA and ICRA AA+.")
    assert extract_from_annual_report([page]) is None


def test_annual_report_without_rating_phrase_gives_none():
    assert extract_from_annual_report(["The Company's facilities are rated CARE A."]) is None
