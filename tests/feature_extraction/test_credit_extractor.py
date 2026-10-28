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


def test_icra_spoken_short_term_echo_is_not_a_long_term_rating():
    """Jet Airways: '[ICRA] A3 (pronounced ICRA A three)' was read as long-term A."""
    text = "The short term rating has been re-affirmed at [ICRA] A3 (pronounced ICRA A three)."
    assert ce.find_ratings(text) == []
    assert ce.extract_rating_and_outlook(text) == (None, None)


def test_revised_to_x_from_y_picks_x():
    """Jet Airways: ICRA writes the new rating first, the old one after 'from'."""
    text = ("The long term rating has been revised to [ICRA] C (pronounced ICRA C) from [ICRA] B "
            "(pronounced ICRA B) (negative outlook). The short term rating has been reaffirmed "
            "at [ICRA] A4 (pronounced ICRA A four).")
    assert [r["rating"] for r in ce.find_ratings(text)] == ["C", "C", "B", "B"]
    assert ce.extract_rating_and_outlook(text)[0] == "C"


def test_spoken_echo_stands_in_for_ocr_damaged_new_rating():
    """Jet Airways: '[lCRA] B' is unreadable, its spoken echo 'ICRA B' is not."""
    text = ("revised to [lCRA] B (pronounced ICRA B) (negative outlook) from [ICRA] BB "
            "(pronounced ICRA B) and reaffirmed at [ICRA] A4 (pronounced ICRA A four).")
    assert ce.extract_rating_and_outlook(text)[0] == "B"


def test_only_the_superseded_rating_legible_gives_no_rating():
    text = "revised to [lCRA] BB+ (pronounced ICRA double B plus) from [ICRA] BBB- (pronounced ICRA triple B minus)"
    assert ce.extract_rating_and_outlook(text) == (None, None)


def test_rating_stated_after_to_beats_a_trailing_instrument_table():
    """Tata Steel: BWR's letter ends with an instrument table that lists 'May 2012 BWR AA+ (Stable)'."""
    text = ("Brickwork has on 31st October, 2016 revised Tata Steel Limited's credit rating to BWR AA "
            "from BWR AA+ for NCD and BWR AA- from BWR AA for the perpetual debt. "
            "Instrument Issue Date Earlier Rating Rating action Rs. 3000 BWR AA Unsecured NCD "
            "Downgraded May 2012 BWR AA+ (Stable)")
    assert ce.extract_rating_and_outlook(text)[0] == "AA"


def test_rating_history_annex_is_ignored():
    """Reliance Capital: CARE's press release ends with a table of past ratings (CARE A, AA+)."""
    text = ("Long Term debt ‘CARE D’ Revised from ‘CARE BB; Stable’. "
            "Rating History for last three years: Please refer Annexure. "
            "1)CARE BB; 1) CARE A+ 1)CARE AA+ (18-May-19) 4)CARE A (Under Credit watch)")
    assert ce.extract_rating_and_outlook(text)[0] == "D"


@pytest.mark.parametrize("text,expected", [
    ("CARE Ratings (CARE) has revised its rating to A+ (credit watch with developing implications)", "A+"),
    ("Brickwork Ratings has revised rating to AA for long-term debt programme", "AA"),
    ("CARE Ratings (CARE) has reaffirmed the rating at BBB and revised the outlook", "BBB"),
])
def test_announcement_sentence_rating(text, expected):
    """Reliance Capital: company-written disclosures name the agency in the sentence."""
    assert ce.extract_rating_and_outlook(text)[0] == expected


@pytest.mark.parametrize("text", [
    "ICRA Limited (ICRA) has revised its rating by one notch to A 1 (A one) for commercial paper",
    "ICRA Limited (ICRA) has revised its rating to A4 (On rating watch with negative implications)",
])
def test_announcement_short_term_rating_is_not_long_term(text):
    assert ce.extract_rating_and_outlook(text) == (None, None)


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


# ---------------------------------------------------------------------------
# Rating letters without a long-term rating: reference-only rows (DHFL / IDEA letters)
# ---------------------------------------------------------------------------

_SEBI = "Securities and Exchange Board of India (Listing Obligations and Disclosure Requirements) Regulations. "


def _ref(tmp_path, text):
    return ce.extract(SimpleNamespace(id=1), None, _jsonl(tmp_path, _SEBI + text), MagicMock())


def test_commercial_paper_only_is_reference_row(tmp_path):
    r = _ref(tmp_path, "CARE Ratings Limited (CARE) has assigned Commercial Paper Rs. 10,000 crore CARE A1+ Assigned")
    assert r.credit_rating is None and r.rating_outlook is None
    assert r.extras["short_term_ratings"] == ["A1+"] and r.extras["agency"] == "CARE"
    assert r.extras["reference_only"] is True


def test_ocr_short_term_symbol_and_negative_watch(tmp_path):
    r = _ref(tmp_path, "ICRA Limited (lCRA) Commercial Paper Programme Rs. 8,000 crore ICRAA1+@, "
                       "(Placed on watch with negative implications.)")
    assert r.extras["short_term_ratings"] == ["A1+"] and r.extras["watch"] == "Negative"
    r = _ref(tmp_path, "CRISIL Limited has re-affirmed Commercial Paper Rs. 15,000 crore CRISIL Al+ Re-affirmed")
    assert r.extras["short_term_ratings"] == ["A1+"] and r.extras["agency"] == "CRISIL"


def test_fixed_deposit_scale_is_kept_as_reference(tmp_path):
    r = _ref(tmp_path, "Brickwork Ratings India Pvt. Ltd (Brickwork) Fixed Deposit Rs. 12,000 crore BWR FAAA Re-affirmed "
                       "(Pronounced as BWR F Triple A) Outlook Stable")
    assert r.credit_rating is None and r.extras["deposit_ratings"] == ["FAAA"] and r.extras["agency"] == "BWR"


def test_withdrawal_and_downgrade_without_symbol(tmp_path):
    r = _ref(tmp_path, "CARE Ratings Limited (CARE) Commercial Paper - - Rating Withdrawn by CARE at the request of the Company.")
    assert r.extras["rating_withdrawn"] is True and r.extras["agency"] == "CARE"
    r = _ref(tmp_path, "Brickwork Ratings, has downgraded its rating on Non-Convertible Debentures aggregating to Rs. 2,500 crores")
    assert r.extras["stated_direction"] == "downgrade" and r.extras["agency"] == "BWR"
    assert r.credit_rating is None


def test_reference_rows_are_ignored_by_rating_rules():
    from risk_engine.scorer import PeriodFeatureSet, evaluate_rules_v2
    ref = SimpleNamespace(id=9, credit_rating=None, rating_outlook=None, rating_change_flag=None,
                          additional_signals={"reference_only": True, "watch": "Negative", "stated_direction": "downgrade"})
    fs = PeriodFeatureSet(company_id=1, fiscal_year="FY2019", fiscal_quarter="Q4", financial_features=[],
                          governance_features=[], credit_features=[ref], shareholding_patterns=[])
    r = evaluate_rules_v2(fs)
    assert r.triggered_rules == []
    assert not {"RATING_DOWNGRADE", "RATING_DEFAULT_GRADE"} & set(r.evaluated_rule_ids)
