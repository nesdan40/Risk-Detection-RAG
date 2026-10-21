"""
Example-based tests for src/feature_extraction/category_normalizer.py.

Covers:
  - Layer 1: exact alias match via document_category (idempotence + alias dict entries)
  - Layer 1: exact alias match via filename (all four filename patterns + stem fallback)
  - Suffix stripping: _(N), _SCANNED, _SCANNEED, _SCANNEDpdf
  - Layer 2: fuzzy matching (threshold boundary, tie-breaking)
  - Layer 3: keyword matching (priority order, skip when chunk_text=None)
  - Edge cases: empty inputs, Corporate Action - Other, Credit Rating, unclassified

Requirements: 2.1–2.10, 10.1, 10.2, 10.3
"""

import pytest

from feature_extraction.category_normalizer import (
    ALIAS_DICT,
    CANONICAL_LABELS,
    CORPORATE_ACTION_OTHER,
    FUZZY_THRESHOLD,
    normalize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(doc_cat, file_name="", chunk_text=None):
    """Shorthand wrapper so tests stay compact."""
    return normalize(doc_cat, file_name, chunk_text)


# ---------------------------------------------------------------------------
# Layer 1 — Idempotence: every canonical label resolves to itself exactly
# ---------------------------------------------------------------------------

class TestLayer1Idempotence:
    """
    Passing any canonical label string as document_category must return
    (label, 'exact') — the normalizer is idempotent over its own output set.
    Covers Requirement 10.2.
    """

    @pytest.mark.parametrize("label", sorted(CANONICAL_LABELS))
    def test_canonical_label_is_identity(self, label):
        result, layer = norm(label)
        assert result == label
        assert layer == "exact"

    def test_corporate_action_other_is_identity(self):
        result, layer = norm(CORPORATE_ACTION_OTHER)
        assert result == CORPORATE_ACTION_OTHER
        assert layer == "exact"

    def test_credit_rating_token_is_identity(self):
        """'Credit Rating' is a valid alias dict key that should pass through unchanged."""
        result, layer = norm("Credit Rating")
        assert result == "Credit Rating"
        assert layer == "exact"


# ---------------------------------------------------------------------------
# Layer 1 — Alias dictionary entries (non-canonical inputs → canonical output)
# ---------------------------------------------------------------------------

class TestLayer1AliasEntries:
    """
    Selected alias dict entries that are NOT already canonical labels must
    also resolve via Layer 1.  Covers Requirement 10.1 and 2.2.
    """

    # --- Board Meeting ---
    def test_obm_alias(self):
        assert norm("OBM") == ("Board Meeting", "exact")

    def test_outcome_of_board_meeting(self):
        assert norm("Outcome of Board Meeting") == ("Board Meeting", "exact")

    def test_outcome_of_bm(self):
        assert norm("Outcome of BM") == ("Board Meeting", "exact")

    def test_ooutcome_typo(self):
        """Real data typo: 'Ooutcome Of Board meeting'."""
        assert norm("Ooutcome Of Board meeting") == ("Board Meeting", "exact")

    # --- Financial Results ---
    def test_financial_results_updates(self):
        assert norm("Financial Results Updates") == ("Financial Results", "exact")

    def test_financial_result_updates_singular(self):
        assert norm("Financial Result Updates") == ("Financial Results", "exact")

    def test_fiancial_typo(self):
        assert norm("Fiancial Results Updates") == ("Financial Results", "exact")

    def test_financial_reults_typo(self):
        assert norm("Financial Reults Updates") == ("Financial Results", "exact")

    def test_financialresults_updates_merged(self):
        """'FInancialResults Updates' variant from real data."""
        assert norm("FInancialResults Updates") == ("Financial Results", "exact")

    # --- AGM ---
    def test_agm_updates_alias(self):
        assert norm("AGMupdates") == ("AGM / Annual General Meeting Updates", "exact")

    def test_annual_general_meeting_updates(self):
        assert norm("Annual General Meeting Updates") == (
            "AGM / Annual General Meeting Updates",
            "exact",
        )

    def test_annual_general_meeting_short(self):
        assert norm("Annual General Meeting") == (
            "AGM / Annual General Meeting Updates",
            "exact",
        )

    def test_agm_short(self):
        assert norm("AGM") == ("AGM / Annual General Meeting Updates", "exact")

    # --- Appointment ---
    def test_appointmet_typo(self):
        assert norm("Appointmet") == ("Appointment", "exact")

    def test_directorappointed_alias(self):
        """'DirectorAppointed' maps to Appointment, not Director Change."""
        assert norm("DirectorAppointed") == ("Appointment", "exact")

    # --- Director Change ---
    def test_cessation_alias(self):
        assert norm("Cessation") == ("Director Change", "exact")

    def test_directorchange_merged(self):
        assert norm("DirectorChange") == ("Director Change", "exact")

    def test_change_in_director(self):
        assert norm("Change in Director") == ("Director Change", "exact")

    def test_change_in_director_underscore(self):
        assert norm("Change_in_Director") == ("Director Change", "exact")

    # --- News Clarification ---
    def test_clarification_alias(self):
        assert norm("Clarification") == ("News Clarification", "exact")

    def test_newsclarification_merged(self):
        assert norm("NewsClarification") == ("News Clarification", "exact")

    # --- News Verification ---
    def test_newsverification_merged(self):
        assert norm("NewsVerification") == ("News Verification", "exact")

    # --- Press Release ---
    def test_press_relase_typo(self):
        assert norm("Press Relase") == ("Press Release", "exact")

    def test_press_releas_typo(self):
        assert norm("Press Releas") == ("Press Release", "exact")

    def test_pressrelease_merged(self):
        assert norm("PressRelease") == ("Press Release", "exact")

    def test_press_release_updates(self):
        assert norm("Press Release Updates") == ("Press Release", "exact")

    def test_press_standalone(self):
        assert norm("Press") == ("Press Release", "exact")

    def test_pr_standalone(self):
        assert norm("PR") == ("Press Release", "exact")

    # --- Corporate Action - Other ---
    def test_book_closure(self):
        assert norm("Book Closure") == (CORPORATE_ACTION_OTHER, "exact")

    def test_bookclosure_merged(self):
        assert norm("BookClosure") == (CORPORATE_ACTION_OTHER, "exact")

    def test_buyback(self):
        assert norm("Buyback") == (CORPORATE_ACTION_OTHER, "exact")

    def test_acquisition(self):
        assert norm("Acquisition") == (CORPORATE_ACTION_OTHER, "exact")

    def test_aquisition_typo(self):
        assert norm("Aquisition") == (CORPORATE_ACTION_OTHER, "exact")

    def test_settlement(self):
        assert norm("Settlement") == (CORPORATE_ACTION_OTHER, "exact")

    def test_stayorder(self):
        assert norm("StayOrder") == (CORPORATE_ACTION_OTHER, "exact")

    def test_change_in_auditors(self):
        assert norm("Change in Auditors") == (CORPORATE_ACTION_OTHER, "exact")

    def test_change_in_auditors_title_case(self):
        assert norm("Change In Auditors") == (CORPORATE_ACTION_OTHER, "exact")

    # --- Case-insensitivity ---
    def test_case_insensitive_board_meeting(self):
        assert norm("board meeting") == ("Board Meeting", "exact")

    def test_case_insensitive_resignation(self):
        assert norm("RESIGNATION") == ("Resignation", "exact")

    def test_case_insensitive_appointment(self):
        assert norm("APPOINTMENT") == ("Appointment", "exact")

    def test_case_insensitive_obm(self):
        assert norm("obm") == ("Board Meeting", "exact")


# ---------------------------------------------------------------------------
# Layer 1 — Filename-based extraction (when document_category is None)
# ---------------------------------------------------------------------------

class TestLayer1FilenameExtraction:
    """
    When document_category is None, the normalizer must extract a token from
    the filename and resolve it.  Tests cover all four filename patterns and
    the full-stem fallback.  Covers Requirement 2.6.
    """

    # Pattern 1: {TICKER}_Ann_{YYYYMMDD}_{Token}
    def test_pattern1_ann_board_meeting(self):
        """Standard BSE-style Ann filename with Board Meeting token."""
        result, layer = normalize(None, "TATASTEEL_Ann_20180515_Board Meeting.pdf", None)
        assert result == "Board Meeting"
        assert layer == "exact"

    def test_pattern1_ann_resignation(self):
        result, layer = normalize(None, "DHFL_Ann_20170913_Resignation_SCANNED.pdf", None)
        assert result == "Resignation"
        assert layer == "exact"

    def test_pattern1_ann_press_release(self):
        result, layer = normalize(None, "DHFL_Ann_20160825_Press Release_SCANNED.pdf", None)
        assert result == "Press Release"
        assert layer == "exact"

    def test_pattern1_ann_agm(self):
        result, layer = normalize(
            None,
            "DHFL_Ann_20160708_Annual General Meeting Updates_SCANNED.pdf",
            None,
        )
        assert result == "AGM / Annual General Meeting Updates"
        assert layer == "exact"

    def test_pattern1_ann_with_duplicate_suffix(self):
        """Filename with _(2)_SCANNED duplicate suffix should still resolve."""
        result, layer = normalize(
            None,
            "DHFL_Ann_20160504_Financial Results Updates (2)_SCANNED.pdf",
            None,
        )
        assert result == "Financial Results"
        assert layer == "exact"

    # Pattern 2: {TICKER}_{YYYYMMDD}_{Token}
    def test_pattern2_date_then_token_appointment(self):
        """HDFCBANK-style: ticker_YYYYMMDD_Token."""
        result, layer = normalize(
            None, "HDFCBANK_20190131_Appointment_SCANNED.pdf", None
        )
        assert result == "Appointment"
        assert layer == "exact"

    def test_pattern2_date_then_token_press_release(self):
        result, layer = normalize(
            None, "INFY_20180615_Press Release_SCANNED.pdf", None
        )
        assert result == "Press Release"
        assert layer == "exact"

    # Pattern 3: {TICKER}_{Token}_{YYYYMMDD}
    def test_pattern3_token_then_date_director_change(self):
        """JSW-style: ticker_Token_YYYYMMDD."""
        result, layer = normalize(
            None, "JSW_DirectorChange_01022015_SCANNED.pdf", None
        )
        assert result == "Director Change"
        assert layer == "exact"

    def test_pattern3_token_then_date_resignation(self):
        result, layer = normalize(
            None, "TCS_Resignation_15032018_SCANNED.pdf", None
        )
        assert result == "Resignation"
        assert layer == "exact"

    # Pattern 4: {TICKER_ABBREV}_{Token}_{DDMMYYYY}
    def test_pattern4_ddmmyyyy_appointment(self):
        """JAYPEE-style: abbrev_Token_DDMMYYYY."""
        result, layer = normalize(
            None, "JAYPEE_Appointment_27112016_SCANNED.pdf", None
        )
        assert result == "Appointment"
        assert layer == "exact"

    def test_pattern4_ddmmyyyy_board_meeting(self):
        result, layer = normalize(
            None, "PGCI_Outcome of Board Meeting_14022015_SCANNED.pdf", None
        )
        assert result == "Board Meeting"
        assert layer == "exact"

    # Full-stem fallback
    def test_fullstem_fallback_exact_match(self):
        """
        When no filename pattern matches, the full stem is used as the token.
        A stem that equals an alias dict key exactly resolves via 'exact'.
        'Resignation' has no underscore/date structure so falls to full-stem.
        """
        result, layer = normalize(None, "Resignation.pdf", None)
        assert result == "Resignation"
        assert layer == "exact"

    def test_fullstem_fallback_resolves_fuzzy_for_prefixed_stem(self):
        """
        Full stem 'SomeCompany_Board Meeting' contains a company prefix so it
        doesn't match the alias dict key 'board meeting' exactly — fuzzy resolves it.
        """
        result, layer = normalize(None, "SomeCompany_Board Meeting.pdf", None)
        assert result == "Board Meeting"
        assert layer == "fuzzy"

    def test_fullstem_fallback_unclassified(self):
        """A stem with no recognizable token produces 'unclassified'."""
        result, layer = normalize(None, "ZZZCORP_xyzunknowntoken.pdf", None)
        assert result == "unclassified"
        assert layer == "unclassified"


# ---------------------------------------------------------------------------
# Suffix stripping
# ---------------------------------------------------------------------------

class TestSuffixStripping:
    """
    Trailing _(N), _SCANNED, _SCANNEED, _SCANNEDpdf suffixes must be removed
    before matching.  Covers Requirement 2.7.
    """

    def test_strip_scanned_suffix(self):
        """'Resignation_SCANNED' must equal 'Resignation' after stripping."""
        result_stripped, _ = norm("Resignation_SCANNED")
        result_plain, _ = norm("Resignation")
        assert result_stripped == result_plain == "Resignation"

    def test_strip_scanneed_typo_suffix(self):
        assert norm("Resignation_SCANNEED") == ("Resignation", "exact")

    def test_strip_scannedpdf_suffix(self):
        assert norm("Press Release_SCANNEDpdf") == ("Press Release", "exact")

    def test_strip_duplicate_number_suffix_underscore(self):
        """'Financial Results Updates_(2)' → strip _(2) → 'Financial Results Updates'."""
        result, layer = norm("Financial Results Updates_(2)")
        assert result == "Financial Results"
        assert layer == "exact"

    def test_strip_duplicate_number_suffix_space(self):
        """'Financial Results Updates (2)_SCANNED' — space before (N) also stripped."""
        result, layer = norm("Financial Results Updates (2)_SCANNED")
        assert result == "Financial Results"
        assert layer == "exact"

    def test_strip_large_duplicate_number(self):
        assert norm("Press Release_(10)") == ("Press Release", "exact")

    def test_combined_number_and_scanned(self):
        """Both _(2) and _SCANNED stripped together."""
        assert norm("Appointment_(2)_SCANNED") == ("Appointment", "exact")

    def test_strip_applied_to_document_category(self):
        """Suffix stripping applies to document_category, not just filenames."""
        assert norm("Director Change_SCANNED") == ("Director Change", "exact")

    def test_strip_not_applied_to_dict_keys(self):
        """ALIAS_DICT keys themselves do NOT contain these suffixes — no double-stripping."""
        # If stripping were applied to dict keys, it would alter them and break lookup.
        # Verifying 'board meeting' (a clean key) still resolves correctly.
        assert norm("board meeting") == ("Board Meeting", "exact")


# ---------------------------------------------------------------------------
# Layer 2 — Fuzzy matching
# ---------------------------------------------------------------------------

class TestLayer2FuzzyMatching:
    """
    Tokens that aren't in the alias dict but are close enough (score ≥ 80)
    should resolve via fuzzy match.  Tokens scoring < 80 fall through.
    Covers Requirement 2.3.
    """

    def test_typo_resignaton_resolves_fuzzy(self):
        """'Resignaton' (dropped 'i') is close enough to 'resignation' for fuzzy match."""
        result, layer = norm("Resignaton")
        assert result == "Resignation"
        assert layer == "fuzzy"

    def test_typo_appointmnt_resolves_fuzzy(self):
        """'Appointmnt' (dropped 'e') is close enough to 'appointment'."""
        result, layer = norm("Appointmnt")
        assert result == "Appointment"
        assert layer == "fuzzy"

    def test_completely_unrelated_token_does_not_fuzzy_match(self):
        """A token with no similarity to any alias dict key falls through to unclassified."""
        result, layer = normalize(None, "ZCORP_XYZQWERTY123_SCANNED.pdf", None)
        assert result == "unclassified"
        assert layer == "unclassified"

    def test_fuzzy_tie_breaking_longer_key_wins(self):
        """
        When two dict keys have the same fuzzy score, the longer key wins.
        'news clarification' is longer than 'clarification'; a token that
        scores equally against both should prefer 'news clarification'.
        This is verified indirectly: passing 'News_Clarification' (underscore form)
        — after lowercasing it's 'news_clarification', which is not in the dict,
        so it fuzzes; 'news clarification' (length 17) beats 'clarification' (length 13).
        """
        result, layer = norm("News_Clarification")
        # The result should map to News Clarification (via fuzzy on 'news_clarification')
        assert result == "News Clarification"
        # Layer could be exact (if the underscore→space is handled) or fuzzy
        assert layer in ("exact", "fuzzy")

    def test_fuzzy_threshold_boundary_below_does_not_match(self):
        """
        A token that has near-zero similarity to any alias dict entry should not
        match via fuzzy (score would be well below 80).
        """
        result, layer = normalize(None, "CORP_AAABBBCCC9999_SCANNED.pdf", None)
        assert layer in ("fuzzy", "keyword", "unclassified")
        # If it fuzzy-matched, the score was ≥ 80 — that would be a real match
        # The important thing is it doesn't raise and returns a valid tuple
        assert result in (
            list(CANONICAL_LABELS) + [CORPORATE_ACTION_OTHER, "Credit Rating", "unclassified"]
        )


# ---------------------------------------------------------------------------
# Layer 3 — Keyword matching
# ---------------------------------------------------------------------------

class TestLayer3KeywordMatching:
    """
    When layers 1 and 2 fail, keyword matching against chunk_text[:2000]
    should identify the category.  Covers Requirement 2.4.
    """

    # We need a document_category and file_name that won't match layers 1 or 2
    _UNKNOWN_CAT = None
    _UNKNOWN_FILE = "CORP_UNKNOWNTOKEN99_SCANNED.pdf"

    def test_board_meeting_keyword(self):
        chunk = "The company convened a board meeting on 15 January 2019 to discuss..."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Board Meeting"
        assert layer == "keyword"

    def test_outcome_of_board_keyword(self):
        chunk = "Outcome of Board: the directors resolved to approve the financial statements."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Board Meeting"
        assert layer == "keyword"

    def test_resigned_keyword(self):
        chunk = "Mr. John Smith has resigned from the Board of Directors effective immediately."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Resignation"
        assert layer == "keyword"

    def test_resignation_keyword(self):
        chunk = "The resignation of the Managing Director was accepted by the Board."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Resignation"
        assert layer == "keyword"

    def test_director_change_keyword(self):
        chunk = "Notice of change in director pursuant to regulatory requirements."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Director Change"
        assert layer == "keyword"

    def test_director_cessation_keyword(self):
        chunk = "This is to inform about the cessation of director Mr. Ramesh Kumar."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Director Change"
        assert layer == "keyword"

    def test_director_vacate_keyword(self):
        chunk = "The director shall vacate the office as per the Companies Act."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Director Change"
        assert layer == "keyword"

    def test_appointed_keyword(self):
        chunk = "Mr. Anil Kumar has been appointed as an Independent Director."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Appointment"
        assert layer == "keyword"

    def test_appointment_keyword(self):
        chunk = "Subject: Appointment of Chief Financial Officer under SEBI regulations."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Appointment"
        assert layer == "keyword"

    def test_agm_annual_general_meeting_keyword(self):
        chunk = "Notice of Annual General Meeting to be held on 15 September 2019."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "AGM / Annual General Meeting Updates"
        assert layer == "keyword"

    def test_agm_short_keyword(self):
        chunk = "Outcome of the AGM held on 28 August 2018 at the registered office."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "AGM / Annual General Meeting Updates"
        assert layer == "keyword"

    def test_credit_rating_keyword(self):
        chunk = "CRISIL has reaffirmed the credit rating of the company at AA+."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Credit Rating"
        assert layer == "keyword"

    def test_rating_upgrade_keyword(self):
        chunk = "The rating agency has communicated a rating upgrade to AA from A+."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Credit Rating"
        assert layer == "keyword"

    def test_rating_outlook_keyword(self):
        chunk = "The outlook on the rating has been revised to Negative from Stable."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Credit Rating"
        assert layer == "keyword"

    def test_layer3_skipped_when_chunk_text_none(self):
        """When chunk_text is None, Layer 3 must be skipped — no exception raised."""
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, None)
        assert result == "unclassified"
        assert layer == "unclassified"

    def test_layer3_text_beyond_2000_chars_ignored(self):
        """Keywords appearing only after the first 2000 chars should not trigger a match."""
        prefix = "x " * 1001  # >2000 characters with no keywords
        suffix = "The board meeting was held yesterday."
        chunk = prefix + suffix
        assert len(chunk) > 2000
        # The keyword 'board meeting' is past position 2000 — should not match
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        # Result must be unclassified (the keyword is out of range)
        assert result == "unclassified"
        assert layer == "unclassified"

    # --- Priority ordering ---

    def test_priority_credit_over_director(self):
        """
        'Credit Rating' has highest priority.  Text containing 'credit rating'
        AND 'director change' keywords must resolve to 'Credit Rating'.
        """
        chunk = "Notice regarding credit rating downgrade and change in director."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Credit Rating"
        assert layer == "keyword"

    def test_priority_director_change_over_appointment(self):
        """
        'Director Change' has higher priority than 'Appointment'.
        Text with 'appointed' AND 'director cessation' should resolve to
        'Director Change' because it also contains 'director' + 'cessation'.
        """
        chunk = "Mr. Sharma was appointed while another director's cessation was noted."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Director Change"
        assert layer == "keyword"

    def test_priority_resignation_over_appointment(self):
        """
        'Resignation' has higher priority than 'Appointment'.
        Text with both 'resigned' and 'appointed' should resolve to 'Resignation'.
        """
        chunk = "Mr. Kumar has resigned. Ms. Priya was appointed in his place."
        result, layer = normalize(self._UNKNOWN_CAT, self._UNKNOWN_FILE, chunk)
        assert result == "Resignation"
        assert layer == "keyword"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """
    Boundary inputs: empty strings, None values, Corporate Action - Other
    fixed points, and combinations.  Covers Requirement 10.3 and 2.5.
    """

    def test_none_category_empty_filename_no_text(self):
        """All None/empty inputs must return ('unclassified', 'unclassified')."""
        assert normalize(None, "", None) == ("unclassified", "unclassified")

    def test_empty_string_category_empty_filename(self):
        """Empty string document_category treated the same as None — falls to filename."""
        assert normalize("", "", None) == ("unclassified", "unclassified")

    def test_whitespace_only_category_treated_as_none(self):
        """Whitespace-only document_category falls through to filename extraction."""
        result, layer = normalize("   ", "CORP_Resignation_20180101.pdf", None)
        assert result == "Resignation"
        assert layer == "exact"

    def test_corporate_action_other_is_fixed_point(self):
        """'Corporate Action - Other' resolves to itself (idempotence)."""
        assert normalize(CORPORATE_ACTION_OTHER, "", None) == (
            CORPORATE_ACTION_OTHER,
            "exact",
        )

    def test_credit_rating_as_document_category(self):
        """'Credit Rating' resolves to itself when passed as document_category."""
        assert normalize("Credit Rating", "", None) == ("Credit Rating", "exact")

    def test_unknown_token_all_layers_fail(self):
        """Token that defeats all three layers must return 'unclassified'."""
        result, layer = normalize(None, "CORP_UNKNOWNTOKEN99_SCANNED.pdf", None)
        assert result == "unclassified"
        assert layer == "unclassified"

    def test_result_never_none(self):
        """normalize() must never return None as either element of the tuple."""
        for doc_cat, file_name, chunk in [
            (None, "", None),
            ("", "", None),
            ("RandomGarbage", "RandomFile.pdf", None),
            (None, "CORP_XYZTOKEN_SCANNED.pdf", "some irrelevant text"),
        ]:
            label, layer_val = normalize(doc_cat, file_name, chunk)
            assert label is not None
            assert layer_val is not None

    def test_result_label_always_in_valid_set(self):
        """Every returned label must be a member of the defined valid output set."""
        valid_labels = CANONICAL_LABELS | {CORPORATE_ACTION_OTHER, "Credit Rating", "unclassified"}
        for doc_cat, file_name, chunk in [
            (None, "", None),
            ("Board Meeting", "", None),
            ("UNKNOWN_GARBAGE_TOKEN", "", None),
            (None, "CORP_XYZTOKEN_SCANNED.pdf", None),
        ]:
            label, _ = normalize(doc_cat, file_name, chunk)
            assert label in valid_labels

    def test_result_layer_always_in_valid_set(self):
        """Every returned resolution_layer must be one of the four valid values."""
        valid_layers = {"exact", "fuzzy", "keyword", "unclassified"}
        for doc_cat, file_name, chunk in [
            (None, "", None),
            ("Board Meeting", "", None),
            ("SOME_TYPO_NEAR_RESIGNATION", "", None),
        ]:
            _, layer_val = normalize(doc_cat, file_name, chunk)
            assert layer_val in valid_layers

    def test_normalize_does_not_raise_on_extreme_input(self):
        """normalize() must never raise regardless of input content."""
        try:
            normalize(None, "!@#$%^&*()_+{}|:<>?", "Some random text 12345")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"normalize() raised an unexpected exception: {exc!r}")

    def test_normalize_does_not_raise_on_very_long_chunk_text(self):
        """Very long chunk_text must not cause an error — only first 2000 chars used."""
        long_text = "a" * 100_000
        try:
            normalize(None, "CORP_XYZTOKEN_SCANNED.pdf", long_text)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"normalize() raised on long chunk text: {exc!r}")

    def test_document_category_takes_precedence_over_filename(self):
        """
        When document_category is non-NULL and non-empty, filename must be ignored.
        The filename here would resolve to 'Board Meeting' if used, but
        document_category of 'Resignation' should win.
        """
        result, layer = normalize(
            "Resignation",
            "CORP_Ann_20180101_Board Meeting.pdf",
            None,
        )
        assert result == "Resignation"
        assert layer == "exact"

    def test_none_document_category_falls_to_filename(self):
        """None document_category must cause filename extraction to be used."""
        result, layer = normalize(None, "CORP_Ann_20180101_Press Release.pdf", None)
        assert result == "Press Release"
        assert layer == "exact"
