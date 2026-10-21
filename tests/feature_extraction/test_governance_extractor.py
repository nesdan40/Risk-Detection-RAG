"""
Example-based tests for src/feature_extraction/governance_extractor.py.

Covers:
  - group_announcements_by_period(): fiscal period bucketing, NULL period_start
    exclusion, empty input, and multiple documents across periods.
  - aggregate(): count correctness for all nine canonical labels, clarification_count
    superset semantics, 'unclassified' exclusion, 'Corporate Action - Other' handling,
    'Financial Results' and 'AGM' no-count behaviour, zero defaults.
  - evaluate_agm_delay_flag(): empty list, boundary dates (Sep 30 / Oct 1),
    multiple dates (earliest used), all-None period_start list.

Requirements: 4.1–4.10
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from feature_extraction.governance_extractor import (
    CategorizedAnnouncement,
    GovernanceResult,
    aggregate,
    evaluate_agm_delay_flag,
    group_announcements_by_period,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(doc_id: int, period_start, company_id: int = 1, file_name: str = "test.pdf"):
    """Create a mock Document ORM object with the minimal attributes used by the extractor."""
    doc = MagicMock()
    doc.id = doc_id
    doc.company_id = company_id
    doc.file_name = file_name
    doc.period_start = period_start
    return doc


def make_ann(
    document_id: int = 1,
    canonical_label: str = "Board Meeting",
    period_start: date | None = date(2018, 5, 15),
    fiscal_year: str = "FY2019",
    fiscal_quarter: str = "Q1",
    file_name: str = "test.pdf",
    company_id: int = 1,
    resolution_layer: str = "exact",
) -> CategorizedAnnouncement:
    """Convenience factory for CategorizedAnnouncement objects."""
    return CategorizedAnnouncement(
        document_id=document_id,
        file_name=file_name,
        company_id=company_id,
        period_start=period_start,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        canonical_label=canonical_label,
        resolution_layer=resolution_layer,
    )


# ---------------------------------------------------------------------------
# group_announcements_by_period
# ---------------------------------------------------------------------------

class TestGroupAnnouncementsByPeriod:
    """Tests for fiscal period bucketing of corporate announcement documents."""

    def test_may_date_maps_to_fy2019_q1(self):
        """date(2018, 5, 15) falls in Q1 (Apr–Jun) of FY2019."""
        doc = make_doc(1, date(2018, 5, 15))
        normalized = {1: ("Board Meeting", "exact")}
        groups = group_announcements_by_period([doc], normalized)
        assert ("FY2019", "Q1") in groups
        assert len(groups[("FY2019", "Q1")]) == 1
        ann = groups[("FY2019", "Q1")][0]
        assert ann.document_id == 1
        assert ann.fiscal_year == "FY2019"
        assert ann.fiscal_quarter == "Q1"

    def test_february_date_maps_to_fy2019_q4(self):
        """date(2019, 2, 1) falls in Q4 (Jan–Mar) of FY2019."""
        doc = make_doc(2, date(2019, 2, 1))
        normalized = {2: ("Resignation", "exact")}
        groups = group_announcements_by_period([doc], normalized)
        assert ("FY2019", "Q4") in groups
        assert len(groups[("FY2019", "Q4")]) == 1
        ann = groups[("FY2019", "Q4")][0]
        assert ann.fiscal_year == "FY2019"
        assert ann.fiscal_quarter == "Q4"

    def test_null_period_start_excluded(self):
        """Document with period_start=None must be silently excluded — no KeyError."""
        doc_null = make_doc(10, None)
        doc_valid = make_doc(11, date(2018, 8, 1))
        normalized = {10: ("Board Meeting", "exact"), 11: ("Resignation", "exact")}
        groups = group_announcements_by_period([doc_null, doc_valid], normalized)
        # No entry should exist for None period_start
        for key, anns in groups.items():
            for ann in anns:
                assert ann.document_id != 10, "NULL period_start doc must be excluded"
        # The valid doc must still be present
        assert ("FY2019", "Q2") in groups
        assert groups[("FY2019", "Q2")][0].document_id == 11

    def test_multiple_documents_grouped_correctly(self):
        """Documents in different quarters end up in separate groups."""
        docs = [
            make_doc(1, date(2018, 5, 15)),   # Q1 FY2019
            make_doc(2, date(2018, 9, 30)),   # Q2 FY2019
            make_doc(3, date(2018, 10, 1)),   # Q3 FY2019
            make_doc(4, date(2019, 1, 31)),   # Q4 FY2019
        ]
        normalized = {
            1: ("Board Meeting", "exact"),
            2: ("Appointment", "exact"),
            3: ("Resignation", "exact"),
            4: ("Director Change", "exact"),
        }
        groups = group_announcements_by_period(docs, normalized)
        assert len(groups) == 4
        assert len(groups[("FY2019", "Q1")]) == 1
        assert len(groups[("FY2019", "Q2")]) == 1
        assert len(groups[("FY2019", "Q3")]) == 1
        assert len(groups[("FY2019", "Q4")]) == 1

    def test_multiple_documents_same_period(self):
        """Two documents in the same period are both in the same group list."""
        docs = [
            make_doc(1, date(2018, 4, 10)),
            make_doc(2, date(2018, 6, 25)),
        ]
        normalized = {
            1: ("Board Meeting", "exact"),
            2: ("Press Release", "exact"),
        }
        groups = group_announcements_by_period(docs, normalized)
        assert ("FY2019", "Q1") in groups
        assert len(groups[("FY2019", "Q1")]) == 2

    def test_empty_document_list_returns_empty_dict(self):
        """Empty input produces an empty groups dict — no error."""
        groups = group_announcements_by_period([], {})
        assert groups == {}

    def test_all_null_period_start_returns_empty_dict(self):
        """When all documents have NULL period_start, the result is empty."""
        docs = [make_doc(1, None), make_doc(2, None)]
        normalized = {1: ("Board Meeting", "exact"), 2: ("Resignation", "exact")}
        groups = group_announcements_by_period(docs, normalized)
        assert groups == {}

    def test_document_not_in_normalized_categories_defaults_unclassified(self):
        """If a doc's id is absent from normalized_categories, it defaults to 'unclassified'."""
        doc = make_doc(99, date(2018, 7, 1))
        groups = group_announcements_by_period([doc], {})  # empty normalized_categories
        assert ("FY2019", "Q2") in groups
        ann = groups[("FY2019", "Q2")][0]
        assert ann.canonical_label == "unclassified"
        assert ann.resolution_layer == "unclassified"

    def test_category_correctly_assigned_to_announcement(self):
        """The resolved canonical_label from normalized_categories is stored on the announcement."""
        doc = make_doc(5, date(2018, 11, 15))
        normalized = {5: ("Director Change", "fuzzy")}
        groups = group_announcements_by_period([doc], normalized)
        ann = groups[("FY2019", "Q3")][0]
        assert ann.canonical_label == "Director Change"
        assert ann.resolution_layer == "fuzzy"

    def test_boundary_april_1_maps_to_q1_fy(self):
        """April 1 is the first day of Q1 and starts a new fiscal year."""
        doc = make_doc(1, date(2019, 4, 1))
        normalized = {1: ("Board Meeting", "exact")}
        groups = group_announcements_by_period([doc], normalized)
        assert ("FY2020", "Q1") in groups

    def test_boundary_march_31_maps_to_q4(self):
        """March 31 is the last day of Q4."""
        doc = make_doc(1, date(2019, 3, 31))
        normalized = {1: ("Board Meeting", "exact")}
        groups = group_announcements_by_period([doc], normalized)
        assert ("FY2019", "Q4") in groups


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    """Tests for governance signal count aggregation."""

    def _all_labels_set(self) -> list[CategorizedAnnouncement]:
        """Create one announcement per each of the nine canonical labels
        plus one 'unclassified' and one 'Corporate Action - Other'."""
        return [
            make_ann(1,  "Board Meeting"),
            make_ann(2,  "Director Change"),
            make_ann(3,  "Appointment"),
            make_ann(4,  "Resignation"),
            make_ann(5,  "Press Release"),
            make_ann(6,  "News Clarification"),
            make_ann(7,  "News Verification"),
            make_ann(8,  "Financial Results"),
            make_ann(9,  "AGM / Annual General Meeting Updates"),
        ]

    # ---- core counts ----

    def test_one_of_each_canonical_label_counts(self):
        """One doc per non-count label + Press Release + NC + NV → correct totals."""
        result = aggregate(self._all_labels_set())
        assert result.board_meeting_count == 1
        assert result.director_change_count == 1
        assert result.appointment_count == 1
        assert result.resignation_count == 1
        assert result.press_release_count == 1

    def test_clarification_count_superset(self):
        """clarification_count = press_release + news_clarification + news_verification."""
        result = aggregate(self._all_labels_set())
        # 1 Press Release + 1 News Clarification + 1 News Verification = 3
        assert result.clarification_count == 3

    def test_clarification_count_equals_pr_plus_nc_plus_nv(self):
        """Verify the superset relationship explicitly against separate counts."""
        announcements = [
            make_ann(1, "Press Release"),
            make_ann(2, "Press Release"),
            make_ann(3, "News Clarification"),
            make_ann(4, "News Verification"),
            make_ann(5, "News Verification"),
        ]
        result = aggregate(announcements)
        assert result.press_release_count == 2
        assert result.clarification_count == 5  # 2 PR + 1 NC + 2 NV

    def test_agm_delay_flag_is_none_from_aggregate(self):
        """aggregate() never sets agm_delay_flag — it stays None always."""
        result = aggregate(self._all_labels_set())
        assert result.agm_delay_flag is None

    def test_corporate_actions_empty_when_no_corporate_action_other(self):
        """No 'Corporate Action - Other' docs → corporate_actions list is empty."""
        result = aggregate(self._all_labels_set())
        assert result.corporate_actions == []

    # ---- unclassified exclusion ----

    def test_unclassified_excluded_from_all_counts(self):
        """'unclassified' docs must not contribute to any count column."""
        announcements = [make_ann(1, "unclassified")]
        result = aggregate(announcements)
        assert result.board_meeting_count == 0
        assert result.director_change_count == 0
        assert result.appointment_count == 0
        assert result.resignation_count == 0
        assert result.press_release_count == 0
        assert result.clarification_count == 0

    def test_unclassified_not_in_corporate_actions(self):
        """'unclassified' docs must not appear in corporate_actions list."""
        result = aggregate([make_ann(1, "unclassified")])
        assert result.corporate_actions == []

    # ---- Corporate Action - Other ----

    def test_corporate_action_other_appears_in_corporate_actions(self):
        """'Corporate Action - Other' must be added to the corporate_actions list."""
        ann = make_ann(
            document_id=42,
            canonical_label="Corporate Action - Other",
            period_start=date(2018, 6, 1),
            file_name="DHFL_Ann_20180601_Book Closure.pdf",
        )
        result = aggregate([ann])
        assert len(result.corporate_actions) == 1
        ca = result.corporate_actions[0]
        assert ca["document_id"] == 42
        assert ca["file_name"] == "DHFL_Ann_20180601_Book Closure.pdf"
        assert ca["date"] == "2018-06-01"
        assert ca["resolved_label"] == "Corporate Action - Other"

    def test_corporate_action_other_excluded_from_counts(self):
        """'Corporate Action - Other' must NOT increment any typed count column."""
        result = aggregate([make_ann(1, "Corporate Action - Other")])
        assert result.board_meeting_count == 0
        assert result.director_change_count == 0
        assert result.appointment_count == 0
        assert result.resignation_count == 0
        assert result.press_release_count == 0
        assert result.clarification_count == 0

    def test_corporate_action_other_null_period_start_date_field(self):
        """When period_start is None, the 'date' field in corporate_actions should be null."""
        ann = make_ann(
            document_id=99,
            canonical_label="Corporate Action - Other",
            period_start=None,
            file_name="CORP_Acquisition.pdf",
        )
        result = aggregate([ann])
        assert len(result.corporate_actions) == 1
        assert result.corporate_actions[0]["date"] is None

    def test_multiple_corporate_actions_all_recorded(self):
        """Multiple 'Corporate Action - Other' docs all appear in the list."""
        announcements = [
            make_ann(1, "Corporate Action - Other", file_name="BookClosure.pdf"),
            make_ann(2, "Corporate Action - Other", file_name="Buyback.pdf"),
        ]
        result = aggregate(announcements)
        assert len(result.corporate_actions) == 2
        ids = {ca["document_id"] for ca in result.corporate_actions}
        assert ids == {1, 2}

    # ---- Financial Results and AGM no-count ----

    def test_financial_results_does_not_increment_any_count(self):
        """'Financial Results' announcements are governance docs but add no count."""
        result = aggregate([make_ann(1, "Financial Results")])
        assert result.board_meeting_count == 0
        assert result.director_change_count == 0
        assert result.appointment_count == 0
        assert result.resignation_count == 0
        assert result.press_release_count == 0
        assert result.clarification_count == 0
        assert result.corporate_actions == []

    def test_agm_does_not_increment_any_count(self):
        """'AGM / Annual General Meeting Updates' has no typed count column."""
        result = aggregate([make_ann(1, "AGM / Annual General Meeting Updates")])
        assert result.board_meeting_count == 0
        assert result.director_change_count == 0
        assert result.appointment_count == 0
        assert result.resignation_count == 0
        assert result.press_release_count == 0
        assert result.clarification_count == 0
        assert result.corporate_actions == []

    # ---- zero defaults ----

    def test_no_announcements_returns_zero_counts(self):
        """Empty announcements list returns all counts at 0."""
        result = aggregate([])
        assert result.board_meeting_count == 0
        assert result.director_change_count == 0
        assert result.appointment_count == 0
        assert result.resignation_count == 0
        assert result.press_release_count == 0
        assert result.clarification_count == 0
        assert result.agm_delay_flag is None
        assert result.corporate_actions == []

    def test_counts_are_int_not_none(self):
        """Count fields are always int, never None."""
        result = aggregate([])
        assert isinstance(result.board_meeting_count, int)
        assert isinstance(result.director_change_count, int)
        assert isinstance(result.appointment_count, int)
        assert isinstance(result.resignation_count, int)
        assert isinstance(result.press_release_count, int)
        assert isinstance(result.clarification_count, int)

    # ---- multi-document accumulation ----

    def test_multiple_board_meetings_counted_correctly(self):
        """Three Board Meeting docs → board_meeting_count = 3."""
        announcements = [make_ann(i, "Board Meeting") for i in range(1, 4)]
        result = aggregate(announcements)
        assert result.board_meeting_count == 3

    def test_mixed_labels_counts_are_independent(self):
        """Counts for different labels don't interfere with each other."""
        announcements = [
            make_ann(1, "Board Meeting"),
            make_ann(2, "Board Meeting"),
            make_ann(3, "Resignation"),
            make_ann(4, "News Clarification"),
        ]
        result = aggregate(announcements)
        assert result.board_meeting_count == 2
        assert result.resignation_count == 1
        assert result.appointment_count == 0
        assert result.clarification_count == 1  # 1 NC (no PR, no NV)

    def test_press_release_contributes_to_both_columns(self):
        """A single Press Release increments both press_release_count and clarification_count."""
        result = aggregate([make_ann(1, "Press Release")])
        assert result.press_release_count == 1
        assert result.clarification_count == 1

    def test_news_clarification_only_increments_clarification_count(self):
        """News Clarification increments clarification_count but NOT press_release_count."""
        result = aggregate([make_ann(1, "News Clarification")])
        assert result.press_release_count == 0
        assert result.clarification_count == 1

    def test_news_verification_only_increments_clarification_count(self):
        """News Verification increments clarification_count but NOT press_release_count."""
        result = aggregate([make_ann(1, "News Verification")])
        assert result.press_release_count == 0
        assert result.clarification_count == 1

    def test_result_is_governance_result_instance(self):
        """aggregate() returns a GovernanceResult instance."""
        result = aggregate([])
        assert isinstance(result, GovernanceResult)


# ---------------------------------------------------------------------------
# evaluate_agm_delay_flag
# ---------------------------------------------------------------------------

class TestEvaluateAgmDelayFlag:
    """Tests for AGM compliance deadline evaluation."""

    def _agm_ann(self, period_start: date | None, fiscal_year: str = "FY2019") -> CategorizedAnnouncement:
        """Helper: create an AGM announcement with the given period_start."""
        return make_ann(
            document_id=1,
            canonical_label="AGM / Annual General Meeting Updates",
            period_start=period_start,
            fiscal_year=fiscal_year,
            fiscal_quarter="Q1",
        )

    # ---- empty / None cases ----

    def test_empty_list_returns_none(self):
        """No AGM announcements → flag is None."""
        assert evaluate_agm_delay_flag([], "FY2019") is None

    def test_all_none_period_start_returns_none(self):
        """When every announcement has period_start=None, flag is None."""
        anns = [self._agm_ann(None), self._agm_ann(None)]
        assert evaluate_agm_delay_flag(anns, "FY2019") is None

    # ---- boundary dates: FY2019 deadline = Sep 30, 2019 ----

    def test_on_deadline_sep_30_returns_false(self):
        """AGM on exactly Sep 30 of the deadline year is on time → False."""
        ann = self._agm_ann(date(2019, 9, 30), "FY2019")
        assert evaluate_agm_delay_flag([ann], "FY2019") is False

    def test_one_day_after_deadline_returns_true(self):
        """AGM on Oct 1 (one day after deadline) is delayed → True."""
        ann = self._agm_ann(date(2019, 10, 1), "FY2019")
        assert evaluate_agm_delay_flag([ann], "FY2019") is True

    def test_well_before_deadline_returns_false(self):
        """AGM in August, well before September 30, is clearly on time."""
        ann = self._agm_ann(date(2019, 8, 15), "FY2019")
        assert evaluate_agm_delay_flag([ann], "FY2019") is False

    def test_well_after_deadline_returns_true(self):
        """AGM in December, well after September 30, is clearly delayed."""
        ann = self._agm_ann(date(2019, 12, 31), "FY2019")
        assert evaluate_agm_delay_flag([ann], "FY2019") is True

    # ---- multiple AGM dates: earliest used ----

    def test_any_late_agm_in_period_makes_flag_true(self):
        """If any AGM in the period was held after 30 Sep of its accounts year -> True."""
        anns = [
            self._agm_ann(date(2019, 10, 1), "FY2019"),   # late
            self._agm_ann(date(2019, 8, 15), "FY2019"),   # on time
        ]
        assert evaluate_agm_delay_flag(anns, "FY2019") is True

    def test_jan_to_mar_agm_belongs_to_previous_accounts_year(self):
        """A February 2017 AGM covers the year ended Mar 2016 (deadline Sep 2016) -> late."""
        assert evaluate_agm_delay_flag([self._agm_ann(date(2017, 2, 1))], "FY2017") is True

    def test_december_2016_agm_is_late(self):
        """Jet Airways: AGM on 28 Dec 2016 for the year ended Mar 2016 -> late."""
        assert evaluate_agm_delay_flag([self._agm_ann(date(2016, 12, 28))], "FY2017") is True

    def test_june_agm_is_on_time(self):
        assert evaluate_agm_delay_flag([self._agm_ann(date(2016, 6, 30))], "FY2017") is False

    def test_multiple_agm_dates_earliest_after_deadline_is_true(self):
        """All AGM dates after deadline → earliest is still after Sep 30 → True."""
        anns = [
            self._agm_ann(date(2019, 10, 5), "FY2019"),
            self._agm_ann(date(2019, 11, 20), "FY2019"),
        ]
        result = evaluate_agm_delay_flag(anns, "FY2019")
        assert result is True

    def test_none_period_start_ignored_uses_valid_date(self):
        """Announcements with period_start=None are ignored; valid dates still evaluated."""
        anns = [
            self._agm_ann(None, "FY2019"),       # ignored
            self._agm_ann(date(2019, 8, 1), "FY2019"),  # on time
        ]
        result = evaluate_agm_delay_flag(anns, "FY2019")
        assert result is False

    def test_none_and_delayed_date_gives_true(self):
        """One None + one delayed date → the delayed date drives the flag."""
        anns = [
            self._agm_ann(None, "FY2019"),
            self._agm_ann(date(2019, 10, 15), "FY2019"),
        ]
        result = evaluate_agm_delay_flag(anns, "FY2019")
        assert result is True

    # ---- different fiscal years ----

    def test_fy2018_deadline_is_sep_30_2018(self):
        """FY2018 ends March 31, 2018 → deadline is Sep 30, 2018."""
        ann = self._agm_ann(date(2018, 9, 30), "FY2018")
        assert evaluate_agm_delay_flag([ann], "FY2018") is False

        ann_late = self._agm_ann(date(2018, 10, 1), "FY2018")
        assert evaluate_agm_delay_flag([ann_late], "FY2018") is True

    def test_fy2020_deadline_is_sep_30_2020(self):
        """FY2020 ends March 31, 2020 → deadline is Sep 30, 2020."""
        ann = self._agm_ann(date(2020, 9, 30), "FY2020")
        assert evaluate_agm_delay_flag([ann], "FY2020") is False

        ann_late = self._agm_ann(date(2020, 10, 1), "FY2020")
        assert evaluate_agm_delay_flag([ann_late], "FY2020") is True

    # ---- return type ----

    def test_returns_bool_not_truthy(self):
        """The return value must be True/False/None — not just truthy/falsy."""
        ann_on_time = self._agm_ann(date(2019, 7, 1), "FY2019")
        ann_late = self._agm_ann(date(2019, 11, 1), "FY2019")

        result_false = evaluate_agm_delay_flag([ann_on_time], "FY2019")
        result_true = evaluate_agm_delay_flag([ann_late], "FY2019")
        result_none = evaluate_agm_delay_flag([], "FY2019")

        assert result_false is False
        assert result_true is True
        assert result_none is None


# ---------------------------------------------------------------------------
# AGM delay flag placement: Q4 receives the flag; Q1–Q3 always receive NULL
# ---------------------------------------------------------------------------

class TestAgmDelayFlagOnQ4Row:
    """
    Verify that upsert_governance passes agm_delay_flag only on Q4 rows
    and stores NULL for Q1–Q3 rows, regardless of what GovernanceResult holds.

    This tests the contract documented in feature_writer.upsert_governance:
      "AGM delay flag: only meaningful on Q4, NULL for all other quarters"

    The test captures the payload handed to pg_insert by patching
    sqlalchemy.dialects.postgresql.insert so no real DB is needed.
    Requirements: 4.4, 4.5
    """

    def _make_result(self, agm_delay_flag: bool | None) -> "GovernanceResult":
        """GovernanceResult with a specific agm_delay_flag."""
        result = GovernanceResult()
        result.agm_delay_flag = agm_delay_flag
        return result

    def _call_upsert_governance(
        self,
        fiscal_quarter: str,
        agm_delay_flag: bool | None,
    ) -> dict:
        """
        Call upsert_governance with a fully mocked session and capture
        the values dict passed to pg_insert(ExtractedFeature).values(**payload).

        Returns the captured payload dict.
        """
        from unittest.mock import MagicMock, patch, call
        from datetime import date as _date
        from feature_extraction.feature_writer import upsert_governance

        captured_payload: dict = {}

        # The savepoint context manager used by begin_nested()
        mock_sp = MagicMock()
        mock_sp.__enter__ = MagicMock(return_value=None)
        mock_sp.__exit__ = MagicMock(return_value=False)

        mock_session = MagicMock()
        mock_session.begin_nested.return_value = mock_sp
        # execute() must return something with .scalar_one()
        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one.return_value = 1
        mock_session.execute.return_value = mock_execute_result
        mock_session.get.return_value = MagicMock()  # the returned ORM object

        # Intercept pg_insert to capture the .values() payload
        original_pg_insert = None

        def mock_insert(model):
            """Return a chain of fluent mocks that capture .values(**payload)."""
            stmt_mock = MagicMock()

            def capture_values(**kwargs):
                captured_payload.update(kwargs)
                return stmt_mock  # keep the chain going

            stmt_mock.values = capture_values
            stmt_mock.on_conflict_do_update.return_value = stmt_mock
            stmt_mock.returning.return_value = stmt_mock
            return stmt_mock

        with patch("feature_extraction.feature_writer.pg_insert", side_effect=mock_insert):
            gov_result = self._make_result(agm_delay_flag)
            upsert_governance(
                session=mock_session,
                company_id=1,
                fiscal_year="FY2019",
                fiscal_quarter=fiscal_quarter,
                period_end=_date(2019, 3, 31),
                result=gov_result,
            )

        return captured_payload

    def test_q4_row_receives_agm_delay_flag_true(self):
        """Q4 row passes the actual agm_delay_flag value (True) to the DB."""
        payload = self._call_upsert_governance("Q4", agm_delay_flag=True)
        assert payload["agm_delay_flag"] is True

    def test_q4_row_receives_agm_delay_flag_false(self):
        """Q4 row passes the actual agm_delay_flag value (False) to the DB."""
        payload = self._call_upsert_governance("Q4", agm_delay_flag=False)
        assert payload["agm_delay_flag"] is False

    def test_q4_row_with_none_flag_passes_none(self):
        """Q4 row with agm_delay_flag=None (no AGM found) stores NULL."""
        payload = self._call_upsert_governance("Q4", agm_delay_flag=None)
        assert payload["agm_delay_flag"] is None

    def test_q1_row_passes_flag_through(self):
        """The flag lives on the quarter the AGM occurred in."""
        payload = self._call_upsert_governance("Q1", agm_delay_flag=True)
        assert payload["agm_delay_flag"] is True

    def test_q2_row_passes_false_through(self):
        payload = self._call_upsert_governance("Q2", agm_delay_flag=False)
        assert payload["agm_delay_flag"] is False

    def test_q3_row_passes_flag_through(self):
        payload = self._call_upsert_governance("Q3", agm_delay_flag=True)
        assert payload["agm_delay_flag"] is True

    def test_feature_type_is_governance_for_all_quarters(self):
        """feature_type must always be 'governance' regardless of quarter."""
        for quarter in ("Q1", "Q2", "Q3", "Q4"):
            payload = self._call_upsert_governance(quarter, agm_delay_flag=None)
            assert payload["feature_type"] == "governance", (
                f"feature_type should be 'governance' for {quarter}"
            )

    def test_document_id_is_null_for_governance_rows(self):
        """Governance rows always have document_id=NULL (period-level aggregate)."""
        for quarter in ("Q1", "Q2", "Q3", "Q4"):
            payload = self._call_upsert_governance(quarter, agm_delay_flag=None)
            assert payload["document_id"] is None, (
                f"document_id should be NULL for governance {quarter} row"
            )
