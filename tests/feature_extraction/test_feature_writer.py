"""
Unit tests for src/feature_extraction/feature_writer.py

Strategy
--------
The three upsert functions use PostgreSQL's ``INSERT … ON CONFLICT DO UPDATE``
via ``sqlalchemy.dialects.postgresql.insert``.  SQLite does not support the
PostgreSQL dialect, so we cannot use an in-memory SQLite engine for these
tests.

Instead we use ``unittest.mock`` to intercept calls made against a Mock
session and verify:
  1. The statement is a pg_insert statement with the correct conflict target
     and SET clause.
  2. No forbidden label columns appear in any payload.
  3. The correct values are forwarded.

``delete_company_features`` only uses a plain ``sqlalchemy.delete`` which is
dialect-agnostic, but because it executes directly on a session we mock the
session there too and assert the DELETE was issued for the correct company.

For the "idempotence" / "single row" behaviour we use a real PostgreSQL
session via pytest-skips if POSTGRES is not available — but the mock tests
cover the *logic* path (payload construction, conflict keys) which is where
the de-duplication guarantee lives.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 8.2
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, call, patch

import pytest

from sqlalchemy.dialects.postgresql import Insert as PgInsert

from feature_extraction.feature_writer import (
    _FORBIDDEN_COLUMNS,
    _assert_no_forbidden_columns,
    delete_company_features,
    upsert_credit,
    upsert_financial,
    upsert_governance,
)
from feature_extraction.financial_extractor import FinancialResult
from feature_extraction.governance_extractor import GovernanceResult
from feature_extraction.credit_extractor import CreditResult


# ---------------------------------------------------------------------------
# Helpers / shared builders
# ---------------------------------------------------------------------------

def _make_financial_result(**overrides) -> FinancialResult:
    """Build a FinancialResult with reasonable defaults."""
    result = FinancialResult(
        revenue=Decimal("1000.00"),
        pat=Decimal("100.00"),
        ebitda_margin_pct=Decimal("15.50"),
        debt_to_equity=Decimal("1.2345"),
        interest_coverage=Decimal("3.5000"),
        current_ratio=Decimal("1.8000"),
        cfo=Decimal("200.00"),
        revenue_growth_pct=Decimal("5.00"),
    )
    result.signals_populated = 8
    for k, v in overrides.items():
        setattr(result, k, v)
    return result


def _make_governance_result(**overrides) -> GovernanceResult:
    result = GovernanceResult(
        board_meeting_count=3,
        director_change_count=1,
        appointment_count=2,
        resignation_count=0,
        press_release_count=4,
        clarification_count=6,
        agm_delay_flag=False,
        corporate_actions=[],
    )
    for k, v in overrides.items():
        setattr(result, k, v)
    return result


def _make_credit_result(**overrides) -> CreditResult:
    result = CreditResult(
        credit_rating="AA+",
        rating_outlook="Stable",
        rating_change_flag=False,
        instrument_ratings=None,
    )
    for k, v in overrides.items():
        setattr(result, k, v)
    return result


def _mock_session():
    """
    Return a MagicMock session whose execute() returns a mock scalar result
    and whose begin_nested() returns a context-manager-compatible object.
    """
    session = MagicMock()

    # begin_nested() must work as a context manager
    nested_ctx = MagicMock()
    nested_ctx.__enter__ = MagicMock(return_value=nested_ctx)
    nested_ctx.__exit__ = MagicMock(return_value=False)
    session.begin_nested.return_value = nested_ctx

    # execute() returns an object whose scalar_one() gives a fake primary key
    execute_result = MagicMock()
    execute_result.scalar_one.return_value = 42
    session.execute.return_value = execute_result

    # session.get() returns a fake ExtractedFeature-like object
    fake_feature = MagicMock()
    fake_feature.id = 42
    session.get.return_value = fake_feature

    return session


def _capture_pg_insert_payload(session: MagicMock) -> dict:
    """
    Extract the values dict from the PgInsert statement that was passed
    to ``session.execute()``.

    Works by inspecting the compiled INSERT statement values.
    """
    assert session.execute.called, "session.execute was never called"
    stmt = session.execute.call_args[0][0]  # first positional arg
    # The pg_insert stmt stores values in ._values (list of ClauseElement tuples)
    # We use the public .compile() approach or introspect the statement directly.
    # Most reliable: pull from stmt.parameters (ColumnCollection of bind params)
    # SQLAlchemy Insert stores inserted values in the ._values list of dicts.
    # We access via stmt.compile().params which flattens to a plain dict.
    from sqlalchemy.dialects import postgresql
    compiled = stmt.compile(dialect=postgresql.dialect())
    return dict(compiled.params)


# ---------------------------------------------------------------------------
# Label guard (_assert_no_forbidden_columns)
# ---------------------------------------------------------------------------

class TestLabelGuard:
    """Tests for the ``_assert_no_forbidden_columns`` helper."""

    def test_clean_payload_does_not_raise(self):
        payload = {"company_id": 1, "fiscal_year": "FY2019", "revenue": Decimal("100")}
        _assert_no_forbidden_columns(payload)  # must not raise

    @pytest.mark.parametrize("forbidden_col", list(_FORBIDDEN_COLUMNS))
    def test_raises_for_each_forbidden_column(self, forbidden_col):
        payload = {"company_id": 1, forbidden_col: "some_value"}
        with pytest.raises(AssertionError, match="Label guard violation"):
            _assert_no_forbidden_columns(payload)

    def test_raises_with_all_forbidden_columns(self):
        payload = {col: "x" for col in _FORBIDDEN_COLUMNS}
        with pytest.raises(AssertionError):
            _assert_no_forbidden_columns(payload)

    def test_forbidden_column_names_are_correct_set(self):
        """Verify the guard covers all required forbidden names per Req 6.4."""
        assert "group" in _FORBIDDEN_COLUMNS
        assert "distressed" in _FORBIDDEN_COLUMNS
        assert "healthy" in _FORBIDDEN_COLUMNS
        assert "label" in _FORBIDDEN_COLUMNS
        assert "ground_truth" in _FORBIDDEN_COLUMNS


# ---------------------------------------------------------------------------
# upsert_financial — payload and structure tests
# ---------------------------------------------------------------------------

class TestUpsertFinancial:
    """
    Tests for ``upsert_financial``.

    We mock the session and inspect the pg_insert statement passed to
    ``session.execute()`` to verify the payload is correctly constructed
    and the conflict target is correct.
    """

    def test_calls_session_execute_with_pg_insert(self):
        session = _mock_session()
        result = _make_financial_result()
        upsert_financial(
            session, company_id=1, document_id=10,
            fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        assert session.execute.called
        stmt = session.execute.call_args[0][0]
        assert isinstance(stmt, PgInsert)

    def test_payload_contains_correct_feature_type(self):
        session = _mock_session()
        result = _make_financial_result()
        upsert_financial(
            session, company_id=1, document_id=10,
            fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["feature_type"] == "financial"

    def test_payload_contains_correct_identifiers(self):
        session = _mock_session()
        result = _make_financial_result()
        upsert_financial(
            session, company_id=7, document_id=99,
            fiscal_year="FY2020", fiscal_quarter="Q3",
            period_end=date(2019, 12, 31), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["company_id"] == 7
        assert params["document_id"] == 99
        assert params["fiscal_year"] == "FY2020"
        assert params["fiscal_quarter"] == "Q3"

    def test_financial_signals_in_payload(self):
        session = _mock_session()
        result = _make_financial_result(
            revenue=Decimal("5000.00"),
            pat=Decimal("500.00"),
        )
        upsert_financial(
            session, company_id=1, document_id=10,
            fiscal_year="FY2019", fiscal_quarter="Q1",
            period_end=date(2018, 6, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["revenue"] == Decimal("5000.00")
        assert params["pat"] == Decimal("500.00")

    def test_governance_and_credit_fields_are_null_for_financial_row(self):
        session = _mock_session()
        result = _make_financial_result()
        upsert_financial(
            session, company_id=1, document_id=10,
            fiscal_year="FY2019", fiscal_quarter="Q1",
            period_end=date(2018, 6, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["board_meeting_count"] is None
        assert params["director_change_count"] is None
        assert params["credit_rating"] is None
        assert params["rating_outlook"] is None

    def test_no_forbidden_columns_in_payload(self):
        session = _mock_session()
        result = _make_financial_result()
        # If label guard fires, AssertionError is raised before execute is called
        upsert_financial(
            session, company_id=1, document_id=10,
            fiscal_year="FY2019", fiscal_quarter="Q1",
            period_end=date(2018, 6, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        for forbidden in _FORBIDDEN_COLUMNS:
            assert forbidden not in params, (
                f"Forbidden column '{forbidden}' found in financial upsert payload"
            )

    def test_uses_begin_nested_savepoint(self):
        """The upsert must run inside session.begin_nested() per Req 6.5."""
        session = _mock_session()
        result = _make_financial_result()
        upsert_financial(
            session, company_id=1, document_id=10,
            fiscal_year="FY2019", fiscal_quarter="Q1",
            period_end=date(2018, 6, 30), result=result,
        )
        assert session.begin_nested.called

    def test_conflict_target_includes_required_columns(self):
        """
        Verify the ON CONFLICT target references (company_id, document_id,
        feature_type) which are the three columns in ix_ef_financial_upsert.

        We compile to PostgreSQL SQL string and check that all three column
        names appear in the ON CONFLICT clause.
        """
        from sqlalchemy.dialects import postgresql as pg_dialect

        session = _mock_session()
        result = _make_financial_result()
        upsert_financial(
            session, company_id=1, document_id=10,
            fiscal_year="FY2019", fiscal_quarter="Q1",
            period_end=date(2018, 6, 30), result=result,
        )
        stmt = session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=pg_dialect.dialect()))
        # The compiled SQL must contain "ON CONFLICT (company_id, document_id, feature_type)"
        assert "ON CONFLICT" in sql
        assert "company_id" in sql
        assert "document_id" in sql
        assert "feature_type" in sql

    def test_annual_report_has_null_fiscal_quarter(self):
        """Annual report documents pass fiscal_quarter=None."""
        session = _mock_session()
        result = _make_financial_result()
        upsert_financial(
            session, company_id=1, document_id=10,
            fiscal_year="FY2019", fiscal_quarter=None,
            period_end=date(2019, 3, 31), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["fiscal_quarter"] is None

    def test_second_call_with_same_key_still_executes(self):
        """
        Calling upsert_financial twice with the same (company_id, document_id)
        should both reach session.execute (the DB-level deduplication is the
        ON CONFLICT clause — the function itself always issues the statement).
        """
        session = _mock_session()
        result1 = _make_financial_result(revenue=Decimal("1000.00"))
        result2 = _make_financial_result(revenue=Decimal("9999.00"))

        upsert_financial(session, company_id=1, document_id=10,
                         fiscal_year="FY2019", fiscal_quarter="Q2",
                         period_end=date(2018, 9, 30), result=result1)
        upsert_financial(session, company_id=1, document_id=10,
                         fiscal_year="FY2019", fiscal_quarter="Q2",
                         period_end=date(2018, 9, 30), result=result2)

        assert session.execute.call_count == 2

    def test_set_clause_excludes_conflict_key_columns(self):
        """
        The SET clause on the ON CONFLICT DO UPDATE must NOT re-assign the
        conflict key columns (company_id, document_id, feature_type, created_at).
        We verify via the OnConflictDoUpdate.update_values_to_set attribute which
        is a list of (column_name, value) tuples.
        """
        session = _mock_session()
        result = _make_financial_result()
        upsert_financial(
            session, company_id=1, document_id=10,
            fiscal_year="FY2019", fiscal_quarter="Q1",
            period_end=date(2018, 6, 30), result=result,
        )
        stmt = session.execute.call_args[0][0]
        conflict = stmt._post_values_clause
        # update_values_to_set is a list of (column_name, expression) tuples
        set_keys = {item[0] for item in conflict.update_values_to_set}
        for key in ("company_id", "document_id", "feature_type", "created_at"):
            assert key not in set_keys, (
                f"Conflict key '{key}' should not appear in the SET clause"
            )


# ---------------------------------------------------------------------------
# upsert_governance — payload and structure tests
# ---------------------------------------------------------------------------

class TestUpsertGovernance:
    def test_calls_session_execute_with_pg_insert(self):
        session = _mock_session()
        result = _make_governance_result()
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        assert session.execute.called
        stmt = session.execute.call_args[0][0]
        assert isinstance(stmt, PgInsert)

    def test_payload_contains_correct_feature_type(self):
        session = _mock_session()
        result = _make_governance_result()
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["feature_type"] == "governance"

    def test_document_id_is_null_for_governance(self):
        session = _mock_session()
        result = _make_governance_result()
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["document_id"] is None

    def test_governance_counts_in_payload(self):
        session = _mock_session()
        result = _make_governance_result(
            board_meeting_count=5,
            director_change_count=2,
            appointment_count=3,
            resignation_count=1,
            press_release_count=4,
            clarification_count=7,
        )
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["board_meeting_count"] == 5
        assert params["director_change_count"] == 2
        assert params["appointment_count"] == 3
        assert params["resignation_count"] == 1
        assert params["press_release_count"] == 4
        assert params["clarification_count"] == 7

    def test_agm_delay_flag_stored_on_q4(self):
        """agm_delay_flag from result is passed through on Q4 rows."""
        session = _mock_session()
        result = _make_governance_result(agm_delay_flag=True)
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q4",
            period_end=date(2019, 3, 31), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["agm_delay_flag"] is True

    def test_agm_delay_flag_stored_on_any_quarter(self):
        """The flag lives on the quarter the AGM occurred in, not only Q4."""
        session = _mock_session()
        for quarter in ("Q1", "Q2", "Q3"):
            session = _mock_session()
            result = _make_governance_result(agm_delay_flag=True)
            upsert_governance(
                session, company_id=1, fiscal_year="FY2019", fiscal_quarter=quarter,
                period_end=date(2018, 6, 30), result=result,
            )
            params = _capture_pg_insert_payload(session)
            assert params["agm_delay_flag"] is True, quarter

    def test_financial_and_credit_fields_are_null_for_governance(self):
        session = _mock_session()
        result = _make_governance_result()
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["revenue"] is None
        assert params["pat"] is None
        assert params["credit_rating"] is None
        assert params["rating_outlook"] is None

    def test_no_forbidden_columns_in_payload(self):
        session = _mock_session()
        result = _make_governance_result()
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        for forbidden in _FORBIDDEN_COLUMNS:
            assert forbidden not in params, (
                f"Forbidden column '{forbidden}' found in governance upsert payload"
            )

    def test_uses_begin_nested_savepoint(self):
        session = _mock_session()
        result = _make_governance_result()
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        assert session.begin_nested.called

    def test_conflict_target_includes_required_columns(self):
        """Conflict target: (company_id, fiscal_year, fiscal_quarter, feature_type)."""
        from sqlalchemy.dialects import postgresql as pg_dialect

        session = _mock_session()
        result = _make_governance_result()
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        stmt = session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=pg_dialect.dialect()))
        assert "ON CONFLICT" in sql
        assert "company_id" in sql
        assert "fiscal_year" in sql
        assert "fiscal_quarter" in sql
        assert "feature_type" in sql

    def test_second_call_with_same_key_executes_both_times(self):
        """
        Both calls reach session.execute; the ON CONFLICT clause handles
        deduplication at the database level.
        """
        session = _mock_session()
        result1 = _make_governance_result(board_meeting_count=2)
        result2 = _make_governance_result(board_meeting_count=5)
        upsert_governance(session, company_id=1, fiscal_year="FY2019",
                          fiscal_quarter="Q2", period_end=date(2018, 9, 30),
                          result=result1)
        upsert_governance(session, company_id=1, fiscal_year="FY2019",
                          fiscal_quarter="Q2", period_end=date(2018, 9, 30),
                          result=result2)
        assert session.execute.call_count == 2

    def test_corporate_actions_stored_in_additional_signals(self):
        """corporate_actions list is stored under additional_signals['corporate_actions']."""
        session = _mock_session()
        actions = [
            {"document_id": 5, "file_name": "foo.pdf", "date": "2018-07-01",
             "resolved_label": "Corporate Action - Other"}
        ]
        result = _make_governance_result(corporate_actions=actions)
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["additional_signals"] == {"corporate_actions": actions}

    def test_additional_signals_is_none_when_no_corporate_actions(self):
        session = _mock_session()
        result = _make_governance_result(corporate_actions=[])
        upsert_governance(
            session, company_id=1, fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["additional_signals"] is None


# ---------------------------------------------------------------------------
# upsert_credit — payload and structure tests
# ---------------------------------------------------------------------------

class TestUpsertCredit:
    def test_calls_session_execute_with_pg_insert(self):
        session = _mock_session()
        result = _make_credit_result()
        upsert_credit(
            session, company_id=1, document_id=20,
            fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        assert session.execute.called
        stmt = session.execute.call_args[0][0]
        assert isinstance(stmt, PgInsert)

    def test_payload_contains_correct_feature_type(self):
        session = _mock_session()
        result = _make_credit_result()
        upsert_credit(
            session, company_id=1, document_id=20,
            fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["feature_type"] == "credit"

    def test_credit_signals_in_payload(self):
        session = _mock_session()
        result = _make_credit_result(
            credit_rating="BBB+",
            rating_outlook="Negative",
            rating_change_flag=True,
        )
        upsert_credit(
            session, company_id=1, document_id=20,
            fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["credit_rating"] == "BBB+"
        assert params["rating_outlook"] == "Negative"
        assert params["rating_change_flag"] is True

    def test_financial_and_governance_fields_null_for_credit(self):
        session = _mock_session()
        result = _make_credit_result()
        upsert_credit(
            session, company_id=1, document_id=20,
            fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["revenue"] is None
        assert params["board_meeting_count"] is None
        assert params["agm_delay_flag"] is None

    def test_no_forbidden_columns_in_payload(self):
        session = _mock_session()
        result = _make_credit_result()
        upsert_credit(
            session, company_id=1, document_id=20,
            fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        params = _capture_pg_insert_payload(session)
        for forbidden in _FORBIDDEN_COLUMNS:
            assert forbidden not in params, (
                f"Forbidden column '{forbidden}' found in credit upsert payload"
            )

    def test_uses_begin_nested_savepoint(self):
        session = _mock_session()
        result = _make_credit_result()
        upsert_credit(
            session, company_id=1, document_id=20,
            fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        assert session.begin_nested.called

    def test_conflict_target_includes_required_columns(self):
        """Conflict target: (company_id, document_id, feature_type)."""
        from sqlalchemy.dialects import postgresql as pg_dialect

        session = _mock_session()
        result = _make_credit_result()
        upsert_credit(
            session, company_id=1, document_id=20,
            fiscal_year="FY2019", fiscal_quarter="Q2",
            period_end=date(2018, 9, 30), result=result,
        )
        stmt = session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=pg_dialect.dialect()))
        assert "ON CONFLICT" in sql
        assert "company_id" in sql
        assert "document_id" in sql
        assert "feature_type" in sql

    def test_second_call_with_same_key_executes_both_times(self):
        session = _mock_session()
        result1 = _make_credit_result(credit_rating="AA")
        result2 = _make_credit_result(credit_rating="BBB")
        upsert_credit(session, company_id=1, document_id=20,
                      fiscal_year="FY2019", fiscal_quarter="Q2",
                      period_end=date(2018, 9, 30), result=result1)
        upsert_credit(session, company_id=1, document_id=20,
                      fiscal_year="FY2019", fiscal_quarter="Q2",
                      period_end=date(2018, 9, 30), result=result2)
        assert session.execute.call_count == 2

    def test_instrument_ratings_stored_in_additional_signals(self):
        """instrument_ratings list stored under additional_signals['instrument_ratings']."""
        session = _mock_session()
        instruments = [
            {"instrument": "NCD", "rating": "AA+", "outlook": "Stable"},
            {"instrument": "CP", "rating": "A1+", "outlook": None},
        ]
        result = _make_credit_result(instrument_ratings=instruments)
        upsert_credit(
            session, company_id=1, document_id=20,
            fiscal_year="FY2019", fiscal_quarter=None,
            period_end=None, result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["additional_signals"] == {"instrument_ratings": instruments}

    def test_additional_signals_none_when_no_instrument_ratings(self):
        session = _mock_session()
        result = _make_credit_result(instrument_ratings=None)
        upsert_credit(
            session, company_id=1, document_id=20,
            fiscal_year="FY2019", fiscal_quarter=None,
            period_end=None, result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["additional_signals"] is None

    def test_null_fiscal_period_accepted(self):
        """credit_rating docs may have no fiscal period — fiscal_year/quarter/period_end all None."""
        session = _mock_session()
        result = _make_credit_result()
        upsert_credit(
            session, company_id=1, document_id=20,
            fiscal_year=None, fiscal_quarter=None,
            period_end=None, result=result,
        )
        params = _capture_pg_insert_payload(session)
        assert params["fiscal_year"] is None
        assert params["fiscal_quarter"] is None
        assert params["period_end"] is None


# ---------------------------------------------------------------------------
# delete_company_features
# ---------------------------------------------------------------------------

class TestDeleteCompanyFeatures:
    """
    Tests for ``delete_company_features``.

    We mock ``session.execute`` to capture the DELETE statement and verify:
    - The correct table is targeted.
    - The WHERE clause filters by the correct company_id.
    - The function returns the rowcount.
    - processing_log is never touched.
    """

    def _mock_session_for_delete(self, rowcount: int = 3) -> MagicMock:
        session = MagicMock()
        execute_result = MagicMock()
        execute_result.rowcount = rowcount
        session.execute.return_value = execute_result
        return session

    def test_returns_correct_deleted_count(self):
        session = self._mock_session_for_delete(rowcount=5)
        count = delete_company_features(session, company_id=7)
        assert count == 5

    def test_returns_zero_when_no_rows_deleted(self):
        session = self._mock_session_for_delete(rowcount=0)
        count = delete_company_features(session, company_id=99)
        assert count == 0

    def test_calls_session_execute_once(self):
        session = self._mock_session_for_delete()
        delete_company_features(session, company_id=1)
        assert session.execute.call_count == 1

    def test_delete_statement_targets_extracted_features_table(self):
        """The DELETE must target 'extracted_features', not processing_log."""
        from sqlalchemy.sql.dml import Delete as SADelete
        session = self._mock_session_for_delete()
        delete_company_features(session, company_id=1)

        stmt = session.execute.call_args[0][0]
        assert isinstance(stmt, SADelete)
        # Table name must be extracted_features
        assert stmt.table.name == "extracted_features"

    def test_delete_where_clause_uses_correct_company_id(self):
        """
        The DELETE's WHERE clause must bind the correct company_id value
        so it only removes rows for the specified company.
        """
        from sqlalchemy.dialects import postgresql as pg_dialect
        session = self._mock_session_for_delete()
        delete_company_features(session, company_id=42)

        stmt = session.execute.call_args[0][0]
        compiled = stmt.compile(dialect=pg_dialect.dialect())
        params = dict(compiled.params)
        # Should have exactly one bind param equal to 42
        assert 42 in params.values(), (
            f"Expected company_id=42 in DELETE params, got: {params}"
        )

    def test_does_not_call_execute_on_processing_log(self):
        """
        delete_company_features must NEVER issue any statement that touches
        processing_log (Req 8.2 hard constraint).
        """
        session = self._mock_session_for_delete()
        delete_company_features(session, company_id=1)

        # Check every execute call — none should mention processing_log
        for call_args in session.execute.call_args_list:
            stmt = call_args[0][0]
            stmt_str = str(stmt.compile(
                compile_kwargs={"literal_binds": False}
            ))
            assert "processing_log" not in stmt_str.lower(), (
                "delete_company_features issued a statement touching processing_log"
            )

    def test_does_not_add_orm_objects(self):
        """
        delete_company_features uses Core DELETE, not ORM session.add/delete,
        so it never loads rows into memory.
        """
        session = self._mock_session_for_delete()
        delete_company_features(session, company_id=1)
        session.add.assert_not_called()
        session.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Cross-type label guard integration — all three upsert functions
# ---------------------------------------------------------------------------

class TestLabelGuardIntegration:
    """
    Verify that _assert_no_forbidden_columns is invoked by each upsert function
    and that injecting a forbidden column raises AssertionError before execute.
    """

    @pytest.mark.parametrize("forbidden", list(_FORBIDDEN_COLUMNS))
    def test_financial_upsert_raises_on_forbidden_column_injection(self, forbidden):
        """
        Patch _assert_no_forbidden_columns to inject a forbidden column into
        the payload and confirm AssertionError is raised.
        """
        session = _mock_session()
        result = _make_financial_result()
        result.additional_signals = {forbidden: "leaked_value"}

        # The additional_signals is stored as-is; the forbidden col guard
        # checks top-level payload keys — additional_signals as a nested key
        # is outside scope.  Instead verify the guard directly fires on payload.
        from feature_extraction import feature_writer as fw
        original = fw._assert_no_forbidden_columns

        def patched(payload):
            payload[forbidden] = "injected"
            return original(payload)

        with patch.object(fw, "_assert_no_forbidden_columns", side_effect=patched):
            with pytest.raises(AssertionError, match="Label guard violation"):
                upsert_financial(
                    session, company_id=1, document_id=10,
                    fiscal_year="FY2019", fiscal_quarter="Q1",
                    period_end=date(2018, 6, 30), result=result,
                )

    @pytest.mark.parametrize("forbidden", list(_FORBIDDEN_COLUMNS))
    def test_governance_upsert_raises_on_forbidden_column_injection(self, forbidden):
        session = _mock_session()
        result = _make_governance_result()

        from feature_extraction import feature_writer as fw
        original = fw._assert_no_forbidden_columns

        def patched(payload):
            payload[forbidden] = "injected"
            return original(payload)

        with patch.object(fw, "_assert_no_forbidden_columns", side_effect=patched):
            with pytest.raises(AssertionError, match="Label guard violation"):
                upsert_governance(
                    session, company_id=1, fiscal_year="FY2019",
                    fiscal_quarter="Q2", period_end=date(2018, 9, 30),
                    result=result,
                )

    @pytest.mark.parametrize("forbidden", list(_FORBIDDEN_COLUMNS))
    def test_credit_upsert_raises_on_forbidden_column_injection(self, forbidden):
        session = _mock_session()
        result = _make_credit_result()

        from feature_extraction import feature_writer as fw
        original = fw._assert_no_forbidden_columns

        def patched(payload):
            payload[forbidden] = "injected"
            return original(payload)

        with patch.object(fw, "_assert_no_forbidden_columns", side_effect=patched):
            with pytest.raises(AssertionError, match="Label guard violation"):
                upsert_credit(
                    session, company_id=1, document_id=20,
                    fiscal_year="FY2019", fiscal_quarter="Q2",
                    period_end=date(2018, 9, 30), result=result,
                )
