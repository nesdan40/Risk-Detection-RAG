"""
Unit tests for src/feature_extraction/feature_logger.py

Uses SQLAlchemy in-memory SQLite so no running PostgreSQL is required.

Only the three tables this component touches are created:
  companies, documents, processing_log

Other models (extracted_features, predictions) use PostgreSQL-specific JSONB
columns that SQLite cannot compile, so we deliberately exclude them by using
a scoped MetaData — the same pattern used in tests/ingestion/test_ingestion_logger.py.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8
"""

from __future__ import annotations

import pytest

from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.orm import Session
from unittest.mock import patch

from models.base import Base
from models.company import Company
from models.document import Document
from models.processing_log import ProcessingLog

from feature_extraction.feature_logger import (
    log_started,
    log_success,
    log_governance_success,
    log_unmapped_category,
    log_failed,
)
from feature_extraction.governance_extractor import GovernanceResult


# ---------------------------------------------------------------------------
# SQLite-compatible metadata (only the three tables we need)
# ---------------------------------------------------------------------------

_TEST_TABLES = {"companies", "documents", "processing_log"}


def _test_metadata() -> MetaData:
    """Return a MetaData instance containing only the SQLite-safe tables."""
    meta = MetaData()
    for table in Base.metadata.tables.values():
        if table.name in _TEST_TABLES:
            table.to_metadata(meta)
    return meta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_session():
    """
    Provide a fresh in-memory SQLite session for each test.

    Only companies / documents / processing_log tables are created so
    SQLite's JSONB limitation doesn't block the fixture setup.
    """
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    meta = _test_metadata()
    meta.create_all(engine)

    with Session(engine) as session:
        yield session

    meta.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def company(db_session):
    """Insert a minimal Company row and return it."""
    c = Company(ticker="TESTCO", name="Test Company Ltd")
    db_session.add(c)
    db_session.flush()
    return c


@pytest.fixture()
def document(db_session, company):
    """Insert a minimal Document row and return it."""
    d = Document(
        company_id=company.id,
        file_name="TESTCO_AnnualReport_FY2019.pdf",
        file_path="data/raw/TESTCO/annual_reports/TESTCO_AnnualReport_FY2019.pdf",
        document_type="annual_report",
    )
    db_session.add(d)
    db_session.flush()
    return d


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _all_logs(session: Session) -> list[ProcessingLog]:
    return session.query(ProcessingLog).order_by(ProcessingLog.id).all()


# ---------------------------------------------------------------------------
# log_started — Requirement 7.1
# ---------------------------------------------------------------------------

class TestLogStarted:
    def test_inserts_one_row(self, db_session, document):
        log_started(document_id=document.id, document_type="annual_report", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1

    def test_stage_is_feature_extraction(self, db_session, document):
        log_started(document_id=document.id, document_type="annual_report", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].stage == "feature_extraction"

    def test_status_is_started(self, db_session, document):
        log_started(document_id=document.id, document_type="annual_report", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].status == "started"

    def test_document_id_is_preserved(self, db_session, document):
        log_started(document_id=document.id, document_type="annual_report", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id == document.id

    def test_message_matches_format(self, db_session, document):
        log_started(document_id=document.id, document_type="annual_report", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        expected = f"Starting feature_extraction for document_id={document.id}, type=annual_report"
        assert logs[0].message == expected

    def test_message_contains_document_id(self, db_session, document):
        log_started(document_id=document.id, document_type="quarterly_result", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert f"document_id={document.id}" in logs[0].message

    def test_message_contains_document_type(self, db_session, document):
        log_started(document_id=document.id, document_type="credit_rating", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "credit_rating" in logs[0].message


# ---------------------------------------------------------------------------
# log_success — Requirement 7.2
# ---------------------------------------------------------------------------

class TestLogSuccess:
    def test_inserts_one_row(self, db_session, document):
        log_success(document_id=document.id, feature_type="financial", signals_populated=5, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1

    def test_status_is_success(self, db_session, document):
        log_success(document_id=document.id, feature_type="financial", signals_populated=5, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].status == "success"

    def test_stage_is_feature_extraction(self, db_session, document):
        log_success(document_id=document.id, feature_type="financial", signals_populated=3, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].stage == "feature_extraction"

    def test_document_id_is_preserved(self, db_session, document):
        log_success(document_id=document.id, feature_type="credit", signals_populated=2, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id == document.id

    def test_message_contains_feature_type(self, db_session, document):
        log_success(document_id=document.id, feature_type="financial", signals_populated=4, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "financial" in logs[0].message

    def test_message_contains_signals_populated_count(self, db_session, document):
        log_success(document_id=document.id, feature_type="financial", signals_populated=7, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "7" in logs[0].message

    def test_message_format(self, db_session, document):
        log_success(document_id=document.id, feature_type="credit", signals_populated=3, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message == "feature_type=credit, signals_populated=3"

    def test_zero_signals_populated(self, db_session, document):
        log_success(document_id=document.id, feature_type="financial", signals_populated=0, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "signals_populated=0" in logs[0].message


# ---------------------------------------------------------------------------
# log_governance_success — Requirement 7.5
# ---------------------------------------------------------------------------

class TestLogGovernanceSuccess:
    def _make_result(self, **kwargs) -> GovernanceResult:
        defaults = dict(
            board_meeting_count=3,
            director_change_count=1,
            appointment_count=2,
            resignation_count=0,
            press_release_count=4,
            clarification_count=6,
            agm_delay_flag=False,
        )
        defaults.update(kwargs)
        return GovernanceResult(**defaults)

    def test_document_id_is_null(self, db_session):
        result = self._make_result()
        log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].document_id is None

    def test_status_is_success(self, db_session):
        result = self._make_result()
        log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].status == "success"

    def test_stage_is_feature_extraction(self, db_session):
        result = self._make_result()
        log_governance_success("TESTCO", "FY2019", "Q2", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].stage == "feature_extraction"

    def test_message_contains_company_ticker(self, db_session):
        result = self._make_result()
        log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "company=TESTCO" in logs[0].message

    def test_message_contains_period(self, db_session):
        result = self._make_result()
        log_governance_success("TESTCO", "FY2019", "Q3", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "period=FY2019/Q3" in logs[0].message

    def test_message_contains_board_meeting_count(self, db_session):
        result = self._make_result(board_meeting_count=5)
        log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "board_meeting_count=5" in logs[0].message

    def test_message_contains_director_change_count(self, db_session):
        result = self._make_result(director_change_count=2)
        log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "director_change_count=2" in logs[0].message

    def test_message_contains_appointment_count(self, db_session):
        result = self._make_result(appointment_count=3)
        log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "appointment_count=3" in logs[0].message

    def test_message_contains_resignation_count(self, db_session):
        result = self._make_result(resignation_count=1)
        log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "resignation_count=1" in logs[0].message

    def test_message_contains_press_release_count(self, db_session):
        result = self._make_result(press_release_count=7)
        log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "press_release_count=7" in logs[0].message

    def test_message_contains_clarification_count(self, db_session):
        result = self._make_result(clarification_count=9)
        log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "clarification_count=9" in logs[0].message

    def test_message_contains_agm_delay_flag(self, db_session):
        result = self._make_result(agm_delay_flag=True)
        log_governance_success("TESTCO", "FY2019", "Q4", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "agm_delay_flag=True" in logs[0].message

    def test_message_contains_agm_delay_flag_none(self, db_session):
        result = self._make_result(agm_delay_flag=None)
        log_governance_success("TESTCO", "FY2019", "Q2", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "agm_delay_flag=None" in logs[0].message

    def test_only_one_row_inserted_no_started_row(self, db_session):
        """log_governance_success must NOT insert a 'started' row — only the terminal success row."""
        result = self._make_result()
        log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].status == "success"
        # No started row
        started = [l for l in logs if l.status == "started"]
        assert len(started) == 0

    def test_message_req75_format_exact(self, db_session):
        """Verify the message matches the Req 7.5 format template exactly."""
        result = GovernanceResult(
            board_meeting_count=2,
            director_change_count=1,
            appointment_count=0,
            resignation_count=1,
            press_release_count=3,
            clarification_count=5,
            agm_delay_flag=False,
        )
        log_governance_success("DHFL", "FY2019", "Q4", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        msg = logs[0].message
        expected = (
            "governance extracted for company=DHFL, "
            "period=FY2019/Q4, "
            "board_meeting_count=2, "
            "director_change_count=1, "
            "appointment_count=0, "
            "resignation_count=1, "
            "resignation_count=1, "
            "press_release_count=3, "
            "clarification_count=5, "
            "agm_delay_flag=False"
        )
        assert msg == expected


# ---------------------------------------------------------------------------
# log_unmapped_category — Requirement 7.3
# ---------------------------------------------------------------------------

class TestLogUnmappedCategory:
    def test_inserts_one_row(self, db_session, document):
        log_unmapped_category(
            document_id=document.id,
            document_category="SomeWeirdCategory",
            file_name="TESTCO_Ann_20190101_SomeWeirdCategory_SCANNED.pdf",
            ticker="TESTCO",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1

    def test_status_is_unmapped_category(self, db_session, document):
        log_unmapped_category(
            document_id=document.id,
            document_category="Unknown",
            file_name="file.pdf",
            ticker="TESTCO",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].status == "unmapped_category"

    def test_stage_is_feature_extraction(self, db_session, document):
        log_unmapped_category(
            document_id=document.id,
            document_category="Unknown",
            file_name="file.pdf",
            ticker="TESTCO",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].stage == "feature_extraction"

    def test_document_id_preserved(self, db_session, document):
        log_unmapped_category(
            document_id=document.id,
            document_category="WeirdToken",
            file_name="file.pdf",
            ticker="TESTCO",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id == document.id

    def test_message_starts_with_unmapped_category_token(self, db_session, document):
        """Message must start with literal 'unmapped_category' as first token (Req 7.3)."""
        log_unmapped_category(
            document_id=document.id,
            document_category="StrangeLabel",
            file_name="file.pdf",
            ticker="TESTCO",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message.startswith("unmapped_category")

    def test_message_first_word_is_exactly_unmapped_category(self, db_session, document):
        log_unmapped_category(
            document_id=document.id,
            document_category="X",
            file_name="file.pdf",
            ticker="TESTCO",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        first_token = logs[0].message.split()[0]
        assert first_token == "unmapped_category"

    def test_message_contains_file_name(self, db_session, document):
        log_unmapped_category(
            document_id=document.id,
            document_category="Whatever",
            file_name="TESTCO_Ann_20190515_Whatever_SCANNED.pdf",
            ticker="TESTCO",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        assert "TESTCO_Ann_20190515_Whatever_SCANNED.pdf" in logs[0].message

    def test_message_contains_ticker(self, db_session, document):
        log_unmapped_category(
            document_id=document.id,
            document_category="Whatever",
            file_name="file.pdf",
            ticker="TESTCO",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        assert "TESTCO" in logs[0].message

    def test_message_contains_raw_category_value(self, db_session, document):
        log_unmapped_category(
            document_id=document.id,
            document_category="BookClosure2019",
            file_name="file.pdf",
            ticker="TESTCO",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        assert "BookClosure2019" in logs[0].message

    def test_none_category_shown_as_null_placeholder(self, db_session, document):
        """When document_category is None, the message should show '(NULL)' not crash."""
        log_unmapped_category(
            document_id=document.id,
            document_category=None,
            file_name="file.pdf",
            ticker="TESTCO",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        assert "(NULL)" in logs[0].message


# ---------------------------------------------------------------------------
# log_failed — Requirement 7.4
# ---------------------------------------------------------------------------

class TestLogFailed:
    def test_inserts_one_row(self, db_session, document):
        log_failed(document_id=document.id, error=ValueError("oops"), session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1

    def test_status_is_failed(self, db_session, document):
        log_failed(document_id=document.id, error=ValueError("oops"), session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].status == "failed"

    def test_stage_is_feature_extraction(self, db_session, document):
        log_failed(document_id=document.id, error=Exception("e"), session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].stage == "feature_extraction"

    def test_document_id_preserved(self, db_session, document):
        log_failed(document_id=document.id, error=Exception("err"), session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id == document.id

    def test_document_id_none_is_allowed(self, db_session):
        """Period-level failures may have document_id=None."""
        log_failed(document_id=None, error=Exception("period fail"), session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id is None

    def test_message_contains_error_detail(self, db_session, document):
        log_failed(document_id=document.id, error=RuntimeError("JSONL missing"), session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "JSONL missing" in logs[0].message

    def test_message_truncated_to_1000_chars(self, db_session, document):
        long_error = Exception("e" * 2000)
        log_failed(document_id=document.id, error=long_error, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs[0].message) == 1000

    def test_message_not_truncated_when_under_limit(self, db_session, document):
        short_error = Exception("short error message")
        log_failed(document_id=document.id, error=short_error, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message == "short error message"

    def test_message_exactly_1000_chars_not_truncated(self, db_session, document):
        exact_error = Exception("a" * 1000)
        log_failed(document_id=document.id, error=exact_error, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs[0].message) == 1000

    def test_accepts_string_error(self, db_session, document):
        log_failed(document_id=document.id, error="plain string error", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "plain string error" in logs[0].message


# ---------------------------------------------------------------------------
# Savepoint pattern / graceful failure — Requirements 7.7, 7.8
# ---------------------------------------------------------------------------

class TestGracefulFailure:
    def test_no_exception_when_log_started_fails(self, db_session, document, capsys):
        """A DB error on the insert must not propagate — fire-and-forget."""
        with patch.object(db_session, "add", side_effect=Exception("DB down")):
            log_started(document_id=document.id, document_type="annual_report", session=db_session)
        # passes silently

    def test_warning_to_stderr_contains_processing_log_when_started_fails(self, db_session, document, capsys):
        with patch.object(db_session, "add", side_effect=Exception("simulated failure")):
            log_started(document_id=document.id, document_type="annual_report", session=db_session)

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "processing_log" in captured.err

    def test_no_exception_when_log_success_fails(self, db_session, document, capsys):
        with patch.object(db_session, "add", side_effect=Exception("DB down")):
            log_success(document_id=document.id, feature_type="financial", signals_populated=5, session=db_session)

    def test_warning_to_stderr_contains_processing_log_when_success_fails(self, db_session, document, capsys):
        with patch.object(db_session, "add", side_effect=Exception("simulated")):
            log_success(document_id=document.id, feature_type="financial", signals_populated=3, session=db_session)

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "processing_log" in captured.err

    def test_no_exception_when_log_governance_fails(self, db_session, capsys):
        result = GovernanceResult()
        with patch.object(db_session, "add", side_effect=Exception("DB down")):
            log_governance_success("TESTCO", "FY2019", "Q1", result, db_session)

    def test_no_exception_when_log_unmapped_fails(self, db_session, document, capsys):
        with patch.object(db_session, "add", side_effect=Exception("DB down")):
            log_unmapped_category(
                document_id=document.id,
                document_category="X",
                file_name="file.pdf",
                ticker="TESTCO",
                session=db_session,
            )

    def test_no_exception_when_log_failed_fails(self, db_session, document, capsys):
        with patch.object(db_session, "add", side_effect=Exception("DB down")):
            log_failed(document_id=document.id, error=ValueError("original"), session=db_session)

    def test_outer_session_still_usable_after_log_failure(self, db_session, document, capsys):
        """
        A log insert failure must not roll back or invalidate the outer session.
        Simulate: log fails → then a real document insert still works.
        """
        with patch.object(db_session, "add", side_effect=Exception("DB down")):
            log_started(document_id=document.id, document_type="annual_report", session=db_session)

        # Outer session should still be usable — add a new company row
        extra_company = Company(ticker="EXTRA", name="Extra Co")
        db_session.add(extra_company)
        db_session.flush()

        result = db_session.query(Company).filter_by(ticker="EXTRA").one()
        assert result is not None

    def test_log_failure_does_not_affect_subsequent_logs(self, db_session, document, capsys):
        """
        After one log call fails, subsequent log calls in the same session still work.
        """
        # First call: patched to fail
        with patch.object(db_session, "add", side_effect=Exception("DB down")):
            log_started(document_id=document.id, document_type="annual_report", session=db_session)

        # Second call: real — should insert normally
        log_failed(document_id=document.id, error="subsequent failure", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        # Only the second call should have produced a row
        assert len(logs) == 1
        assert logs[0].status == "failed"


# ---------------------------------------------------------------------------
# No started row for governance (per note in Req 7)
# ---------------------------------------------------------------------------

class TestGovernanceNoStartedRow:
    def test_governance_success_produces_exactly_one_row(self, db_session):
        result = GovernanceResult(
            board_meeting_count=4,
            director_change_count=0,
            appointment_count=1,
            resignation_count=2,
            press_release_count=3,
            clarification_count=5,
            agm_delay_flag=None,
        )
        log_governance_success("TESTCO", "FY2020", "Q2", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1

    def test_governance_success_row_is_not_started(self, db_session):
        result = GovernanceResult()
        log_governance_success("TESTCO", "FY2020", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].status != "started"

    def test_governance_success_row_has_null_document_id(self, db_session):
        result = GovernanceResult()
        log_governance_success("TESTCO", "FY2020", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id is None

    def test_governance_success_combined_with_document_log(self, db_session, document):
        """
        The governance log (document_id=NULL) and a document log (document_id set)
        can both be in the same session — verify they coexist correctly.
        """
        log_success(
            document_id=document.id,
            feature_type="financial",
            signals_populated=3,
            session=db_session,
        )
        result = GovernanceResult(board_meeting_count=2)
        log_governance_success("TESTCO", "FY2020", "Q1", result, db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 2

        doc_log = next(l for l in logs if l.document_id is not None)
        gov_log = next(l for l in logs if l.document_id is None)

        assert doc_log.document_id == document.id
        assert gov_log.status == "success"
        assert "governance extracted" in gov_log.message
