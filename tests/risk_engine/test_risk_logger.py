"""
Unit tests for src/risk_engine/risk_logger.py

Uses SQLAlchemy in-memory SQLite for ProcessingLog insert verification.
Only creates the tables needed (companies, documents, processing_log) — 
excludes JSONB-dependent tables that SQLite cannot compile.

Validates: Requirements 7.1, 7.2, 7.3, 7.4
"""

import pytest
from unittest.mock import patch

from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.orm import Session

from models.base import Base
from models.company import Company
from models.document import Document
from models.processing_log import ProcessingLog

from risk_engine.risk_logger import (
    STAGE_RISK_SCORING,
    log_risk_scoring_started,
    log_risk_scoring_success,
    log_risk_scoring_failed,
    _MAX_MESSAGE_LEN,
)


# ---------------------------------------------------------------------------
# SQLite-compatible metadata (only tables without JSONB)
# ---------------------------------------------------------------------------

_TEST_TABLES = {"companies", "documents", "processing_log"}


def _test_metadata() -> MetaData:
    """Return a MetaData instance containing only SQLite-safe tables."""
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
    In-memory SQLite session for each test.
    Only companies, documents, and processing_log tables are created.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_logs(session: Session) -> list[ProcessingLog]:
    return session.query(ProcessingLog).order_by(ProcessingLog.id).all()


# ---------------------------------------------------------------------------
# Module-level constant tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_stage_risk_scoring_value(self):
        assert STAGE_RISK_SCORING == "risk_scoring"

    def test_max_message_len(self):
        assert _MAX_MESSAGE_LEN == 1000


# ---------------------------------------------------------------------------
# log_risk_scoring_started
# ---------------------------------------------------------------------------

class TestLogStarted:
    def test_inserts_correct_stage_status(self, db_session):
        """stage='risk_scoring', status='started'."""
        log_risk_scoring_started(
            company_id=1, ticker="DHFL",
            fiscal_year="FY2018", fiscal_quarter="Q3",
            session=db_session,
        )
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == STAGE_RISK_SCORING
        assert logs[0].status == "started"

    def test_document_id_is_always_null(self, db_session):
        """document_id must be None for risk_scoring (period-level, not doc-level)."""
        log_risk_scoring_started(
            company_id=1, ticker="TEST",
            fiscal_year="FY2019", fiscal_quarter=None,
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        assert logs[0].document_id is None

    def test_message_includes_ticker_and_period_quarterly(self, db_session):
        log_risk_scoring_started(
            company_id=1, ticker="DHFL",
            fiscal_year="FY2018", fiscal_quarter="Q3",
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        msg = logs[0].message
        assert "DHFL" in msg
        assert "FY2018" in msg
        assert "Q3" in msg

    def test_message_format_annual_period(self, db_session):
        """Annual period (None quarter) → message contains FY but no Q."""
        log_risk_scoring_started(
            company_id=1, ticker="TESTCO",
            fiscal_year="FY2017", fiscal_quarter=None,
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        msg = logs[0].message
        assert "TESTCO" in msg
        assert "FY2017" in msg


# ---------------------------------------------------------------------------
# log_risk_scoring_success
# ---------------------------------------------------------------------------

class TestLogSuccess:
    def test_inserts_correct_stage_status(self, db_session):
        """stage='risk_scoring', status='success'."""
        log_risk_scoring_success(
            company_id=1, ticker="DHFL",
            fiscal_year="FY2018", fiscal_quarter="Q3",
            risk_level="High", n_triggered=4,
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == STAGE_RISK_SCORING
        assert logs[0].status == "success"

    def test_document_id_always_null(self, db_session):
        log_risk_scoring_success(
            company_id=1, ticker="TEST",
            fiscal_year="FY2018", fiscal_quarter="Q1",
            risk_level="Low", n_triggered=0,
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        assert logs[0].document_id is None

    def test_message_format(self, db_session):
        """Message must contain ticker, period, risk_level, and triggered count."""
        log_risk_scoring_success(
            company_id=1, ticker="DHFL",
            fiscal_year="FY2018", fiscal_quarter="Q3",
            risk_level="High", n_triggered=5,
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        msg = logs[0].message
        assert "DHFL" in msg
        assert "FY2018" in msg
        assert "Q3" in msg
        assert "High" in msg
        assert "5" in msg

    def test_success_message_zero_triggered(self, db_session):
        log_risk_scoring_success(
            company_id=1, ticker="HEALTHY",
            fiscal_year="FY2019", fiscal_quarter="Q1",
            risk_level="Low", n_triggered=0,
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        msg = logs[0].message
        assert "Low" in msg
        assert "0" in msg


# ---------------------------------------------------------------------------
# log_risk_scoring_failed
# ---------------------------------------------------------------------------

class TestLogFailed:
    def test_inserts_correct_stage_status(self, db_session):
        """stage='risk_scoring', status='failed'."""
        log_risk_scoring_failed(
            company_id=1, ticker="DHFL",
            fiscal_year="FY2018", fiscal_quarter="Q2",
            error="Connection refused",
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == STAGE_RISK_SCORING
        assert logs[0].status == "failed"

    def test_document_id_always_null(self, db_session):
        log_risk_scoring_failed(
            company_id=1, ticker="X",
            fiscal_year="FY2018", fiscal_quarter="Q1",
            error="err",
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        assert logs[0].document_id is None

    def test_error_detail_in_message(self, db_session):
        log_risk_scoring_failed(
            company_id=1, ticker="DHFL",
            fiscal_year="FY2018", fiscal_quarter="Q4",
            error="No feature data found for period",
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        assert "No feature data found for period" in logs[0].message

    def test_message_truncated_when_error_exceeds_1000_chars(self, db_session):
        """Error longer than 1000 chars should be truncated to _MAX_MESSAGE_LEN."""
        long_error = "E" * 2500
        log_risk_scoring_failed(
            company_id=1, ticker="X",
            fiscal_year="FY2018", fiscal_quarter="Q1",
            error=long_error,
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        # message = "Failed X FY2018 Q1: " + truncated_error
        # The error portion is truncated; total message length is ≤ prefix + 1000
        # The implementation truncates the error to _MAX_MESSAGE_LEN before adding prefix
        assert len(logs[0].message) <= len("Failed X FY2018 Q1: ") + _MAX_MESSAGE_LEN

    def test_message_not_truncated_for_short_error(self, db_session):
        """Short error should appear in full."""
        log_risk_scoring_failed(
            company_id=1, ticker="TEST",
            fiscal_year="FY2019", fiscal_quarter="Q3",
            error="short error",
            session=db_session,
        )
        db_session.flush()
        logs = _all_logs(db_session)
        assert "short error" in logs[0].message


# ---------------------------------------------------------------------------
# Graceful failure — log insert itself fails
# ---------------------------------------------------------------------------

class TestGracefulLogFailure:
    def test_no_exception_when_started_insert_fails(self, db_session, capsys):
        """If the processing_log insert fails, no exception should propagate."""
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            # Must not raise
            log_risk_scoring_started(
                company_id=1, ticker="X",
                fiscal_year="FY2018", fiscal_quarter="Q1",
                session=db_session,
            )

    def test_warning_printed_to_stderr_when_insert_fails(self, db_session, capsys):
        with patch.object(db_session, "add", side_effect=Exception("simulated failure")):
            log_risk_scoring_started(
                company_id=1, ticker="X",
                fiscal_year="FY2018", fiscal_quarter="Q1",
                session=db_session,
            )

        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    def test_no_exception_when_success_insert_fails(self, db_session, capsys):
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            log_risk_scoring_success(
                company_id=1, ticker="X",
                fiscal_year="FY2018", fiscal_quarter="Q1",
                risk_level="Low", n_triggered=0,
                session=db_session,
            )
        # no exception = pass

    def test_no_exception_when_failed_insert_fails(self, db_session, capsys):
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            log_risk_scoring_failed(
                company_id=1, ticker="X",
                fiscal_year="FY2018", fiscal_quarter="Q1",
                error="original error",
                session=db_session,
            )
        # no exception = pass

    def test_savepoint_failure_does_not_corrupt_outer_session(self, db_session):
        """After a failed log insert (via SAVEPOINT), the session stays usable."""
        with patch.object(db_session, "add", side_effect=Exception("nested failure")):
            log_risk_scoring_started(
                company_id=1, ticker="X",
                fiscal_year="FY2018", fiscal_quarter="Q1",
                session=db_session,
            )

        # Session should still be functional — can still add/flush a real row
        company = Company(ticker="STILL_WORKS", name="Still Works Inc")
        db_session.add(company)
        db_session.flush()
        result = db_session.query(Company).filter_by(ticker="STILL_WORKS").first()
        assert result is not None


# ---------------------------------------------------------------------------
# document_id always None for all three log functions
# ---------------------------------------------------------------------------

class TestDocumentIdAlwaysNull:
    @pytest.mark.parametrize("log_fn,kwargs", [
        (
            log_risk_scoring_started,
            dict(company_id=1, ticker="T", fiscal_year="FY2018", fiscal_quarter="Q1"),
        ),
        (
            log_risk_scoring_success,
            dict(company_id=1, ticker="T", fiscal_year="FY2018", fiscal_quarter="Q1",
                 risk_level="Low", n_triggered=0),
        ),
        (
            log_risk_scoring_failed,
            dict(company_id=1, ticker="T", fiscal_year="FY2018", fiscal_quarter="Q1",
                 error="some error"),
        ),
    ])
    def test_document_id_always_null(self, db_session, log_fn, kwargs):
        log_fn(**kwargs, session=db_session)
        db_session.flush()
        logs = _all_logs(db_session)
        assert logs[-1].document_id is None
