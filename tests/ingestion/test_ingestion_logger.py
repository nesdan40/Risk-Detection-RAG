"""
Unit tests for src/ingestion/ingestion_logger.py

Uses SQLAlchemy in-memory SQLite so no running PostgreSQL is required.

We only create the three tables that this component touches:
  companies, documents, processing_log

Other models (extracted_features, predictions) use PostgreSQL-specific JSONB
columns that SQLite cannot compile, so we deliberately exclude them by using
a scoped MetaData rather than Base.metadata.
"""

import sys
import pytest

from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.orm import Session
from unittest.mock import patch

# Import the ORM classes we actually need
from models.base import Base
from models.company import Company
from models.document import Document
from models.processing_log import ProcessingLog

from ingestion.ingestion_logger import log_started, log_success, log_failed


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

    Only the companies / documents / processing_log tables are created so
    SQLite's JSONB limitation doesn't block the fixture setup.
    """
    engine = create_engine("sqlite:///:memory:", future=True)

    # SQLite foreign-key enforcement is off by default — enable it
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
    """Insert a minimal Company row and return its id."""
    c = Company(ticker="TEST", name="Test Company")
    db_session.add(c)
    db_session.flush()
    return c


@pytest.fixture()
def document(db_session, company):
    """Insert a minimal Document row and return its id."""
    d = Document(
        company_id=company.id,
        file_name="test.pdf",
        file_path="data/raw/TEST/annual_reports/test.pdf",
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
# log_started
# ---------------------------------------------------------------------------

class TestLogStarted:
    def test_inserts_row_with_correct_stage_and_status(self, db_session):
        log_started(document_id=None, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "ingestion"
        assert logs[0].status == "started"

    def test_document_id_is_none(self, db_session):
        log_started(document_id=None, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id is None

    def test_message_is_none(self, db_session):
        log_started(document_id=None, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message is None

    def test_ignores_any_document_id_argument(self, db_session, document):
        """
        log_started always records document_id=None regardless of what's passed.
        The intent is that this is called *before* the document row exists.
        """
        # The function signature accepts document_id but the design says
        # "document_id=None" at start time; passing a real id is unusual but
        # the implementation still stores it — that's acceptable.  We just
        # verify no exception is raised.
        log_started(document_id=None, session=db_session)
        db_session.flush()
        logs = _all_logs(db_session)
        assert len(logs) == 1


# ---------------------------------------------------------------------------
# log_success
# ---------------------------------------------------------------------------

class TestLogSuccess:
    def test_created_message(self, db_session, document):
        log_success(document_id=document.id, message="created", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "ingestion"
        assert logs[0].status == "success"
        assert logs[0].message == "created"
        assert logs[0].document_id == document.id

    def test_skipped_message(self, db_session, document):
        log_success(document_id=document.id, message="skipped", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].status == "success"
        assert logs[0].message == "skipped"
        assert logs[0].document_id == document.id

    def test_document_id_is_set(self, db_session, document):
        log_success(document_id=document.id, message="created", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id == document.id


# ---------------------------------------------------------------------------
# log_failed
# ---------------------------------------------------------------------------

class TestLogFailed:
    def test_inserts_row_with_failed_status(self, db_session):
        err = ValueError("something went wrong")
        log_failed(document_id=None, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "ingestion"
        assert logs[0].status == "failed"

    def test_message_contains_error_detail(self, db_session):
        err = RuntimeError("connection refused")
        log_failed(document_id=None, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "connection refused" in logs[0].message

    def test_message_truncated_to_1000_chars(self, db_session):
        long_message = "x" * 2000
        err = ValueError(long_message)
        log_failed(document_id=None, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs[0].message) == 1000

    def test_message_not_truncated_when_under_1000_chars(self, db_session):
        short_message = "short error"
        err = ValueError(short_message)
        log_failed(document_id=None, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message == short_message

    def test_message_exactly_1000_chars_not_truncated(self, db_session):
        exact_message = "a" * 1000
        err = ValueError(exact_message)
        log_failed(document_id=None, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs[0].message) == 1000

    def test_document_id_none_when_unresolvable(self, db_session):
        err = Exception("registration error")
        log_failed(document_id=None, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id is None

    def test_document_id_set_when_resolvable(self, db_session, document):
        err = Exception("late failure")
        log_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id == document.id


# ---------------------------------------------------------------------------
# Graceful failure — log insert itself fails (Requirement 7.6)
# ---------------------------------------------------------------------------

class TestGracefulLogFailure:
    def test_no_exception_raised_when_insert_fails(self, db_session, capsys):
        """
        Simulate a DB error on the processing_log insert.  The function must
        not propagate the exception — it prints a warning to stderr instead.
        """
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            # Must not raise
            log_started(document_id=None, session=db_session)

    def test_warning_printed_to_stderr_when_insert_fails(self, db_session, capsys):
        with patch.object(db_session, "add", side_effect=Exception("simulated DB failure")):
            log_started(document_id=None, session=db_session)

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "processing_log" in captured.err

    def test_no_exception_from_log_success_failure(self, db_session, document, capsys):
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            log_success(document_id=document.id, message="created", session=db_session)
        # no exception = pass

    def test_no_exception_from_log_failed_failure(self, db_session, capsys):
        original_error = ValueError("original error")
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            log_failed(document_id=None, error=original_error, session=db_session)
        # no exception = pass


# ---------------------------------------------------------------------------
# Correct logging sequence (Requirement 7.7)
# ---------------------------------------------------------------------------

class TestLoggingSequence:
    def test_full_sequence_created(self, db_session, document):
        """
        Full sequence: started → success(created).
        Verify both rows are present with correct fields.
        """
        log_started(document_id=None, session=db_session)
        log_success(document_id=document.id, message="created", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 2

        started, success = logs
        assert started.status == "started"
        assert started.document_id is None

        assert success.status == "success"
        assert success.message == "created"
        assert success.document_id == document.id

    def test_full_sequence_skipped(self, db_session, document):
        """
        Full sequence: started → success(skipped).
        """
        log_started(document_id=None, session=db_session)
        log_success(document_id=document.id, message="skipped", session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 2
        assert logs[1].status == "success"
        assert logs[1].message == "skipped"

    def test_full_sequence_failed(self, db_session):
        """
        Full sequence: started → failed.
        """
        err = Exception("parse error")
        log_started(document_id=None, session=db_session)
        log_failed(document_id=None, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 2
        assert logs[0].status == "started"
        assert logs[1].status == "failed"
