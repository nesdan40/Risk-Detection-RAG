"""
Unit tests for src/preprocessing/extraction_logger.py

Uses SQLAlchemy in-memory SQLite so no running PostgreSQL is required.

Only the three tables this component touches are created:
  companies, documents, processing_log

Other models (extracted_features, predictions) use PostgreSQL-specific JSONB
columns that SQLite cannot compile, so they are excluded via a scoped MetaData
— exactly the same pattern used in tests/ingestion/test_ingestion_logger.py.
"""

import pytest

from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.orm import Session
from unittest.mock import patch

from models.base import Base
from models.company import Company
from models.document import Document
from models.processing_log import ProcessingLog

from preprocessing.extraction_logger import (
    log_extraction_started,
    log_extraction_success,
    log_extraction_failed,
    log_chunking_started,
    log_chunking_success,
    log_chunking_failed,
)


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
    SQLite's JSONB limitation doesn't block fixture setup.
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
    """Insert a minimal Company row and return the ORM object."""
    c = Company(ticker="TEST", name="Test Company")
    db_session.add(c)
    db_session.flush()
    return c


@pytest.fixture()
def document(db_session, company):
    """Insert a minimal Document row and return the ORM object."""
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
# 1. log_extraction_started
# ---------------------------------------------------------------------------


class TestLogExtractionStarted:
    def test_inserts_row_with_correct_stage_and_status(self, db_session, document):
        log_extraction_started(document_id=document.id, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "text_extraction"
        assert logs[0].status == "started"

    def test_document_id_is_stored(self, db_session, document):
        log_extraction_started(document_id=document.id, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id == document.id

    def test_message_is_none(self, db_session, document):
        log_extraction_started(document_id=document.id, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message is None


# ---------------------------------------------------------------------------
# 2. log_extraction_success — above threshold (char_count >= 50)
# ---------------------------------------------------------------------------


class TestLogExtractionSuccessAboveThreshold:
    def test_message_is_extracted_at_threshold(self, db_session, document):
        log_extraction_success(document_id=document.id, char_count=50, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].status == "success"
        assert logs[0].stage == "text_extraction"
        assert logs[0].message == "extracted"

    def test_message_is_extracted_well_above_threshold(self, db_session, document):
        log_extraction_success(document_id=document.id, char_count=5000, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message == "extracted"

    def test_document_id_is_stored(self, db_session, document):
        log_extraction_success(document_id=document.id, char_count=100, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id == document.id


# ---------------------------------------------------------------------------
# 3. log_extraction_success — below threshold (char_count < 50)
# ---------------------------------------------------------------------------


class TestLogExtractionSuccessBelowThreshold:
    def test_message_contains_low_yield(self, db_session, document):
        log_extraction_success(document_id=document.id, char_count=10, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "low_yield" in logs[0].message

    def test_message_contains_char_count(self, db_session, document):
        log_extraction_success(document_id=document.id, char_count=23, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "23" in logs[0].message

    def test_status_is_still_success(self, db_session, document):
        log_extraction_success(document_id=document.id, char_count=0, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].status == "success"

    def test_message_at_boundary_49(self, db_session, document):
        log_extraction_success(document_id=document.id, char_count=49, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "low_yield" in logs[0].message
        assert "49" in logs[0].message


# ---------------------------------------------------------------------------
# 4. log_extraction_failed
# ---------------------------------------------------------------------------


class TestLogExtractionFailed:
    def test_inserts_row_with_failed_status(self, db_session, document):
        err = ValueError("PDF parse error")
        log_extraction_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "text_extraction"
        assert logs[0].status == "failed"

    def test_message_contains_error_detail(self, db_session, document):
        err = RuntimeError("file not readable")
        log_extraction_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "file not readable" in logs[0].message

    def test_message_truncated_to_1000_chars(self, db_session, document):
        long_msg = "e" * 2000
        err = ValueError(long_msg)
        log_extraction_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs[0].message) == 1000

    def test_message_not_truncated_when_under_limit(self, db_session, document):
        err = ValueError("short error")
        log_extraction_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message == "short error"

    def test_message_exactly_1000_chars_unchanged(self, db_session, document):
        exact_msg = "x" * 1000
        err = ValueError(exact_msg)
        log_extraction_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs[0].message) == 1000


# ---------------------------------------------------------------------------
# 5. log_chunking_started
# ---------------------------------------------------------------------------


class TestLogChunkingStarted:
    def test_inserts_row_with_correct_stage_and_status(self, db_session, document):
        log_chunking_started(document_id=document.id, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "chunking"
        assert logs[0].status == "started"

    def test_document_id_is_stored(self, db_session, document):
        log_chunking_started(document_id=document.id, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id == document.id

    def test_message_is_none(self, db_session, document):
        log_chunking_started(document_id=document.id, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message is None


# ---------------------------------------------------------------------------
# 6. log_chunking_success — chunk_count > 0
# ---------------------------------------------------------------------------


class TestLogChunkingSuccessWithChunks:
    def test_message_format_chunks_written(self, db_session, document):
        log_chunking_success(document_id=document.id, chunk_count=7, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "chunking"
        assert logs[0].status == "success"
        assert logs[0].message == "chunks_written: 7"

    def test_chunk_count_in_message(self, db_session, document):
        log_chunking_success(document_id=document.id, chunk_count=42, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message == "chunks_written: 42"

    def test_document_id_stored(self, db_session, document):
        log_chunking_success(document_id=document.id, chunk_count=3, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].document_id == document.id


# ---------------------------------------------------------------------------
# 7. log_chunking_success — chunk_count == 0
# ---------------------------------------------------------------------------


class TestLogChunkingSuccessZeroChunks:
    def test_message_is_no_chunks(self, db_session, document):
        log_chunking_success(document_id=document.id, chunk_count=0, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].status == "success"
        assert logs[0].message == "no_chunks: text below threshold"

    def test_stage_is_chunking(self, db_session, document):
        log_chunking_success(document_id=document.id, chunk_count=0, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].stage == "chunking"


# ---------------------------------------------------------------------------
# 8. log_chunking_failed
# ---------------------------------------------------------------------------


class TestLogChunkingFailed:
    def test_inserts_row_with_failed_status(self, db_session, document):
        err = OSError("disk full")
        log_chunking_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "chunking"
        assert logs[0].status == "failed"

    def test_message_contains_error_detail(self, db_session, document):
        err = ValueError("chunk write failed")
        log_chunking_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert "chunk write failed" in logs[0].message

    def test_message_truncated_to_1000_chars(self, db_session, document):
        long_msg = "c" * 2000
        err = OSError(long_msg)
        log_chunking_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs[0].message) == 1000

    def test_message_not_truncated_when_under_limit(self, db_session, document):
        err = OSError("short chunk error")
        log_chunking_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message == "short chunk error"


# ---------------------------------------------------------------------------
# 9. Savepoint rollback — log insert itself fails (Requirements 3.6, 7.6)
# ---------------------------------------------------------------------------


class TestSavepointRollbackNoException:
    def test_no_exception_raised_when_insert_fails(self, db_session, document, capsys):
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            # Must not raise
            log_extraction_started(document_id=document.id, session=db_session)

    def test_warning_printed_to_stderr_on_insert_failure(self, db_session, document, capsys):
        with patch.object(db_session, "add", side_effect=Exception("simulated failure")):
            log_extraction_started(document_id=document.id, session=db_session)

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "processing_log" in captured.err

    def test_no_exception_from_success_insert_failure(self, db_session, document, capsys):
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            log_extraction_success(document_id=document.id, char_count=200, session=db_session)
        # no exception = pass

    def test_no_exception_from_failed_insert_failure(self, db_session, document, capsys):
        err = ValueError("original error")
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            log_extraction_failed(document_id=document.id, error=err, session=db_session)
        # no exception = pass

    def test_no_exception_from_chunking_insert_failure(self, db_session, document, capsys):
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            log_chunking_started(document_id=document.id, session=db_session)
        # no exception = pass


# ---------------------------------------------------------------------------
# 10. Sequence: extraction started → success (Requirement 7.7)
# ---------------------------------------------------------------------------


class TestExtractionStartedSuccessSequence:
    def test_two_rows_in_correct_order(self, db_session, document):
        log_extraction_started(document_id=document.id, session=db_session)
        log_extraction_success(document_id=document.id, char_count=300, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 2

        started, success = logs
        assert started.stage == "text_extraction"
        assert started.status == "started"
        assert started.message is None

        assert success.stage == "text_extraction"
        assert success.status == "success"
        assert success.message == "extracted"
        assert success.document_id == document.id


# ---------------------------------------------------------------------------
# 11. Sequence: extraction started → failed (Requirement 7.7)
# ---------------------------------------------------------------------------


class TestExtractionStartedFailedSequence:
    def test_two_rows_in_correct_order(self, db_session, document):
        err = Exception("parse error")
        log_extraction_started(document_id=document.id, session=db_session)
        log_extraction_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 2

        started, failed = logs
        assert started.status == "started"
        assert failed.status == "failed"
        assert "parse error" in failed.message


# ---------------------------------------------------------------------------
# 12. Sequence: chunking started → success (Requirement 7.7)
# ---------------------------------------------------------------------------


class TestChunkingStartedSuccessSequence:
    def test_two_rows_in_correct_order(self, db_session, document):
        log_chunking_started(document_id=document.id, session=db_session)
        log_chunking_success(document_id=document.id, chunk_count=15, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 2

        started, success = logs
        assert started.stage == "chunking"
        assert started.status == "started"
        assert started.message is None

        assert success.stage == "chunking"
        assert success.status == "success"
        assert success.message == "chunks_written: 15"
        assert success.document_id == document.id


# ---------------------------------------------------------------------------
# 13. Sequence: chunking started → failed (Requirement 7.7)
# ---------------------------------------------------------------------------


class TestChunkingStartedFailedSequence:
    def test_two_rows_in_correct_order(self, db_session, document):
        err = OSError("write failed")
        log_chunking_started(document_id=document.id, session=db_session)
        log_chunking_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 2

        started, failed = logs
        assert started.stage == "chunking"
        assert started.status == "started"

        assert failed.stage == "chunking"
        assert failed.status == "failed"
        assert "write failed" in failed.message
