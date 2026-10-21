"""
Unit tests for src/embedding/embedding_logger.py

Uses SQLAlchemy in-memory SQLite so no running PostgreSQL is required.

Only the three tables this component touches are created:
  companies, documents, processing_log

Other models (extracted_features, predictions, shareholding_patterns) use
PostgreSQL-specific JSONB columns that SQLite cannot compile, so they are
excluded via a scoped MetaData — exactly the same pattern used in
tests/preprocessing/test_extraction_logger.py.
"""

import pytest

from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.orm import Session
from unittest.mock import patch

from models.base import Base
from models.company import Company
from models.document import Document
from models.processing_log import ProcessingLog

from embedding.embedding_logger import (
    log_embedding_started,
    log_embedding_success,
    log_embedding_failed,
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
    c = Company(ticker="EMBTEST", name="Embedding Test Company")
    db_session.add(c)
    db_session.flush()
    return c


@pytest.fixture()
def document(db_session, company):
    """Insert a minimal Document row and return the ORM object."""
    d = Document(
        company_id=company.id,
        file_name="test_embed.pdf",
        file_path="data/raw/EMBTEST/annual_reports/test_embed.pdf",
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
# 1. test_log_embedding_started
# ---------------------------------------------------------------------------


class TestLogEmbeddingStarted:
    def test_log_embedding_started(self, db_session, document):
        """Inserts row with stage='embedding', status='started', message=None."""
        log_embedding_started(document_id=document.id, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "embedding"
        assert logs[0].status == "started"
        assert logs[0].message is None
        assert logs[0].document_id == document.id


# ---------------------------------------------------------------------------
# 2. test_log_embedding_success_with_chunks
# ---------------------------------------------------------------------------


class TestLogEmbeddingSuccessWithChunks:
    def test_log_embedding_success_with_chunks(self, db_session, document):
        """chunk_count=5 → message='embedded: 5 chunks'"""
        log_embedding_success(document_id=document.id, chunk_count=5, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "embedding"
        assert logs[0].status == "success"
        assert logs[0].message == "embedded: 5 chunks"
        assert logs[0].document_id == document.id


# ---------------------------------------------------------------------------
# 3. test_log_embedding_success_zero_chunks
# ---------------------------------------------------------------------------


class TestLogEmbeddingSuccessZeroChunks:
    def test_log_embedding_success_zero_chunks(self, db_session, document):
        """chunk_count=0 → message='no_chunks: nothing to embed'"""
        log_embedding_success(document_id=document.id, chunk_count=0, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "embedding"
        assert logs[0].status == "success"
        assert logs[0].message == "no_chunks: nothing to embed"


# ---------------------------------------------------------------------------
# 4. test_log_embedding_failed_message_truncated
# ---------------------------------------------------------------------------


class TestLogEmbeddingFailedMessageTruncated:
    def test_log_embedding_failed_message_truncated(self, db_session, document):
        """Error string of 2000 chars → stored message is ≤ 1000 chars."""
        long_msg = "x" * 2000
        err = ValueError(long_msg)
        log_embedding_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].stage == "embedding"
        assert logs[0].status == "failed"
        assert len(logs[0].message) <= 1000
        assert len(logs[0].message) == 1000


# ---------------------------------------------------------------------------
# 5. test_log_embedding_failed_exception_object
# ---------------------------------------------------------------------------


class TestLogEmbeddingFailedExceptionObject:
    def test_log_embedding_failed_exception_object(self, db_session, document):
        """Pass an Exception instance; assert message == str(exception)[:1000]."""
        err = Exception("embedding failed due to OOM")
        log_embedding_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 1
        assert logs[0].message == str(err)[:1000]
        assert logs[0].message == "embedding failed due to OOM"

    def test_log_embedding_failed_message_equals_str_error_truncated(self, db_session, document):
        """Verify truncation formula: message == str(error)[:1000] exactly."""
        err = RuntimeError("z" * 1500)
        log_embedding_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert logs[0].message == str(err)[:1000]


# ---------------------------------------------------------------------------
# 6. test_savepoint_rollback_outer_unaffected
# ---------------------------------------------------------------------------


class TestSavepointRollbackOuterUnaffected:
    def test_savepoint_rollback_outer_unaffected(self, db_session, document, capsys):
        """Mock session.add to raise; assert no exception propagated, warning to stderr,
        outer session state is unaffected."""
        with patch.object(db_session, "add", side_effect=Exception("DB write error")):
            # Must not raise
            log_embedding_started(document_id=document.id, session=db_session)

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "processing_log" in captured.err

        # Outer session unaffected — the document we inserted earlier is still accessible
        db_session.flush()
        retrieved = db_session.get(Document, document.id)
        assert retrieved is not None
        assert retrieved.file_name == "test_embed.pdf"

    def test_no_exception_on_success_insert_failure(self, db_session, document):
        """log_embedding_success must not raise even when insert fails."""
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            log_embedding_success(document_id=document.id, chunk_count=3, session=db_session)
        # no exception = pass

    def test_no_exception_on_failed_insert_failure(self, db_session, document):
        """log_embedding_failed must not raise even when insert fails."""
        err = ValueError("original error")
        with patch.object(db_session, "add", side_effect=Exception("DB is down")):
            log_embedding_failed(document_id=document.id, error=err, session=db_session)
        # no exception = pass

    def test_warning_contains_stage_on_failure(self, db_session, document, capsys):
        """Warning message should reference the stage name."""
        with patch.object(db_session, "add", side_effect=Exception("simulated failure")):
            log_embedding_success(document_id=document.id, chunk_count=10, session=db_session)

        captured = capsys.readouterr()
        assert "WARNING" in captured.err


# ---------------------------------------------------------------------------
# 7. test_full_sequence_started_then_success
# ---------------------------------------------------------------------------


class TestFullSequenceStartedThenSuccess:
    def test_full_sequence_started_then_success(self, db_session, document):
        """Call started then success; assert two rows in processing_log in order."""
        log_embedding_started(document_id=document.id, session=db_session)
        log_embedding_success(document_id=document.id, chunk_count=12, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 2

        started, success = logs
        assert started.stage == "embedding"
        assert started.status == "started"
        assert started.message is None
        assert started.document_id == document.id

        assert success.stage == "embedding"
        assert success.status == "success"
        assert success.message == "embedded: 12 chunks"
        assert success.document_id == document.id


# ---------------------------------------------------------------------------
# 8. test_full_sequence_started_then_failed
# ---------------------------------------------------------------------------


class TestFullSequenceStartedThenFailed:
    def test_full_sequence_started_then_failed(self, db_session, document):
        """Call started then failed; assert two rows in processing_log in order."""
        err = RuntimeError("model encode failed")
        log_embedding_started(document_id=document.id, session=db_session)
        log_embedding_failed(document_id=document.id, error=err, session=db_session)
        db_session.flush()

        logs = _all_logs(db_session)
        assert len(logs) == 2

        started, failed = logs
        assert started.stage == "embedding"
        assert started.status == "started"
        assert started.message is None

        assert failed.stage == "embedding"
        assert failed.status == "failed"
        assert "model encode failed" in failed.message
        assert failed.document_id == document.id
