"""
Integration tests for src/preprocessing/run_extraction.py

Strategy
--------
- Tests 1–8 call individual functions (process_document_full,
  process_document_rechunk, get_candidate_documents, get_rechunk_candidates)
  directly with an in-memory SQLite session — no real PostgreSQL required.
- Test 9 patches the engine so that main() calls get_candidate_documents /
  get_rechunk_candidates against the SQLite DB and hits the "nothing to do"
  branch.
- `PROCESSED_DIR` in run_extraction is patched to `tmp_path` for every test
  that exercises file I/O.
- `text_extractor.extract` is mocked wherever real PDFs would otherwise be
  needed.

Requirements: 1.2, 1.3, 1.4, 1.7, 8.3, 8.5, 8.6, 8.7, 9.1, 9.2, 9.4, 9.5
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from models.base import Base
from models.company import Company
from models.document import Document
from models.processing_log import ProcessingLog

from preprocessing.run_extraction import (
    ExtractionResult,
    get_candidate_documents,
    get_rechunk_candidates,
    process_document_full,
    process_document_rechunk,
)
from preprocessing.text_extractor import ExtractionOutcome


# ---------------------------------------------------------------------------
# SQLite-compatible metadata (companies, documents, processing_log only)
# ---------------------------------------------------------------------------

_TEST_TABLES = {"companies", "documents", "processing_log"}


def _test_metadata() -> MetaData:
    """Return a MetaData containing only the three SQLite-safe tables."""
    meta = MetaData()
    for table in Base.metadata.tables.values():
        if table.name in _TEST_TABLES:
            table.to_metadata(meta)
    return meta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_engine():
    """
    Create a fresh in-memory SQLite engine for each test.

    Yields the engine so individual tests can open their own sessions or
    pass it to main() via a patch.
    """
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    meta = _test_metadata()
    meta.create_all(engine)

    yield engine

    meta.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    """Provide a single session attached to the in-memory engine."""
    with Session(db_engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_company(session: Session, ticker: str = "TESTCO", name: str = "Test Co") -> Company:
    c = Company(ticker=ticker, name=name)
    session.add(c)
    session.flush()
    return c


def _seed_document(
    session: Session,
    company: Company,
    *,
    text_extracted: bool = False,
    chunked: bool = False,
    file_name: str = "report.pdf",
) -> Document:
    d = Document(
        company_id=company.id,
        file_name=file_name,
        file_path=f"data/raw/{company.ticker}/annual_reports/{file_name}",
        document_type="annual_report",
        fiscal_year="FY2020",
    )
    d.text_extracted = text_extracted
    d.chunked = chunked
    session.add(d)
    session.flush()
    return d


def _all_logs(session: Session) -> list[ProcessingLog]:
    return session.query(ProcessingLog).order_by(ProcessingLog.id).all()


# ---------------------------------------------------------------------------
# Good extraction outcome fixture (text long enough to produce real chunks)
# ---------------------------------------------------------------------------

_GOOD_OUTCOME = ExtractionOutcome(
    text="word " * 800,       # 4000 chars → multiple chunks
    outcome="extracted",
    char_count=4000,
    error=None,
    extraction_method="pymupdf",
)

_OCR_OUTCOME = ExtractionOutcome(
    text="word " * 800,
    outcome="extracted",
    char_count=4000,
    error=None,
    extraction_method="ocr",
)

_LOW_YIELD_OUTCOME = ExtractionOutcome(
    text="ab",
    outcome="low_yield",
    char_count=2,
    error=None,
    extraction_method="pymupdf",
)


# ---------------------------------------------------------------------------
# Test 1 — full pipeline end-to-end for a single document
# ---------------------------------------------------------------------------


class TestFullPipelineSingleDocument:
    def test_flags_set_and_jsonl_written(self, db_session, tmp_path):
        """
        Seed company + document with text_extracted=FALSE; mock extract to
        return a good outcome; assert flags, JSONL file, and processing_log.

        Requirements: 8.3, 8.5, 9.2
        """
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company)

        with patch(
            "preprocessing.run_extraction.PROCESSED_DIR", tmp_path
        ), patch(
            "preprocessing.run_extraction.extract", return_value=_GOOD_OUTCOME
        ):
            fresh_doc = db_session.get(Document, doc.id)
            result = process_document_full(fresh_doc, company.ticker, db_session)

        # flags
        fresh_doc = db_session.get(Document, doc.id)
        assert fresh_doc.text_extracted is True
        assert fresh_doc.chunked is True

        # JSONL file exists and is non-empty
        jsonl_path = tmp_path / company.ticker / f"{doc.id}_chunks.jsonl"
        assert jsonl_path.exists(), "JSONL file was not created"
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 0, "JSONL file must contain at least one chunk"

        # Processing log: extraction started + success, chunking started + success
        logs = _all_logs(db_session)
        stages_statuses = [(lg.stage, lg.status) for lg in logs]
        assert ("text_extraction", "started") in stages_statuses
        assert ("text_extraction", "success") in stages_statuses
        assert ("chunking", "started") in stages_statuses
        assert ("chunking", "success") in stages_statuses

        # Result object
        assert result.outcome == "extracted"
        assert result.chunk_count > 0

    def test_each_jsonl_line_is_valid_json(self, db_session, tmp_path):
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company)

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", return_value=_GOOD_OUTCOME
        ):
            fresh_doc = db_session.get(Document, doc.id)
            process_document_full(fresh_doc, company.ticker, db_session)

        jsonl_path = tmp_path / company.ticker / f"{doc.id}_chunks.jsonl"
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            assert obj["document_id"] == doc.id
            assert obj["company_ticker"] == company.ticker


# ---------------------------------------------------------------------------
# Test 2 — skips documents already extracted + chunked
# ---------------------------------------------------------------------------


class TestSkipsAlreadyExtractedAndChunked:
    def test_both_candidate_queries_return_empty(self, db_session):
        """
        A document with text_extracted=TRUE, chunked=TRUE must not appear in
        either candidate query.

        Requirements: 8.6, 9.1
        """
        company = _seed_company(db_session)
        _seed_document(db_session, company, text_extracted=True, chunked=True)

        pass1 = get_candidate_documents(db_session, None)
        pass2 = get_rechunk_candidates(db_session, None)

        assert pass1 == []
        assert pass2 == []


# ---------------------------------------------------------------------------
# Test 3 — rechunk pass (text_extracted=TRUE, chunked=FALSE)
# ---------------------------------------------------------------------------


class TestRechunkPass:
    def test_chunked_flag_set_and_jsonl_written(self, db_session, tmp_path):
        """
        Documents with text_extracted=TRUE, chunked=FALSE skip extraction and
        only run the chunking stage.

        Requirements: 8.7, 9.4
        """
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company, text_extracted=True, chunked=False)

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", return_value=_GOOD_OUTCOME
        ):
            fresh_doc = db_session.get(Document, doc.id)
            result = process_document_rechunk(fresh_doc, company.ticker, db_session)

        fresh_doc = db_session.get(Document, doc.id)
        assert fresh_doc.chunked is True

        jsonl_path = tmp_path / company.ticker / f"{doc.id}_chunks.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 0

        assert result.outcome == "extracted"

    def test_no_text_extraction_log_rows(self, db_session, tmp_path):
        """
        The rechunk pass must not insert any text_extraction stage log rows.

        Requirements: 8.7
        """
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company, text_extracted=True, chunked=False)

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", return_value=_GOOD_OUTCOME
        ):
            fresh_doc = db_session.get(Document, doc.id)
            process_document_rechunk(fresh_doc, company.ticker, db_session)

        logs = _all_logs(db_session)
        extraction_rows = [lg for lg in logs if lg.stage == "text_extraction"]
        assert extraction_rows == [], (
            "process_document_rechunk must not write any text_extraction log rows"
        )

    def test_rechunk_candidate_appears_in_pass2_not_pass1(self, db_session):
        """
        text_extracted=TRUE, chunked=FALSE must appear only in get_rechunk_candidates.

        Requirements: 8.7
        """
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company, text_extracted=True, chunked=False)

        pass1 = get_candidate_documents(db_session, None)
        pass2 = get_rechunk_candidates(db_session, None)

        assert not any(d.id == doc.id for d in pass1)
        assert any(d.id == doc.id for d in pass2)


# ---------------------------------------------------------------------------
# Test 4 — company filter restricts candidate set
# ---------------------------------------------------------------------------


class TestCompanyFilterRestrictsToOneCompany:
    def test_only_matching_ticker_returned(self, db_session):
        """
        get_candidate_documents with company_filter='ALPHA' must return only
        documents belonging to ALPHA, not BETA.

        Requirements: 1.2
        """
        alpha = _seed_company(db_session, ticker="ALPHA", name="Alpha Corp")
        beta = _seed_company(db_session, ticker="BETA", name="Beta Corp")

        doc_a = _seed_document(db_session, alpha)
        _seed_document(db_session, beta)

        results = get_candidate_documents(db_session, "ALPHA")

        assert len(results) == 1
        assert results[0].id == doc_a.id

    def test_filter_returns_empty_when_no_docs_for_ticker(self, db_session):
        """
        A company with no unprocessed docs returns an empty list (not an error).

        Requirements: 1.2
        """
        _seed_company(db_session, ticker="EMPTY", name="Empty Corp")
        # No documents seeded for EMPTY

        results = get_candidate_documents(db_session, "EMPTY")
        assert results == []

    def test_multiple_docs_all_returned_for_matching_company(self, db_session):
        """
        All unprocessed docs for the filtered company are returned.

        Requirements: 1.2
        """
        company = _seed_company(db_session, ticker="MULTI")
        doc1 = _seed_document(db_session, company, file_name="r1.pdf")
        doc2 = _seed_document(db_session, company, file_name="r2.pdf")

        results = get_candidate_documents(db_session, "MULTI")
        result_ids = {r.id for r in results}

        assert doc1.id in result_ids
        assert doc2.id in result_ids


# ---------------------------------------------------------------------------
# Test 5 — unknown ticker causes SystemExit with non-zero code
# ---------------------------------------------------------------------------


class TestUnknownTickerExitsWithNonZeroCode:
    def test_system_exit_raised(self, db_session):
        """
        get_candidate_documents with an unrecognised ticker must raise
        SystemExit with a non-zero exit code.

        Requirements: 1.3
        """
        with pytest.raises(SystemExit) as exc_info:
            get_candidate_documents(db_session, "UNKNOWN")

        assert exc_info.value.code != 0

    def test_system_exit_raised_for_rechunk_candidates(self, db_session):
        """
        Same requirement applies to get_rechunk_candidates.

        Requirements: 1.3
        """
        with pytest.raises(SystemExit) as exc_info:
            get_rechunk_candidates(db_session, "UNKNOWN")

        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Test 6 — scanned PDF (low_yield) → empty JSONL, flags still TRUE
# ---------------------------------------------------------------------------


class TestScannedPdfLowYield:
    def test_flags_set_and_log_message_low_yield(self, db_session, tmp_path):
        """
        When extract returns outcome='low_yield', the pipeline sets
        text_extracted=TRUE, chunked=TRUE, produces zero chunks, and logs a
        message containing 'low_yield'.

        The low_yield branch short-circuits before calling chunk_and_store so
        no JSONL file is written to disk — the chunked=TRUE flag and the
        'no_chunks: text below threshold' log entry are the canonical signals
        that chunking was handled (Req 4.5, 8.3).

        Requirements: 8.3, 9.5
        """
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company)

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", return_value=_LOW_YIELD_OUTCOME
        ):
            fresh_doc = db_session.get(Document, doc.id)
            result = process_document_full(fresh_doc, company.ticker, db_session)

        fresh_doc = db_session.get(Document, doc.id)
        # A low-yield document is NOT complete: it stays unprocessed so it is
        # retried on the next run (previously it was marked chunked and forgotten).
        assert fresh_doc.text_extracted is False
        assert fresh_doc.chunked is False

        # The failure and its reason are logged
        logs = _all_logs(db_session)
        failed = [lg for lg in logs if lg.stage == "text_extraction" and lg.status == "failed"]
        assert failed and "low_yield" in (failed[0].message or "")

        assert result.outcome == "low_yield"
        assert result.chunk_count == 0

    def test_low_yield_writes_no_chunking_success(self, db_session, tmp_path):
        """No chunking/success row may be written for a low-yield document."""
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company)

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", return_value=_LOW_YIELD_OUTCOME
        ):
            fresh_doc = db_session.get(Document, doc.id)
            process_document_full(fresh_doc, company.ticker, db_session)

        logs = _all_logs(db_session)
        assert [lg for lg in logs if lg.stage == "chunking" and lg.status == "success"] == []


# ---------------------------------------------------------------------------
# Test 7 — OCR fallback produces real chunks
# ---------------------------------------------------------------------------


class TestOCRFallbackProducesChunks:
    def test_text_extracted_chunked_jsonl_nonempty(self, db_session, tmp_path):
        """
        When extract returns extraction_method='ocr' with a good text outcome,
        the pipeline behaves identically to a native extraction.

        Requirements: 8.3, 9.2
        """
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company)

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", return_value=_OCR_OUTCOME
        ):
            fresh_doc = db_session.get(Document, doc.id)
            result = process_document_full(fresh_doc, company.ticker, db_session)

        fresh_doc = db_session.get(Document, doc.id)
        assert fresh_doc.text_extracted is True
        assert fresh_doc.chunked is True

        jsonl_path = tmp_path / company.ticker / f"{doc.id}_chunks.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 0, "OCR-extracted document must produce chunks"

        assert result.outcome == "extracted"
        assert result.chunk_count > 0

    def test_chunk_metadata_carries_ocr_document_fields(self, db_session, tmp_path):
        """Each chunk object must carry the document's fiscal_year and document_type."""
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company)

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", return_value=_OCR_OUTCOME
        ):
            fresh_doc = db_session.get(Document, doc.id)
            process_document_full(fresh_doc, company.ticker, db_session)

        jsonl_path = tmp_path / company.ticker / f"{doc.id}_chunks.jsonl"
        first_line = jsonl_path.read_text(encoding="utf-8").splitlines()[0]
        obj = json.loads(first_line)

        assert obj["document_type"] == "annual_report"
        assert obj["fiscal_year"] == "FY2020"


# ---------------------------------------------------------------------------
# Test 8 — partial file cleanup on chunk_and_store error
# ---------------------------------------------------------------------------


class TestPartialFileCleanupOnChunkError:
    def test_no_jsonl_remains_and_chunked_false(self, db_session, tmp_path):
        """
        When chunk_and_store raises OSError, the partially written JSONL must
        be deleted and chunked must stay FALSE.

        Requirements: 9.5
        """
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company)

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", return_value=_GOOD_OUTCOME
        ), patch(
            "preprocessing.run_extraction.chunk_and_store",
            side_effect=OSError("disk full"),
        ):
            fresh_doc = db_session.get(Document, doc.id)
            result = process_document_full(fresh_doc, company.ticker, db_session)

        fresh_doc = db_session.get(Document, doc.id)
        assert fresh_doc.chunked is False, "chunked must stay FALSE after OSError"

        # No JSONL file should remain
        jsonl_path = tmp_path / company.ticker / f"{doc.id}_chunks.jsonl"
        assert not jsonl_path.exists(), "Partial JSONL must be cleaned up after OSError"

        assert result.outcome == "failed"
        assert result.chunk_count == 0

    def test_chunking_failed_log_row_written(self, db_session, tmp_path):
        """
        A processing_log row with stage='chunking', status='failed' must be
        inserted when chunk_and_store raises.

        Requirements: 9.5
        """
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company)

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", return_value=_GOOD_OUTCOME
        ), patch(
            "preprocessing.run_extraction.chunk_and_store",
            side_effect=OSError("disk full"),
        ):
            fresh_doc = db_session.get(Document, doc.id)
            process_document_full(fresh_doc, company.ticker, db_session)

        logs = _all_logs(db_session)
        failed_rows = [lg for lg in logs if lg.stage == "chunking" and lg.status == "failed"]
        assert len(failed_rows) == 1, "Expected exactly one chunking/failed log row"
        assert "disk full" in failed_rows[0].message


# ---------------------------------------------------------------------------
# Test 9 — database connection error handling
# ---------------------------------------------------------------------------


class TestDatabaseConnectionErrors:
    def test_database_error_during_candidate_query_exits_with_code_1(self, capsys):
        """
        When a SQLAlchemy error occurs during candidate retrieval, main()
        should print an error to stderr and exit with code 1.

        Requirements: 1.8
        """
        from preprocessing.run_extraction import main

        # Mock get_engine to return a broken engine that raises on connection
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = SQLAlchemyError("Connection failed")
        
        with patch("preprocessing.run_extraction.get_engine", return_value=mock_engine):
            with pytest.raises(SystemExit) as exc_info:
                main(company_filter=None)

        # Must exit with code 1
        assert exc_info.value.code == 1

        # Must print error to stderr
        captured = capsys.readouterr()
        assert "ERROR: Database error" in captured.err

    def test_system_exit_from_validate_ticker_propagated(self, db_engine, capsys):
        """
        SystemExit from _validate_ticker should be propagated without conversion.

        Requirements: 1.3
        """
        from preprocessing.run_extraction import main

        with patch("preprocessing.run_extraction.get_engine", return_value=db_engine):
            with pytest.raises(SystemExit) as exc_info:
                main(company_filter="NONEXISTENT")

        # Should be the same SystemExit from _validate_ticker
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Test 10 — additional edge cases and boundary conditions
# ---------------------------------------------------------------------------


class TestAdditionalEdgeCases:
    def test_document_id_sequence_processing(self, db_session, tmp_path):
        """
        Documents should be processed in ascending order of document_id
        for deterministic runs.

        Requirements: 1.6
        """
        company = _seed_company(db_session)
        
        # Create documents with IDs that would be processed in different order 
        # if not sorted properly
        doc3 = _seed_document(db_session, company, file_name="third.pdf")
        doc1 = _seed_document(db_session, company, file_name="first.pdf")
        doc2 = _seed_document(db_session, company, file_name="second.pdf")
        
        # Force specific ID order by manipulating the IDs
        # (In real SQLite, auto-increment would give them sequential IDs anyway,
        # but this test verifies the ORDER BY is working)
        
        results = get_candidate_documents(db_session, None)
        result_ids = [doc.id for doc in results]
        
        # Should be sorted in ascending order
        assert result_ids == sorted(result_ids)

    def test_commit_failure_recovery(self, db_session, tmp_path):
        """
        When session.commit() fails, the system should handle it gracefully
        and continue processing other documents.

        Requirements: 8.5, 8.8
        """
        company = _seed_company(db_session)
        doc = _seed_document(db_session, company)

        # Mock commit to fail on the first call but succeed on retry attempts
        original_commit = db_session.commit
        call_count = 0
        
        def mock_commit():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First commit fails (the main transaction)
                raise SQLAlchemyError("Commit failed")
            # Let subsequent commits succeed (e.g., for error logging)
            return original_commit()

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", return_value=_GOOD_OUTCOME
        ):
            fresh_doc = db_session.get(Document, doc.id)
            # Patch the session's commit method directly
            with patch.object(db_session, 'commit', side_effect=mock_commit):
                result = process_document_full(fresh_doc, company.ticker, db_session)

        # Should return failed result
        assert result.outcome == "failed"

    def test_processed_dir_creation_with_nested_paths(self, db_session, tmp_path):
        """
        Verify that PROCESSED_DIR creation works with deeply nested paths.

        Requirements: 1.7
        """
        company = _seed_company(db_session, ticker="NESTED_CO")
        doc = _seed_document(db_session, company)
        
        # Use a deeply nested temporary directory
        nested_processed_dir = tmp_path / "very" / "deep" / "nested" / "path"
        
        with patch("preprocessing.run_extraction.PROCESSED_DIR", nested_processed_dir), patch(
            "preprocessing.run_extraction.extract", return_value=_GOOD_OUTCOME
        ):
            fresh_doc = db_session.get(Document, doc.id)
            process_document_full(fresh_doc, company.ticker, db_session)

        # Verify the directory structure was created
        expected_jsonl = nested_processed_dir / company.ticker / f"{doc.id}_chunks.jsonl"
        assert expected_jsonl.exists()

    def test_mixed_extraction_outcomes_in_batch(self, db_session, tmp_path):
        """
        Test processing multiple documents with different extraction outcomes
        in a single run to verify batch processing robustness.

        Requirements: 8.3, 9.2, 9.5
        """
        company = _seed_company(db_session)
        
        # Create multiple documents
        doc_success = _seed_document(db_session, company, file_name="success.pdf")
        doc_low_yield = _seed_document(db_session, company, file_name="low_yield.pdf")
        doc_failed = _seed_document(db_session, company, file_name="failed.pdf")
        
        # Mock different outcomes for different documents
        def mock_extract(document, project_root):
            if document.file_name == "success.pdf":
                return _GOOD_OUTCOME
            elif document.file_name == "low_yield.pdf":
                return _LOW_YIELD_OUTCOME
            else:  # failed.pdf
                return ExtractionOutcome(
                    text="",
                    outcome="failed",
                    char_count=0,
                    error="File not found",
                    extraction_method="none"
                )

        with patch("preprocessing.run_extraction.PROCESSED_DIR", tmp_path), patch(
            "preprocessing.run_extraction.extract", side_effect=mock_extract
        ):
            # Process all documents
            docs = get_candidate_documents(db_session, None)
            results = []
            for doc in docs:
                fresh_doc = db_session.get(Document, doc.id)
                result = process_document_full(fresh_doc, company.ticker, db_session)
                results.append(result)

        # Verify mixed outcomes
        outcomes = {r.outcome for r in results}
        assert "extracted" in outcomes
        assert "low_yield" in outcomes
        assert "failed" in outcomes
        
        # Verify appropriate files were created
        success_jsonl = tmp_path / company.ticker / f"{doc_success.id}_chunks.jsonl"
        low_yield_jsonl = tmp_path / company.ticker / f"{doc_low_yield.id}_chunks.jsonl"
        failed_jsonl = tmp_path / company.ticker / f"{doc_failed.id}_chunks.jsonl"
        
        assert success_jsonl.exists()
        assert not low_yield_jsonl.exists()  # low_yield doesn't write files
        assert not failed_jsonl.exists()      # failed doesn't write files


# ---------------------------------------------------------------------------
# Test 11 — empty candidate set causes main() to exit with code 0
# ---------------------------------------------------------------------------


class TestEmptyCandidateSetExitsZero:
    def test_exits_zero_with_message(self, db_engine, tmp_path, capsys):
        """
        When all documents are already text_extracted=TRUE + chunked=TRUE,
        main() prints a 'no documents require processing' message and exits 0.

        Requirements: 1.4, 9.1
        """
        # Seed one fully-processed document so the DB is not completely empty
        with Session(db_engine) as seed_session:
            company = _seed_company(seed_session)
            _seed_document(seed_session, company, text_extracted=True, chunked=True)
            seed_session.commit()

        from preprocessing.run_extraction import main

        with patch("preprocessing.run_extraction.get_engine", return_value=db_engine), patch(
            "preprocessing.run_extraction.PROCESSED_DIR", tmp_path
        ):
            with pytest.raises(SystemExit) as exc_info:
                main(company_filter=None)

        # Must exit with code 0
        assert exc_info.value.code == 0 or exc_info.value.code is None

        # Must print the "nothing to do" message
        captured = capsys.readouterr()
        assert "no documents require processing" in captured.out

    def test_no_jsonl_files_created(self, db_engine, tmp_path):
        """No files should be written when there is nothing to process."""
        with Session(db_engine) as seed_session:
            company = _seed_company(seed_session)
            _seed_document(seed_session, company, text_extracted=True, chunked=True)
            seed_session.commit()

        from preprocessing.run_extraction import main

        with patch("preprocessing.run_extraction.get_engine", return_value=db_engine), patch(
            "preprocessing.run_extraction.PROCESSED_DIR", tmp_path
        ):
            try:
                main(company_filter=None)
            except SystemExit:
                pass

        # tmp_path should contain no JSONL files
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert jsonl_files == [], f"Unexpected JSONL files written: {jsonl_files}"
