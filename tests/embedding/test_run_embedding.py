"""
Integration tests for src/embedding/run_embedding.py

Strategy
--------
All external dependencies are mocked so no real PostgreSQL, Qdrant, or
embedding model is required:

  - SQLite in-memory engine (same pattern as test_run_extraction.py)
  - embedding.run_embedding.get_engine     → returns the SQLite engine
  - embedding.run_embedding.get_qdrant_client → returns an in-memory mock QdrantClient
  - embedding.run_embedding.ensure_collection → no-op
  - embedding.run_embedding.load_model     → returns a mock that generates
                                             np.zeros((n, 768)) vectors

Tests are marked @pytest.mark.integration for organisation but do not skip —
all dependencies are mocked, so they run in any environment.

JSONL files are written to tmp_path / company_ticker / f"{doc_id}_chunks.jsonl",
mirroring the path that process_document() constructs internally.

Requirements covered: 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 4.2, 4.3, 4.4, 4.6, 4.9,
                       6.1, 6.4, 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.5
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.orm import Session

from models.base import Base
from models.company import Company
from models.document import Document
from models.processing_log import ProcessingLog

from embedding.embedder import NAMESPACE_UUID


# ---------------------------------------------------------------------------
# SQLite-compatible metadata (companies, documents, processing_log only)
# ---------------------------------------------------------------------------

_TEST_TABLES = {"companies", "documents", "processing_log"}


def _test_metadata() -> MetaData:
    """Return MetaData containing only the three SQLite-safe tables."""
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
    """In-memory SQLite engine, fresh per test."""
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
def mock_qdrant_client():
    """
    Mock QdrantClient that stores upserted points in memory.

    Attributes exposed for assertions:
      .stored_points  — dict[str, PointStruct] keyed by point ID string
      .upserted_calls — list of (collection_name, points) tuples
    """
    client = MagicMock()
    client.stored_points: dict[str, object] = {}
    client.upserted_calls: list = []

    def _upsert(collection_name, points, **kwargs):
        for p in points:
            client.stored_points[str(p.id)] = p
        client.upserted_calls.append((collection_name, list(points)))

    client.upsert.side_effect = _upsert
    client.get_collections.return_value.collections = []
    return client


@pytest.fixture()
def mock_model():
    """Mock SentenceTransformer that returns np.zeros((n, 768)) for any batch."""
    model = MagicMock()
    model.encode.side_effect = lambda batch: np.zeros((len(batch), 768), dtype=float)
    return model


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
    chunked: bool = True,
    embedded: bool = False,
    file_name: str = "report.pdf",
    document_type: str = "annual_report",
    fiscal_year: str = "FY2022",
) -> Document:
    d = Document(
        company_id=company.id,
        file_name=file_name,
        file_path=f"data/raw/{company.ticker}/annual_reports/{file_name}",
        document_type=document_type,
        fiscal_year=fiscal_year,
    )
    d.chunked = chunked
    d.embedded = embedded
    session.add(d)
    session.flush()
    return d


def _make_chunk(
    document_id: int,
    company_id: int,
    ticker: str,
    chunk_index: int = 0,
    chunk_text: str = "Sample chunk text.",
    document_type: str = "annual_report",
    fiscal_year: str = "FY2022",
) -> dict:
    """Return a valid chunk dict with all 11 required JSONL fields."""
    return {
        "document_id": document_id,
        "company_id": company_id,
        "company_ticker": ticker,
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "document_type": document_type,
        "document_category": None,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": None,
        "source": "BSE",
        "file_path": f"data/raw/{ticker}/annual_reports/report.pdf",
    }


def _write_jsonl(path: Path, chunks: list[dict]) -> None:
    """Write a list of chunk dicts to a JSONL file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def _all_logs(session: Session, document_id: int | None = None) -> list[ProcessingLog]:
    q = session.query(ProcessingLog).order_by(ProcessingLog.id)
    if document_id is not None:
        q = q.filter(ProcessingLog.document_id == document_id)
    return q.all()


# ---------------------------------------------------------------------------
# Shared patch context manager for main()
# ---------------------------------------------------------------------------


def _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
    """
    Return a context manager that patches all external dependencies of
    run_embedding.main() so it runs against the SQLite engine, mock model,
    and mock Qdrant client.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with patch("embedding.run_embedding.get_engine", return_value=db_engine), \
             patch("embedding.run_embedding.get_qdrant_client", return_value=mock_qdrant_client), \
             patch("embedding.run_embedding.ensure_collection"), \
             patch("embedding.run_embedding.load_model", return_value=mock_model), \
             patch("embedding.run_embedding.PROCESSED_DIR", tmp_path):
            yield

    return _ctx()


# ---------------------------------------------------------------------------
# Test 1 — full pipeline end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullPipeline:
    def test_full_pipeline(self, db_engine, mock_model, mock_qdrant_client, tmp_path):
        """
        Seed one company + document (chunked=TRUE, embedded=FALSE) with a valid
        3-chunk JSONL → run main() → assert embedded=TRUE, processing_log success
        row with correct message, and Qdrant points with correct payload fields.

        Requirements: 4.2, 4.3, 6.1, 6.4, 8.1, 8.2
        """
        from embedding.run_embedding import main

        ticker = "TESTCO"
        with Session(db_engine) as session:
            company = _seed_company(session, ticker=ticker)
            doc = _seed_document(session, company)
            doc_id = doc.id
            company_id = company.id
            session.commit()

        # Write 3-chunk JSONL to the expected location
        jsonl_path = tmp_path / ticker / f"{doc_id}_chunks.jsonl"
        chunks = [
            _make_chunk(doc_id, company_id, ticker, chunk_index=i, chunk_text=f"Chunk {i} text.")
            for i in range(3)
        ]
        _write_jsonl(jsonl_path, chunks)

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            main()

        # Assert embedded=TRUE in DB
        with Session(db_engine) as session:
            doc_row = session.get(Document, doc_id)
            assert doc_row.embedded is True, "embedded flag must be TRUE after successful pipeline"

            # Assert processing_log success row with correct message
            logs = _all_logs(session, document_id=doc_id)
            stage_statuses = [(lg.stage, lg.status) for lg in logs]
            assert ("embedding", "started") in stage_statuses
            assert ("embedding", "success") in stage_statuses

            success_rows = [lg for lg in logs if lg.stage == "embedding" and lg.status == "success"]
            assert len(success_rows) == 1
            assert success_rows[0].message == "embedded: 3 chunks"

        # Assert Qdrant received 3 upserted points
        assert len(mock_qdrant_client.stored_points) == 3

        # Assert correct payload fields on each point
        for i, chunk in enumerate(chunks):
            expected_id = str(uuid.uuid5(NAMESPACE_UUID, f"{doc_id}:{i}"))
            assert expected_id in mock_qdrant_client.stored_points, \
                f"Expected point ID {expected_id} not found in Qdrant"

            point = mock_qdrant_client.stored_points[expected_id]
            payload = point.payload
            assert payload["company_ticker"] == ticker
            assert payload["document_type"] == "annual_report"
            assert payload["fiscal_year"] == "FY2022"
            assert payload["document_id"] == doc_id
            assert payload["chunk_index"] == i

            # No prohibited keys in payload
            _PROHIBITED = {"group", "distressed", "healthy", "label", "ground_truth"}
            assert not (set(payload.keys()) & _PROHIBITED), \
                f"Prohibited keys found in Qdrant payload: {set(payload.keys()) & _PROHIBITED}"


# ---------------------------------------------------------------------------
# Test 2 — idempotency (run pipeline twice)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIdempotency:
    def test_idempotency(self, db_engine, mock_model, mock_qdrant_client, tmp_path):
        """
        Run pipeline twice on the same document. Second run finds no candidates,
        writes no new log rows, and Qdrant point count is unchanged.

        Requirements: 4.9
        """
        from embedding.run_embedding import main

        ticker = "IDEMPCO"
        with Session(db_engine) as session:
            company = _seed_company(session, ticker=ticker)
            doc = _seed_document(session, company)
            doc_id = doc.id
            company_id = company.id
            session.commit()

        jsonl_path = tmp_path / ticker / f"{doc_id}_chunks.jsonl"
        chunks = [_make_chunk(doc_id, company_id, ticker, chunk_index=i) for i in range(2)]
        _write_jsonl(jsonl_path, chunks)

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            main()  # first run

        # Capture state after first run
        with Session(db_engine) as session:
            log_count_after_first = len(_all_logs(session, document_id=doc_id))

        point_count_after_first = len(mock_qdrant_client.stored_points)

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            with pytest.raises(SystemExit) as exc_info:
                main()  # second run — nothing to do
            assert exc_info.value.code == 0 or exc_info.value.code is None

        # No new processing_log rows written
        with Session(db_engine) as session:
            log_count_after_second = len(_all_logs(session, document_id=doc_id))

        assert log_count_after_second == log_count_after_first, \
            "Second run must not add new processing_log rows for already-embedded document"

        # Qdrant point count unchanged
        assert len(mock_qdrant_client.stored_points) == point_count_after_first, \
            "Second run must not add new Qdrant points"


# ---------------------------------------------------------------------------
# Test 3 — empty JSONL
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEmptyJsonl:
    def test_empty_jsonl(self, db_engine, mock_model, mock_qdrant_client, tmp_path):
        """
        JSONL file with zero non-empty lines → embedded=TRUE,
        processing_log message='no_chunks: nothing to embed', no Qdrant points written.

        Requirements: 4.6, 8.3
        """
        from embedding.run_embedding import main

        ticker = "EMPTCO"
        with Session(db_engine) as session:
            company = _seed_company(session, ticker=ticker)
            doc = _seed_document(session, company)
            doc_id = doc.id
            session.commit()

        # Write JSONL with only blank lines
        jsonl_path = tmp_path / ticker / f"{doc_id}_chunks.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("\n\n   \n", encoding="utf-8")

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            main()

        with Session(db_engine) as session:
            doc_row = session.get(Document, doc_id)
            assert doc_row.embedded is True, \
                "embedded must be TRUE for empty JSONL (skipped outcome)"

            logs = _all_logs(session, document_id=doc_id)
            success_rows = [lg for lg in logs if lg.stage == "embedding" and lg.status == "success"]
            assert len(success_rows) == 1
            assert success_rows[0].message == "no_chunks: nothing to embed"

        # No Qdrant points upserted
        assert mock_qdrant_client.upsert.call_count == 0, \
            "upsert must not be called for an empty JSONL"


# ---------------------------------------------------------------------------
# Test 4 — missing JSONL
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMissingJsonl:
    def test_missing_jsonl(self, db_engine, mock_model, mock_qdrant_client, tmp_path):
        """
        JSONL file absent → embedded=FALSE, processing_log status='failed',
        message includes the missing file path.

        Requirements: 4.3, 8.4
        """
        from embedding.run_embedding import main

        ticker = "MISSCO"
        with Session(db_engine) as session:
            company = _seed_company(session, ticker=ticker)
            doc = _seed_document(session, company)
            doc_id = doc.id
            session.commit()

        # Do NOT write any JSONL file — it should be missing

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            main()

        with Session(db_engine) as session:
            doc_row = session.get(Document, doc_id)
            assert doc_row.embedded is False, \
                "embedded must remain FALSE when JSONL is missing"

            logs = _all_logs(session, document_id=doc_id)
            failed_rows = [lg for lg in logs if lg.stage == "embedding" and lg.status == "failed"]
            assert len(failed_rows) == 1

            # Message must include the expected file path
            expected_fragment = f"{doc_id}_chunks.jsonl"
            assert expected_fragment in failed_rows[0].message, \
                f"Failed message should include the missing file path. Got: {failed_rows[0].message!r}"


# ---------------------------------------------------------------------------
# Test 5 — malformed JSON line
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMalformedJsonLine:
    def test_malformed_json_line(self, db_engine, mock_model, mock_qdrant_client, tmp_path):
        """
        JSONL with one valid line then one invalid JSON line → embedded=FALSE,
        processing_log failed row message includes line number 2.

        Requirements: 4.4
        """
        from embedding.run_embedding import main

        ticker = "MALCO"
        with Session(db_engine) as session:
            company = _seed_company(session, ticker=ticker)
            doc = _seed_document(session, company)
            doc_id = doc.id
            company_id = company.id
            session.commit()

        jsonl_path = tmp_path / ticker / f"{doc_id}_chunks.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        valid_line = json.dumps(_make_chunk(doc_id, company_id, ticker, chunk_index=0))
        invalid_line = "{ this is not valid JSON !!!"
        jsonl_path.write_text(valid_line + "\n" + invalid_line + "\n", encoding="utf-8")

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            main()

        with Session(db_engine) as session:
            doc_row = session.get(Document, doc_id)
            assert doc_row.embedded is False, \
                "embedded must remain FALSE when JSONL has a malformed line"

            logs = _all_logs(session, document_id=doc_id)
            failed_rows = [lg for lg in logs if lg.stage == "embedding" and lg.status == "failed"]
            assert len(failed_rows) == 1

            # Message must reference line 2
            assert "2" in failed_rows[0].message, \
                f"Failed message should include line number 2. Got: {failed_rows[0].message!r}"


# ---------------------------------------------------------------------------
# Test 6 — missing chunk_text field
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMissingChunkTextField:
    def test_missing_chunk_text_field(self, db_engine, mock_model, mock_qdrant_client, tmp_path):
        """
        JSONL line missing 'chunk_text' key → embedded=FALSE,
        processing_log failed row identifies the line.

        Requirements: 4.4
        """
        from embedding.run_embedding import main

        ticker = "NOCTCO"
        with Session(db_engine) as session:
            company = _seed_company(session, ticker=ticker)
            doc = _seed_document(session, company)
            doc_id = doc.id
            company_id = company.id
            session.commit()

        jsonl_path = tmp_path / ticker / f"{doc_id}_chunks.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        # Build a chunk dict without chunk_text
        chunk = _make_chunk(doc_id, company_id, ticker, chunk_index=0)
        del chunk["chunk_text"]
        jsonl_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            main()

        with Session(db_engine) as session:
            doc_row = session.get(Document, doc_id)
            assert doc_row.embedded is False, \
                "embedded must remain FALSE when chunk_text field is missing"

            logs = _all_logs(session, document_id=doc_id)
            failed_rows = [lg for lg in logs if lg.stage == "embedding" and lg.status == "failed"]
            assert len(failed_rows) == 1
            # Message should identify the line
            assert "1" in failed_rows[0].message or "chunk_text" in failed_rows[0].message, \
                f"Failed message should identify the offending line. Got: {failed_rows[0].message!r}"


# ---------------------------------------------------------------------------
# Test 7 — company filter
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCompanyFilter:
    def test_company_filter(self, db_engine, mock_model, mock_qdrant_client, tmp_path):
        """
        Two companies, both with eligible documents.
        main(company_filter='ALPHA') processes only ALPHA documents;
        BETA documents remain embedded=FALSE with no new processing_log rows.

        Requirements: 1.2
        """
        from embedding.run_embedding import main

        with Session(db_engine) as session:
            company_a = _seed_company(session, ticker="ALPHA", name="Alpha Corp")
            company_b = _seed_company(session, ticker="BETA", name="Beta Corp")
            doc_a = _seed_document(session, company_a)
            doc_b = _seed_document(session, company_b)
            doc_a_id, doc_b_id = doc_a.id, doc_b.id
            company_a_id, company_b_id = company_a.id, company_b.id
            session.commit()

        # Write valid JSONL for both companies
        for ticker, doc_id, company_id in [
            ("ALPHA", doc_a_id, company_a_id),
            ("BETA", doc_b_id, company_b_id),
        ]:
            jsonl_path = tmp_path / ticker / f"{doc_id}_chunks.jsonl"
            chunks = [_make_chunk(doc_id, company_id, ticker, chunk_index=0)]
            _write_jsonl(jsonl_path, chunks)

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            main(company_filter="ALPHA")

        with Session(db_engine) as session:
            doc_a_row = session.get(Document, doc_a_id)
            doc_b_row = session.get(Document, doc_b_id)

            # ALPHA processed
            assert doc_a_row.embedded is True, "ALPHA document must be embedded"

            # BETA untouched
            assert doc_b_row.embedded is False, \
                "BETA document must remain embedded=FALSE when filtered to ALPHA"

            # No processing_log rows for BETA
            beta_logs = _all_logs(session, document_id=doc_b_id)
            assert beta_logs == [], \
                "No processing_log rows should be written for BETA when filter is ALPHA"


# ---------------------------------------------------------------------------
# Test 8 — Qdrant upsert overwrite (idempotent point IDs)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestQdrantUpsertOverwrite:
    def test_qdrant_upsert_overwrite(self, db_engine, mock_model, mock_qdrant_client, tmp_path):
        """
        Pre-seed mock Qdrant storage with a point at the same uuid5 ID as
        what the pipeline will produce. Run pipeline → no error, point
        overwritten, embedded=TRUE.

        Requirements: 4.2, 6.4
        """
        from embedding.run_embedding import main

        ticker = "OVERWCO"
        with Session(db_engine) as session:
            company = _seed_company(session, ticker=ticker)
            doc = _seed_document(session, company)
            doc_id = doc.id
            company_id = company.id
            session.commit()

        jsonl_path = tmp_path / ticker / f"{doc_id}_chunks.jsonl"
        chunks = [_make_chunk(doc_id, company_id, ticker, chunk_index=0)]
        _write_jsonl(jsonl_path, chunks)

        # Pre-seed Qdrant with a stale point at the same uuid5 ID
        expected_point_id = str(uuid.uuid5(NAMESPACE_UUID, f"{doc_id}:0"))
        stale_point = MagicMock()
        stale_point.id = expected_point_id
        stale_point.payload = {"stale": True}
        mock_qdrant_client.stored_points[expected_point_id] = stale_point

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            main()  # must not raise

        with Session(db_engine) as session:
            doc_row = session.get(Document, doc_id)
            assert doc_row.embedded is True, \
                "embedded must be TRUE even when point ID already existed in Qdrant"

        # Point must have been overwritten (upsert was called)
        assert mock_qdrant_client.upsert.call_count >= 1
        # The stored point should now be the new version (not the stale one)
        assert expected_point_id in mock_qdrant_client.stored_points
        refreshed_point = mock_qdrant_client.stored_points[expected_point_id]
        assert refreshed_point.payload.get("stale") is not True, \
            "Stale point payload should have been overwritten by upsert"


# ---------------------------------------------------------------------------
# Test 9 — unknown ticker causes non-zero SystemExit
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUnknownTickerExitsNonzero:
    def test_unknown_ticker_exits_nonzero(self, db_engine, mock_model, mock_qdrant_client, tmp_path):
        """
        main(company_filter='UNKNOWN') → SystemExit with non-zero code.
        No documents are processed.

        Requirements: 1.3
        """
        from embedding.run_embedding import main

        with Session(db_engine) as session:
            company = _seed_company(session, ticker="REALCO")
            doc = _seed_document(session, company)
            doc_id = doc.id
            session.commit()

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            with pytest.raises(SystemExit) as exc_info:
                main(company_filter="UNKNOWN")

        assert exc_info.value.code != 0, \
            "SystemExit code must be non-zero for unrecognised ticker"

        # No documents processed — embedded still FALSE
        with Session(db_engine) as session:
            doc_row = session.get(Document, doc_id)
            assert doc_row.embedded is False


# ---------------------------------------------------------------------------
# Test 10 — empty candidate set exits with code 0
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEmptyCandidateSetExitsZero:
    def test_empty_candidate_set_exits_zero(
        self, db_engine, mock_model, mock_qdrant_client, tmp_path, capsys
    ):
        """
        All documents already embedded=TRUE → stdout contains
        "no documents require embedding", exit code 0.

        Requirements: 1.4, 9.1
        """
        from embedding.run_embedding import main

        with Session(db_engine) as session:
            company = _seed_company(session, ticker="DONECO")
            _seed_document(session, company, chunked=True, embedded=True)
            session.commit()

        with _patched_main(db_engine, mock_model, mock_qdrant_client, tmp_path):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0 or exc_info.value.code is None, \
            "Exit code must be 0 when there are no candidate documents"

        captured = capsys.readouterr()
        assert "no documents require embedding" in captured.out, \
            f"Expected 'no documents require embedding' in stdout. Got: {captured.out!r}"
