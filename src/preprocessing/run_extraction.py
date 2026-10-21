"""
Extraction pipeline orchestrator — entry point for PDF text extraction and chunking.

Usage
-----
    python src/preprocessing/run_extraction.py                # all companies
    python src/preprocessing/run_extraction.py --company DHFL # single company

Exit codes
----------
    0  — success (including "nothing new to process")
    1  — database connection error, or --company ticker not found

Two-pass design
---------------
Pass 1: documents WHERE text_extracted = FALSE
    → full extraction + chunking

Pass 2: documents WHERE text_extracted = TRUE AND chunked = FALSE
    → re-extract text (not stored on disk), chunking only

Label-leakage guarantee
-----------------------
This module never opens, reads, or passes ``ground_truth.json`` to any function.
No field named 'group', 'distressed', 'healthy', 'label', or 'ground_truth' is
ever written to any output file or database row.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import json

from config.settings import EXTRACTED_DIR, PROCESSED_DIR, PROJECT_ROOT
from database.postgres import get_engine
from models.company import Company
from models.document import Document
from preprocessing.chunker import chunk_and_store
from preprocessing.extraction_logger import (
    log_chunking_failed,
    log_chunking_started,
    log_chunking_success,
    log_extraction_failed,
    log_extraction_started,
    log_extraction_success,
)
from preprocessing.text_extractor import extract


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Outcome of processing a single document through extraction + chunking."""

    document_id: int
    file_name: str
    outcome: Literal["extracted", "low_yield", "failed"]
    chunk_count: int


# ---------------------------------------------------------------------------
# Document candidate queries
# ---------------------------------------------------------------------------


def get_candidate_documents(
    session: Session,
    company_filter: str | None,
) -> list[Document]:
    """
    Pass 1: return documents WHERE text_extracted = FALSE.

    Joins companies so the ticker is available for path construction and
    filtering.  Results are ordered by document.id ASC for deterministic runs
    (Req 1.6).

    Args:
        session:        Active SQLAlchemy session.
        company_filter: If provided, restrict to documents whose associated
                        company ticker matches exactly.

    Returns:
        Ordered list of Document ORM objects.

    Raises:
        SystemExit(1)       if company_filter is given but the ticker does not
                            exist in the companies table (Req 1.3).
        SQLAlchemyError     on DB failure — caller catches and exits (Req 1.8).
    """
    if company_filter is not None:
        _validate_ticker(session, company_filter)

    stmt = (
        select(Document)
        .join(Company, Document.company_id == Company.id)
        .where(Document.text_extracted.is_(False))
        .order_by(Document.id.asc())
    )

    if company_filter is not None:
        stmt = stmt.where(Company.ticker == company_filter)

    return list(session.scalars(stmt).all())


def get_rechunk_candidates(
    session: Session,
    company_filter: str | None,
) -> list[Document]:
    """
    Pass 2: return documents WHERE text_extracted = TRUE AND chunked = FALSE.

    These are documents whose extraction succeeded on a previous run but
    chunking did not complete (e.g., crash or OSError).  Text is re-extracted
    from disk rather than cached; only the chunking stage is run (Req 8.7).

    Args / Raises: same as get_candidate_documents.
    """
    if company_filter is not None:
        _validate_ticker(session, company_filter)

    stmt = (
        select(Document)
        .join(Company, Document.company_id == Company.id)
        .where(Document.text_extracted.is_(True))
        .where(Document.chunked.is_(False))
        .order_by(Document.id.asc())
    )

    if company_filter is not None:
        stmt = stmt.where(Company.ticker == company_filter)

    return list(session.scalars(stmt).all())


def _validate_ticker(session: Session, ticker: str) -> None:
    """
    Exit with code 1 if the given ticker does not exist in the companies table.

    Args:
        session: Active SQLAlchemy session.
        ticker:  Ticker string to look up.

    Raises:
        SystemExit(1) if no matching company row found.
    """
    stmt = select(Company).where(Company.ticker == ticker)
    company = session.scalars(stmt).first()
    if company is None:
        print(
            f"ERROR: --company '{ticker}' did not match any ticker in the "
            "companies table. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-document processing: full extraction + chunking
# ---------------------------------------------------------------------------


def _extraction_detail(o) -> str:
    """Compact method/page summary stored in processing_log."""
    bits = [f"method={o.extraction_method}", f"pages={o.page_count}", f"ocr_pages={o.ocr_pages}"]
    if o.truncated:
        bits.append("ocr_truncated=true")
    if o.error:
        bits.append(f"note={o.error}")
    return " ".join(bits)


def save_pages(document: Document, company_ticker: str, o) -> None:
    """Write per-page, row-preserving text for the feature extractors."""
    out_dir = EXTRACTED_DIR / company_ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_id": document.id,
        "method": o.extraction_method,
        "page_count": o.page_count,
        "ocr_pages": o.ocr_pages,
        "pages": o.pages,
    }
    (out_dir / f"{document.id}_pages.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def process_document_full(
    document: Document,
    company_ticker: str,
    session: Session,
) -> ExtractionResult:
    """
    Full extraction + chunking for a document with text_extracted = FALSE.

    Transaction pattern (Req 8.5):
        All flag updates and processing_log rows for this document are committed
        in a single session.commit() at the end of each branch.

    Args:
        document:        Document ORM object (attached to session).
        company_ticker:  Ticker string used for JSONL output path.
        session:         Active SQLAlchemy session (caller owns lifecycle).

    Returns:
        ExtractionResult with outcome and chunk count.
    """
    # ---- Step 1: log extraction started ------------------------------------
    log_extraction_started(document.id, session)

    # ---- Step 2: attempt extraction ----------------------------------------
    outcome_obj = extract(document, PROJECT_ROOT)

    # ---- Branch: extraction failed -----------------------------------------
    if outcome_obj.outcome == "failed":
        log_extraction_failed(document.id, outcome_obj.error or "unknown error", session)
        document.text_extracted = False  # already False; explicit for clarity
        try:
            session.commit()
        except Exception:  # noqa: BLE001
            session.rollback()
        return ExtractionResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="failed",
            chunk_count=0,
        )

    # ---- Extraction succeeded (extracted or low_yield) ---------------------
    # ---- Branch: low_yield — treat as a failure so it is retried on the next run.
    # (Previously these were marked chunked=TRUE with zero chunks and never revisited.)
    if outcome_obj.outcome == "low_yield":
        log_extraction_failed(
            document.id,
            f"low_yield: {outcome_obj.char_count} chars | {_extraction_detail(outcome_obj)}",
            session,
        )
        document.text_extracted = False
        document.chunked = False
        try:
            session.commit()
        except Exception:  # noqa: BLE001
            session.rollback()
        return ExtractionResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="low_yield",
            chunk_count=0,
        )

    log_extraction_success(
        document.id, outcome_obj.char_count, session, detail=_extraction_detail(outcome_obj)
    )
    document.text_extracted = True
    try:
        save_pages(document, company_ticker, outcome_obj)
    except OSError as exc:
        log_extraction_failed(document.id, f"could not save page text: {exc}", session)
        document.text_extracted = False
        session.commit()
        return ExtractionResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="failed",
            chunk_count=0,
        )

    # ---- Branch: extracted — proceed to chunking ---------------------------
    log_chunking_started(document.id, session)

    jsonl_path = PROCESSED_DIR / company_ticker / f"{document.id}_chunks.jsonl"

    try:
        result = chunk_and_store(
            document, outcome_obj.text, company_ticker, PROCESSED_DIR,
            pages=outcome_obj.pages,
        )
        document.chunked = True
        log_chunking_success(document.id, result.chunk_count, session)
        try:
            session.commit()
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            _handle_commit_failure(document, exc, session)
            return ExtractionResult(
                document_id=document.id,
                file_name=document.file_name or "",
                outcome="failed",
                chunk_count=0,
            )
        return ExtractionResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="extracted",
            chunk_count=result.chunk_count,
        )

    except (OSError, ValueError) as exc:
        # Partial file cleanup (Req 9.5)
        jsonl_path.unlink(missing_ok=True)
        document.chunked = False
        log_chunking_failed(document.id, exc, session)
        try:
            session.commit()
        except Exception as commit_exc:  # noqa: BLE001
            session.rollback()
            _handle_commit_failure(document, commit_exc, session)
        return ExtractionResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="failed",
            chunk_count=0,
        )


# ---------------------------------------------------------------------------
# Per-document processing: rechunk only (pass 2)
# ---------------------------------------------------------------------------


def process_document_rechunk(
    document: Document,
    company_ticker: str,
    session: Session,
) -> ExtractionResult:
    """
    Chunking-only pass for documents with text_extracted=TRUE, chunked=FALSE.

    Re-extracts text via text_extractor.extract() because extracted text is
    not cached on disk.  Does NOT write extraction log entries and does NOT
    update text_extracted (Req 8.7).

    Args:
        document:        Document ORM object (attached to session).
        company_ticker:  Ticker string used for JSONL output path.
        session:         Active SQLAlchemy session (caller owns lifecycle).

    Returns:
        ExtractionResult with outcome and chunk count.
    """
    # Re-extract text (no extraction log entries, no text_extracted update)
    outcome_obj = extract(document, PROJECT_ROOT)

    # If re-extraction itself failed or yielded low text, can't chunk
    if outcome_obj.outcome in ("failed", "low_yield"):
        error_msg = (
            outcome_obj.error
            if outcome_obj.outcome == "failed"
            else f"re-extraction yielded low_yield: {outcome_obj.char_count} chars"
        )
        log_chunking_failed(document.id, error_msg, session)
        document.chunked = False
        try:
            session.commit()
        except Exception:  # noqa: BLE001
            session.rollback()
        return ExtractionResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="failed",
            chunk_count=0,
        )

    # Re-extraction succeeded — proceed to chunking
    try:
        save_pages(document, company_ticker, outcome_obj)
    except OSError:
        pass  # page text is regenerated on the next full extraction
    log_chunking_started(document.id, session)

    jsonl_path = PROCESSED_DIR / company_ticker / f"{document.id}_chunks.jsonl"

    try:
        result = chunk_and_store(
            document, outcome_obj.text, company_ticker, PROCESSED_DIR,
            pages=outcome_obj.pages,
        )
        document.chunked = True
        log_chunking_success(document.id, result.chunk_count, session)
        try:
            session.commit()
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            _handle_commit_failure(document, exc, session)
            return ExtractionResult(
                document_id=document.id,
                file_name=document.file_name or "",
                outcome="failed",
                chunk_count=0,
            )
        return ExtractionResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="extracted",
            chunk_count=result.chunk_count,
        )

    except (OSError, ValueError) as exc:
        # Partial file cleanup (Req 9.5)
        jsonl_path.unlink(missing_ok=True)
        document.chunked = False
        log_chunking_failed(document.id, exc, session)
        try:
            session.commit()
        except Exception as commit_exc:  # noqa: BLE001
            session.rollback()
            _handle_commit_failure(document, commit_exc, session)
        return ExtractionResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="failed",
            chunk_count=0,
        )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_document_status(result: ExtractionResult) -> None:
    """
    Print one status line to stdout per Req 10.1.

    Format:
        [{doc_id}] {file_name} | outcome: {outcome} | chunks: {chunk_count}

    Also prints a warning to stderr for low_yield or failed outcomes (Req 10.3).
    """
    print(
        f"[{result.document_id}] {result.file_name} | "
        f"outcome: {result.outcome} | chunks: {result.chunk_count}"
    )

    if result.outcome in ("low_yield", "failed"):
        print(
            f"WARNING: [{result.document_id}] {result.file_name} — "
            f"outcome={result.outcome}",
            file=sys.stderr,
        )


def print_final_summary(results: list[ExtractionResult]) -> None:
    """
    Print final summary block to stdout per Req 10.2.

    Counts:
        - Total processed
        - Extracted / Low-yield / Failed
        - Total chunks written
    """
    total = len(results)
    extracted = sum(1 for r in results if r.outcome == "extracted")
    low_yield = sum(1 for r in results if r.outcome == "low_yield")
    failed = sum(1 for r in results if r.outcome == "failed")
    total_chunks = sum(r.chunk_count for r in results)

    print()
    print("=" * 44)
    print("EXTRACTION SUMMARY")
    print("=" * 44)
    print(f"Total processed  : {total}")
    print(f"Extracted        : {extracted}")
    print(f"Low-yield        : {low_yield}")
    print(f"Failed           : {failed}")
    print(f"Total chunks     : {total_chunks}")
    print("=" * 44)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _handle_commit_failure(
    document: Document,
    exc: Exception,
    session: Session,
) -> None:
    """
    Best-effort commit-error logging after a rollback.

    Called when session.commit() raises — at this point we have already
    rolled back.  We attempt to write a failed log entry in a fresh savepoint.
    If that also fails, we print to stderr and continue (Req 8.8).
    """
    try:
        log_chunking_failed(document.id, f"commit failed: {exc!s}"[:1000], session)
        session.commit()
    except Exception as log_exc:  # noqa: BLE001
        print(
            f"WARNING: [{document.id}] {document.file_name} — "
            f"commit error could not be logged: {log_exc}",
            file=sys.stderr,
        )


def _get_company_ticker(document: Document, session: Session) -> str:
    """
    Resolve the company ticker for a document via its company relationship.

    Falls back to querying by company_id if the relationship isn't loaded.
    Returns empty string if resolution fails (caller uses it only for path
    construction; errors will surface when the path is used).
    """
    # Try the already-loaded relationship first
    if document.company is not None:
        return document.company.ticker

    if document.company_id is None:
        return ""

    stmt = select(Company).where(Company.id == document.company_id)
    company = session.scalars(stmt).first()
    return company.ticker if company else ""


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main(company_filter: str | None = None) -> None:
    """
    Run the full extraction and chunking pipeline.

    Two-pass design:
        Pass 1 — documents WHERE text_extracted = FALSE (full extraction + chunking)
        Pass 2 — documents WHERE text_extracted = TRUE AND chunked = FALSE (rechunk only)

    Args:
        company_filter:  If provided, only documents for this company ticker
                         will be processed.

    Exit codes
    ----------
    0 — success (or no new data)
    1 — database error at startup, or unrecognised --company ticker
    """
    # Req 1.7: ensure PROCESSED_DIR exists before any file writes
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    engine = get_engine()

    # ---- Retrieve candidate documents (pass 1 and pass 2) ------------------
    try:
        with Session(engine) as session:
            pass1_docs = get_candidate_documents(session, company_filter)
            pass2_docs = get_rechunk_candidates(session, company_filter)

            # Build ticker lookup while session is still open (lazy load guard)
            pass1_tickers: dict[int, str] = {
                doc.id: _get_company_ticker(doc, session) for doc in pass1_docs
            }
            pass2_tickers: dict[int, str] = {
                doc.id: _get_company_ticker(doc, session) for doc in pass2_docs
            }
    except SystemExit:
        raise  # propagate sys.exit() from _validate_ticker
    except SQLAlchemyError as exc:
        print(
            f"ERROR: Database error retrieving candidate documents: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Req 1.4: nothing to do --------------------------------------------
    if not pass1_docs and not pass2_docs:
        print("no documents require processing")
        sys.exit(0)

    # ---- Req 10.5: company-filter notice -----------------------------------
    if company_filter is not None:
        print(f"Processing restricted to company: {company_filter}")

    all_results: list[ExtractionResult] = []

    # ---- Pass 1: full extraction + chunking --------------------------------
    for document in pass1_docs:
        ticker = pass1_tickers.get(document.id, "")
        with Session(engine) as session:
            # Re-attach the document to this session
            doc = session.get(Document, document.id)
            if doc is None:
                continue
            result = process_document_full(doc, ticker, session)
        print_document_status(result)
        all_results.append(result)

    # ---- Pass 2: rechunk only ----------------------------------------------
    for document in pass2_docs:
        ticker = pass2_tickers.get(document.id, "")
        with Session(engine) as session:
            doc = session.get(Document, document.id)
            if doc is None:
                continue
            result = process_document_rechunk(doc, ticker, session)
        print_document_status(result)
        all_results.append(result)

    print_final_summary(all_results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract text and chunk documents from data/raw/ into data/processed/."
        )
    )
    parser.add_argument(
        "--company",
        metavar="TICKER",
        default=None,
        help="Restrict processing to documents for this company ticker.",
    )
    args = parser.parse_args()
    main(company_filter=args.company)
