"""
Embedding pipeline orchestrator — entry point for chunk embedding and Qdrant upsert.

Usage
-----
    python src/embedding/run_embedding.py                # all companies
    python src/embedding/run_embedding.py --company DHFL # single company

Exit codes
----------
    0  — success (including "nothing new to process")
    1  — Qdrant connection error, collection mismatch, model load error,
         database connection error, or --company ticker not found

Label-leakage guarantee
-----------------------
This module never opens, reads, or passes ``ground_truth.json`` to any function.
No field named 'group', 'distressed', 'healthy', 'label', or 'ground_truth' is
ever written to any Qdrant payload, processing_log row, or database column.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config.settings import PROCESSED_DIR
from database.postgres import get_engine
from embedding.embedder import build_points, embed_texts, load_model
from embedding.embedding_logger import (
    log_embedding_failed,
    log_embedding_started,
    log_embedding_success,
)
from embedding.qdrant_client_factory import (
    COLLECTION_NAME,
    ensure_collection,
    get_qdrant_client,
)
from models.company import Company
from models.document import Document


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingResult:
    """Outcome of processing a single document through embedding + Qdrant upsert."""

    document_id: int
    file_name: str
    outcome: Literal["embedded", "failed", "skipped"]
    chunk_count: int


# ---------------------------------------------------------------------------
# Document candidate query
# ---------------------------------------------------------------------------


def get_candidate_documents(
    session: Session,
    company_filter: str | None,
) -> list[Document]:
    """
    Return documents WHERE chunked = TRUE AND embedded = FALSE.

    Joins companies so the ticker is available for path construction and
    filtering. Results are ordered by document.id ASC for deterministic runs
    (Req 1.5).

    Args:
        session:        Active SQLAlchemy session.
        company_filter: If provided, restrict to documents whose associated
                        company ticker matches exactly.

    Returns:
        Ordered list of Document ORM objects.

    Raises:
        SystemExit(1)    if company_filter is given but the ticker does not
                         exist in the companies table (Req 1.3).
        SQLAlchemyError  on DB failure — caller catches and exits (Req 1.7).
    """
    if company_filter is not None:
        _validate_ticker(session, company_filter)

    stmt = (
        select(Document)
        .join(Company, Document.company_id == Company.id)
        .where(Document.chunked.is_(True))
        .where(Document.embedded.is_(False))
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
# Per-document processing
# ---------------------------------------------------------------------------


def process_document(
    document: Document,
    company_ticker: str,
    model: SentenceTransformer,
    qdrant_client: QdrantClient,
    session: Session,
) -> EmbeddingResult:
    """
    Full embedding + Qdrant upsert flow for a single document.

    Transaction pattern (Req 6.4):
        All flag updates and processing_log rows for this document are committed
        in a single session.commit() at the end of each branch.

    Args:
        document:       Document ORM object (attached to session).
        company_ticker: Ticker string used for JSONL path construction.
        model:          Loaded SentenceTransformer instance.
        qdrant_client:  Connected QdrantClient.
        session:        Active SQLAlchemy session (caller owns lifecycle).

    Returns:
        EmbeddingResult with outcome and chunk count.
    """
    # Step 1: log embedding started ----------------------------------------
    log_embedding_started(document.id, session)

    # Step 2: resolve JSONL path -------------------------------------------
    jsonl_path: Path = PROCESSED_DIR / company_ticker / f"{document.id}_chunks.jsonl"

    # Step 3a: missing JSONL -----------------------------------------------
    if not jsonl_path.exists():
        error_msg = f"JSONL file not found: {jsonl_path}"
        log_embedding_failed(document.id, Exception(error_msg), session)
        document.embedded = False
        try:
            session.commit()
        except Exception:  # noqa: BLE001
            session.rollback()
        return EmbeddingResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="failed",
            chunk_count=0,
        )

    # Step 3b: read and parse JSONL ----------------------------------------
    try:
        raw_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        log_embedding_failed(document.id, exc, session)
        document.embedded = False
        try:
            session.commit()
        except Exception:  # noqa: BLE001
            session.rollback()
        return EmbeddingResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="failed",
            chunk_count=0,
        )

    # Filter blank lines — blank lines are silently skipped (not malformed)
    non_empty_lines: list[tuple[int, str]] = [
        (i + 1, line) for i, line in enumerate(raw_lines) if line.strip()
    ]

    # Step 3c: empty JSONL (zero non-empty lines) --------------------------
    if not non_empty_lines:
        document.embedded = True
        log_embedding_success(document.id, 0, session)
        try:
            session.commit()
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            _handle_commit_failure(document, exc, session)
            return EmbeddingResult(
                document_id=document.id,
                file_name=document.file_name or "",
                outcome="failed",
                chunk_count=0,
            )
        return EmbeddingResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="skipped",
            chunk_count=0,
        )

    # Step 3d: parse each non-empty line as JSON; fail on first bad line ---
    chunks: list[dict] = []
    for line_number, line_text in non_empty_lines:
        try:
            obj = json.loads(line_text)
        except json.JSONDecodeError as exc:
            error_msg = f"Line {line_number}: invalid JSON — {exc}"
            log_embedding_failed(document.id, Exception(error_msg), session)
            document.embedded = False
            try:
                session.commit()
            except Exception:  # noqa: BLE001
                session.rollback()
            return EmbeddingResult(
                document_id=document.id,
                file_name=document.file_name or "",
                outcome="failed",
                chunk_count=0,
            )

        if "chunk_text" not in obj:
            error_msg = f"Line {line_number}: missing 'chunk_text' field"
            log_embedding_failed(document.id, Exception(error_msg), session)
            document.embedded = False
            try:
                session.commit()
            except Exception:  # noqa: BLE001
                session.rollback()
            return EmbeddingResult(
                document_id=document.id,
                file_name=document.file_name or "",
                outcome="failed",
                chunk_count=0,
            )

        chunks.append(obj)

    # Steps 4–6: embed → build points → upsert ----------------------------
    try:
        chunk_texts = [c["chunk_text"] for c in chunks]
        embeddings = embed_texts(model, chunk_texts)
        points = build_points(chunks, embeddings)
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as exc:  # noqa: BLE001
        log_embedding_failed(document.id, exc, session)
        document.embedded = False
        try:
            session.commit()
        except Exception:  # noqa: BLE001
            session.rollback()
        return EmbeddingResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="failed",
            chunk_count=0,
        )

    # Steps 7–9: flag + log + commit ---------------------------------------
    document.embedded = True
    log_embedding_success(document.id, len(chunks), session)

    try:
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        _handle_commit_failure(document, exc, session)
        return EmbeddingResult(
            document_id=document.id,
            file_name=document.file_name or "",
            outcome="failed",
            chunk_count=0,
        )

    return EmbeddingResult(
        document_id=document.id,
        file_name=document.file_name or "",
        outcome="embedded",
        chunk_count=len(chunks),
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_document_status(result: EmbeddingResult) -> None:
    """
    Print one status line to stdout per Req 9.1.

    Format:
        [{doc_id}] {file_name} | chunks: {chunk_count} | status: {outcome}

    Also prints a warning to stderr for failed outcomes (Req 9.3).
    """
    print(
        f"[{result.document_id}] {result.file_name} | "
        f"chunks: {result.chunk_count} | status: {result.outcome}"
    )

    if result.outcome == "failed":
        print(
            f"WARNING: [{result.document_id}] {result.file_name} — "
            f"outcome={result.outcome}",
            file=sys.stderr,
        )


def print_final_summary(results: list[EmbeddingResult]) -> None:
    """
    Print final summary block to stdout per Req 9.2.

    Counts:
        - Total processed
        - Embedded / Skipped / Failed
        - Total chunks embedded
    """
    total = len(results)
    embedded = sum(1 for r in results if r.outcome == "embedded")
    skipped = sum(1 for r in results if r.outcome == "skipped")
    failed = sum(1 for r in results if r.outcome == "failed")
    total_chunks = sum(r.chunk_count for r in results)

    print()
    print("=" * 44)
    print("EMBEDDING SUMMARY")
    print("=" * 44)
    print(f"Total processed : {total}")
    print(f"Embedded        : {embedded}")
    print(f"Skipped         : {skipped}")
    print(f"Failed          : {failed}")
    print(f"Total chunks    : {total_chunks}")
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

    Called when session.commit() raises after a successful Qdrant upsert — at
    this point we have already rolled back. We attempt to write a failed log
    entry in a fresh savepoint. If that also fails, we print to stderr and
    continue (Req 6.2).
    """
    print(
        f"WARNING: [{document.id}] {document.file_name} — "
        f"commit failed after successful Qdrant upsert: {exc}",
        file=sys.stderr,
    )
    try:
        log_embedding_failed(
            document.id,
            Exception(f"commit failed: {exc!s}"[:1000]),
            session,
        )
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
    Run the full embedding pipeline.

    Startup sequence:
        1. get_qdrant_client()       — on connection error: stderr + sys.exit(1)
        2. ensure_collection(client) — on ValueError (mismatch): stderr + sys.exit(1)
        3. load_model()              — on load error: stderr + sys.exit(1)
        4. get_candidate_documents   — on SQLAlchemyError: stderr + sys.exit(1)
        5. Empty set                 — stdout "no documents require embedding" + summary + exit(0)
        6. company_filter given      — print restriction notice with ticker to stdout
        7. For each doc: fresh session, session.get(Document, id), process_document(...)
        8. print_final_summary(all_results)

    Args:
        company_filter: If provided, only documents for this company ticker
                        will be processed.

    Exit codes
    ----------
    0 — success (or no new data)
    1 — Qdrant/DB/model error at startup, or unrecognised --company ticker
    """
    # Step 1: Qdrant client ------------------------------------------------
    try:
        qdrant_client = get_qdrant_client()
        # Probe the connection by listing collections (cheap health check)
        qdrant_client.get_collections()
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: Could not connect to Qdrant: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 2: ensure collection --------------------------------------------
    try:
        ensure_collection(qdrant_client)
    except ValueError as exc:
        print(
            f"ERROR: Qdrant collection configuration mismatch: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 3: load model ---------------------------------------------------
    try:
        model = load_model()
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: Failed to load embedding model: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 4: retrieve candidate documents ---------------------------------
    engine = get_engine()

    try:
        with Session(engine) as session:
            candidate_docs = get_candidate_documents(session, company_filter)

            # Build ticker lookup while session is still open (lazy load guard)
            ticker_map: dict[int, str] = {
                doc.id: _get_company_ticker(doc, session) for doc in candidate_docs
            }
    except SystemExit:
        raise  # propagate sys.exit() from _validate_ticker
    except SQLAlchemyError as exc:
        print(
            f"ERROR: Database error retrieving candidate documents: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 5: empty candidate set ------------------------------------------
    if not candidate_docs:
        print("no documents require embedding")
        print_final_summary([])
        sys.exit(0)

    # Step 6: company-filter notice ----------------------------------------
    if company_filter is not None:
        print(f"Processing restricted to company: {company_filter}")

    all_results: list[EmbeddingResult] = []

    # Step 7: process each document ----------------------------------------
    for document in candidate_docs:
        ticker = ticker_map.get(document.id, "")
        with Session(engine) as session:
            # Re-attach the document to this fresh session
            doc = session.get(Document, document.id)
            if doc is None:
                continue
            result = process_document(doc, ticker, model, qdrant_client, session)
        print_document_status(result)
        all_results.append(result)

    # Step 8: final summary ------------------------------------------------
    print_final_summary(all_results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Embed document chunks from data/processed/ and upsert into Qdrant."
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
