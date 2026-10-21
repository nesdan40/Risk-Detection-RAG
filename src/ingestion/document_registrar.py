"""
Document registrar for the document ingestion pipeline.

Responsible for scanning company subdirectories, deriving metadata from PDF
filenames, and registering document rows in the ``documents`` table.

Key conventions
---------------
* ``file_path`` is stored as a forward-slash-separated path relative to
  ``PROJECT_ROOT`` regardless of the host OS path separator (Req 4.5).
* Deduplication is by ``file_path``: a file already present in the table is
  skipped (returns ``'skipped'``), never duplicated (Req 4.2, 8.4).
* Newly inserted rows always have ``text_extracted``, ``chunked``, and
  ``embedded`` set to ``False`` — these are controlled by the ORM defaults
  and are never explicitly set here (Req 4.3).
* Processing-log events are written inside ``register_document`` for every
  attempt: ``started`` → ``success`` (created/skipped) or ``failed``
  (Req 7.1–7.7).
* Subdirectory → document_type mapping (Req 4.1, 5.1–5.4):
    annual_reports/          → 'annual_report'
    quarterly_results/       → 'quarterly_result'
    corporate_announcements/ → 'corporate_announcement'
    credit_ratings/          → 'credit_rating'

Label-leakage note
------------------
This module never reads or references ``ground_truth.json`` or any
distress/group/label information.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config.settings import PROJECT_ROOT
from ingestion.filename_parser import parse_filename
from ingestion.ingestion_logger import log_failed, log_started, log_success
from ingestion.metadata_reader import CompanyMetadata
from models.document import Document


# ---------------------------------------------------------------------------
# Subdirectory → document_type mapping (Req 4.1, 5.1–5.4)
# ---------------------------------------------------------------------------

SUBDIR_TO_DOCTYPE: dict[str, str] = {
    "annual_reports": "annual_report",
    "quarterly_results": "quarterly_result",
    "corporate_announcements": "corporate_announcement",
    "credit_ratings": "credit_rating",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DocumentResult:
    """Aggregated outcome of registering all documents for a company."""

    registered: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Subdirectory scanner
# ---------------------------------------------------------------------------


def scan_subdirectory(
    company_dir: Path,
    subdir_name: str,
    document_type: str,  # noqa: ARG001 — kept for caller symmetry / future use
) -> list[Path]:
    """
    Return a list of ``.pdf`` files found in *company_dir / subdir_name*.

    If the subdirectory does not exist, a warning is printed to stderr and an
    empty list is returned without raising an error (Req 4.7).

    Args:
        company_dir:   Absolute path to the company's root directory.
        subdir_name:   Name of the subdirectory to scan (e.g. ``'annual_reports'``).
        document_type: Document type string for the subdirectory — accepted for
                       interface consistency with the caller but not used here.

    Returns:
        List of absolute :class:`~pathlib.Path` objects for each ``.pdf`` file
        found, or an empty list if the subdirectory is absent.
    """
    subdir = company_dir / subdir_name

    if not subdir.exists():
        print(
            f"WARNING: Subdirectory '{subdir_name}' not found in "
            f"'{company_dir.name}'. Skipping.",
            file=sys.stderr,
        )
        return []

    return sorted(subdir.glob("*.pdf"))


# ---------------------------------------------------------------------------
# Single-document registration
# ---------------------------------------------------------------------------


def register_document(
    company_id: int,
    file_path: Path,
    document_type: str,
    exchange: str,
    session: Session,
) -> str:
    """
    Register a single PDF document in the ``documents`` table.

    Steps
    -----
    1. Log ``started`` event (document_id=None — row may not exist yet).
    2. Normalise *file_path* to a forward-slash relative path from PROJECT_ROOT.
    3. Check whether a row with that ``file_path`` already exists.
       * If it exists: log ``success`` / ``'skipped'``, return ``'skipped'``.
    4. Parse the filename to derive metadata (fiscal periods, category, etc.).
    5. Insert a new ``Document`` row; flush to obtain the new PK.
    6. Log ``success`` / ``'created'``, return ``'created'``.
    7. On any exception: log ``failed``, return ``'failed'`` (never re-raises).

    Args:
        company_id:    FK to ``companies.id`` (Req 4.4).
        file_path:     Absolute path to the PDF file on the filesystem.
        document_type: One of ``'annual_report'``, ``'quarterly_result'``,
                       ``'corporate_announcement'``, ``'credit_rating'``.
        exchange:      Exchange value from ``metadata.json`` (e.g. ``'BSE'``),
                       passed to :func:`~ingestion.filename_parser.parse_filename`
                       for source derivation.
        session:       Active SQLAlchemy session.

    Returns:
        ``'created'`` — new row inserted.
        ``'skipped'`` — row with same ``file_path`` already existed.
        ``'failed'``  — an error occurred; details logged in processing_log.
    """
    # (a) Log started before any work — document_id is None at this point
    # (Req 7.1, 7.7)
    log_started(document_id=None, session=session)

    try:
        # (b) Normalise file_path to forward-slash relative path (Req 4.5)
        try:
            relative_path = file_path.relative_to(PROJECT_ROOT)
        except ValueError:
            # file_path is not under PROJECT_ROOT; use as-is but normalised
            relative_path = file_path
        normalised_path: str = relative_path.as_posix()

        # (c) Deduplication check — query-then-insert for SQLite compatibility
        # (Req 4.2, 8.4)
        existing: Document | None = (
            session.query(Document).filter_by(file_path=normalised_path).first()
        )

        if existing is not None:
            # (c-i) Already registered — log skipped and return
            log_success(document_id=existing.id, message="skipped", session=session)
            return "skipped"

        # (d) Parse filename to derive metadata (Req 5.1–5.4)
        parsed = parse_filename(
            file_path=file_path,
            document_type=document_type,
            exchange=exchange,
        )

        # (e) Insert new Document row (Req 4.3, 4.4)
        doc = Document(
            company_id=company_id,
            file_name=file_path.name,
            file_path=normalised_path,
            document_type=parsed.document_type,
            document_category=parsed.document_category,
            fiscal_year=parsed.fiscal_year,
            fiscal_quarter=parsed.fiscal_quarter,
            period_start=parsed.period_start,
            period_end=parsed.period_end,
            source=parsed.source,
            # text_extracted, chunked, embedded — rely on ORM server_default=False
        )
        session.add(doc)
        session.flush()  # Populate doc.id without committing

        # (f) Log success with the new document's PK (Req 7.2, 7.7)
        log_success(document_id=doc.id, message="created", session=session)
        return "created"

    except Exception as exc:  # noqa: BLE001
        # (g) Log failure; never re-raise (Req 7.4, 7.5)
        log_failed(document_id=None, error=exc, session=session)
        return "failed"


# ---------------------------------------------------------------------------
# Bulk registration for a company
# ---------------------------------------------------------------------------


def register_documents(
    company_id: int,
    company_dir: Path,
    metadata: CompanyMetadata,
    session: Session,
) -> DocumentResult:
    """
    Scan all four standard subdirectories and register every PDF found.

    For each subdirectory in :data:`SUBDIR_TO_DOCTYPE`, this function calls
    :func:`scan_subdirectory` to collect PDF paths, then calls
    :func:`register_document` for each file.  Results are aggregated into a
    :class:`DocumentResult`.

    Missing subdirectories are silently skipped with a stderr warning (handled
    inside :func:`scan_subdirectory` — Req 4.7).

    Args:
        company_id:   FK to ``companies.id`` — set on every inserted document
                      row (Req 4.4).
        company_dir:  Absolute path to the company's root directory (e.g.
                      ``data/raw/DHFL/``).
        metadata:     Parsed :class:`~ingestion.metadata_reader.CompanyMetadata`
                      for this company; provides the ``exchange`` value used for
                      source derivation.
        session:      Active SQLAlchemy session.

    Returns:
        :class:`DocumentResult` with ``registered``, ``skipped``, and
        ``errors`` counts.
    """
    result = DocumentResult()

    for subdir_name, document_type in SUBDIR_TO_DOCTYPE.items():
        pdf_files = scan_subdirectory(
            company_dir=company_dir,
            subdir_name=subdir_name,
            document_type=document_type,
        )

        for pdf_path in pdf_files:
            outcome = register_document(
                company_id=company_id,
                file_path=pdf_path,
                document_type=document_type,
                exchange=metadata.exchange,
                session=session,
            )

            if outcome == "created":
                result.registered += 1
            elif outcome == "skipped":
                result.skipped += 1
            else:
                # 'failed'
                result.errors.append(pdf_path.name)

    return result
