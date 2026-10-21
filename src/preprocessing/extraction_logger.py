"""
ExtractionLogger — writes processing_log rows for the PDF text extraction
and chunking pipeline.

Two stages are covered by this module:

  text_extraction
    log_extraction_started  — before attempting extraction (status='started')
    log_extraction_success  — after a successful extraction attempt (status='success')
    log_extraction_failed   — when extraction raises an exception (status='failed')

  chunking
    log_chunking_started    — before chunking begins (status='started')
    log_chunking_success    — after chunks are written to disk (status='success')
    log_chunking_failed     — when chunking raises an exception (status='failed')

If a processing_log INSERT itself fails, a warning is printed to stderr and
the method returns silently — a log failure must never abort document processing
(Requirements 3.6, 7.7).
"""

import sys

from sqlalchemy.orm import Session

from models.processing_log import ProcessingLog

# Stage name constants used as the `stage` column value in processing_log.
STAGE_EXTRACTION: str = "text_extraction"
STAGE_CHUNKING: str = "chunking"

# Maximum number of characters stored in the `message` column.  Prevents
# pathological exception messages from overflowing the Text column.
_MAX_MESSAGE_LEN: int = 1000

# Minimum character count for extracted text to be considered non-low-yield.
_LOW_YIELD_THRESHOLD: int = 50


# ---------------------------------------------------------------------------
# Public API — text_extraction stage
# ---------------------------------------------------------------------------


def log_extraction_started(document_id: int, session: Session) -> None:
    """
    Insert a processing_log row with stage='text_extraction', status='started'.

    Called immediately before any extraction attempt begins for a document.

    Args:
        document_id: ID of the document row being processed.
        session:     SQLAlchemy session to use for the insert.
    """
    _insert_log(
        session=session,
        document_id=document_id,
        stage=STAGE_EXTRACTION,
        status="started",
        message=None,
    )


def log_extraction_success(
    document_id: int,
    char_count: int,
    session: Session,
    detail: str | None = None,
) -> None:
    """
    Insert a processing_log row with stage='text_extraction', status='success'.

    The message distinguishes a normal extraction from a low-yield outcome:
      - char_count >= 50  →  message = 'extracted'
      - char_count <  50  →  message = 'low_yield: {char_count} chars extracted;
                              likely scanned PDF'

    Args:
        document_id: ID of the document row being processed.
        char_count:  Number of characters in the stripped extracted text.
        session:     SQLAlchemy session to use for the insert.
    """
    if char_count >= _LOW_YIELD_THRESHOLD:
        message = "extracted"
    else:
        message = f"low_yield: {char_count} chars extracted; likely scanned PDF"
    if detail:
        message = f"{message} | {detail}"[:1000]

    _insert_log(
        session=session,
        document_id=document_id,
        stage=STAGE_EXTRACTION,
        status="success",
        message=message,
    )


def log_extraction_failed(
    document_id: int,
    error: Exception,
    session: Session,
) -> None:
    """
    Insert a processing_log row with stage='text_extraction', status='failed'.

    The error detail is stored in message, truncated to _MAX_MESSAGE_LEN
    characters so that pathological exception messages don't blow out the
    Text column.

    Args:
        document_id: ID of the document row being processed.
        error:       The exception that caused the failure.
        session:     SQLAlchemy session to use for the insert.
    """
    _insert_log(
        session=session,
        document_id=document_id,
        stage=STAGE_EXTRACTION,
        status="failed",
        message=str(error)[:_MAX_MESSAGE_LEN],
    )


# ---------------------------------------------------------------------------
# Public API — chunking stage
# ---------------------------------------------------------------------------


def log_chunking_started(document_id: int, session: Session) -> None:
    """
    Insert a processing_log row with stage='chunking', status='started'.

    Called immediately before chunking begins for a document.

    Args:
        document_id: ID of the document row being processed.
        session:     SQLAlchemy session to use for the insert.
    """
    _insert_log(
        session=session,
        document_id=document_id,
        stage=STAGE_CHUNKING,
        status="started",
        message=None,
    )


def log_chunking_success(
    document_id: int,
    chunk_count: int,
    session: Session,
) -> None:
    """
    Insert a processing_log row with stage='chunking', status='success'.

    The message distinguishes a normal write from a zero-chunk outcome:
      - chunk_count > 0  →  message = 'chunks_written: {chunk_count}'
      - chunk_count == 0 →  message = 'no_chunks: text below threshold'

    Args:
        document_id: ID of the document row being processed.
        chunk_count: Number of chunks written to the JSONL file.
        session:     SQLAlchemy session to use for the insert.
    """
    if chunk_count > 0:
        message = f"chunks_written: {chunk_count}"
    else:
        message = "no_chunks: text below threshold"

    _insert_log(
        session=session,
        document_id=document_id,
        stage=STAGE_CHUNKING,
        status="success",
        message=message,
    )


def log_chunking_failed(
    document_id: int,
    error: Exception,
    session: Session,
) -> None:
    """
    Insert a processing_log row with stage='chunking', status='failed'.

    The error detail is stored in message, truncated to _MAX_MESSAGE_LEN
    characters.

    Args:
        document_id: ID of the document row being processed.
        error:       The exception that caused the failure.
        session:     SQLAlchemy session to use for the insert.
    """
    _insert_log(
        session=session,
        document_id=document_id,
        stage=STAGE_CHUNKING,
        status="failed",
        message=str(error)[:_MAX_MESSAGE_LEN],
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _insert_log(
    session: Session,
    document_id: int | None,
    stage: str,
    status: str,
    message: str | None,
) -> None:
    """
    Core insert logic shared by all public functions.

    Uses a nested savepoint so that a log-insert failure doesn't roll back
    any work already flushed to the outer transaction.  If the insert itself
    raises, the savepoint is rolled back, a warning is printed to stderr, and
    the function returns without re-raising (Requirements 3.6, 7.7).

    Args:
        session:     SQLAlchemy session to use for the insert.
        document_id: ID of the document row being processed (may be None).
        stage:       Processing stage name ('text_extraction' or 'chunking').
        status:      Event status ('started', 'success', or 'failed').
        message:     Optional detail message; None for started rows.
    """
    try:
        log_entry = ProcessingLog(
            document_id=document_id,
            stage=stage,
            status=status,
            message=message,
        )
        # Use a nested transaction (SAVEPOINT) so a failure here doesn't
        # corrupt the outer session state.
        with session.begin_nested():
            session.add(log_entry)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: Failed to write processing_log entry "
            f"(stage={stage!r}, status={status!r}, document_id={document_id}): "
            f"{exc}",
            file=sys.stderr,
        )
