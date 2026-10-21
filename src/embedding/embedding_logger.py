"""
EmbeddingLogger — writes processing_log rows for the embedding pipeline.

Stage covered by this module:

  embedding
    log_embedding_started  — before attempting embedding (status='started')
    log_embedding_success  — after a successful embedding attempt (status='success')
    log_embedding_failed   — when embedding raises an exception (status='failed')

If a processing_log INSERT itself fails, a warning is printed to stderr and
the method returns silently — a log failure must never abort document processing
(Requirements 7.1–7.6).
"""

import sys

from sqlalchemy.orm import Session

from models.processing_log import ProcessingLog

# Stage name constant used as the `stage` column value in processing_log.
STAGE_EMBEDDING: str = "embedding"

# Maximum number of characters stored in the `message` column.  Prevents
# pathological exception messages from overflowing the Text column.
_MAX_MESSAGE_LEN: int = 1000


# ---------------------------------------------------------------------------
# Public API — embedding stage
# ---------------------------------------------------------------------------


def log_embedding_started(document_id: int, session: Session) -> None:
    """
    Insert a processing_log row with stage='embedding', status='started'.

    Called immediately before any embedding attempt begins for a document.

    Args:
        document_id: ID of the document row being processed.
        session:     SQLAlchemy session to use for the insert.
    """
    _insert_log(
        session=session,
        document_id=document_id,
        stage=STAGE_EMBEDDING,
        status="started",
        message=None,
    )


def log_embedding_success(
    document_id: int,
    chunk_count: int,
    session: Session,
) -> None:
    """
    Insert a processing_log row with stage='embedding', status='success'.

    The message distinguishes a normal embedding from a zero-chunk outcome:
      - chunk_count == 0  →  message = 'no_chunks: nothing to embed'
      - chunk_count >  0  →  message = 'embedded: {chunk_count} chunks'

    Args:
        document_id: ID of the document row being processed.
        chunk_count: Number of chunks that were embedded.
        session:     SQLAlchemy session to use for the insert.
    """
    if chunk_count == 0:
        message = "no_chunks: nothing to embed"
    else:
        message = f"embedded: {chunk_count} chunks"

    _insert_log(
        session=session,
        document_id=document_id,
        stage=STAGE_EMBEDDING,
        status="success",
        message=message,
    )


def log_embedding_failed(
    document_id: int,
    error: Exception,
    session: Session,
) -> None:
    """
    Insert a processing_log row with stage='embedding', status='failed'.

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
        stage=STAGE_EMBEDDING,
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
    the function returns without re-raising (Requirements 7.1–7.6).

    Args:
        session:     SQLAlchemy session to use for the insert.
        document_id: ID of the document row being processed (may be None).
        stage:       Processing stage name ('embedding').
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
