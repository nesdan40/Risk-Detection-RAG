"""
IngestionLogger — writes processing_log rows for the document ingestion pipeline.

Every document ingestion attempt is recorded in three events:
  1. log_started  — before attempting registration (document_id=None)
  2. log_success  — after a successful insert ('created') or skip ('skipped')
  3. log_failed   — when registration fails (message truncated to 1000 chars)

If the processing_log INSERT itself fails, a warning is printed to stderr and
the method returns silently — a log failure must never abort document processing
(Requirement 7.6).
"""

import sys

from sqlalchemy.orm import Session

from models.processing_log import ProcessingLog

_STAGE = "ingestion"
_MAX_MESSAGE_LEN = 1000


def log_started(document_id: int | None, session: Session) -> None:
    """
    Insert a processing_log row with status='started'.

    Called before attempting document registration.  document_id is always
    None at this point because the documents row may not exist yet.

    Args:
        document_id: Always None when called at the start of ingestion.
        session:     SQLAlchemy session to use for the insert.
    """
    _insert_log(
        session=session,
        document_id=document_id,
        status="started",
        message=None,
    )


def log_success(document_id: int, message: str, session: Session) -> None:
    """
    Insert a processing_log row with status='success'.

    Args:
        document_id: ID of the newly created or already-existing document row.
        message:     'created' if a new row was inserted, 'skipped' if the
                     document already existed and was left unchanged.
        session:     SQLAlchemy session to use for the insert.
    """
    _insert_log(
        session=session,
        document_id=document_id,
        status="success",
        message=message,
    )


def log_failed(
    document_id: int | None, error: Exception, session: Session
) -> None:
    """
    Insert a processing_log row with status='failed'.

    The error detail is stored in message, truncated to 1000 characters so
    that pathological exception messages don't blow out the Text column.

    Args:
        document_id: ID of the document row if it could be resolved, else None.
        error:       The exception that caused the failure.
        session:     SQLAlchemy session to use for the insert.
    """
    detail = str(error)[:_MAX_MESSAGE_LEN]
    _insert_log(
        session=session,
        document_id=document_id,
        status="failed",
        message=detail,
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _insert_log(
    session: Session,
    document_id: int | None,
    status: str,
    message: str | None,
) -> None:
    """
    Core insert logic shared by all three public functions.

    Uses a nested savepoint so that a log-insert failure doesn't roll back
    any work already flushed to the outer transaction.  If the insert itself
    raises, the savepoint is rolled back, a warning is printed to stderr, and
    the function returns without re-raising (Requirement 7.6).
    """
    try:
        log_entry = ProcessingLog(
            document_id=document_id,
            stage=_STAGE,
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
            f"(stage={_STAGE!r}, status={status!r}, document_id={document_id}): "
            f"{exc}",
            file=sys.stderr,
        )
