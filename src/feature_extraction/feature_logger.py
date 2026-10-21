"""
feature_logger — writes processing_log rows for the feature extraction pipeline.

Every feature extraction event is recorded here using the same savepoint pattern
as ingestion_logger.py: each insert runs inside ``session.begin_nested()`` so a
log failure never rolls back an ``extracted_features`` write that has already been
staged in the outer transaction (Requirements 7.7, 7.8).

Public functions
----------------
log_started             — status='started' for document-level extraction begin
log_success             — status='success' for document-level extraction complete
log_governance_success  — status='success', document_id=NULL, for a governance period
log_unmapped_category   — status='unmapped_category' when category cannot be resolved
log_failed              — status='failed' for any extraction failure

All functions are fire-and-forget: they never re-raise and never abort the caller.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from models.processing_log import ProcessingLog

if TYPE_CHECKING:
    # GovernanceResult is defined in governance_extractor to avoid circular
    # imports at runtime; we only need the type for annotation purposes.
    from feature_extraction.governance_extractor import GovernanceResult

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

STAGE: str = "feature_extraction"
MAX_MESSAGE_LEN: int = 1000


# ---------------------------------------------------------------------------
# Public logging functions
# ---------------------------------------------------------------------------


def log_started(document_id: int, document_type: str, session: Session) -> None:
    """
    Insert a processing_log row with status='started'.

    Called before any extraction is attempted for a document (Req 7.1, 7.6).
    Message format: 'Starting feature_extraction for document_id={id}, type={document_type}'

    Args:
        document_id:   ID of the document about to be processed.
        document_type: Value of ``documents.document_type`` (e.g. 'annual_report').
        session:       Active SQLAlchemy session.
    """
    message = (
        f"Starting feature_extraction for document_id={document_id}, "
        f"type={document_type}"
    )
    _insert_log(
        session=session,
        document_id=document_id,
        status="started",
        message=message,
    )


def log_success(
    document_id: int,
    feature_type: str,
    signals_populated: int,
    session: Session,
) -> None:
    """
    Insert a processing_log row with status='success' for a document-level extraction.

    Called after at least one ``extracted_features`` row is written (Req 7.2).
    Message format: 'feature_type={type}, signals_populated={N}'

    Args:
        document_id:       ID of the processed document.
        feature_type:      One of 'financial', 'credit', etc.
        signals_populated: Count of non-NULL typed columns written.
        session:           Active SQLAlchemy session.
    """
    message = f"feature_type={feature_type}, signals_populated={signals_populated}"
    _insert_log(
        session=session,
        document_id=document_id,
        status="success",
        message=message,
    )


def log_governance_success(
    ticker: str,
    fiscal_year: str,
    fiscal_quarter: str,
    result: "GovernanceResult",
    session: Session,
) -> None:
    """
    Insert a processing_log row for a governance period aggregate.

    Governance rows are period-level (not document-level), so document_id=NULL.
    Only a terminal success row is written — no 'started' row (per Req 7 note).
    Message format (Req 7.5):
        'governance extracted for company={ticker}, period={fy}/{fq},
         board_meeting_count={N}, director_change_count={N}, appointment_count={N},
         resignation_count={N}, resignation_count={N}, press_release_count={N},
         clarification_count={N}, agm_delay_flag={value}'

    Note: The requirements spec intentionally repeats 'resignation_count' twice
    in the message format template; the format below matches it exactly.

    Args:
        ticker:         Company ticker string.
        fiscal_year:    e.g. 'FY2019'.
        fiscal_quarter: e.g. 'Q1'.
        result:         GovernanceResult dataclass instance with all counts.
        session:        Active SQLAlchemy session.
    """
    message = (
        f"governance extracted for company={ticker}, "
        f"period={fiscal_year}/{fiscal_quarter}, "
        f"board_meeting_count={result.board_meeting_count}, "
        f"director_change_count={result.director_change_count}, "
        f"appointment_count={result.appointment_count}, "
        f"resignation_count={result.resignation_count}, "
        f"resignation_count={result.resignation_count}, "
        f"press_release_count={result.press_release_count}, "
        f"clarification_count={result.clarification_count}, "
        f"agm_delay_flag={result.agm_delay_flag}"
    )
    _insert_log(
        session=session,
        document_id=None,  # governance rows are not document-level
        status="success",
        message=message,
    )


def log_unmapped_category(
    document_id: int,
    document_category: str | None,
    file_name: str,
    ticker: str,
    session: Session,
) -> None:
    """
    Insert a processing_log row when a document's category cannot be resolved.

    The message MUST start with the literal token 'unmapped_category' as its
    first word so that the orchestrator can identify these rows reliably (Req 7.3).
    Message format:
        'unmapped_category {raw_category} | file={file_name} | company={ticker}
         | manual review required'

    Args:
        document_id:       ID of the unclassified document.
        document_category: Raw ``documents.document_category`` value (may be None).
        file_name:         ``documents.file_name`` value.
        ticker:            Company ticker for context.
        session:           Active SQLAlchemy session.
    """
    raw = document_category if document_category is not None else "(NULL)"
    message = (
        f"unmapped_category {raw} | file={file_name} | company={ticker} "
        f"| manual review required"
    )
    _insert_log(
        session=session,
        document_id=document_id,
        status="unmapped_category",
        message=message,
    )


def log_failed(
    document_id: int | None,
    error: Exception | str,
    session: Session,
) -> None:
    """
    Insert a processing_log row with status='failed'.

    The error detail is stored in message, truncated to MAX_MESSAGE_LEN characters
    so pathological exception messages don't overflow the Text column (Req 7.4).

    Args:
        document_id: ID of the document that failed, or None for period-level failures.
        error:       The exception or error string describing the failure.
        session:     Active SQLAlchemy session.
    """
    detail = str(error)[:MAX_MESSAGE_LEN]
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
    Core insert logic shared by all public logging functions.

    Uses a nested savepoint (``session.begin_nested()``) so that a log-insert
    failure does not roll back any work already flushed to the outer transaction
    (Requirements 7.7, 7.8).

    If the insert raises for any reason, a WARNING is printed to stderr and the
    function returns without re-raising — a log failure must never abort the
    feature extraction run.

    Args:
        session:     Active SQLAlchemy session.
        document_id: FK to documents.id; None for period-level events.
        status:      Processing log status string.
        message:     Free-text detail; may be None.
    """
    try:
        log_entry = ProcessingLog(
            document_id=document_id,
            stage=STAGE,
            status=status,
            message=message,
        )
        with session.begin_nested():
            session.add(log_entry)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: Failed to write processing_log entry "
            f"(stage={STAGE!r}, status={status!r}, document_id={document_id}): "
            f"{exc}",
            file=sys.stderr,
        )
