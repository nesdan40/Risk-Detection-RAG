"""
RiskLogger — writes processing_log rows for the risk scoring pipeline.

Stage covered by this module:

  risk_scoring
    log_risk_scoring_started  — before rule evaluation begins (status='started')
    log_risk_scoring_success  — after prediction is successfully upserted (status='success')
    log_risk_scoring_failed   — when scoring raises an exception (status='failed')

All rows written here set document_id=None because risk scoring is a
company+period operation, not a document-level operation (Requirement 7.4).

If a processing_log INSERT itself fails, a warning is printed to stderr and
the function returns silently — a log failure must never abort the outer
period-scoring transaction (Requirements 7.1–7.4).
"""

import sys

from sqlalchemy.orm import Session

from models import ProcessingLog

# Stage name constant used as the `stage` column value in processing_log.
STAGE_RISK_SCORING: str = "risk_scoring"

# Maximum number of characters stored in the `message` column.  Prevents
# pathological exception messages from overflowing the Text column.
_MAX_MESSAGE_LEN: int = 1000


# ---------------------------------------------------------------------------
# Public API — risk_scoring stage
# ---------------------------------------------------------------------------


def log_risk_scoring_started(
    company_id: int,
    ticker: str,
    fiscal_year: str,
    fiscal_quarter: str | None,
    session: Session,
) -> None:
    """
    Insert a processing_log row with stage='risk_scoring', status='started'.

    Called immediately before rule evaluation begins for a period.

    Args:
        company_id:     Company being scored (unused in log message; document_id=None).
        ticker:         Company ticker for human-readable log message.
        fiscal_year:    Fiscal year string, e.g. 'FY2018'.
        fiscal_quarter: Fiscal quarter string ('Q1'–'Q4') or None for annual periods.
        session:        SQLAlchemy session to use for the insert.
    """
    period_str = f"{fiscal_year} {fiscal_quarter}" if fiscal_quarter else fiscal_year
    message = f"Scoring {ticker} {period_str}"
    _insert_log(
        session=session,
        document_id=None,
        stage=STAGE_RISK_SCORING,
        status="started",
        message=message,
    )


def log_risk_scoring_success(
    company_id: int,
    ticker: str,
    fiscal_year: str,
    fiscal_quarter: str | None,
    risk_level: str,
    n_triggered: int,
    session: Session,
) -> None:
    """
    Insert a processing_log row with stage='risk_scoring', status='success'.

    Called immediately after the prediction row is successfully upserted.

    Message format: "Scored {ticker} {period_str}: {risk_level}, {n_triggered} rules triggered"
    e.g. "Scored DHFL FY2018 Q3: High, 5 rules triggered"

    Args:
        company_id:     Company being scored (unused in log message; document_id=None).
        ticker:         Company ticker.
        fiscal_year:    Fiscal year string, e.g. 'FY2018'.
        fiscal_quarter: Fiscal quarter string ('Q1'–'Q4') or None for annual periods.
        risk_level:     Assigned risk level ('Low', 'Medium', 'High', 'Critical').
        n_triggered:    Number of rules that fired for this period.
        session:        SQLAlchemy session to use for the insert.
    """
    period_str = f"{fiscal_year} {fiscal_quarter}" if fiscal_quarter else fiscal_year
    message = f"Scored {ticker} {period_str}: {risk_level}, {n_triggered} rules triggered"
    _insert_log(
        session=session,
        document_id=None,
        stage=STAGE_RISK_SCORING,
        status="success",
        message=message,
    )


def log_risk_scoring_failed(
    company_id: int,
    ticker: str,
    fiscal_year: str,
    fiscal_quarter: str | None,
    error: str,
    session: Session,
) -> None:
    """
    Insert a processing_log row with stage='risk_scoring', status='failed'.

    The error detail is stored in message, truncated to _MAX_MESSAGE_LEN
    characters so that pathological exception messages don't overflow the
    Text column.

    Args:
        company_id:     Company being scored (unused in log message; document_id=None).
        ticker:         Company ticker.
        fiscal_year:    Fiscal year string, e.g. 'FY2018'.
        fiscal_quarter: Fiscal quarter string ('Q1'–'Q4') or None for annual periods.
        error:          Error detail string (exception message or custom message).
        session:        SQLAlchemy session to use for the insert.
    """
    period_str = f"{fiscal_year} {fiscal_quarter}" if fiscal_quarter else fiscal_year
    error_truncated = error[:_MAX_MESSAGE_LEN]
    message = f"Failed {ticker} {period_str}: {error_truncated}"
    _insert_log(
        session=session,
        document_id=None,
        stage=STAGE_RISK_SCORING,
        status="failed",
        message=message,
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

    Uses a nested savepoint (session.begin_nested()) so that a log-insert
    failure doesn't roll back any work already flushed to the outer
    transaction.  If the insert itself raises, the savepoint is rolled back,
    a warning is printed to stderr, and the function returns without
    re-raising — a log failure must never abort period scoring.

    Args:
        session:     SQLAlchemy session to use for the insert.
        document_id: Always None for risk_scoring stage (period-level, not doc-level).
        stage:       Processing stage name ('risk_scoring').
        status:      Event status ('started', 'success', or 'failed').
        message:     Detail message describing the event.
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
