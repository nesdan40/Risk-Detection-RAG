"""
prediction_loader.py — PostgreSQL data access for the Streamlit demo UI.

Provides all database queries needed by the Risk Report View and Evidence View.
Never uses SELECT * — all queries name columns explicitly.
Never references column names: group, label, distressed, healthy, ground_truth.
"""

from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy import text

# ── Rule weights ─────────────────────────────────────────────────────────────
# Defined independently — do NOT import from src/risk_engine/rules.py.
# Used by the UI to sort triggered rules by weight descending.

RULE_WEIGHTS: dict[str, float] = {
    "RATING_DEFAULT_GRADE":   0.12,
    "PLEDGE_RISING":          0.10,
    "INTEREST_COVERAGE_LOW":  0.10,
    "RATING_DOWNGRADE":       0.10,
    "PLEDGE_HIGH":            0.08,
    "NEGATIVE_PAT":           0.08,
    "DEBT_TO_EQUITY_HIGH":    0.08,
    "CFO_NEGATIVE":           0.08,
    "RESIGNATION_SPIKE":      0.07,
    "DIRECTOR_CHURN":         0.06,
    "CURRENT_RATIO_LOW":      0.06,
    "REVENUE_DECLINING":      0.06,
    "PROMOTER_HOLDING_LOW":   0.05,
    "AGM_DELAY":              0.05,
    "CLARIFICATION_SPIKE":    0.05,
}

TOTAL_RULES = 15


# ── Public API ────────────────────────────────────────────────────────────────


def get_companies(session: Session) -> list[dict]:
    """
    Return companies that have at least one rule_based_v1 prediction,
    ordered alphabetically by ticker.

    Selected columns: id, ticker, name only — no sector, listing_type,
    collection_window_start, collection_window_end, or created_at.

    Returns:
        List of dicts with keys 'id', 'ticker', 'name'.
        Empty list when no companies have predictions.
    """
    sql = text(
        "SELECT DISTINCT c.id, c.ticker, c.name "
        "FROM companies c "
        "INNER JOIN predictions p ON p.company_id = c.id "
        "WHERE p.model_version = 'rule_based_v1' "
        "ORDER BY c.ticker ASC"
    )
    result = session.execute(sql)
    rows = result.fetchall()
    return [{"id": row.id, "ticker": row.ticker, "name": row.name} for row in rows]


def get_latest_prediction(company_id: int, session: Session) -> dict | None:
    """
    Return the single most recent rule_based_v1 prediction for the company.

    Query:
        SELECT id, company_id, fiscal_year, fiscal_quarter, risk_score,
               risk_level, risk_flags, evidence_chunks, model_version, created_at
        FROM predictions
        WHERE company_id = :company_id
          AND model_version = 'rule_based_v1'
        ORDER BY created_at DESC
        LIMIT 1

    Returns:
        Dict with keys matching the selected columns, or None if no row exists.

    Raises:
        SQLAlchemyError — propagates to caller; app.py catches and displays error.
    """
    sql = text(
        "SELECT id, company_id, fiscal_year, fiscal_quarter, risk_score, "
        "       risk_level, risk_flags, evidence_chunks, model_version, created_at "
        "FROM predictions "
        "WHERE company_id = :company_id "
        "  AND model_version = 'rule_based_v1' "
        "ORDER BY created_at DESC "
        "LIMIT 1"
    )
    result = session.execute(sql, {"company_id": company_id})
    row = result.fetchone()
    if row is None:
        return None
    return {
        "id":              row.id,
        "company_id":      row.company_id,
        "fiscal_year":     row.fiscal_year,
        "fiscal_quarter":  row.fiscal_quarter,
        "risk_score":      row.risk_score,
        "risk_level":      row.risk_level,
        "risk_flags":      row.risk_flags,
        "evidence_chunks": row.evidence_chunks,
        "model_version":   row.model_version,
        "created_at":      row.created_at,
    }


def get_extracted_features(
    company_id: int,
    fiscal_year: str,
    fiscal_quarter: str | None,
    session: Session,
) -> list[dict]:
    """
    Return extracted_features rows for the given (company, period).

    Selected columns: id, company_id, document_id, fiscal_year, fiscal_quarter,
    feature_type, revenue, revenue_growth_pct, pat, ebitda_margin_pct,
    debt_to_equity, interest_coverage, current_ratio, cfo,
    board_meeting_count, director_change_count, appointment_count,
    resignation_count, agm_delay_flag, clarification_count, press_release_count,
    credit_rating, rating_outlook, rating_change_flag, created_at.

    Never selects additional_signals or any unlisted column.

    When fiscal_quarter is None, filters WHERE fiscal_quarter IS NULL.
    When fiscal_quarter is a string, filters WHERE fiscal_quarter = :fiscal_quarter.

    Returns:
        List of dicts. Empty list when no rows match.
    """
    if fiscal_quarter is None:
        sql = text(
            "SELECT id, company_id, document_id, fiscal_year, fiscal_quarter, "
            "       feature_type, revenue, revenue_growth_pct, pat, ebitda_margin_pct, "
            "       debt_to_equity, interest_coverage, current_ratio, cfo, "
            "       board_meeting_count, director_change_count, appointment_count, "
            "       resignation_count, agm_delay_flag, clarification_count, "
            "       press_release_count, credit_rating, rating_outlook, "
            "       rating_change_flag, created_at "
            "FROM extracted_features "
            "WHERE company_id = :company_id "
            "  AND fiscal_year = :fiscal_year "
            "  AND fiscal_quarter IS NULL"
        )
        params: dict = {"company_id": company_id, "fiscal_year": fiscal_year}
    else:
        sql = text(
            "SELECT id, company_id, document_id, fiscal_year, fiscal_quarter, "
            "       feature_type, revenue, revenue_growth_pct, pat, ebitda_margin_pct, "
            "       debt_to_equity, interest_coverage, current_ratio, cfo, "
            "       board_meeting_count, director_change_count, appointment_count, "
            "       resignation_count, agm_delay_flag, clarification_count, "
            "       press_release_count, credit_rating, rating_outlook, "
            "       rating_change_flag, created_at "
            "FROM extracted_features "
            "WHERE company_id = :company_id "
            "  AND fiscal_year = :fiscal_year "
            "  AND fiscal_quarter = :fiscal_quarter"
        )
        params = {
            "company_id":     company_id,
            "fiscal_year":    fiscal_year,
            "fiscal_quarter": fiscal_quarter,
        }

    result = session.execute(sql, params)
    rows = result.fetchall()

    columns = [
        "id", "company_id", "document_id", "fiscal_year", "fiscal_quarter",
        "feature_type", "revenue", "revenue_growth_pct", "pat", "ebitda_margin_pct",
        "debt_to_equity", "interest_coverage", "current_ratio", "cfo",
        "board_meeting_count", "director_change_count", "appointment_count",
        "resignation_count", "agm_delay_flag", "clarification_count",
        "press_release_count", "credit_rating", "rating_outlook",
        "rating_change_flag", "created_at",
    ]

    return [dict(zip(columns, row)) for row in rows]
