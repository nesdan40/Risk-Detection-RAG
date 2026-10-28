"""
CLI entry point and orchestrator for the rule-based risk engine.

This module ties together all risk engine components:
  - Period discovery from PostgreSQL (extracted_features + shareholding_patterns)
  - Feature loading per (company_id, fiscal_year, fiscal_quarter) period
  - Rule evaluation via scorer.py
  - Evidence retrieval via evidence_retriever.py
  - Prediction upsert via prediction_writer.py
  - Audit logging via risk_logger.py

Usage:
    python -m src.risk_engine.run_risk_engine
    python -m src.risk_engine.run_risk_engine --company DHFL
    python -m src.risk_engine.run_risk_engine --fiscal-year FY2018
    python -m src.risk_engine.run_risk_engine --company DHFL --fiscal-year FY2018
    python -m src.risk_engine.run_risk_engine --company DHFL --model-version rule_based_v2
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from database.postgres import get_engine
from models import Company, ExtractedFeature, ShareholdingPattern
from risk_engine.scorer import evaluate_rules, evaluate_rules_v2
from risk_engine.evidence_retriever import retrieve_evidence
from risk_engine.prediction_writer import (
    MODEL_VERSION,
    MODEL_VERSION_V2,
    MODEL_VERSIONS,
    upsert_prediction,
)
from risk_engine.risk_logger import (
    log_risk_scoring_started,
    log_risk_scoring_success,
    log_risk_scoring_failed,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PeriodIdentifier:
    """
    Represents a single scoreable (company, period) combination discovered
    from the database during period selection.

    Attributes:
        company_id:     Primary key from companies table.
        ticker:         Company ticker symbol (e.g. 'DHFL').
        fiscal_year:    e.g. 'FY2018'.
        fiscal_quarter: e.g. 'Q3', or None for annual-level periods.
    """

    company_id: int
    ticker: str
    fiscal_year: str
    fiscal_quarter: str | None


@dataclass
class ScoringResult:
    """
    Result of processing a single (company, period) combination.

    Attributes:
        company_id:     Primary key of the scored company.
        ticker:         Company ticker symbol.
        fiscal_year:    Fiscal year of the scored period.
        fiscal_quarter: Fiscal quarter of the scored period (None for annual).
        outcome:        'scored' — prediction written successfully.
                        'failed' — unhandled exception; prediction not written.
                        'skipped' — zero feature rows; scoring not attempted.
        risk_level:     'Low' | 'Medium' | 'High' | 'Critical', or None on failure/skip.
        n_triggered:    Number of rules that fired (0 on failure or skip).
        coverage:       v2 only — evaluable rule weight / total weight; None otherwise.
    """

    company_id: int
    ticker: str
    fiscal_year: str
    fiscal_quarter: str | None
    outcome: Literal["scored", "failed", "skipped"]
    risk_level: str | None
    n_triggered: int
    coverage: float | None = None


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


def get_qdrant_client() -> QdrantClient:
    """
    Create a QdrantClient using QDRANT_HOST and QDRANT_PORT environment variables.

    Environment variables:
        QDRANT_HOST: Hostname or IP of the Qdrant server. Default: 'localhost'.
        QDRANT_PORT: Port number of the Qdrant server. Default: 6333.

    Returns:
        A connected QdrantClient instance.

    Raises:
        Exception: If the underlying QdrantClient constructor fails.
    """
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def load_embedding_model() -> SentenceTransformer:
    """
    Load the BAAI/bge-base-en-v1.5 sentence transformer model for evidence
    retrieval query embedding.

    This is the same model used during the embedding pipeline, ensuring that
    query vectors are in the same vector space as the stored chunk embeddings.

    Returns:
        A loaded SentenceTransformer instance.

    Raises:
        Exception: If the model cannot be loaded (e.g. network or disk error).
    """
    from config.settings import EMBEDDING_MODEL

    return SentenceTransformer(EMBEDDING_MODEL or "BAAI/bge-base-en-v1.5")


# ---------------------------------------------------------------------------
# Period sorting helper
# ---------------------------------------------------------------------------


def _quarter_sort_key(quarter: str | None) -> tuple:
    """
    Sort key for fiscal quarters, ordering Q1 < Q2 < Q3 < Q4 < None.

    None (annual period) sorts after all named quarters so that quarterly
    periods are processed before the annual roll-up for the same fiscal year.

    Returns a 2-tuple for standard tuple comparison:
        'Q1'  → (0, 0)
        'Q2'  → (0, 1)
        'Q3'  → (0, 2)
        'Q4'  → (0, 3)
        None  → (1, 0)   — sorts after all named quarters
    """
    if quarter is None:
        return (1, 0)
    quarter_order = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    return (0, quarter_order.get(quarter, 999))


# ---------------------------------------------------------------------------
# Ticker validation
# ---------------------------------------------------------------------------


def _validate_ticker(ticker: str, session: Session) -> None:
    """
    Verify that a ticker exists in the companies table.

    If the ticker is not found, prints an error message to stderr and calls
    sys.exit(1) so the engine halts before attempting any scoring.

    Args:
        ticker:  Ticker string to look up — matched case-sensitively against
                 companies.ticker.
        session: An active SQLAlchemy Session.

    Returns:
        None — returns normally when the ticker is found.

    Raises:
        SystemExit(1): If the ticker is not present in the companies table.
    """
    count = session.scalar(
        select(func.count()).select_from(Company).where(Company.ticker == ticker)
    )
    if not count:
        print(
            f"ERROR: Company ticker '{ticker}' not found in database.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Period discovery
# ---------------------------------------------------------------------------


def get_scoreable_periods(
    session: Session,
    company_filter: str | None = None,
    fiscal_year_filter: str | None = None,
) -> list[PeriodIdentifier]:
    """
    Discover all (company_id, ticker, fiscal_year, fiscal_quarter) combinations
    that have at least one row in extracted_features OR shareholding_patterns.

    Filters are applied before deduplication:
    - company_filter: restrict to the exact ticker (case-sensitive); calls
      _validate_ticker and exits with code 1 if not found.
    - fiscal_year_filter: restrict to the given fiscal year string (e.g. 'FY2018').

    Periods are sorted ascending by (fiscal_year, fiscal_quarter), with
    fiscal_quarter=None sorting after 'Q4' within the same fiscal year.

    Args:
        session:             Active SQLAlchemy Session.
        company_filter:      Optional ticker to restrict processing.
        fiscal_year_filter:  Optional fiscal year string to restrict processing.

    Returns:
        List of PeriodIdentifier, sorted deterministically.

    Raises:
        SystemExit(1): If company_filter ticker is not found in companies table,
                       or if a SQLAlchemyError occurs during querying.
        SystemExit(0): If no scoreable periods are found after applying all filters.
    """
    # Validate ticker first — exits immediately if not found.
    if company_filter is not None:
        _validate_ticker(company_filter, session)

    try:
        # --- Query 1: distinct periods from extracted_features ---
        q_features = (
            select(
                ExtractedFeature.company_id,
                Company.ticker,
                ExtractedFeature.fiscal_year,
                ExtractedFeature.fiscal_quarter,
            )
            .join(Company, ExtractedFeature.company_id == Company.id)
            .distinct()
        )

        if company_filter is not None:
            q_features = q_features.where(Company.ticker == company_filter)
        if fiscal_year_filter is not None:
            q_features = q_features.where(
                ExtractedFeature.fiscal_year == fiscal_year_filter
            )

        # --- Query 2: distinct periods from shareholding_patterns ---
        q_shareholding = (
            select(
                ShareholdingPattern.company_id,
                Company.ticker,
                ShareholdingPattern.fiscal_year,
                ShareholdingPattern.fiscal_quarter,
            )
            .join(Company, ShareholdingPattern.company_id == Company.id)
            .distinct()
        )

        if company_filter is not None:
            q_shareholding = q_shareholding.where(Company.ticker == company_filter)
        if fiscal_year_filter is not None:
            q_shareholding = q_shareholding.where(
                ShareholdingPattern.fiscal_year == fiscal_year_filter
            )

        # --- Union into a Python set to deduplicate ---
        periods_set: set[tuple[int, str, str, str | None]] = set()

        for row in session.execute(q_features):
            periods_set.add(
                (row.company_id, row.ticker, row.fiscal_year, row.fiscal_quarter)
            )

        for row in session.execute(q_shareholding):
            periods_set.add(
                (row.company_id, row.ticker, row.fiscal_year, row.fiscal_quarter)
            )

    except SQLAlchemyError as exc:
        print(
            f"ERROR: Failed to retrieve scoreable periods: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # No periods found after all filters.
    if not periods_set:
        print("No scoreable periods found for the given filters.")
        sys.exit(0)

    # Convert to PeriodIdentifier list and sort deterministically.
    periods = [
        PeriodIdentifier(
            company_id=cid,
            ticker=ticker,
            fiscal_year=fy,
            fiscal_quarter=fq,
        )
        for cid, ticker, fy, fq in periods_set
    ]

    periods.sort(key=lambda p: (p.fiscal_year or "", _quarter_sort_key(p.fiscal_quarter)))

    return periods


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_period_features(
    company_id: int,
    fiscal_year: str,
    fiscal_quarter: str | None,
    session: Session,
) -> "PeriodFeatureSet":
    """
    Load all structured feature data for a single (company, period) from PostgreSQL.

    Queries `extracted_features` for the exact (company_id, fiscal_year,
    fiscal_quarter) combination across all feature_type values, then queries
    `shareholding_patterns` for the entire fiscal year (not just the scored
    quarter) so that PLEDGE_RISING can compare across consecutive quarters.

    The extracted_features rows are split into three typed lists by feature_type:
        - financial_features:   feature_type = 'financial'
        - governance_features:  feature_type = 'governance'
        - credit_features:      feature_type = 'credit'

    Shareholding patterns are sorted ascending by fiscal_quarter (NULL / annual
    rows sort last via _quarter_sort_key) so the PLEDGE_RISING evaluator can
    iterate them in quarter order without re-sorting.

    The Scorer receives the returned PeriodFeatureSet and never issues its own
    database queries (Requirement 3.4).

    Args:
        company_id:     Primary key of the company being scored.
        fiscal_year:    e.g. 'FY2018'.
        fiscal_quarter: e.g. 'Q3', or None for an annual-level period.
        session:        Active SQLAlchemy Session.

    Returns:
        PeriodFeatureSet populated with all available rows for the period.
        Any of the feature lists may be empty if no data exists for that type —
        the Scorer handles partial feature sets gracefully (Requirement 3.5).
    """
    from risk_engine.scorer import PeriodFeatureSet
    from sqlalchemy import and_

    # ------------------------------------------------------------------
    # Query 1: extracted_features — all feature_type values for this period
    # ------------------------------------------------------------------
    stmt_features = select(ExtractedFeature).where(
        and_(
            ExtractedFeature.company_id == company_id,
            ExtractedFeature.fiscal_year == fiscal_year,
            ExtractedFeature.fiscal_quarter == fiscal_quarter,
        )
    )
    all_features: list[ExtractedFeature] = list(session.scalars(stmt_features))

    financial_features = [f for f in all_features if f.feature_type == "financial"]
    governance_features = [f for f in all_features if f.feature_type == "governance"]
    credit_features = [f for f in all_features if f.feature_type == "credit"]

    # Governance rows exist per quarter only; an annual period sees the year's worst quarter.
    if fiscal_quarter is None and not governance_features:
        quarterly_gov = list(session.scalars(
            select(ExtractedFeature).where(
                and_(
                    ExtractedFeature.company_id == company_id,
                    ExtractedFeature.fiscal_year == fiscal_year,
                    ExtractedFeature.fiscal_quarter.is_not(None),
                    ExtractedFeature.feature_type == "governance",
                )
            )
        ))
        rolled = rollup_governance(quarterly_gov)
        if rolled is not None:
            governance_features = [rolled]

    # Credit ratings are dated documents.  Only ratings issued on or before the end
    # of the scored period may be used (otherwise a later downgrade would "explain"
    # an earlier period), and the most recent such rating wins.
    from feature_extraction.fiscal_utils import calculate_period_end

    period_end = calculate_period_end(fiscal_year, fiscal_quarter)
    credit_features = list(session.scalars(
        select(ExtractedFeature)
        .where(
            and_(
                ExtractedFeature.company_id == company_id,
                ExtractedFeature.feature_type == "credit",
                ExtractedFeature.period_end.is_not(None),
                ExtractedFeature.period_end <= period_end,
            )
        )
        .order_by(ExtractedFeature.period_end.desc(), ExtractedFeature.document_id.desc())
    ))

    # ------------------------------------------------------------------
    # Query 2: shareholding_patterns — full FY (all quarters) for the company.
    # Fetch all rows for the fiscal year regardless of quarter so that
    # cross-quarter rules (e.g. PLEDGE_RISING) see the full quarterly history.
    # ------------------------------------------------------------------
    stmt_shareholding = (
        select(ShareholdingPattern)
        .where(
            and_(
                ShareholdingPattern.company_id == company_id,
                ShareholdingPattern.fiscal_year == fiscal_year,
            )
        )
    )
    shareholding_rows: list[ShareholdingPattern] = list(
        session.scalars(stmt_shareholding)
    )

    # Sort at the Python level so NULL fiscal_quarter sorts after Q4 (not before
    # Q1 as most SQL engines do for NULL in ascending ORDER BY).
    shareholding_rows.sort(key=lambda sp: _quarter_sort_key(sp.fiscal_quarter))

    # ------------------------------------------------------------------
    # Assemble and return the PeriodFeatureSet
    # ------------------------------------------------------------------
    return PeriodFeatureSet(
        company_id=company_id,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        financial_features=financial_features,
        governance_features=governance_features,
        credit_features=credit_features,
        shareholding_patterns=shareholding_rows,
    )


def rollup_governance(rows: list[ExtractedFeature]) -> ExtractedFeature | None:
    """
    Annual governance signals from the fiscal year's quarterly rows, as the worst quarter.

    The governance rules are single-quarter spikes, so counts are never summed:
      - resignation_count   = max over quarters
      - clarification_count = max over quarters
      - director_change_count is set so resignation + director_change equals the largest
        single-quarter churn (DIRECTOR_CHURN keeps its one-quarter meaning)
      - agm_delay_flag      = True if any quarter is True, else False if any is False, else None

    Returns a transient (never added to the session) row carrying the id and quarter of
    the worst-churn quarter, so evidence is retrieved from that quarter's announcements.
    Returns None when *rows* is empty.
    """
    if not rows:
        return None

    def _max(attr: str) -> int | None:
        vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        return max(vals) if vals else None

    def _churn(r: ExtractedFeature) -> int | None:
        if r.resignation_count is None and r.director_change_count is None:
            return None
        return (r.resignation_count or 0) + (r.director_change_count or 0)

    churned = [r for r in rows if _churn(r) is not None]
    worst = max(churned, key=_churn) if churned else rows[0]
    max_resignation = _max("resignation_count")
    director_change = None
    if churned:
        director_change = _churn(worst) - (max_resignation or 0)

    flags = [r.agm_delay_flag for r in rows if r.agm_delay_flag is not None]
    agm_flag = True if any(flags) else (False if flags else None)

    return ExtractedFeature(
        id=worst.id,
        company_id=worst.company_id,
        document_id=None,
        fiscal_year=worst.fiscal_year,
        fiscal_quarter=worst.fiscal_quarter,
        feature_type="governance",
        resignation_count=max_resignation,
        director_change_count=director_change,
        clarification_count=_max("clarification_count"),
        agm_delay_flag=agm_flag,
        additional_signals={
            "rollup": "annual = worst quarter (max), not a sum",
            "source_row_ids": sorted(r.id for r in rows),
        },
    )


# ---------------------------------------------------------------------------
# Period processing
# ---------------------------------------------------------------------------


_SHAREHOLDING_RULES = {"PLEDGE_RISING", "PLEDGE_HIGH", "PROMOTER_HOLDING_LOW"}
_CREDIT_RULES = {"RATING_DOWNGRADE", "RATING_DEFAULT_GRADE"}
# financial rule -> the feature whose provenance (page, row label) locates its evidence
_FINANCIAL_RULE_METRIC = {
    "INTEREST_COVERAGE_LOW": "interest_coverage", "NEGATIVE_PAT": "pat",
    "DEBT_TO_EQUITY_HIGH": "debt_to_equity", "CURRENT_RATIO_LOW": "current_ratio",
    "CFO_NEGATIVE": "cfo", "REVENUE_DECLINING": "revenue",
}
# computed ratios record a page but no row label; anchor on the row they are built from
_COMPUTED_RATIO_ANCHOR = {
    "INTEREST_COVERAGE_LOW": "finance cost", "CURRENT_RATIO_LOW": "current assets",
    "DEBT_TO_EQUITY_HIGH": "borrowings",
}
_GOVERNANCE_RULE_CATEGORIES = {
    "RESIGNATION_SPIKE": ["Resignation"],
    "DIRECTOR_CHURN": ["Resignation", "Director Change"],
    "CLARIFICATION_SPIKE": ["Press Release", "News Clarification", "News Verification"],
    "AGM_DELAY": ["AGM / Annual General Meeting Updates"],
}


def build_evidence_scopes(feature_set, triggered_rules) -> dict[str, dict]:
    """
    For each triggered rule, describe where its evidence must come from, based on the
    feature row that fired it (financial/credit -> that source document; governance ->
    that period's announcements of the relevant categories; shareholding -> the
    shareholding row itself, which has no PDF).
    """
    by_id = {
        f.id: f
        for f in (feature_set.financial_features + feature_set.governance_features
                  + feature_set.credit_features)
    }
    sp_by_id = {sp.id: sp for sp in feature_set.shareholding_patterns}
    scopes: dict[str, dict] = {}
    for rule in triggered_rules:
        if rule.rule_id in _SHAREHOLDING_RULES:
            sp = sp_by_id.get(rule.feature_row_id)
            if sp is not None:
                scopes[rule.rule_id] = {
                    "kind": "shareholding",
                    "fiscal_year": sp.fiscal_year,
                    "fiscal_quarter": sp.fiscal_quarter,
                    "note": (
                        f"shareholding_patterns {sp.fiscal_year} {sp.fiscal_quarter or ''}: "
                        f"promoter holding {sp.promoter_holding_pct}%, pledge {sp.pledge_pct}% "
                        "(manually entered, metadata.json)"
                    ).replace("  ", " "),
                }
            continue
        row = by_id.get(rule.feature_row_id)
        if row is None:
            continue
        if row.document_id is not None:
            scope = {"document_id": row.document_id}
            metric = _FINANCIAL_RULE_METRIC.get(rule.rule_id)
            if metric:
                # the page and row label the value was extracted from (feature provenance)
                src = (((row.additional_signals or {}).get("sources") or {}).get(metric)) or {}
                if src.get("page") is not None:
                    scope["page"] = src["page"]
                    scope["label"] = src.get("label") or _COMPUTED_RATIO_ANCHOR.get(rule.rule_id)
            elif rule.rule_id in _CREDIT_RULES and row.credit_rating:
                scope["rating"] = row.credit_rating
            scopes[rule.rule_id] = scope
        elif rule.rule_id in _GOVERNANCE_RULE_CATEGORIES:
            scopes[rule.rule_id] = {
                "document_type": "corporate_announcement",
                "fiscal_year": row.fiscal_year,
                "fiscal_quarter": row.fiscal_quarter,
                "categories": _GOVERNANCE_RULE_CATEGORIES[rule.rule_id],
            }
    return scopes


def process_period(
    period: PeriodIdentifier,
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    engine,
    model_version: str = MODEL_VERSION,
) -> ScoringResult:
    """
    Score a single (company, fiscal_year, fiscal_quarter) period end-to-end.

    Sequence:
      1. log started
      2. load features
      3. skip if zero rows (log failed, commit, return skipped)
      4. evaluate rules
      5. retrieve evidence
      6. upsert prediction
      7. log success
      8. commit
      9. return ScoringResult(outcome='scored')

    On any exception after step 1:
      - rollback the outer transaction
      - log failure (the logger's SAVEPOINT protects the log insert)
      - commit the failure log
      - return ScoringResult(outcome='failed')

    Args:
        period:          The period to score.
        qdrant_client:   Connected QdrantClient.
        embedding_model: Loaded SentenceTransformer.
        engine:          SQLAlchemy Engine used to open a fresh Session per period.
        model_version:   'rule_based_v1' (evaluate_rules) or 'rule_based_v2'
                         (evaluate_rules_v2); written to predictions.model_version.

    Returns:
        ScoringResult with outcome 'scored', 'failed', or 'skipped'.
    """
    with Session(engine) as session:
        try:
            # Step 1 — log started (SAVEPOINT-protected inside logger)
            log_risk_scoring_started(
                company_id=period.company_id,
                ticker=period.ticker,
                fiscal_year=period.fiscal_year,
                fiscal_quarter=period.fiscal_quarter,
                session=session,
            )

            # Step 2 — load features
            feature_set = load_period_features(
                company_id=period.company_id,
                fiscal_year=period.fiscal_year,
                fiscal_quarter=period.fiscal_quarter,
                session=session,
            )

            # Step 3 — skip if zero rows
            total_rows = (
                len(feature_set.financial_features)
                + len(feature_set.governance_features)
                + len(feature_set.credit_features)
                + len(feature_set.shareholding_patterns)
            )
            if total_rows == 0:
                log_risk_scoring_failed(
                    company_id=period.company_id,
                    ticker=period.ticker,
                    fiscal_year=period.fiscal_year,
                    fiscal_quarter=period.fiscal_quarter,
                    error="No feature data found for period",
                    session=session,
                )
                session.commit()
                return ScoringResult(
                    company_id=period.company_id,
                    ticker=period.ticker,
                    fiscal_year=period.fiscal_year,
                    fiscal_quarter=period.fiscal_quarter,
                    outcome="skipped",
                    risk_level=None,
                    n_triggered=0,
                )

            # Step 4 — evaluate rules
            scorer = evaluate_rules_v2 if model_version == MODEL_VERSION_V2 else evaluate_rules
            scorer_result = scorer(feature_set)

            # Step 5 — retrieve evidence
            from feature_extraction.fiscal_utils import calculate_period_end

            evidence = retrieve_evidence(
                triggered_rules=scorer_result.triggered_rules,
                company_id=period.company_id,
                qdrant_client=qdrant_client,
                embedding_model=embedding_model,
                scopes=build_evidence_scopes(feature_set, scorer_result.triggered_rules),
                period_cap=calculate_period_end(
                    period.fiscal_year, period.fiscal_quarter
                ).isoformat(),
            )

            # Step 6 — upsert prediction
            upsert_prediction(
                company_id=period.company_id,
                fiscal_year=period.fiscal_year,
                fiscal_quarter=period.fiscal_quarter,
                risk_score=scorer_result.risk_score,
                risk_level=scorer_result.risk_level,
                triggered_rules=scorer_result.triggered_rules,
                evidence_chunks=evidence,
                session=session,
                model_version=model_version,
            )

            # Step 7 — log success
            log_risk_scoring_success(
                company_id=period.company_id,
                ticker=period.ticker,
                fiscal_year=period.fiscal_year,
                fiscal_quarter=period.fiscal_quarter,
                risk_level=scorer_result.risk_level,
                n_triggered=len(scorer_result.triggered_rules),
                session=session,
            )

            # Step 8 — commit prediction + success log atomically
            session.commit()

            # Step 9 — return result
            return ScoringResult(
                company_id=period.company_id,
                ticker=period.ticker,
                fiscal_year=period.fiscal_year,
                fiscal_quarter=period.fiscal_quarter,
                outcome="scored",
                risk_level=scorer_result.risk_level,
                n_triggered=len(scorer_result.triggered_rules),
                coverage=scorer_result.coverage,
            )

        except Exception as exc:  # noqa: BLE001
            # Roll back the prediction upsert (if any) so the DB stays clean
            session.rollback()
            print(
                f"ERROR: scoring {period.ticker} {period.fiscal_year} "
                f"{period.fiscal_quarter or 'annual'} failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

            # Log failure — the logger uses a SAVEPOINT so this survives the rollback
            log_risk_scoring_failed(
                company_id=period.company_id,
                ticker=period.ticker,
                fiscal_year=period.fiscal_year,
                fiscal_quarter=period.fiscal_quarter,
                error=str(exc),
                session=session,
            )
            try:
                session.commit()
            except Exception:  # noqa: BLE001
                pass  # If even the failure log can't commit, carry on

            return ScoringResult(
                company_id=period.company_id,
                ticker=period.ticker,
                fiscal_year=period.fiscal_year,
                fiscal_quarter=period.fiscal_quarter,
                outcome="failed",
                risk_level=None,
                n_triggered=0,
            )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_period_status(result: ScoringResult) -> None:
    """
    Print a single status line for a scored (or skipped/failed) period.

    Format examples:
        [scored]  DHFL FY2018 Q3 — High (5 rules)
        [scored]  DHFL FY2018    — Low (0 rules)
        [skipped] DHFL FY2018 Q1 — no feature data
        [failed]  DHFL FY2018 Q2

    Args:
        result: ScoringResult from process_period.
    """
    period_str = (
        f"{result.fiscal_year} {result.fiscal_quarter}"
        if result.fiscal_quarter
        else result.fiscal_year
    )
    tag = f"[{result.outcome}]"

    if result.outcome == "scored":
        line = f"{tag:<10} {result.ticker} {period_str} — {result.risk_level} ({result.n_triggered} rules)"
        if result.coverage is not None:
            line += f" coverage={result.coverage:.0%}"
    elif result.outcome == "skipped":
        line = f"{tag:<10} {result.ticker} {period_str} — no feature data"
    else:  # failed
        line = f"{tag:<10} {result.ticker} {period_str}"
        print(
            f"  WARNING: Scoring failed for {result.ticker} {period_str}",
            file=sys.stderr,
        )

    print(line)


def print_final_summary(results: list[ScoringResult]) -> None:
    """
    Print the canonical end-of-run summary line.

    Format (Requirement 7.6):
        "Risk scoring complete. Scored {N} periods. {S} succeeded, {F} failed."

    where N = total periods processed, S = outcome == 'scored',
    F = outcome == 'failed'.  Skipped periods count toward N but not S or F.

    Args:
        results: All ScoringResult objects from the run.
    """
    n_total = len(results)
    n_scored = sum(1 for r in results if r.outcome == "scored")
    n_failed = sum(1 for r in results if r.outcome == "failed")
    print(
        f"Risk scoring complete. Scored {n_total} periods. "
        f"{n_scored} succeeded, {n_failed} failed."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    CLI orchestrator for the rule-based risk engine.

    Startup sequence:
      1. Parse --company and --fiscal-year args.
      2. Obtain the SQLAlchemy engine.
      3. Connect to Qdrant (exit 1 on failure).
      4. Load the embedding model (exit 1 on failure).
      5. Discover scoreable periods via get_scoreable_periods().
      6. Loop over periods, calling process_period() for each.
      7. Print per-period status lines and the final summary.
      8. Exit with code 1 if any period failed, 0 otherwise.

    Never reads ground_truth.json.
    Never queries or writes group / label / distressed / healthy columns.
    """
    # ------------------------------------------------------------------
    # 1. Parse CLI arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Rule-based risk engine: score companies and write predictions."
    )
    parser.add_argument(
        "--company",
        metavar="TICKER",
        default=None,
        help="Restrict processing to this company ticker (case-sensitive).",
    )
    parser.add_argument(
        "--fiscal-year",
        metavar="FY{YYYY}",
        default=None,
        help="Restrict processing to this fiscal year (e.g., FY2018).",
    )
    parser.add_argument(
        "--model-version",
        choices=MODEL_VERSIONS,
        default=MODEL_VERSION,
        help="Scoring version to run and write (default: rule_based_v1).",
    )
    args = parser.parse_args()

    company_filter: str | None = args.company
    fiscal_year_filter: str | None = args.fiscal_year

    # ------------------------------------------------------------------
    # 2. Build SQLAlchemy engine
    # ------------------------------------------------------------------
    try:
        engine = get_engine()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to create database engine: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Connect to Qdrant
    # ------------------------------------------------------------------
    try:
        qdrant_client = get_qdrant_client()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to connect to Qdrant: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Load embedding model
    # ------------------------------------------------------------------
    try:
        embedding_model = load_embedding_model()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Failed to load embedding model: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Discover scoreable periods
    # ------------------------------------------------------------------
    try:
        with Session(engine) as session:
            periods = get_scoreable_periods(
                session=session,
                company_filter=company_filter,
                fiscal_year_filter=fiscal_year_filter,
            )
    except SQLAlchemyError as exc:
        print(
            f"ERROR: Failed to retrieve scoreable periods: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # get_scoreable_periods already calls sys.exit(0) when no periods are
    # found, so reaching here means we have at least one period to process.

    # Print filter context so the user knows what was applied
    if company_filter:
        print(f"Company filter : {company_filter}")
    if fiscal_year_filter:
        print(f"Fiscal year    : {fiscal_year_filter}")
    print(f"Model version  : {args.model_version}")
    print(f"Periods found  : {len(periods)}\n")

    # ------------------------------------------------------------------
    # 6–7. Score each period and print progress
    # ------------------------------------------------------------------
    all_results: list[ScoringResult] = []
    for period in periods:
        result = process_period(
            period=period,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            engine=engine,
            model_version=args.model_version,
        )
        all_results.append(result)
        print_period_status(result)

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print()
    print_final_summary(all_results)

    # ------------------------------------------------------------------
    # 9. Exit code
    # ------------------------------------------------------------------
    if any(r.outcome == "failed" for r in all_results):
        sys.exit(1)
    else:
        sys.exit(0)


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
