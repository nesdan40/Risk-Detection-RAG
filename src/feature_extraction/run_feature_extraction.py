"""
Feature extraction pipeline orchestrator.

Usage
-----
    python src/feature_extraction/run_feature_extraction.py
    python src/feature_extraction/run_feature_extraction.py --company DHFL
    python src/feature_extraction/run_feature_extraction.py --company DHFL --force

Exit codes
----------
    0 — success (including "nothing new to process")
    1 — DB error, unrecognised --company ticker, or --force without --company

Phase execution order (hard constraint from design)
----------------------------------------------------
    A. Category normalization for all corporate_announcement docs
    B. Financial extraction (annual_report, quarterly_result)
    C. Governance aggregation (period-level, after all announcements categorised)
    D. Credit extraction (credit_rating docs first, then corp-announcement fallback)

Label-leakage guarantee
-----------------------
This module never opens, reads, or passes ``ground_truth.json`` to any
function.  Every JSONL path passed to extractors is checked at the call site
to confirm it does not reference ground_truth.json (Req 1.6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import and_, not_, select, exists
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config.settings import EXTRACTED_DIR, PROCESSED_DIR
from database.postgres import get_engine
from models.company import Company
from models.document import Document
from models.feature import ExtractedFeature

from feature_extraction import (
    category_normalizer,
    credit_extractor,
    feature_logger,
    feature_writer,
    financial_extractor,
    governance_extractor,
)
from feature_extraction.governance_extractor import (
    CategorizedAnnouncement,
    evaluate_agm_delay_flag,
    group_announcements_by_period,
)
from feature_extraction.fiscal_utils import calculate_period_end

# ---------------------------------------------------------------------------
# Eligible document types
# ---------------------------------------------------------------------------

_ELIGIBLE_TYPES = (
    "annual_report",
    "quarterly_result",
    "corporate_announcement",
    "credit_rating",
)


# ---------------------------------------------------------------------------
# Ground-truth guard
# ---------------------------------------------------------------------------


def _assert_not_ground_truth(path: Path) -> None:
    """
    Hard runtime guard: raises AssertionError if path references ground_truth.json.

    Called before every JSONL path is passed to an extractor (Req 1.6).
    """
    assert "ground_truth.json" not in path.name.lower(), (
        f"INVARIANT VIOLATION: ground_truth.json must never be opened "
        f"by the feature extraction pipeline. Got path: {path}"
    )


# ---------------------------------------------------------------------------
# Document selection
# ---------------------------------------------------------------------------


def get_candidate_documents(
    session: Session,
    company_filter: str | None,
) -> list[Document]:
    """
    Return all eligible documents that have not yet had any extracted_features row.

    Eligibility:
    - ``embedded = TRUE``
    - ``document_type`` in eligible types
    - no ``extracted_features`` row exists for ``document_id`` (any feature_type)

    If ``company_filter`` is provided, restrict to that ticker.  Exits 1 if
    the ticker is not found in the companies table.

    Results ordered by ``documents.id ASC`` for deterministic runs (Req 1.5).

    Args:
        session:        Active SQLAlchemy session.
        company_filter: Optional ticker string to restrict candidates.

    Returns:
        List of :class:`~models.document.Document` ORM objects.

    Raises:
        SystemExit(1) — ticker not found, or SQLAlchemy error.
    """
    try:
        # -- Validate company filter -----------------------------------------
        company_id_filter: int | None = None
        if company_filter is not None:
            company_row = session.execute(
                select(Company.id).where(Company.ticker == company_filter)
            ).first()
            if company_row is None:
                print(
                    f"ERROR: --company '{company_filter}' does not match any "
                    "ticker in the companies table. Aborting.",
                    file=sys.stderr,
                )
                sys.exit(1)
            company_id_filter = company_row[0]

        # -- Build candidate query -------------------------------------------
        # Idempotency: NOT EXISTS (any extracted_features row for this document)
        ef_subq = exists(
            select(ExtractedFeature.id).where(
                ExtractedFeature.document_id == Document.id
            )
        )

        stmt = (
            select(Document)
            .where(
                and_(
                    Document.embedded.is_(True),
                    Document.document_type.in_(_ELIGIBLE_TYPES),
                    not_(ef_subq),
                )
            )
            .order_by(Document.id.asc())
        )

        if company_id_filter is not None:
            stmt = stmt.where(Document.company_id == company_id_filter)

        rows = session.execute(stmt).scalars().all()
        return list(rows)

    except SQLAlchemyError as exc:
        print(
            f"ERROR: Database error retrieving candidate documents: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Process a single document (Phase B / D dispatcher)
# ---------------------------------------------------------------------------


def process_document(
    document: Document,
    company: Company,
    normalized_categories: dict[int, tuple[str, str]],
    session: Session,
) -> tuple[str, int]:
    """
    Dispatch a single document to the appropriate extractor.

    Routing:
        ``annual_report`` / ``quarterly_result`` → :func:`financial_extractor.extract`
        ``credit_rating``                         → :func:`credit_extractor.extract`
        ``corporate_announcement``                → category already resolved in Phase A;
                                                    no per-doc write here

    Returns:
        ``(outcome, signals_populated)`` where outcome is one of
        ``'success'``, ``'failed'``, ``'unmapped_category'``, ``'skipped'``.
    """
    doc_type = document.document_type

    if doc_type in ("annual_report", "quarterly_result"):
        pages_path = _build_pages_path(company.ticker, document.id)
        _assert_not_ground_truth(pages_path)

        result = financial_extractor.extract(document, company, pages_path, session)
        if result is None:
            return "failed", 0

        # Validate required metadata fields (Req 3.12)
        if document.fiscal_year is None or document.period_end is None:
            feature_logger.log_failed(
                document.id,
                f"NULL metadata field for document_id={document.id}: "
                f"fiscal_year={document.fiscal_year}, period_end={document.period_end}",
                session,
            )
            return "failed", 0

        period_end = _coerce_date(document.period_end)
        try:
            feature_writer.upsert_financial(
                session=session,
                company_id=company.id,
                document_id=document.id,
                fiscal_year=document.fiscal_year,
                fiscal_quarter=document.fiscal_quarter,
                period_end=period_end,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001
            feature_logger.log_failed(document.id, exc, session)
            return "failed", 0

        return "success", result.signals_populated

    elif doc_type == "credit_rating":
        # Handled in Phase D — this path should not normally be called for
        # credit_rating docs from process_document (they are handled directly
        # in main()). Return skipped if called outside Phase D context.
        return "skipped", 0

    elif doc_type == "corporate_announcement":
        # Category was resolved in Phase A; no per-doc DB write for announcements
        canonical_label, _ = normalized_categories.get(
            document.id, ("unclassified", "unclassified")
        )
        if canonical_label == "unclassified":
            return "unmapped_category", 0
        return "success", 0

    return "skipped", 0


# ---------------------------------------------------------------------------
# CLI output helpers
# ---------------------------------------------------------------------------


def print_per_doc_status(
    outcome: str,
    doc_id: int,
    file_name: str,
    feature_type: str,
    signals_populated: int,
) -> None:
    """
    Print one per-document status line to stdout (Req 9.1).

    Format::

        [{outcome}] doc_id={id} | {file_name} | type={feature_type} | signals={N} non-null columns written
    """
    print(
        f"[{outcome}] doc_id={doc_id} | {file_name} | "
        f"type={feature_type} | signals={signals_populated} non-null columns written"
    )


def print_category_mapping_table(category_counts: dict[str, int]) -> None:
    """
    Print the canonical-category mapping summary table to stdout (Req 9.5).

    Sorted by count descending, ties broken alphabetically by category name.
    """
    if not category_counts:
        print("\nCategory Mapping Summary: (no corporate announcements processed)")
        return

    print("\nCategory Mapping Summary:")
    print(f"  {'Category':<50} {'Count':>5}")
    print(f"  {'-'*50} {'-'*5}")

    sorted_items = sorted(
        category_counts.items(),
        key=lambda kv: (-kv[1], kv[0]),  # descending count, then alpha
    )
    for label, count in sorted_items:
        print(f"  {label:<50} {count:>5}")


def print_final_summary(
    total: int,
    success: int,
    failed: int,
    unmapped: int,
    features_written: int,
    governance_periods: int,
) -> None:
    """
    Print the EXTRACTION SUMMARY block to stdout (Req 9.2).

    Called AFTER :func:`print_category_mapping_table`.
    """
    print()
    print("EXTRACTION SUMMARY:")
    print(f"  Documents processed      : {total}")
    print(f"  Success                  : {success}")
    print(f"  Failed                   : {failed}")
    print(f"  Unmapped category        : {unmapped}")
    print(f"  extracted_features rows  : {features_written}")
    print(f"  Governance periods       : {governance_periods}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def delete_company_features_for_force(
    session: Session,
    company_id: int,
) -> int:
    """
    Delete all ``extracted_features`` rows for a company prior to a forced
    re-extraction run.

    This is the public entry point used by ``main()`` when ``--force`` is
    provided.  It delegates to :func:`feature_writer.delete_company_features`
    and preserves the ``processing_log`` rows intact (Req 8.2).

    Args:
        session:    Active SQLAlchemy session.
        company_id: PK of the company whose features should be deleted.

    Returns:
        Count of deleted rows.
    """
    return feature_writer.delete_company_features(session, company_id)


def _build_pages_path(ticker: str, document_id: int) -> Path:
    """Per-page, row-preserving text written by the extraction stage."""
    return EXTRACTED_DIR / ticker / f"{document_id}_pages.json"


def _build_jsonl_path(ticker: str, document_id: int) -> Path:
    """Build the JSONL chunk file path for a document."""
    return PROCESSED_DIR / ticker / f"{document_id}_chunks.jsonl"


def _coerce_date(value):
    """Coerce a datetime or date value to a plain date object."""
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date()
    return value


def _load_company_map(session: Session, company_ids: set[int]) -> dict[int, Company]:
    """Fetch Company rows for the given set of IDs."""
    if not company_ids:
        return {}
    rows = session.execute(
        select(Company).where(Company.id.in_(company_ids))
    ).scalars().all()
    return {c.id: c for c in rows}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main(
    company_filter: str | None = None,
    force: bool = False,
) -> None:
    """
    Orchestrate the full feature extraction run.

    Execution order
    ---------------
    1.  Validate ``--force`` requires ``--company`` (Req 6.7).
    2.  Retrieve candidate documents from PostgreSQL.
    3.  If ``--force``: delete existing ``extracted_features`` for the company,
        then re-query to get all (now un-extracted) docs.
    4.  If ``--company`` and non-empty set: print company filter notice (Req 1.8).
    5.  Phase A: normalise all corporate_announcement categories.
    6.  Phase B: extract financial features (annual_report, quarterly_result).
    7.  Phase C: aggregate governance per fiscal period.
    8.  Phase D: extract credit features; fallback to rated announcements.
    9.  Print category mapping table, then EXTRACTION SUMMARY.

    Exit codes
    ----------
    0 — success
    1 — --force without --company; unrecognised ticker; DB error
    """
    # -----------------------------------------------------------------------
    # Step 1: Validate --force requires --company (Req 6.7)
    # -----------------------------------------------------------------------
    if force and not company_filter:
        print(
            "ERROR: --force requires --company to be specified. "
            "Use: --force --company TICKER",
            file=sys.stderr,
        )
        sys.exit(1)

    engine = get_engine()

    with Session(engine) as session:
        # -------------------------------------------------------------------
        # Step 2: Retrieve candidate documents
        # -------------------------------------------------------------------
        candidates = get_candidate_documents(session, company_filter)

        # -------------------------------------------------------------------
        # Step 3: --force: delete existing rows then re-query
        # -------------------------------------------------------------------
        if force and company_filter:
            # Get company_id for the filter
            company_row = session.execute(
                select(Company.id).where(Company.ticker == company_filter)
            ).first()
            if company_row:
                deleted = delete_company_features_for_force(
                    session, company_row[0]
                )
                session.commit()
                print(
                    f"[--force] Deleted {deleted} extracted_features rows "
                    f"for company '{company_filter}'. Re-querying candidates..."
                )
            # Re-query after deletion so all docs are now eligible
            candidates = get_candidate_documents(session, company_filter)

        # -------------------------------------------------------------------
        # Step 4: Empty candidate set guard (Req 1.4, 8.5)
        # -------------------------------------------------------------------
        if not candidates:
            print("0 documents require feature extraction")
            return

        # -------------------------------------------------------------------
        # Step 4: Print company filter notice (Req 1.8)
        # -------------------------------------------------------------------
        if company_filter:
            print(f"Processing documents for company: {company_filter}")

        # -------------------------------------------------------------------
        # Load company objects for all candidates (single batch query)
        # -------------------------------------------------------------------
        company_ids = {doc.company_id for doc in candidates if doc.company_id}
        company_map = _load_company_map(session, company_ids)

        # Partition candidates by document type
        announcement_docs = [d for d in candidates if d.document_type == "corporate_announcement"]
        financial_docs = [d for d in candidates if d.document_type in ("annual_report", "quarterly_result")]
        credit_docs = [d for d in candidates if d.document_type == "credit_rating"]

        # -------------------------------------------------------------------
        # Phase A: Category normalization for corporate_announcement docs
        # -------------------------------------------------------------------
        normalized_categories: dict[int, tuple[str, str]] = {}
        category_counts: dict[str, int] = {}

        for doc in announcement_docs:
            company = company_map.get(doc.company_id)
            if company is None:
                continue

            # For Layer 3 we need the first chunk's text — load it if available
            jsonl_path = _build_jsonl_path(company.ticker, doc.id)
            _assert_not_ground_truth(jsonl_path)
            chunk_text: str | None = None
            try:
                chunks = financial_extractor.load_chunks(jsonl_path)
                if chunks:
                    chunk_text = chunks[0].get("chunk_text")
                    if chunk_text:
                        chunk_text = chunk_text[:2000]
            except Exception:  # noqa: BLE001
                chunk_text = None

            canonical_label, resolution_layer = category_normalizer.normalize(
                document_category=doc.document_category,
                file_name=doc.file_name or "",
                chunk_text=chunk_text,
            )
            normalized_categories[doc.id] = (canonical_label, resolution_layer)

            # Track category counts for the mapping table
            category_counts[canonical_label] = category_counts.get(canonical_label, 0) + 1

            # Log and warn for unclassified docs (Req 2.5, 7.3, 9.3)
            if canonical_label == "unclassified":
                feature_logger.log_unmapped_category(
                    document_id=doc.id,
                    document_category=doc.document_category,
                    file_name=doc.file_name or "",
                    ticker=company.ticker,
                    session=session,
                )
                print(
                    f"WARNING: Unmapped category — doc_id={doc.id} | "
                    f"company={company.ticker} | file={doc.file_name} | "
                    f"raw_category={doc.document_category!r}",
                    file=sys.stderr,
                )

        # Commit Phase A log entries
        try:
            session.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Failed to commit Phase A log entries: {exc}", file=sys.stderr)
            session.rollback()

        # -------------------------------------------------------------------
        # Phase B: Financial extraction (annual_report, quarterly_result)
        # -------------------------------------------------------------------
        total_docs = 0
        success_count = 0
        failed_count = 0
        unmapped_count = 0
        features_written = 0

        for doc in financial_docs:
            company = company_map.get(doc.company_id)
            if company is None:
                print(
                    f"WARNING: No company found for document_id={doc.id} "
                    f"(company_id={doc.company_id}). Skipping.",
                    file=sys.stderr,
                )
                continue

            total_docs += 1
            feature_type = doc.document_type  # 'annual_report' or 'quarterly_result'

            # Log started
            feature_logger.log_started(doc.id, doc.document_type or "", session)

            # Validate required metadata
            if doc.fiscal_year is None or doc.period_end is None:
                msg = (
                    f"NULL metadata for document_id={doc.id}: "
                    f"fiscal_year={doc.fiscal_year}, period_end={doc.period_end}"
                )
                feature_logger.log_failed(doc.id, msg, session)
                failed_count += 1
                print_per_doc_status("failed", doc.id, doc.file_name or "", feature_type, 0)
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                continue

            pages_path = _build_pages_path(company.ticker, doc.id)
            _assert_not_ground_truth(pages_path)

            result = financial_extractor.extract(doc, company, pages_path, session)

            if result is None:
                feature_logger.log_failed(
                    doc.id,
                    f"No result from financial_extractor for document_id={doc.id} "
                    f"(JSONL missing, empty, or extraction error)",
                    session,
                )
                failed_count += 1
                print_per_doc_status("failed", doc.id, doc.file_name or "", feature_type, 0)
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                continue

            # Write extracted_features row
            try:
                period_end = _coerce_date(doc.period_end)
                feature_writer.upsert_financial(
                    session=session,
                    company_id=company.id,
                    document_id=doc.id,
                    fiscal_year=doc.fiscal_year,
                    fiscal_quarter=doc.fiscal_quarter,
                    period_end=period_end,
                    result=result,
                )
                feature_logger.log_success(
                    doc.id, "financial", result.signals_populated, session
                )
                session.commit()
                success_count += 1
                features_written += 1
                print_per_doc_status(
                    "success", doc.id, doc.file_name or "", feature_type,
                    result.signals_populated
                )
            except Exception as exc:  # noqa: BLE001
                feature_logger.log_failed(doc.id, exc, session)
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                failed_count += 1
                print_per_doc_status("failed", doc.id, doc.file_name or "", feature_type, 0)
                print(
                    f"ERROR: DB error for doc_id={doc.id} "
                    f"({type(exc).__name__}): {exc}",
                    file=sys.stderr,
                )

        # Quarterly rows need the prior-year row, which may have been extracted later.
        try:
            for cid in sorted({d.company_id for d in financial_docs if d.company_id}):
                financial_extractor.backfill_revenue_growth(session, cid)
            session.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: revenue growth backfill failed: {exc}", file=sys.stderr)
            session.rollback()

        # -------------------------------------------------------------------
        # Phase C: Governance aggregation
        # -------------------------------------------------------------------
        governance_periods = 0

        # group_announcements_by_period needs Document objects and the category map
        grouped = group_announcements_by_period(announcement_docs, normalized_categories)

        # For AGM delay evaluation we need all AGM docs per (company_id, fiscal_year)
        # Build: company_id → fiscal_year → list of CategorizedAnnouncement
        agm_by_company_fy: dict[int, dict[str, list[CategorizedAnnouncement]]] = {}
        for (fy, fq), anns in grouped.items():
            for ann in anns:
                if ann.canonical_label == "AGM / Annual General Meeting Updates":
                    agm_by_company_fy.setdefault(ann.company_id, {})
                    agm_by_company_fy[ann.company_id].setdefault(fy, [])
                    agm_by_company_fy[ann.company_id][fy].append(ann)

        # We need to group by (company_id, fy, fq) — the grouped dict is keyed by
        # (fy, fq) but may contain announcements from multiple companies
        # (when running without --company filter).  Re-group by company.
        company_period_groups: dict[
            tuple[int, str, str], list[CategorizedAnnouncement]
        ] = {}
        for (fy, fq), anns in grouped.items():
            for ann in anns:
                key = (ann.company_id, fy, fq)
                company_period_groups.setdefault(key, [])
                company_period_groups[key].append(ann)

        for (company_id, fy, fq), period_anns in company_period_groups.items():
            company = company_map.get(company_id)
            if company is None:
                continue

            # Aggregate counts for this period
            gov_result = governance_extractor.aggregate(period_anns)

            # AGM delay flag: only meaningful for Q4, evaluated at annual level
            # The flag belongs to the period the AGM took place in (not to the year-end
            # quarter): it is only knowable once the meeting has happened.
            agm_in_period = [
                a for a in period_anns
                if a.canonical_label == "AGM / Annual General Meeting Updates"
            ]
            gov_result.agm_delay_flag = evaluate_agm_delay_flag(agm_in_period, fy)

            # Skip periods with zero meaningful activity
            # (per Req 4.6 only write rows for periods with at least one announcement)
            # Already guaranteed by the grouping logic, but check anyway
            if not period_anns:
                continue

            # Compute period_end for governance row
            try:
                period_end = calculate_period_end(fy, fq)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARNING: Could not compute period_end for {fy}/{fq}: {exc}",
                    file=sys.stderr,
                )
                continue

            try:
                feature_writer.upsert_governance(
                    session=session,
                    company_id=company_id,
                    fiscal_year=fy,
                    fiscal_quarter=fq,
                    period_end=period_end,
                    result=gov_result,
                )
                feature_logger.log_governance_success(
                    ticker=company.ticker,
                    fiscal_year=fy,
                    fiscal_quarter=fq,
                    result=gov_result,
                    session=session,
                )
                session.commit()
                governance_periods += 1
                features_written += 1
            except Exception as exc:  # noqa: BLE001
                feature_logger.log_failed(None, exc, session)
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                print(
                    f"ERROR: DB error writing governance row for "
                    f"company_id={company_id} {fy}/{fq}: {exc}",
                    file=sys.stderr,
                )

        # Count announcement docs in totals
        for doc in announcement_docs:
            total_docs += 1
            label, _ = normalized_categories.get(doc.id, ("unclassified", "unclassified"))
            if label == "unclassified":
                unmapped_count += 1
            else:
                success_count += 1

        # -------------------------------------------------------------------
        # Phase D: Credit extraction
        # -------------------------------------------------------------------
        # D1: Primary — credit_rating document_type docs
        for doc in credit_docs:
            company = company_map.get(doc.company_id)
            if company is None:
                continue

            total_docs += 1

            feature_logger.log_started(doc.id, doc.document_type or "", session)

            jsonl_path = _build_jsonl_path(company.ticker, doc.id)
            _assert_not_ground_truth(jsonl_path)

            result = credit_extractor.extract(doc, company, jsonl_path, session)

            if result is None:
                feature_logger.log_failed(
                    doc.id,
                    f"No credit result for document_id={doc.id} "
                    f"(JSONL missing, empty, or no rating pattern matched)",
                    session,
                )
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                failed_count += 1
                print_per_doc_status("failed", doc.id, doc.file_name or "", "credit_rating", 0)
                continue

            try:
                period_end = _coerce_date(doc.period_end)
                feature_writer.upsert_credit(
                    session=session,
                    company_id=company.id,
                    document_id=doc.id,
                    fiscal_year=doc.fiscal_year,
                    fiscal_quarter=doc.fiscal_quarter,
                    period_end=period_end,
                    result=result,
                )
                signals = sum(
                    1 for v in [result.credit_rating, result.rating_outlook, result.rating_change_flag]
                    if v is not None
                )
                feature_logger.log_success(doc.id, "credit", signals, session)
                session.commit()
                success_count += 1
                features_written += 1
                print_per_doc_status(
                    "success", doc.id, doc.file_name or "", "credit_rating", signals
                )
            except Exception as exc:  # noqa: BLE001
                feature_logger.log_failed(doc.id, exc, session)
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                failed_count += 1
                print_per_doc_status("failed", doc.id, doc.file_name or "", "credit_rating", 0)
                print(
                    f"ERROR: DB error for credit doc_id={doc.id} "
                    f"({type(exc).__name__}): {exc}",
                    file=sys.stderr,
                )

        # D2: Fallback — corporate_announcement docs resolved to 'Credit Rating'
        # These are announcements (not standalone credit_rating doc_type) that
        # contain rating language and were categorised as 'Credit Rating' in Phase A.
        credit_announcement_doc_ids = {
            doc_id
            for doc_id, (label, _) in normalized_categories.items()
            if label == "Credit Rating"
        }

        # Build a lookup of announcement docs by id
        ann_by_id = {doc.id: doc for doc in announcement_docs}

        for doc_id in sorted(credit_announcement_doc_ids):
            doc = ann_by_id.get(doc_id)
            if doc is None:
                continue
            company = company_map.get(doc.company_id)
            if company is None:
                continue

            # Only use as fallback if no credit_rating doc was already processed
            # for this company (i.e., credit_docs list had no doc for this company)
            company_has_primary_credit = any(
                d.company_id == doc.company_id for d in credit_docs
            )
            if company_has_primary_credit:
                # Primary already handled; skip announcement fallback (Req 5.2)
                continue

            jsonl_path = _build_jsonl_path(company.ticker, doc.id)
            _assert_not_ground_truth(jsonl_path)

            result = credit_extractor.extract(doc, company, jsonl_path, session)
            if result is None:
                continue

            try:
                period_end = _coerce_date(doc.period_end)
                feature_writer.upsert_credit(
                    session=session,
                    company_id=company.id,
                    document_id=doc.id,
                    fiscal_year=doc.fiscal_year,
                    fiscal_quarter=doc.fiscal_quarter,
                    period_end=period_end,
                    result=result,
                )
                signals = sum(
                    1 for v in [result.credit_rating, result.rating_outlook, result.rating_change_flag]
                    if v is not None
                )
                feature_logger.log_success(doc.id, "credit", signals, session)
                session.commit()
                features_written += 1
            except Exception as exc:  # noqa: BLE001
                feature_logger.log_failed(doc.id, exc, session)
                try:
                    session.commit()
                except Exception:
                    session.rollback()

        # D3: Last resort — the company's own rating disclosed in its annual reports,
        # only for companies that got no credit row from D1 or D2.
        # only rows that carry a long-term rating count (reference-only letters do not)
        companies_with_credit = set(session.scalars(
            select(ExtractedFeature.company_id).where(
                ExtractedFeature.company_id.in_(company_ids),
                ExtractedFeature.feature_type == "credit",
                ExtractedFeature.credit_rating.is_not(None),
            )
        ))
        for doc in financial_docs:
            if doc.document_type != "annual_report" or doc.company_id in companies_with_credit:
                continue
            company = company_map.get(doc.company_id)
            if company is None:
                continue
            pages_path = _build_pages_path(company.ticker, doc.id)
            _assert_not_ground_truth(pages_path)
            result = credit_extractor.extract_from_annual_report(financial_extractor.load_pages(pages_path))
            if result is None:
                continue
            try:
                feature_writer.upsert_credit(
                    session=session,
                    company_id=company.id,
                    document_id=doc.id,
                    fiscal_year=doc.fiscal_year,
                    fiscal_quarter=doc.fiscal_quarter,
                    period_end=_coerce_date(doc.period_end),
                    result=result,
                )
                session.commit()
                features_written += 1
            except Exception as exc:  # noqa: BLE001
                session.rollback()
                print(
                    f"ERROR: DB error for annual-report credit doc_id={doc.id} "
                    f"({type(exc).__name__}): {exc}",
                    file=sys.stderr,
                )

        # -------------------------------------------------------------------
        # Step 9: Print category mapping table, then EXTRACTION SUMMARY
        # -------------------------------------------------------------------
        # Rating changes depend on document order in time, not processing order.
        try:
            for cid in sorted(company_ids):
                credit_extractor.recompute_rating_changes(session, cid)
            session.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: rating change recompute failed: {exc}", file=sys.stderr)
            session.rollback()

        print_category_mapping_table(category_counts)
        print_final_summary(
            total=total_docs,
            success=success_count,
            failed=failed_count,
            unmapped=unmapped_count,
            features_written=features_written,
            governance_periods=governance_periods,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract structured financial, governance, and credit features "
            "from pre-embedded documents into extracted_features."
        )
    )
    parser.add_argument(
        "--company",
        metavar="TICKER",
        default=None,
        help="Process only documents for this company ticker (case-sensitive).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Delete all existing extracted_features rows for the company "
            "and re-extract from scratch.  Requires --company."
        ),
    )
    args = parser.parse_args()
    main(company_filter=args.company, force=args.force)
