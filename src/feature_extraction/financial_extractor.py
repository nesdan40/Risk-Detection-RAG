"""
Financial feature extractor for the structured feature extraction pipeline.

Reads the per-page, row-preserving text written by the extraction stage
(``data/extracted/{ticker}/{doc_id}_pages.json``) and parses the eight typed
financial signals with :mod:`feature_extraction.statement_parser`.

Design notes
------------
- Works from page text, not from chunks: chunking collapses whitespace, which
  destroys the table rows the numbers live in.
- All money values are normalised to INR crore using the unit declared in the
  table header; a value whose unit cannot be determined is stored as NULL.
- Every populated signal records where it came from (page, scope, unit, label,
  or the formula used) under ``additional_signals["sources"]``, so each number
  can be audited against the source PDF.
- Lenders (NBFCs, housing finance, banks): EBITDA margin and current ratio are
  not meaningful and are left NULL; gross/net NPA go to ``additional_signals``.
- Never raises: ``extract()`` catches everything and returns ``None``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from feature_extraction.statement_parser import parse_statements
from models.company import Company
from models.document import Document
from models.feature import ExtractedFeature

# Numeric(8,4) / Numeric(8,2) columns cannot hold more than this
_RATIO_CAP = Decimal("9999.9999")


@dataclass
class FinancialResult:
    """
    Typed container for extracted financial signals.

    ``None`` means the signal could not be extracted (never zero).
    ``additional_signals`` carries lender extras (gnpa_pct, net_npa_pct) and a
    ``sources`` provenance map.  ``signals_populated`` counts non-None typed fields.
    """

    revenue: Decimal | None = None
    revenue_growth_pct: Decimal | None = None
    pat: Decimal | None = None
    ebitda_margin_pct: Decimal | None = None
    debt_to_equity: Decimal | None = None
    interest_coverage: Decimal | None = None
    current_ratio: Decimal | None = None
    cfo: Decimal | None = None
    additional_signals: dict = field(default_factory=dict)
    signals_populated: int = 0

    def count_signals_populated(self) -> int:
        typed = [
            self.revenue, self.revenue_growth_pct, self.pat, self.ebitda_margin_pct,
            self.debt_to_equity, self.interest_coverage, self.current_ratio, self.cfo,
        ]
        return sum(1 for v in typed if v is not None)


def is_nbfc(company: Company) -> bool:
    """True for lenders, whose statements differ (interest is cost of goods; no current/non-current split)."""
    if company.listing_type == "debt_only":
        return True
    sector = (company.sector or "").lower()
    return any(k in sector for k in ("nbfc", "housing finance", "bank", "diversified finance"))


def convert_to_crores(value: Decimal, unit: str) -> Decimal | None:
    """Normalise a value to INR crore; None for an unrecognised unit."""
    from feature_extraction.statement_parser import _UNIT_TO_CRORE

    factor = _UNIT_TO_CRORE.get(unit.strip().lower())
    return None if factor is None else value * factor


def load_chunks(jsonl_path: Path) -> list[dict]:
    """Read all chunk records from a JSONL file (empty list when missing/empty)."""
    if not jsonl_path.exists():
        return []
    chunks: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_no} of {jsonl_path}: {exc}") from exc
    return chunks


def concatenate_chunk_text(chunks: list[dict]) -> str:
    """Join the ``chunk_text`` of all chunks with single spaces."""
    return " ".join(c["chunk_text"] for c in chunks if c.get("chunk_text") is not None)


def load_pages(pages_path: Path) -> list[str]:
    """Per-page text saved by the extraction stage; [] when missing or unreadable."""
    if not pages_path.exists():
        return []
    try:
        return list(json.loads(pages_path.read_text(encoding="utf-8")).get("pages") or [])
    except (ValueError, OSError):
        return []


def _query_prior_revenue(
    session: Session, company_id: int, fiscal_year: str, fiscal_quarter: str | None
) -> Decimal | None:
    """Revenue of the same period one fiscal year earlier, if already extracted."""
    try:
        prior_fy = f"FY{int(fiscal_year[2:]) - 1}"
    except (ValueError, IndexError):
        return None
    row = session.execute(
        select(ExtractedFeature.revenue).where(and_(
            ExtractedFeature.company_id == company_id,
            ExtractedFeature.feature_type == "financial",
            ExtractedFeature.fiscal_year == prior_fy,
            ExtractedFeature.fiscal_quarter == fiscal_quarter,
        )).limit(1)
    ).first()
    if row is None or row[0] is None:
        return None
    try:
        return Decimal(str(row[0]))
    except InvalidOperation:
        return None


def backfill_revenue_growth(session: Session, company_id: int) -> int:
    """
    Fill revenue_growth_pct for quarterly rows whose prior-year row was extracted
    later (documents are processed in id order, not period order).  Returns rows updated.
    """
    rows = session.execute(
        select(ExtractedFeature).where(and_(
            ExtractedFeature.company_id == company_id,
            ExtractedFeature.feature_type == "financial",
            ExtractedFeature.revenue.is_not(None),
            ExtractedFeature.revenue_growth_pct.is_(None),
        ))
    ).scalars().all()
    updated = 0
    for row in rows:
        prior = _query_prior_revenue(session, company_id, row.fiscal_year or "", row.fiscal_quarter)
        if prior and prior > 0:
            row.revenue_growth_pct = ((Decimal(str(row.revenue)) - prior) / prior * 100).quantize(Decimal("0.01"))
            updated += 1
    return updated


def _revenue_scope(row: ExtractedFeature) -> str | None:
    return (((row.additional_signals or {}).get("sources") or {}).get("revenue") or {}).get("scope")


def reject_impossible_annual_revenue(session: Session, company_id: int) -> int:
    """
    An annual total can never be smaller than one of its own quarters.  A garbled annual
    report can still yield a tiny figure (Jet Airways FY2019: "Total Income 2, �- H,411" read
    as Rs 0.02 crore against quarters of ~6,000 crore).  Such annual revenue, and the growth
    and EBITDA margin computed from it, are set to NULL; the rejected value stays in the
    provenance.  Quarters are compared only when their scope matches the annual row's, since a
    holding company's standalone year can be smaller than a consolidated quarter.  A next-year
    growth computed from the rejected figure is cleared too.  Returns rows changed.
    """
    rows = session.execute(
        select(ExtractedFeature).where(and_(
            ExtractedFeature.company_id == company_id,
            ExtractedFeature.feature_type == "financial",
        ))
    ).scalars().all()
    quarters: dict[str, list[tuple[Decimal, str | None]]] = {}
    for row in rows:
        if row.fiscal_quarter and row.revenue is not None:
            quarters.setdefault(row.fiscal_year, []).append((Decimal(str(row.revenue)), _revenue_scope(row)))

    rejected_years: set[str] = set()
    changed = 0
    for row in rows:
        if row.fiscal_quarter is not None or row.revenue is None:
            continue
        scope = _revenue_scope(row)
        same_scope = [v for v, s in quarters.get(row.fiscal_year, []) if s == scope]
        if not same_scope or Decimal(str(row.revenue)) >= max(same_scope):
            continue
        extra = dict(row.additional_signals or {})
        sources = dict(extra.get("sources") or {})
        sources["revenue"] = {**(sources.get("revenue") or {}),
                              "how": "rejected: annual revenue below a quarter of the same year",
                              "rejected_value": str(row.revenue), "largest_quarter": str(max(same_scope))}
        sources.pop("revenue_growth_pct", None)
        row.revenue = None
        row.revenue_growth_pct = None
        if "revenue" in ((sources.get("ebitda_margin_pct") or {}).get("how") or ""):
            sources.pop("ebitda_margin_pct", None)
            row.ebitda_margin_pct = None
        extra["sources"] = sources
        row.additional_signals = extra
        rejected_years.add(row.fiscal_year)
        changed += 1

    # the next year's growth, when it was computed from the rejected annual figure
    for row in rows:
        if row.fiscal_quarter is not None or row.revenue_growth_pct is None:
            continue
        try:
            prior_fy = f"FY{int(row.fiscal_year[2:]) - 1}"
        except (ValueError, IndexError, TypeError):
            continue
        sources = dict((row.additional_signals or {}).get("sources") or {})
        if prior_fy in rejected_years and "database" in ((sources.get("revenue_growth_pct") or {}).get("how") or ""):
            sources.pop("revenue_growth_pct", None)
            row.revenue_growth_pct = None
            row.additional_signals = {**(row.additional_signals or {}), "sources": sources}
            changed += 1
    return changed


def extract(
    document: Document,
    company: Company,
    pages_path: Path,
    session: Session,
) -> FinancialResult | None:
    """
    Extract financial signals for one annual-report / quarterly-result document.

    Returns None when the page file is missing/empty or on any unexpected error
    (the caller logs the failure).  A FinancialResult with every signal None is
    a valid outcome: the document was read but no statement values were found.
    """
    try:
        return _extract_impl(document, company, pages_path, session)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: Unexpected exception in financial_extractor.extract() "
            f"for document_id={document.id}: {exc!r}",
            file=sys.stderr,
        )
        return None


def _cap(v: Decimal | None) -> Decimal | None:
    if v is None:
        return None
    return max(min(v, _RATIO_CAP), -_RATIO_CAP)


def _extract_impl(
    document: Document, company: Company, pages_path: Path, session: Session
) -> FinancialResult | None:
    pages = load_pages(pages_path)
    if not pages or not any(p.strip() for p in pages):
        return None

    parsed = parse_statements(pages, document.document_type or "", is_nbfc(company))
    q2 = Decimal("0.01")
    result = FinancialResult(
        revenue=parsed.revenue.quantize(q2) if parsed.revenue is not None else None,
        pat=parsed.pat.quantize(q2) if parsed.pat is not None else None,
        cfo=parsed.cfo.quantize(q2) if parsed.cfo is not None else None,
        ebitda_margin_pct=_cap(parsed.ebitda_margin_pct),
        debt_to_equity=_cap(parsed.debt_to_equity),
        interest_coverage=_cap(parsed.interest_coverage),
        current_ratio=_cap(parsed.current_ratio),
    )

    # Revenue growth: annual statements carry the prior year in the second column; results
    # carry the same quarter a year earlier (located by header dates), else the DB is used.
    if result.revenue is not None:
        if parsed.revenue_prev and parsed.revenue_prev > 0:
            result.revenue_growth_pct = (
                (result.revenue - parsed.revenue_prev) / parsed.revenue_prev * 100
            ).quantize(q2)
            parsed.sources["revenue_growth_pct"] = {"how": (
                "annual statement, current vs previous year column"
                if document.document_type == "annual_report"
                else "results, current vs same quarter previous year column"
            )}
        elif document.fiscal_year:
            prior = _query_prior_revenue(session, company.id, document.fiscal_year, document.fiscal_quarter)
            if prior and prior > 0:
                result.revenue_growth_pct = ((result.revenue - prior) / prior * 100).quantize(q2)
                parsed.sources["revenue_growth_pct"] = {"how": "vs same period prior fiscal year (database)"}

    result.additional_signals.update(parsed.additional)
    if parsed.sources:
        result.additional_signals["sources"] = parsed.sources
    result.signals_populated = result.count_signals_populated()
    return result
