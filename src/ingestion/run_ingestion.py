"""
Ingestion pipeline orchestrator — entry point for document ingestion.

Usage
-----
    python src/ingestion/run_ingestion.py                  # all companies
    python src/ingestion/run_ingestion.py --company DHFL   # single company

Exit codes
----------
    0  — success (including "nothing new to process")
    1  — RAW_DATA_DIR does not exist or is not accessible
    2  — --company filter matched no directory

Label-leakage guarantee
-----------------------
This module never opens, reads, or passes ``ground_truth.json`` to any
function.  The only file explicitly opened from a company directory is
``metadata.json``.  A runtime guard in :func:`discover_companies` explicitly
verifies this by checking that none of the paths yielded by directory
iteration are named ``ground_truth.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from config.settings import RAW_DATA_DIR
from database.postgres import get_engine
from ingestion.company_registrar import register_company
from ingestion.document_registrar import register_documents
from ingestion.metadata_reader import read_metadata
from ingestion.shareholding_registrar import register_shareholding_patterns


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CompanyResult:
    """Outcome of processing a single company directory."""

    ticker: str
    docs_registered: int = 0
    docs_skipped: int = 0
    shareholding_registered: int = 0
    errors: list[str] = field(default_factory=list)
    skipped: bool = False  # True when the company was skipped entirely


# ---------------------------------------------------------------------------
# Company discovery
# ---------------------------------------------------------------------------


def discover_companies(
    raw_data_dir: Path,
    company_filter: str | None = None,
) -> list[Path]:
    """
    Return a list of company directories to process.

    Behaviour
    ---------
    * If *raw_data_dir* does not exist → print error to stderr, ``sys.exit(1)``
      (Req 1.5).
    * Scan *raw_data_dir* for immediate subdirectories only (not recursive).
    * Skip any subdirectory that lacks a ``metadata.json`` file — warn to
      stdout (Req 1.2).
    * Never yield or open a path named ``ground_truth.json`` (Req 1.3, 2.1).
    * If *company_filter* is given, return only the matching directory or
      exit with code 2 if no match (Req 9.5, 9.6).
    * If no subdirectories remain after filtering, warn to stdout and return
      an empty list (caller will exit 0) (Req 1.6).

    Args:
        raw_data_dir:    Absolute path to the ``data/raw/`` directory.
        company_filter:  Optional directory name to restrict processing to.

    Returns:
        Ordered list of :class:`~pathlib.Path` objects for company directories
        to process (may be empty).
    """
    # --- Req 1.5: RAW_DATA_DIR must exist -----------------------------------
    if not raw_data_dir.exists() or not raw_data_dir.is_dir():
        print(
            f"ERROR: RAW_DATA_DIR '{raw_data_dir}' does not exist or is not "
            "accessible. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Req 9.5/9.6: filter to a single company ----------------------------
    if company_filter is not None:
        target = raw_data_dir / company_filter
        if not target.is_dir():
            print(
                f"ERROR: --company '{company_filter}' did not match any "
                f"directory under '{raw_data_dir}'. Aborting.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Validate metadata.json exists for the filtered directory
        if not (target / "metadata.json").exists():
            print(
                f"WARNING: Directory '{company_filter}' has no metadata.json. "
                "Skipping.",
            )
            return []
        return [target]

    # --- Req 1.1: scan all immediate subdirectories -------------------------
    candidate_dirs = sorted(p for p in raw_data_dir.iterdir() if p.is_dir())

    if not candidate_dirs:
        print(
            "WARNING: No company directories found in "
            f"'{raw_data_dir}'. Nothing to process."
        )
        return []

    # --- Req 1.2: skip dirs without metadata.json; Req 1.3: skip ground_truth.json ---
    valid_dirs: list[Path] = []
    for company_dir in candidate_dirs:
        # Ground-truth guard: never open or pass this file (Req 1.3, 2.1)
        # We explicitly assert we are NOT targeting it — defensive check.
        assert company_dir.name != "ground_truth.json", (
            "INVARIANT VIOLATION: ground_truth.json must never be processed "
            "as a company directory."
        )

        metadata_path = company_dir / "metadata.json"
        if not metadata_path.exists():
            print(
                f"WARNING: No metadata.json found in '{company_dir.name}'. "
                "Skipping.",
            )
            continue

        valid_dirs.append(company_dir)

    # --- Req 1.6: warn if nothing left after filtering ----------------------
    if not valid_dirs:
        print(
            "WARNING: No valid company directories (with metadata.json) found "
            f"in '{raw_data_dir}'. Nothing to process."
        )

    return valid_dirs


# ---------------------------------------------------------------------------
# Single-company processing
# ---------------------------------------------------------------------------


def process_company(company_dir: Path, session: Session) -> CompanyResult:
    """
    Process a single company directory end-to-end.

    Steps
    -----
    1. Read and parse ``metadata.json`` via :func:`~ingestion.metadata_reader.read_metadata`.
    2. Register (upsert) the company row via :func:`~ingestion.company_registrar.register_company`.
    3. Register all PDF documents via :func:`~ingestion.document_registrar.register_documents`.
    4. Register shareholding patterns via
       :func:`~ingestion.shareholding_registrar.register_shareholding_patterns`.
    5. Commit the session (per-company commit so partial failures are isolated).
    6. Return a :class:`CompanyResult` with all counts.

    Any exception that is not already handled by the sub-components is caught
    here and turned into a ``skipped=True`` result so the pipeline can
    continue with subsequent companies (Req 9.4).

    Args:
        company_dir:  Absolute path to the company's root directory.
        session:      Active SQLAlchemy session (caller owns lifecycle).

    Returns:
        :class:`CompanyResult` — never raises.
    """
    ticker = company_dir.name  # fallback identifier before metadata is parsed

    # ---- Step 1: Read metadata --------------------------------------------
    try:
        metadata = read_metadata(company_dir)
    except FileNotFoundError:
        print(
            f"WARNING: [{ticker}] metadata.json not found. Skipping.",
            file=sys.stderr,
        )
        return CompanyResult(ticker=ticker, skipped=True)
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: [{ticker}] Failed to parse metadata.json: {exc}. Skipping.",
            file=sys.stderr,
        )
        return CompanyResult(ticker=ticker, skipped=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: [{ticker}] Unexpected error reading metadata: {exc}. Skipping.",
            file=sys.stderr,
        )
        return CompanyResult(ticker=ticker, skipped=True)

    if metadata is None:
        # read_metadata already logged the reason (missing required fields)
        return CompanyResult(ticker=ticker, skipped=True)

    ticker = metadata.ticker  # use the real ticker now that we have it

    # ---- Step 2: Register company -----------------------------------------
    try:
        company_id = register_company(metadata, session)
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: [{ticker}] Unexpected error registering company: {exc}. Skipping.",
            file=sys.stderr,
        )
        session.rollback()
        return CompanyResult(ticker=ticker, skipped=True)

    if company_id is None:
        # register_company already logged and rolled back
        return CompanyResult(ticker=ticker, skipped=True)

    # ---- Step 3: Register documents ---------------------------------------
    try:
        doc_result = register_documents(
            company_id=company_id,
            company_dir=company_dir,
            metadata=metadata,
            session=session,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: [{ticker}] Unexpected error registering documents: {exc}.",
            file=sys.stderr,
        )
        doc_result = None  # type: ignore[assignment]

    docs_registered = doc_result.registered if doc_result else 0
    docs_skipped = doc_result.skipped if doc_result else 0
    doc_errors: list[str] = doc_result.errors if doc_result else []

    # ---- Step 4: Register shareholding patterns ---------------------------
    shareholding_registered = 0
    try:
        shareholding_registered = register_shareholding_patterns(
            company_id=company_id,
            shareholding_data=metadata.shareholding_data,
            session=session,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: [{ticker}] Unexpected error registering shareholding patterns: "
            f"{exc}.",
            file=sys.stderr,
        )

    # ---- Step 5: Commit per-company ----------------------------------------
    try:
        session.commit()
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: [{ticker}] Failed to commit transaction: {exc}.",
            file=sys.stderr,
        )
        session.rollback()
        return CompanyResult(ticker=ticker, skipped=True)

    return CompanyResult(
        ticker=ticker,
        docs_registered=docs_registered,
        docs_skipped=docs_skipped,
        shareholding_registered=shareholding_registered,
        errors=doc_errors,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_company_summary(result: CompanyResult) -> None:
    """
    Print a one-line per-company summary to stdout (Req 9.1).

    Format (non-skipped)::

        [TICKER] registered: N docs, skipped: N docs, shareholding: N patterns

    Format (skipped)::

        [TICKER] SKIPPED (see stderr for details)
    """
    if result.skipped:
        print(f"[{result.ticker}] SKIPPED (see stderr for details)")
    else:
        error_note = f", errors: {len(result.errors)}" if result.errors else ""
        print(
            f"[{result.ticker}] registered: {result.docs_registered} docs, "
            f"skipped: {result.docs_skipped} docs, "
            f"shareholding: {result.shareholding_registered} patterns"
            f"{error_note}"
        )


def print_final_summary(results: list[CompanyResult]) -> None:
    """
    Print the overall totals block to stdout (Req 9.2).

    Counts skipped companies separately; only non-skipped companies
    contribute to document and shareholding totals.
    """
    companies_processed = len(results)
    companies_skipped = sum(1 for r in results if r.skipped)
    total_docs_registered = sum(r.docs_registered for r in results if not r.skipped)
    total_docs_skipped = sum(r.docs_skipped for r in results if not r.skipped)
    total_shareholding = sum(r.shareholding_registered for r in results if not r.skipped)
    total_errors = sum(len(r.errors) for r in results if not r.skipped)

    print()
    print("=" * 44)
    print("INGESTION SUMMARY")
    print("=" * 44)
    print(f"Companies processed  : {companies_processed}")
    if companies_skipped:
        print(f"Companies skipped    : {companies_skipped}")
    print(f"Documents registered : {total_docs_registered}")
    print(f"Documents skipped    : {total_docs_skipped}")
    print(f"Shareholding patterns: {total_shareholding}")
    print(f"Errors               : {total_errors}")
    print("=" * 44)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main(company_filter: str | None = None) -> None:
    """
    Run the full document ingestion pipeline.

    Opens a SQLAlchemy session, discovers company directories, processes each
    one in sequence, and prints summaries.  Commits per company so that a
    failure in one company does not roll back others.

    Args:
        company_filter:  If provided, only the company directory with this
                         exact name will be processed.

    Exit codes
    ----------
    0 — success (or no new data; set by sys.exit in helpers for codes 1/2)
    1 — RAW_DATA_DIR missing (raised inside discover_companies)
    2 — --company filter matched no directory (raised inside discover_companies)
    """
    print(f"Scanning '{RAW_DATA_DIR}' for companies...")

    company_dirs = discover_companies(RAW_DATA_DIR, company_filter)

    if not company_dirs:
        # Req 1.6: no valid dirs — warn already printed; exit 0
        sys.exit(0)

    print(f"Found {len(company_dirs)} company director{'y' if len(company_dirs) == 1 else 'ies'}.")
    print()

    engine = get_engine()
    results: list[CompanyResult] = []

    for company_dir in company_dirs:
        print(f"Processing {company_dir.name}...")

        with Session(engine) as session:
            result = process_company(company_dir, session)

        print_company_summary(result)
        results.append(result)

    print_final_summary(results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest company documents from data/raw/ into PostgreSQL.",
    )
    parser.add_argument(
        "--company",
        metavar="COMPANY_NAME",
        default=None,
        help=(
            "Process only the company directory with this exact name. "
            "If omitted, all company directories are processed."
        ),
    )
    args = parser.parse_args()
    main(company_filter=args.company)
