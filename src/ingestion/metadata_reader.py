"""
Metadata reader for the document ingestion pipeline.

Loads and parses metadata.json from a company directory.
Never reads or references ground_truth.json (label leakage prevention).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


# Fields that must never be read or persisted — strict label-leakage prevention
PROHIBITED_FIELDS: frozenset[str] = frozenset(
    {"group", "distressed", "healthy", "label", "ground_truth"}
)


@dataclass
class CompanyMetadata:
    """Parsed contents of a company's metadata.json."""

    ticker: str
    company_name: str
    exchange: str
    sector: str | None
    collection_window_start: date | None
    collection_window_end: date | None
    annual_reports: list[str]
    quarterly_results: list[str]
    corporate_announcements_dir: bool
    credit_ratings_dir: bool
    # quarter_key -> {"promoter_holding_pct": float | None, "pledge_pct": float | None}
    shareholding_data: dict[str, dict[str, float | None]] = field(default_factory=dict)


def _parse_date(value: str | None, field_name: str, company_dir: Path) -> date | None:
    """Parse a YYYY-MM-DD date string; return None and log a warning on failure."""
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        print(
            f"WARNING: Could not parse date for '{field_name}' in "
            f"{company_dir.name}/metadata.json (value: {value!r}). Storing as NULL.",
            file=sys.stderr,
        )
        return None


def _parse_shareholding_data(
    docs_collected: dict,
    company_dir: Path,
) -> dict[str, dict[str, float | None]]:
    """
    Extract shareholding patterns from the documents_collected block.

    Returns an empty dict if the block is absent or empty.
    pledge_pct values are kept as-is here (normalisation to 0.0 for "-"
    is the responsibility of ShareholdingRegistrar).
    """
    try:
        sp_block = docs_collected.get("shareholding_patterns", {})
        data_block: dict = sp_block.get("data", {})
    except AttributeError:
        return {}

    if not data_block:
        return {}

    result: dict[str, dict[str, float | None]] = {}
    for quarter_key, values in data_block.items():
        if not isinstance(values, dict):
            print(
                f"WARNING: Malformed shareholding entry for key '{quarter_key}' in "
                f"{company_dir.name}/metadata.json. Skipping.",
                file=sys.stderr,
            )
            continue

        promoter = values.get("promoter_holding_pct")
        pledge = values.get("pledge_pct")

        # Treat empty string as absent
        promoter_pct: float | None = None
        if promoter is not None and promoter != "":
            try:
                promoter_pct = float(promoter)
            except (ValueError, TypeError):
                print(
                    f"WARNING: Cannot convert promoter_holding_pct={promoter!r} "
                    f"for key '{quarter_key}' in {company_dir.name}/metadata.json. "
                    "Storing as NULL.",
                    file=sys.stderr,
                )

        # pledge_pct: pass raw value through (ShareholdingRegistrar handles "-" → 0.0)
        pledge_pct: float | None = None
        if pledge is not None and pledge != "":
            if pledge == "-":
                pledge_pct = 0.0  # "-" means nil pledge (schema: 0.0, not NULL)
            else:
                try:
                    pledge_pct = float(pledge)
                except (ValueError, TypeError):
                    print(
                        f"WARNING: Cannot convert pledge_pct={pledge!r} "
                        f"for key '{quarter_key}' in {company_dir.name}/metadata.json. "
                        "Storing as NULL.",
                        file=sys.stderr,
                    )

        result[quarter_key] = {
            "promoter_holding_pct": promoter_pct,
            "pledge_pct": pledge_pct,
        }

    return result


def read_metadata(company_dir: Path) -> CompanyMetadata | None:
    """
    Load and parse metadata.json from *company_dir*.

    Raises
    ------
    FileNotFoundError
        If metadata.json does not exist (caught by the caller).
    json.JSONDecodeError
        If the file contains invalid JSON (caught by the caller).

    Returns
    -------
    CompanyMetadata
        On success.
    None
        If required fields (ticker or company_name) are missing — an error
        is printed to stderr and processing continues for other companies.

    Notes
    -----
    * Never reads, opens, or references ground_truth.json (Req 2.1).
    * Prohibited fields (group, distressed, healthy, label, ground_truth)
      are detected, warned about, and silently dropped (Req 2.5).
    """
    metadata_path = company_dir / "metadata.json"

    # Hard guard: never touch ground_truth.json
    # (The open() call below explicitly targets metadata.json only.)
    with open(metadata_path, encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    # ------------------------------------------------------------------ #
    # Prohibited-field check — warn and ignore (Req 2.5)                 #
    # ------------------------------------------------------------------ #
    found_prohibited = PROHIBITED_FIELDS & set(raw.keys())
    if found_prohibited:
        print(
            f"WARNING: Prohibited fields {sorted(found_prohibited)!r} found in "
            f"{company_dir.name}/metadata.json. These fields will be ignored.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ #
    # Required fields (Req 3.5)                                           #
    # ------------------------------------------------------------------ #
    ticker: str | None = raw.get("ticker")
    company_name: str | None = raw.get("company_name")

    missing = []
    if not ticker:
        missing.append("ticker")
    if not company_name:
        missing.append("company_name")

    if missing:
        print(
            f"ERROR: {company_dir.name}/metadata.json is missing required fields: "
            f"{missing}. Skipping company.",
            file=sys.stderr,
        )
        return None

    # ------------------------------------------------------------------ #
    # Optional top-level fields                                            #
    # ------------------------------------------------------------------ #
    exchange: str = raw.get("exchange") or ""
    sector: str | None = raw.get("sector") or None

    collection_window_start = _parse_date(
        raw.get("collection_window_start"), "collection_window_start", company_dir
    )
    collection_window_end = _parse_date(
        raw.get("collection_window_end"), "collection_window_end", company_dir
    )

    # ------------------------------------------------------------------ #
    # documents_collected block                                            #
    # ------------------------------------------------------------------ #
    docs_collected: dict = raw.get("documents_collected") or {}

    annual_block = docs_collected.get("annual_reports") or {}
    annual_reports: list[str] = annual_block.get("files") or []

    quarterly_block = docs_collected.get("quarterly_results") or {}
    quarterly_results: list[str] = quarterly_block.get("files") or []

    corporate_announcements_dir: bool = "corporate_announcements" in docs_collected
    credit_ratings_dir: bool = "credit_ratings" in docs_collected

    shareholding_data = _parse_shareholding_data(docs_collected, company_dir)

    return CompanyMetadata(
        ticker=ticker,
        company_name=company_name,
        exchange=exchange,
        sector=sector,
        collection_window_start=collection_window_start,
        collection_window_end=collection_window_end,
        annual_reports=annual_reports,
        quarterly_results=quarterly_results,
        corporate_announcements_dir=corporate_announcements_dir,
        credit_ratings_dir=credit_ratings_dir,
        shareholding_data=shareholding_data,
    )
