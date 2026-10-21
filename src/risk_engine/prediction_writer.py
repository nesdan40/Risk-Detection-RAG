"""
Prediction writer for the rule-based risk engine.

Writes scored predictions to the `predictions` table using an upsert
(INSERT ... ON CONFLICT DO UPDATE) so that re-running the engine for the
same (company_id, fiscal_year, fiscal_quarter) period cleanly overwrites
the existing row rather than creating a duplicate.

Key design notes:
- MODEL_VERSION is the default model_version; the caller may pass
  MODEL_VERSION_V2 instead. No other value is ever written.
- risk_flags and evidence_chunks are serialized to plain Python lists (JSON-
  compatible) before being handed to SQLAlchemy. None fields become JSON null.
- The unique index ix_predictions_upsert on
  (company_id, fiscal_year, fiscal_quarter, model_version) — created in
  migration 0003 — is the required conflict target for the upsert.
- This module never writes group, label, distressed, healthy, or ground_truth
  columns; the predictions table does not have those columns, and no synonyms
  are used either.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from models import Prediction
from risk_engine.scorer import RuleResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION: str = "rule_based_v1"
MODEL_VERSION_V2: str = "rule_based_v2"
MODEL_VERSIONS: tuple[str, ...] = (MODEL_VERSION, MODEL_VERSION_V2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upsert_prediction(
    company_id: int,
    fiscal_year: str,
    fiscal_quarter: str | None,
    risk_score: float,
    risk_level: str,
    triggered_rules: list[RuleResult],
    evidence_chunks: list[Any],
    session: Session,
    model_version: str = MODEL_VERSION,
) -> None:
    """
    Insert or update one row in the ``predictions`` table for the given period.

    Conflict resolution is on ``(company_id, fiscal_year, fiscal_quarter,
    model_version)``.  When a matching row already exists, all non-key columns
    are overwritten (risk_score, risk_level, risk_flags, evidence_chunks,
    created_at).

    Args:
        company_id:      FK into companies.id.
        fiscal_year:     e.g. 'FY2018'.
        fiscal_quarter:  e.g. 'Q3', or None for an annual-level period.
        risk_score:      Normalized score in [0.0, 1.0].
        risk_level:      One of 'Low', 'Medium', 'High', 'Critical'.
        triggered_rules: List of RuleResult objects that fired during scoring.
                         An empty list is valid and produces risk_flags=[].
        evidence_chunks: List of ChunkReference objects from the evidence
                         retriever.  An empty list produces evidence_chunks=[].
                         Typed as Any to avoid a circular import with
                         evidence_retriever.py (which is not yet imported here).
        session:         Active SQLAlchemy session.  The caller is responsible
                         for committing or rolling back.
        model_version:   One of MODEL_VERSIONS; defaults to 'rule_based_v1'.

    Raises:
        ValueError if model_version is not in MODEL_VERSIONS.
        Any SQLAlchemy exception raised during the upsert propagates to the
        caller (run_risk_engine.py), which logs the failure and continues to
        the next period.
    """
    if model_version not in MODEL_VERSIONS:
        raise ValueError(f"Unsupported model_version: {model_version!r}")

    risk_flags_payload = _serialize_rule_results(triggered_rules)
    evidence_payload = _serialize_evidence_chunks(evidence_chunks)

    stmt = (
        pg_insert(Prediction)
        .values(
            company_id=company_id,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_flags=risk_flags_payload,
            evidence_chunks=evidence_payload,
            model_version=model_version,
        )
        .on_conflict_do_update(
            index_elements=["company_id", "fiscal_year", "fiscal_quarter", "model_version"],
            set_={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_flags": risk_flags_payload,
                "evidence_chunks": evidence_payload,
                "created_at": func.now(),
            },
        )
    )

    session.execute(stmt)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_rule_results(rules: list[RuleResult]) -> list[dict]:
    """
    Serialize a list of RuleResult dataclass objects to a JSON-compatible
    list of dicts.

    Keys per element: rule_id, description, triggered_value, threshold,
    fiscal_year, fiscal_quarter, feature_row_id.
    None values are preserved as Python None (JSON null after serialization
    by SQLAlchemy/psycopg2).
    """
    result = []
    for r in rules:
        result.append(
            {
                "rule_id": r.rule_id,
                "description": r.description,
                "triggered_value": r.triggered_value,
                "threshold": r.threshold,
                "fiscal_year": r.fiscal_year,
                "fiscal_quarter": r.fiscal_quarter,
                "feature_row_id": r.feature_row_id,
            }
        )
    return result


def _serialize_evidence_chunks(chunks: list[Any]) -> list[dict]:
    """
    Serialize a list of ChunkReference objects to a JSON-compatible list of
    dicts.

    Keys per element: rule_id, chunk_text, document_id, file_name,
    fiscal_year, fiscal_quarter, document_type, score.
    None values are preserved as Python None (JSON null).

    Handles two cases:
    - ChunkReference dataclass instances (normal path)
    - Plain dicts (test / future path)
    """
    result = []
    for chunk in chunks:
        if dataclasses.is_dataclass(chunk) and not isinstance(chunk, type):
            # Dataclass instance — convert to dict
            d = dataclasses.asdict(chunk)
        elif isinstance(chunk, dict):
            d = chunk
        else:
            # Fallback: try __dict__
            d = vars(chunk)

        result.append(
            {
                "rule_id": d.get("rule_id"),
                "chunk_text": d.get("chunk_text"),
                "document_id": d.get("document_id"),
                "file_name": d.get("file_name"),
                "fiscal_year": d.get("fiscal_year"),
                "fiscal_quarter": d.get("fiscal_quarter"),
                "document_type": d.get("document_type"),
                "score": d.get("score"),
                "point_id": d.get("point_id"),
                "page_start": d.get("page_start"),
                "page_end": d.get("page_end"),
                "source_kind": d.get("source_kind"),
                "source_note": d.get("source_note"),
            }
        )
    return result



