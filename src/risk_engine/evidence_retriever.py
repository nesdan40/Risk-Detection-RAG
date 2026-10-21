"""
Evidence retriever for the rule-based risk engine.

For each triggered rule returned by the Scorer, this module queries the Qdrant
`financial_documents` collection to find the document chunk that best
semantically supports the signal — linking every risk flag back to a concrete
passage in the original filings.

Design principles:
  - Evidence retrieval is best-effort: a failed Qdrant query never aborts
    prediction writing.  Errors produce a stub ChunkReference and are logged
    to stderr.
  - QUERY_PREFIX is prepended to every query string before embedding, matching
    the BAAI/bge-base-en-v1.5 query encoding convention.
  - The chunk text field in the Qdrant payload is "chunk_text" (set by the
    embedding pipeline's build_points() in src/embedding/embedder.py).
  - Label-leakage prevention: no payload key named "group", "distressed",
    "healthy", "label", or "ground_truth" is ever read or stored.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchAny, MatchValue
from sentence_transformers import SentenceTransformer

from risk_engine.scorer import RuleResult

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

from embedding.embedder import QUERY_PREFIX  # noqa: E402  (single shared definition)
"""
Prefix prepended to every Qdrant search query string before embedding.
Mirrors the convention in src/embedding/embedder.py.
Never applied to chunk texts during ingestion — only at query time.
"""

COLLECTION_NAME: str = "financial_documents"
"""Qdrant collection holding 768-dim BAAI/bge-base-en-v1.5 embeddings."""

# ---------------------------------------------------------------------------
# Rule-specific query templates
# ---------------------------------------------------------------------------

RULE_QUERY_TEMPLATES: dict[str, str] = {
    "PLEDGE_RISING": (
        "promoter shareholding pledged to lenders increasing quarter over quarter"
    ),
    "PLEDGE_HIGH": (
        "promoter pledge percentage high proportion of shares pledged as collateral"
    ),
    "PROMOTER_HOLDING_LOW": (
        "promoter holding percentage declined low insider ownership stake"
    ),
    "INTEREST_COVERAGE_LOW": (
        "interest coverage ratio declined unable to service debt interest expense exceeds earnings"
    ),
    "NEGATIVE_PAT": (
        "net loss profit after tax negative loss making quarter year"
    ),
    "DEBT_TO_EQUITY_HIGH": (
        "debt to equity ratio high excessive leverage borrowings relative to net worth"
    ),
    "CURRENT_RATIO_LOW": (
        "current ratio low liquidity shortfall current liabilities exceed current assets working capital deficit"
    ),
    "CFO_NEGATIVE": (
        "cash flow from operations negative operating activities cash outflow"
    ),
    "REVENUE_DECLINING": (
        "revenue declined sales contraction turnover fell year on year degrowth"
    ),
    "RATING_DOWNGRADE": (
        "credit rating downgraded rating outlook negative watch downgrade"
    ),
    "RATING_DEFAULT_GRADE": (
        "credit rating default grade D SD C NM non-meaningful default on debt obligations"
    ),
    "RESIGNATION_SPIKE": (
        "director resigned board member departure multiple resignations from board"
    ),
    "DIRECTOR_CHURN": (
        "director changes board turnover appointments and resignations high churn"
    ),
    "AGM_DELAY": (
        "annual general meeting AGM postponed delayed not held within statutory deadline"
    ),
    "CLARIFICATION_SPIKE": (
        "press release news clarification regulatory query media scrutiny company response to rumours"
    ),
}

# ---------------------------------------------------------------------------
# ChunkReference dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChunkReference:
    """
    Represents a Qdrant chunk that substantiates a triggered rule.

    All fields except `rule_id` and `score` may be None when no evidence is
    found or when retrieval fails.

    Attributes:
        rule_id:        rule_id of the triggered rule this evidence supports.
        chunk_text:     First 500 characters of the chunk's text payload.
                        None when no result found or on error.
        document_id:    PostgreSQL documents.id from the chunk payload.
                        None when no result found or on error.
        file_name:      file_path from the chunk payload (stored as file_name
                        in ChunkReference for display purposes).
                        None when no result found or on error.
        fiscal_year:    fiscal_year from the chunk payload (e.g., 'FY2018').
                        None when no result found or on error.
        fiscal_quarter: fiscal_quarter from the chunk payload (e.g., 'Q3').
                        None when no result found or on error.
        document_type:  document_type from the chunk payload.
                        None when no result found or on error.
        score:          Cosine similarity score from Qdrant. 0.0 when no result
                        found or on error.
    """

    rule_id: str
    chunk_text: str | None
    document_id: int | None
    file_name: str | None
    fiscal_year: str | None
    fiscal_quarter: str | None
    document_type: str | None
    score: float
    # Traceability (all optional; added after the original fields)
    point_id: str | None = None          # Qdrant point id of the cited chunk
    page_start: int | None = None        # 1-based PDF pages the chunk came from
    page_end: int | None = None
    source_kind: str = "document_chunk"  # document_chunk | company_search | shareholding_pattern
    source_note: str | None = None       # human-readable description of the cited source

# Number of chunks to retrieve per triggered rule
EVIDENCE_TOP_K: int = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_evidence(
    triggered_rules: list[RuleResult],
    company_id: int,
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    scopes: dict[str, dict] | None = None,
    period_cap: str | None = None,
) -> list[ChunkReference]:
    """
    Retrieve a supporting Qdrant chunk for each triggered rule.

    Iterates `triggered_rules` in order, calling `_retrieve_single_evidence`
    for each.  If that call raises any exception, a stub `ChunkReference` is
    appended (only `rule_id` populated, all other fields None / 0.0) and the
    error is logged to stderr.  The function never raises.

    Args:
        triggered_rules:  List of RuleResult objects from the Scorer.
        company_id:       Filter Qdrant results to chunks from this company.
        qdrant_client:    Connected QdrantClient instance.
        embedding_model:  Loaded SentenceTransformer for query embedding.

    Returns:
        List of ChunkReference, one per triggered rule, in the same order as
        `triggered_rules`.  Returns [] when `triggered_rules` is empty.

    Raises:
        Never — all exceptions are caught, logged to stderr, and converted to
        stub ChunkReference objects.
    """
    if not triggered_rules:
        return []

    evidence_chunks: list[ChunkReference] = []

    for rule_result in triggered_rules:
        try:
            chunk_ref = _retrieve_single_evidence(
                rule_result=rule_result,
                company_id=company_id,
                qdrant_client=qdrant_client,
                embedding_model=embedding_model,
                scope=(scopes or {}).get(rule_result.rule_id),
                period_cap=period_cap,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: Evidence retrieval failed for rule '{rule_result.rule_id}': {exc}",
                file=sys.stderr,
            )
            chunk_ref = _make_stub(rule_result.rule_id)

        evidence_chunks.append(chunk_ref)

    return evidence_chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _retrieve_single_evidence(
    rule_result: RuleResult,
    company_id: int,
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    scope: dict | None = None,
    period_cap: str | None = None,
) -> ChunkReference:
    """
    Find the chunk that supports one triggered rule.

    ``scope`` (built from the feature row that fired the rule) restricts the search:
      kind == "shareholding": no source PDF exists; cite the shareholding row itself.
      document_id            : search only inside that source document.
      fiscal_year / quarter / categories : search that period's announcements.
    ``period_cap`` (ISO date) excludes anything dated after the scored period, so a
    flag can never cite evidence from the future.  If the scoped search finds nothing
    the search widens to the company (still capped) and the result is labelled
    ``source_kind="company_search"`` so it is not mistaken for the exact source.
    """
    scope = scope or {}
    if scope.get("kind") == "shareholding":
        ref = _make_stub(rule_result.rule_id)
        ref.source_kind = "shareholding_pattern"
        ref.source_note = scope.get("note")
        ref.fiscal_year = scope.get("fiscal_year")
        ref.fiscal_quarter = scope.get("fiscal_quarter")
        return ref

    query_text = RULE_QUERY_TEMPLATES.get(rule_result.rule_id, rule_result.description)
    query_vector: list[float] = embedding_model.encode(
        QUERY_PREFIX + query_text, convert_to_tensor=False
    ).tolist()

    base = [FieldCondition(key="company_id", match=MatchValue(value=company_id))]
    if period_cap:
        base.append(FieldCondition(key="period_end", range=DatetimeRange(lte=period_cap)))

    scoped = list(base)
    if scope.get("document_id") is not None:
        scoped.append(FieldCondition(key="document_id", match=MatchValue(value=scope["document_id"])))
    else:
        if scope.get("fiscal_year"):
            scoped.append(FieldCondition(key="fiscal_year", match=MatchValue(value=scope["fiscal_year"])))
        if scope.get("fiscal_quarter"):
            scoped.append(FieldCondition(key="fiscal_quarter", match=MatchValue(value=scope["fiscal_quarter"])))
        if scope.get("document_type"):
            scoped.append(FieldCondition(key="document_type", match=MatchValue(value=scope["document_type"])))
        if scope.get("categories"):
            scoped.append(FieldCondition(key="document_category", match=MatchAny(any=list(scope["categories"]))))

    def _search(conditions):
        return qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(must=conditions),
            limit=EVIDENCE_TOP_K,
        ).points

    kind = "document_chunk"
    results = _search(scoped) if len(scoped) > len(base) else []
    if not results:
        results = _search(base)
        kind = "company_search"
    if not results:
        return _make_stub(rule_result.rule_id)

    point = results[0]
    payload: dict = point.payload or {}
    raw_text: str | None = payload.get("chunk_text")
    return ChunkReference(
        rule_id=rule_result.rule_id,
        chunk_text=raw_text[:500] if raw_text else None,
        document_id=payload.get("document_id"),
        file_name=payload.get("file_path"),
        fiscal_year=payload.get("fiscal_year"),
        fiscal_quarter=payload.get("fiscal_quarter"),
        document_type=payload.get("document_type"),
        score=point.score,
        point_id=str(point.id),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        source_kind=kind,
    )


def _make_stub(rule_id: str) -> ChunkReference:
    """
    Return a stub ChunkReference with only `rule_id` populated.

    Used when Qdrant returns zero results or when retrieval raises an exception.
    All text/ID fields are None; score is 0.0.

    Args:
        rule_id: The rule_id of the triggered rule.

    Returns:
        ChunkReference with rule_id set and all other fields None / 0.0.
    """
    return ChunkReference(
        rule_id=rule_id,
        chunk_text=None,
        document_id=None,
        file_name=None,
        fiscal_year=None,
        fiscal_quarter=None,
        document_type=None,
        score=0.0,
    )
