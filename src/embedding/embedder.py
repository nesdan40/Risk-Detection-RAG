"""
Embedder — pure-function core of the embedding pipeline.

This module contains the model loader and three pure functions used to convert
JSONL chunk objects into Qdrant-ready PointStruct objects:

  load_model()     — load BAAI/bge-base-en-v1.5 once at startup
  embed_texts()    — batch-encode a list of chunk texts into float vectors
  make_point_id()  — derive a deterministic UUID for a (document_id, chunk_index) pair
  build_points()   — zip chunk metadata with their embeddings into PointStruct objects

None of these functions perform I/O beyond model inference; all database and
filesystem interactions are handled by run_embedding.py.

Label-leakage note: build_points() never writes keys named "group", "distressed",
"healthy", "label", or "ground_truth" into any payload, regardless of what appears
in the input chunk dict.

QUERY_PREFIX is defined here for the retrieval layer to discover — it must NOT be
applied to chunk texts during ingestion.  BAAI/bge-base-en-v1.5 encodes passage
texts as-is; the prefix is only relevant at query time.
"""

import math
import sys
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct

from config.settings import EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "
"""
Prefix to prepend to query strings at retrieval time only.
NEVER apply this prefix to chunk texts during ingestion.
"""

EMBEDDING_BATCH_SIZE: int = 32
"""Maximum number of chunk texts passed to model.encode() in a single call."""

NAMESPACE_UUID: uuid.UUID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
"""
Fixed namespace for uuid5 point ID generation.
MUST NOT change between pipeline runs or code revisions — changing it would
invalidate all existing Qdrant point IDs.
"""

# The 11 metadata fields sourced from each JSONL chunk object and written into
# each Qdrant point payload.  This constant is used by build_points() to pick
# only these keys, preventing any label or label-adjacent field from leaking in.
_PAYLOAD_FIELDS: tuple[str, ...] = (
    "company_id",
    "document_id",
    "company_ticker",
    "chunk_index",
    "document_type",
    "document_category",
    "fiscal_year",
    "fiscal_quarter",
    "source",
    "file_path",
    "chunk_text",
    "period_end",
    "page_start",
    "page_end",
)

# Keys that must never appear in any Qdrant payload — enforced defensively in
# build_points() via the allowlist in _PAYLOAD_FIELDS.
_PROHIBITED_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {"group", "distressed", "healthy", "label", "ground_truth"}
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model() -> SentenceTransformer:
    """
    Load the BAAI/bge-base-en-v1.5 model exactly once.

    On first call this will download ~440 MB of model weights if they are not
    already cached locally.  The returned instance should be passed into
    embed_texts() for each document rather than calling this function again.

    Returns:
        Loaded SentenceTransformer instance.

    Raises:
        OSError, ValueError, or model-specific exception if the model cannot
        be loaded.  The caller (main()) catches this and exits with code 1.
    """
    model = SentenceTransformer(EMBEDDING_MODEL or "BAAI/bge-base-en-v1.5")
    model.max_seq_length = 512  # BGE limit; chunks are sized to fit inside it
    return model


# ---------------------------------------------------------------------------
# Point ID generation
# ---------------------------------------------------------------------------


def make_point_id(document_id: int, chunk_index: int) -> uuid.UUID:
    """
    Compute a deterministic UUID for a (document_id, chunk_index) pair.

    Algorithm: uuid.uuid5(NAMESPACE_UUID, f"{document_id}:{chunk_index}")

    The same pair always produces the same UUID across runs and environments,
    making Qdrant upserts idempotent without a separate ID-tracking table.

    Args:
        document_id:  PostgreSQL documents.id value.  Must be >= 1.
        chunk_index:  Zero-based index of the chunk within its document.
                      Must be >= 0.

    Returns:
        UUID version 5 derived from NAMESPACE_UUID and the name string.

    Raises:
        ValueError if document_id < 1 or chunk_index < 0.
    """
    if document_id < 1:
        raise ValueError(
            f"document_id must be >= 1, got {document_id!r}"
        )
    if chunk_index < 0:
        raise ValueError(
            f"chunk_index must be >= 0, got {chunk_index!r}"
        )
    return uuid.uuid5(NAMESPACE_UUID, f"{document_id}:{chunk_index}")


# ---------------------------------------------------------------------------
# Batch text embedding
# ---------------------------------------------------------------------------


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
) -> list[list[float]]:
    """
    Embed a list of chunk texts in batches of EMBEDDING_BATCH_SIZE.

    Texts are validated before any model call: whitespace-only strings are
    rejected because they produce meaningless embeddings and typically indicate
    an upstream chunking bug.

    Batching is done manually (rather than relying on sentence-transformers'
    internal batching) so that the exact batch size is under our control and
    testable via property tests.

    Args:
        model: Loaded SentenceTransformer instance (from load_model()).
        texts: List of chunk text strings.

    Returns:
        List of 768-dimensional float vectors, one per input text, in order.
        Returns [] when texts is empty — no model.encode() calls are made.

    Raises:
        ValueError   if any text in texts is empty or whitespace-only.
        Exception    if model.encode() raises; the failed batch index is logged
                     to stderr before the exception is re-raised.  No partial
                     results are returned.
    """
    if not texts:
        return []

    # Validate all texts before starting any encoding so that we either
    # succeed fully or fail fast — no partial results.
    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(
                f"texts[{i}] is empty or whitespace-only; "
                "blank texts must not be passed to embed_texts()"
            )

    results: list[list[float]] = []
    num_batches = math.ceil(len(texts) / EMBEDDING_BATCH_SIZE)

    for batch_idx in range(num_batches):
        start = batch_idx * EMBEDDING_BATCH_SIZE
        end = start + EMBEDDING_BATCH_SIZE
        batch = texts[start:end]

        try:
            encoded = model.encode(batch)
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: model.encode() failed on batch index {batch_idx} "
                f"(texts[{start}:{end}]): {exc}",
                file=sys.stderr,
            )
            raise

        # model.encode() returns a numpy array of shape (batch_size, dims).
        # Convert each row to a plain Python list[float] so the return type
        # is consistent and JSON-serialisable.
        for vector in encoded:
            results.append(vector.tolist())

    return results


# ---------------------------------------------------------------------------
# PointStruct construction
# ---------------------------------------------------------------------------


def build_points(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> list[PointStruct]:
    """
    Zip chunk metadata dicts with their corresponding embeddings into a list
    of Qdrant PointStruct objects ready for upsert.

    Per point:
      - id:      make_point_id(chunk["document_id"], chunk["chunk_index"])
      - vector:  embeddings[i]  (must have exactly 768 elements)
      - payload: all 11 metadata fields from the chunk dict (_PAYLOAD_FIELDS)

    Null handling: JSON null values arrive as Python None from the JSONL
    parser; None is written into the payload as-is.  No substitution.

    Label-leakage prevention: only the 11 fields listed in _PAYLOAD_FIELDS are
    included in the payload.  Any other keys in the chunk dict (including
    prohibited keys such as "group", "distressed", "healthy", "label", or
    "ground_truth") are silently excluded.

    Args:
        chunks:     List of chunk dicts read from JSONL.
        embeddings: Parallel list of float vectors; len must equal len(chunks).

    Returns:
        List of PointStruct, one per chunk, in input order.

    Raises:
        ValueError if any vector in embeddings has length != 768, identifying
                   the document_id, chunk_index, and actual dimension count in
                   the error message.
    """
    # Validate all vector dimensions before building any PointStructs so that
    # we either return a complete list or raise — no partial output.
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        dim = len(vector)
        if dim != 768:
            doc_id = chunk.get("document_id", "<unknown>")
            chunk_idx = chunk.get("chunk_index", "<unknown>")
            raise ValueError(
                f"Expected 768-dimensional vector for "
                f"document_id={doc_id}, chunk_index={chunk_idx} "
                f"(embeddings[{i}]), but got {dim} dimensions"
            )

    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, embeddings):
        point_id = make_point_id(chunk["document_id"], chunk["chunk_index"])

        # Build payload from the allowlisted fields only.
        # - Fields absent from the chunk dict are included as None (preserving
        #   the "missing == null" contract from the JSONL spec).
        # - Fields that are present but not in _PAYLOAD_FIELDS are excluded,
        #   which naturally blocks any prohibited key from leaking in.
        payload: dict = {
            field: chunk.get(field)
            for field in _PAYLOAD_FIELDS
        }

        points.append(
            PointStruct(
                id=str(point_id),
                vector=vector,
                payload=payload,
            )
        )

    return points
