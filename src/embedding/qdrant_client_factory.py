"""
qdrant_client_factory.py
------------------------
Creates the QdrantClient and guarantees that the `financial_documents`
collection exists with the required configuration (768-dim, Cosine) before
any document is processed.

Config is read directly from environment variables — NOT from the QDRANT dict
in settings.py, whose collection name ("financial_risk_chunks") differs from
the one used by this pipeline ("financial_documents").
"""

import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

COLLECTION_NAME: str = "financial_documents"
VECTOR_DIM: int = 768
DISTANCE: Distance = Distance.COSINE


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a QdrantClient using QDRANT_HOST and QDRANT_PORT
    environment variables, defaulting to "localhost" and 6333.

    Connection exceptions are allowed to propagate to the caller (main()),
    which catches them and exits with code 1.
    """
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def ensure_collection(client: QdrantClient) -> None:
    """
    Check whether the `financial_documents` collection exists and has the
    expected configuration.

    - Absent  → create with VectorParams(size=768, distance=Distance.COSINE).
    - Present with correct config (dims == 768, distance == COSINE) → no-op.
    - Present with mismatched config → raise ValueError describing the
      mismatch (found vs expected); the collection is never dropped or
      recreated.

    Args:
        client: An already-connected QdrantClient.

    Raises:
        ValueError: If the existing collection has incompatible parameters.
    """
    existing_names = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=DISTANCE),
        )
        _ensure_payload_indexes(client)
        return

    # Collection exists — verify it matches the required config.
    collection_info = client.get_collection(COLLECTION_NAME)
    params = collection_info.config.params.vectors

    found_size: int = params.size
    found_distance: Distance = params.distance

    mismatches: list[str] = []

    if found_size != VECTOR_DIM:
        mismatches.append(
            f"vector dimensions: found {found_size}, expected {VECTOR_DIM}"
        )

    # Distance comparison: normalise both sides to the enum member name so
    # that string vs enum comparisons don't cause false positives.
    found_distance_name = (
        found_distance.value
        if hasattr(found_distance, "value")
        else str(found_distance)
    )
    expected_distance_name = DISTANCE.value

    if found_distance_name.upper() != expected_distance_name.upper():
        mismatches.append(
            f"distance metric: found {found_distance_name}, expected {expected_distance_name}"
        )

    if mismatches:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' exists but has incompatible "
            f"configuration — {'; '.join(mismatches)}. "
            f"The pipeline will not drop or recreate the collection automatically."
        )


_PAYLOAD_INDEXES = {
    "company_id": PayloadSchemaType.INTEGER,
    "document_id": PayloadSchemaType.INTEGER,
    "fiscal_year": PayloadSchemaType.KEYWORD,
    "document_type": PayloadSchemaType.KEYWORD,
    "period_end": PayloadSchemaType.DATETIME,  # ISO date; enables "on or before" filters
    "fiscal_quarter": PayloadSchemaType.KEYWORD,
    "document_category": PayloadSchemaType.KEYWORD,
}


def _ensure_payload_indexes(client: QdrantClient) -> None:
    """Index the payload fields used by retrieval filters (idempotent)."""
    for field_name, schema in _PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=COLLECTION_NAME, field_name=field_name, field_schema=schema
        )
