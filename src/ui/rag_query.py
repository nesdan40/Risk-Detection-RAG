"""
rag_query.py — RAG query service for the Streamlit demo UI.

Orchestrates the full retrieval-augmented generation pipeline for ad-hoc
questions about a selected company:
  embed → retrieve (Qdrant) → build prompt → generate (Ollama) → return

Module-level constants
----------------------
QUERY_PREFIX      : Prefix prepended to every query before embedding.
QDRANT_COLLECTION : Qdrant collection that holds ingested document chunks.
TOP_K             : Number of chunks to retrieve per query.
MAX_QUERY_LENGTH  : Maximum allowed query string length (characters).

Label-leakage contract
-----------------------
This module MUST NOT read, include in any prompt, or surface in any return
value any payload key named 'group', 'label', 'distressed', 'healthy', or
'ground_truth'.  Enforcement is at code-review time and via the property-based
test suite.
"""

from __future__ import annotations

import os

from qdrant_client.models import FieldCondition, Filter, MatchValue

from ui import ollama_client

# ── Module-level constants ────────────────────────────────────────────────────

from embedding.embedder import QUERY_PREFIX  # noqa: E402  (single shared definition)
QDRANT_COLLECTION = "financial_documents"
TOP_K = 5
MAX_QUERY_LENGTH = 500

# Fields that must never be read from a Qdrant payload or included anywhere.
_FORBIDDEN_FIELDS = frozenset({"group", "label", "distressed", "healthy", "ground_truth"})


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_prompt(query_text: str, chunks: list[dict]) -> str:
    """
    Construct the LLM prompt from retrieved chunks and the user's question.

    Each chunk is formatted as a numbered passage with its source metadata.
    ``chunk_text`` is included in full (no truncation) — the UI display layer
    is responsible for any truncation shown to the user.

    Forbidden fields (``group``, ``label``, ``distressed``, ``healthy``,
    ``ground_truth``) are never read from *chunks* or included in the output.

    Args:
        query_text: The user's original question (not prefixed with QUERY_PREFIX).
        chunks:     List of chunk payload dicts retrieved from Qdrant.  Each dict
                    may contain: ``chunk_text``, ``document_type``, ``fiscal_year``,
                    ``fiscal_quarter``, ``file_path``.

    Returns:
        A fully-formatted prompt string ready to be passed to
        ``ollama_client.generate()``.

    Example::

        prompt = _build_prompt("What was the CFO trend?", chunks)
    """
    # Build the numbered context passages.
    passage_lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        # ── Source metadata ───────────────────────────────────────────────
        document_type = chunk.get("document_type") or "Unknown"
        fiscal_year = chunk.get("fiscal_year") or "Unknown"
        fiscal_quarter = chunk.get("fiscal_quarter")
        fiscal_quarter_or_annual = fiscal_quarter if fiscal_quarter is not None else "Annual"
        file_path = chunk.get("file_path") or ""
        basename = os.path.basename(file_path) if file_path else "Unknown"

        # ── Chunk text — truncated to 800 chars to keep prompt size manageable ──
        chunk_text = (chunk.get("chunk_text") or "")[:800]

        passage_lines.append(
            f"[{i}] Source: {document_type} | {fiscal_year} {fiscal_quarter_or_annual} | {basename}\n"
            f"{chunk_text}"
        )

    context_block = "\n\n".join(passage_lines)

    prompt = (
        "You are a financial analyst assistant. Answer the following question using ONLY\n"
        "the context passages provided below. For each claim in your answer, cite which\n"
        "passage (by its number) supports it. If the context does not contain enough\n"
        "information to answer the question, say so clearly — do not invent information.\n"
        "\n"
        "Context:\n"
        "---\n"
        f"{context_block}\n"
        "---\n"
        "\n"
        f"Question: {query_text}\n"
        "\n"
        "Answer (cite passages by number, e.g. [1], [2]):"
    )

    return prompt


# ── Public API ────────────────────────────────────────────────────────────────

def query_company(
    query_text: str,
    company_id: int,
    qdrant_client,
    embedding_model,
) -> dict:
    """
    Execute a RAG query for the given company.

    Steps:
      1. Validate query_text: non-empty, <= MAX_QUERY_LENGTH characters.
      2. Embed: QUERY_PREFIX + query_text via embedding_model.encode().
      3. Retrieve: top-TOP_K Qdrant chunks filtered by company_id payload field.
      4. Guard: if zero chunks found, return early with a "no chunks" message.
      5. Build prompt using _build_prompt().
      6. Call ollama_client.generate(prompt).
      7. Return result dict.

    Args:
        query_text:      User's natural-language question.
        company_id:      Filter Qdrant results to this company.
        qdrant_client:   Connected QdrantClient (from st.cache_resource).
        embedding_model: Loaded SentenceTransformer (from st.cache_resource).

    Returns:
        {
            "answer":  str,   # LLM-generated response text
            "sources": list,  # one dict per retrieved chunk
        }

    Raises:
        ValueError:   If query_text is empty or exceeds MAX_QUERY_LENGTH.
        RuntimeError: Propagated from ollama_client.generate() on LLM failure.
    """
    # ── Step 1: Validate query_text ──────────────────────────────────────────
    if not query_text:
        raise ValueError("query_text must be a non-empty string.")
    if len(query_text) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"query_text exceeds the maximum allowed length of {MAX_QUERY_LENGTH} characters "
            f"(got {len(query_text)})."
        )

    # ── Step 2: Embed ─────────────────────────────────────────────────────────
    query_vector: list[float] = embedding_model.encode(
        QUERY_PREFIX + query_text,
        convert_to_tensor=False,
    ).tolist()

    # ── Step 3: Retrieve from Qdrant ──────────────────────────────────────────
    company_filter = Filter(
        must=[
            FieldCondition(
                key="company_id",
                match=MatchValue(value=company_id),
            )
        ]
    )
    try:
        # Modern API (qdrant-client >= 1.7)
        result = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=company_filter,
            limit=TOP_K,
        )
        search_results = result.points
    except AttributeError:
        # Legacy API fallback
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=company_filter,
            limit=TOP_K,
        )

    # ── Step 4: Guard — no chunks found ──────────────────────────────────────
    if not search_results:
        return {
            "answer": "No document chunks found for this company in the vector store.",
            "sources": [],
        }

    # ── Step 5: Build sources list (forbidden fields excluded) ────────────────
    sources: list[dict] = []
    for hit in search_results:
        payload = hit.payload or {}
        source = {
            "chunk_text":     payload.get("chunk_text"),
            "document_type":  payload.get("document_type"),
            "fiscal_year":    payload.get("fiscal_year"),
            "fiscal_quarter": payload.get("fiscal_quarter"),
            "file_path":      payload.get("file_path"),
        }
        # Explicitly confirm no forbidden field is included
        for forbidden in _FORBIDDEN_FIELDS:
            source.pop(forbidden, None)
        sources.append(source)

    # ── Step 6: Build prompt and call LLM ────────────────────────────────────
    prompt = _build_prompt(query_text, sources)
    answer = ollama_client.generate(prompt)  # RuntimeError propagates to caller

    # ── Step 7: Return result ─────────────────────────────────────────────────
    return {"answer": answer, "sources": sources}
