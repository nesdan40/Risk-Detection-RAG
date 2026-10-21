"""
tests/ui/test_integration.py — Integration tests for the Streamlit Demo UI.

Every test in this module exercises live services:
  - PostgreSQL (via SQLAlchemy)
  - Qdrant (vector store)
  - Ollama (LLM)

Run with:
    pytest tests/ui/ -m integration

These tests are intentionally excluded from the default run
(``pytest tests/ui/ -m "not integration"``) so the CI / offline suite
is not broken when Docker services are unavailable.

Each test skips gracefully when:
  - a required service is unreachable, or
  - pre-conditions are not met (e.g., no companies ingested, no prediction row).

Requirements: 2.1, 2.2, 2.3, 3.1, 6.2
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session


# ── Helpers — detect service availability ─────────────────────────────────────


def _get_engine():
    """Return a SQLAlchemy engine or raise if PostgreSQL is unreachable."""
    from database.postgres import get_engine
    engine = get_engine()
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def _get_qdrant_client():
    """Return a QdrantClient or raise if Qdrant is unreachable."""
    from database.qdrant import get_client
    client = get_client()
    client.get_collections()          # raises on connection failure
    return client


def _get_embedding_model():
    """Return a loaded SentenceTransformer or raise if model cannot be loaded."""
    from sentence_transformers import SentenceTransformer
    from config.settings import EMBEDDING_MODEL
    return SentenceTransformer(EMBEDDING_MODEL)


def _check_ollama() -> None:
    """Raise RuntimeError if Ollama is unreachable or model not pulled."""
    from ui.ollama_client import check_model_available
    ok, msg = check_model_available()
    if not ok:
        raise RuntimeError(msg)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_engine():
    """
    Module-scoped SQLAlchemy engine connected to the real PostgreSQL instance.
    Skips the entire module if the database is unreachable.
    """
    try:
        return _get_engine()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unreachable: {exc}")


@pytest.fixture(scope="module")
def live_qdrant_client():
    """
    Module-scoped QdrantClient connected to the real Qdrant instance.
    Skips the entire module if Qdrant is unreachable.
    """
    try:
        return _get_qdrant_client()
    except Exception as exc:
        pytest.skip(f"Qdrant unreachable: {exc}")


@pytest.fixture(scope="module")
def live_embedding_model():
    """
    Module-scoped SentenceTransformer. Skips if model cannot be loaded.
    """
    try:
        return _get_embedding_model()
    except Exception as exc:
        pytest.skip(f"Embedding model could not be loaded: {exc}")


# ── Test 1: get_companies — sorted by ticker ──────────────────────────────────


@pytest.mark.integration
def test_get_companies_live(live_engine):
    """
    Calls get_companies() against real PostgreSQL and verifies:
      - the result is a list (possibly empty)
      - every element has the expected keys: 'id', 'ticker', 'name'
      - the list is sorted alphabetically (case-sensitive) by ticker

    Requirements: 3.1
    """
    from ui.prediction_loader import get_companies

    with Session(live_engine) as session:
        companies = get_companies(session)

    assert isinstance(companies, list), "get_companies() must return a list"

    for c in companies:
        assert "id" in c, f"Missing 'id' key in {c}"
        assert "ticker" in c, f"Missing 'ticker' key in {c}"
        assert "name" in c, f"Missing 'name' key in {c}"
        # Ensure no forbidden/extra columns sneak in
        for forbidden in ("sector", "listing_type", "collection_window_start",
                          "collection_window_end", "created_at"):
            assert forbidden not in c, f"Forbidden key '{forbidden}' present in {c}"

    # Verify alphabetical order by ticker
    tickers = [c["ticker"] for c in companies]
    assert tickers == sorted(tickers), (
        f"Companies are not sorted by ticker: {tickers}"
    )


# ── Test 2: get_latest_prediction — dict shape ────────────────────────────────


@pytest.mark.integration
def test_get_latest_prediction_live(live_engine):
    """
    Calls get_latest_prediction() for the first available company and verifies
    the returned dict has the correct shape.

    Skips if:
      - no companies are ingested, or
      - no rule_based_v1 prediction exists for the first company.

    Requirements: 4.1 (implicitly via 8.4)
    """
    from ui.prediction_loader import get_companies, get_latest_prediction

    with Session(live_engine) as session:
        companies = get_companies(session)
        if not companies:
            pytest.skip("No companies ingested — skipping prediction shape test.")

        company_id = companies[0]["id"]
        prediction = get_latest_prediction(company_id, session)

    if prediction is None:
        pytest.skip(
            f"No rule_based_v1 prediction for company_id={company_id} — "
            "run the risk engine first."
        )

    # Verify the expected columns are present
    expected_keys = {
        "id", "company_id", "fiscal_year", "fiscal_quarter",
        "risk_score", "risk_level", "risk_flags", "evidence_chunks",
        "model_version", "created_at",
    }
    actual_keys = set(prediction.keys())
    missing = expected_keys - actual_keys
    assert not missing, f"Prediction dict is missing keys: {missing}"

    # Verify model_version is rule_based_v1 (since that's what the query filters on)
    assert prediction["model_version"] == "rule_based_v1", (
        f"Expected model_version='rule_based_v1', got {prediction['model_version']!r}"
    )

    # risk_score should be numeric (Decimal or float), between 0 and 1
    score = prediction["risk_score"]
    if score is not None:
        assert 0.0 <= float(score) <= 1.0, (
            f"risk_score out of [0,1] range: {score}"
        )

    # risk_level should be one of the defined levels (if set)
    level = prediction["risk_level"]
    if level is not None:
        assert level in {"Low", "Medium", "High", "Critical"}, (
            f"Unexpected risk_level: {level!r}"
        )

    # No label-leakage fields in the returned dict
    for forbidden in ("group", "label", "distressed", "healthy", "ground_truth"):
        assert forbidden not in prediction, (
            f"Forbidden field '{forbidden}' present in prediction dict"
        )


# ── Test 3: run_startup_checks — all green ────────────────────────────────────


@pytest.mark.integration
def test_startup_checks_live():
    """
    Calls run_startup_checks() with all three services running and asserts the
    returned error list is empty.

    This test avoids triggering Streamlit's @st.cache_resource decorator by
    patching the three cached helpers (get_db_engine, get_qdrant_client) inside
    app.py with direct factory calls, ensuring the test works outside a live
    Streamlit session.

    Skips if any required service is unreachable (detected via the helper
    functions before the main assertion so the skip message is informative).

    Requirements: 2.1, 2.2, 2.3
    """
    from unittest.mock import patch

    # Pre-flight: check each service so we can skip with a clear message
    try:
        engine = _get_engine()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unreachable (pre-check): {exc}")

    try:
        qdrant_client = _get_qdrant_client()
    except Exception as exc:
        pytest.skip(f"Qdrant unreachable (pre-check): {exc}")

    try:
        _check_ollama()
    except Exception as exc:
        pytest.skip(f"Ollama unreachable or model not pulled (pre-check): {exc}")

    # Patch the @st.cache_resource helpers inside app.py so we bypass Streamlit
    with (
        patch("ui.app.get_db_engine", return_value=engine),
        patch("ui.app.get_qdrant_client", return_value=qdrant_client),
    ):
        from ui.app import run_startup_checks
        errors = run_startup_checks()

    assert errors == [], (
        f"run_startup_checks() reported failures even though all services are up: {errors}"
    )


# ── Test 4: query_company — live RAG round-trip ───────────────────────────────


@pytest.mark.integration
def test_query_company_live(live_engine, live_qdrant_client, live_embedding_model):
    """
    Submits a short generic query against the first available company using:
      - real Qdrant collection ("financial_documents")
      - real SentenceTransformer embedding
      - real Ollama model

    Asserts:
      - result["answer"] is a non-empty string
      - result["sources"] is a list

    Skips if:
      - no companies are ingested, or
      - Ollama is unreachable / model not pulled, or
      - the Qdrant collection "financial_documents" is empty for the company
        (test proceeds but passes the "no chunks" fast-path as valid too)

    Requirements: 6.2
    """
    from ui.prediction_loader import get_companies
    from ui.rag_query import query_company, QDRANT_COLLECTION

    # Pre-flight: Ollama
    try:
        _check_ollama()
    except Exception as exc:
        pytest.skip(f"Ollama not available: {exc}")

    # Resolve the company to query
    with Session(live_engine) as session:
        companies = get_companies(session)

    if not companies:
        pytest.skip("No companies ingested — skipping RAG round-trip test.")

    company_id = companies[0]["id"]
    ticker = companies[0]["ticker"]

    # Confirm the Qdrant collection exists (skip rather than error if absent)
    try:
        collections = live_qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
    except Exception as exc:
        pytest.skip(f"Could not list Qdrant collections: {exc}")

    if QDRANT_COLLECTION not in collection_names:
        pytest.skip(
            f"Qdrant collection '{QDRANT_COLLECTION}' does not exist yet — "
            "run the embedding pipeline first."
        )

    query_text = "What are the key financial risks?"

    result = query_company(
        query_text=query_text,
        company_id=company_id,
        qdrant_client=live_qdrant_client,
        embedding_model=live_embedding_model,
    )

    # Shape assertions
    assert isinstance(result, dict), "query_company() must return a dict"
    assert "answer" in result, "Result dict missing 'answer' key"
    assert "sources" in result, "Result dict missing 'sources' key"

    answer = result["answer"]
    assert isinstance(answer, str), f"'answer' must be a str, got {type(answer).__name__}"
    assert len(answer) > 0, "'answer' must be a non-empty string"

    sources = result["sources"]
    assert isinstance(sources, list), f"'sources' must be a list, got {type(sources).__name__}"

    # Verify source dicts have the correct shape and no forbidden fields
    for src in sources:
        assert isinstance(src, dict), f"Each source must be a dict, got {type(src)}"
        for expected_key in ("chunk_text", "document_type", "fiscal_year",
                             "fiscal_quarter", "file_path"):
            assert expected_key in src, (
                f"Source dict missing expected key '{expected_key}': {src}"
            )
        for forbidden in ("group", "label", "distressed", "healthy", "ground_truth"):
            assert forbidden not in src, (
                f"Forbidden field '{forbidden}' present in source dict: {src}"
            )
