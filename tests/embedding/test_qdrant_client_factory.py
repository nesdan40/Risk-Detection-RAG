"""
tests/embedding/test_qdrant_client_factory.py
---------------------------------------------
Unit tests for src/embedding/qdrant_client_factory.py.

All Qdrant network calls are mocked — no real Qdrant instance required.
"""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import Distance

from src.embedding.qdrant_client_factory import (
    COLLECTION_NAME,
    VECTOR_DIM,
    ensure_collection,
    get_qdrant_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client_with_collection(size: int, distance: Distance) -> MagicMock:
    """Return a mock QdrantClient that reports one existing collection whose
    config matches the given *size* and *distance*."""
    mock_client = MagicMock()

    # get_collections() side
    mock_item = MagicMock()
    mock_item.name = COLLECTION_NAME
    mock_client.get_collections.return_value.collections = [mock_item]

    # get_collection() side
    mock_info = MagicMock()
    mock_info.config.params.vectors.size = size
    mock_info.config.params.vectors.distance = distance
    mock_client.get_collection.return_value = mock_info

    return mock_client


def _make_mock_client_empty() -> MagicMock:
    """Return a mock QdrantClient that reports no existing collections."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    return mock_client


# ---------------------------------------------------------------------------
# ensure_collection tests
# ---------------------------------------------------------------------------


class TestEnsureCollection:
    def test_ensure_collection_creates_when_absent(self):
        """Collection absent → create_collection called with correct params."""
        mock_client = _make_mock_client_empty()

        ensure_collection(mock_client)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args

        # Positional-or-keyword: tolerate both call styles
        args, kwargs = call_kwargs
        # collection_name may be positional arg[0] or keyword
        name_arg = args[0] if args else kwargs.get("collection_name")
        assert name_arg == COLLECTION_NAME

        # vectors_config is always a keyword in the production code
        vectors_config = kwargs.get("vectors_config") or (args[1] if len(args) > 1 else None)
        assert vectors_config is not None
        assert vectors_config.size == 768
        assert vectors_config.distance == Distance.COSINE

    def test_ensure_collection_skips_when_present_correct(self):
        """Collection already exists with matching config → create_collection NOT called."""
        mock_client = _make_mock_client_with_collection(768, Distance.COSINE)

        ensure_collection(mock_client)

        mock_client.create_collection.assert_not_called()

    def test_ensure_collection_raises_on_dimension_mismatch(self):
        """Existing collection with 512 dims → ValueError; message cites both 768 and 512."""
        mock_client = _make_mock_client_with_collection(512, Distance.COSINE)

        with pytest.raises(ValueError) as exc_info:
            ensure_collection(mock_client)

        msg = str(exc_info.value)
        assert "768" in msg
        assert "512" in msg

    def test_ensure_collection_raises_on_distance_mismatch(self):
        """Existing collection with DOT distance → ValueError; message mentions COSINE."""
        mock_client = _make_mock_client_with_collection(768, Distance.DOT)

        with pytest.raises(ValueError) as exc_info:
            ensure_collection(mock_client)

        msg = str(exc_info.value).upper()
        assert "COSINE" in msg


# ---------------------------------------------------------------------------
# get_qdrant_client tests
# ---------------------------------------------------------------------------


class TestGetQdrantClient:
    def test_get_qdrant_client_default_env(self, monkeypatch):
        """No env vars set → QdrantClient constructed with localhost:6333."""
        monkeypatch.delenv("QDRANT_HOST", raising=False)
        monkeypatch.delenv("QDRANT_PORT", raising=False)

        with patch("src.embedding.qdrant_client_factory.QdrantClient") as mock_cls:
            get_qdrant_client()

        mock_cls.assert_called_once_with(host="localhost", port=6333)

    def test_get_qdrant_client_custom_env(self, monkeypatch):
        """Custom env vars → QdrantClient constructed with those values."""
        monkeypatch.setenv("QDRANT_HOST", "myhost")
        monkeypatch.setenv("QDRANT_PORT", "9999")

        with patch("src.embedding.qdrant_client_factory.QdrantClient") as mock_cls:
            get_qdrant_client()

        mock_cls.assert_called_once_with(host="myhost", port=9999)


# ---------------------------------------------------------------------------
# Module-level constant tests
# ---------------------------------------------------------------------------


class TestConstants:
    def test_collection_name_constant(self):
        assert COLLECTION_NAME == "financial_documents"

    def test_vector_dim_constant(self):
        assert VECTOR_DIM == 768
