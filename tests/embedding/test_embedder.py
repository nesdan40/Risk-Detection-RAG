"""
Unit tests for src/embedding/embedder.py — example-based tests.

All tests use mock SentenceTransformer instances so the real BAAI/bge-base-en-v1.5
model is never loaded.  Tests cover:

  - embed_texts():     empty input, single-batch, multi-batch, blank-text guard,
                       encode() failure propagation
  - make_point_id():  determinism, uniqueness, invalid-argument guards, namespace
  - build_points():   null payload preservation, wrong-dimension guard, required
                       payload field coverage, prohibited-key exclusion
  - module constants: QUERY_PREFIX, EMBEDDING_BATCH_SIZE
"""

import sys
import uuid
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from embedding.embedder import (
    EMBEDDING_BATCH_SIZE,
    NAMESPACE_UUID,
    QUERY_PREFIX,
    build_points,
    embed_texts,
    make_point_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The 11 required payload field names (spec §5.3, task 7.1 note)
_REQUIRED_PAYLOAD_FIELDS = (
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
)

# Prohibited keys that must never appear in any Qdrant payload
_PROHIBITED_PAYLOAD_KEYS = frozenset(
    {"group", "distressed", "healthy", "label", "ground_truth"}
)


def _make_mock_model(batch_size_or_sizes=None):
    """
    Return a MagicMock that mimics SentenceTransformer.encode().

    `encode(batch)` returns a numpy array of shape (len(batch), 768) filled
    with zeros.  This is enough for embed_texts() to convert to list[float].
    """
    model = MagicMock()
    model.encode.side_effect = lambda batch: np.zeros((len(batch), 768), dtype=float)
    return model


def _minimal_chunk(document_id: int = 1, chunk_index: int = 0, **overrides) -> dict:
    """Return a minimal chunk dict with all 11 payload fields populated."""
    base = {
        "company_id": 10,
        "document_id": document_id,
        "company_ticker": "TEST",
        "chunk_index": chunk_index,
        "document_type": "annual_report",
        "document_category": None,
        "fiscal_year": "FY2022",
        "fiscal_quarter": "Q1",
        "source": "BSE",
        "file_path": "data/raw/TEST/annual_reports/test.pdf",
        "chunk_text": "Sample chunk text for testing.",
    }
    base.update(overrides)
    return base


def _dummy_vector(dims: int = 768) -> list[float]:
    return [0.0] * dims


# ---------------------------------------------------------------------------
# embed_texts — empty input
# ---------------------------------------------------------------------------


class TestEmbedTextsEmpty:
    def test_embed_texts_empty(self):
        """texts=[] → no model.encode() calls, returns []."""
        model = _make_mock_model()
        result = embed_texts(model, [])
        assert result == []
        model.encode.assert_not_called()


# ---------------------------------------------------------------------------
# embed_texts — single batch (N <= 32)
# ---------------------------------------------------------------------------


class TestEmbedTextsSingleBatch:
    def test_embed_texts_single_batch(self):
        """N=3 texts → exactly one model.encode() call with all 3 texts."""
        model = _make_mock_model()
        texts = ["text one", "text two", "text three"]

        result = embed_texts(model, texts)

        assert model.encode.call_count == 1
        # The single call must have received all 3 texts
        (actual_batch,), _ = model.encode.call_args
        assert list(actual_batch) == texts
        # Result has one vector per input
        assert len(result) == 3


# ---------------------------------------------------------------------------
# embed_texts — multiple batches (N > 32)
# ---------------------------------------------------------------------------


class TestEmbedTextsMultipleBatches:
    def test_embed_texts_multiple_batches(self):
        """N=70 texts → 3 model.encode() calls (32, 32, 6); no batch exceeds 32."""
        model = _make_mock_model()
        texts = [f"chunk text number {i}" for i in range(70)]

        result = embed_texts(model, texts)

        # Exactly 3 calls
        assert model.encode.call_count == 3

        # Inspect individual batch sizes from call_args_list
        batch_sizes = [len(c.args[0]) for c in model.encode.call_args_list]
        assert batch_sizes == [32, 32, 6], f"expected [32, 32, 6], got {batch_sizes}"

        # No single batch exceeds EMBEDDING_BATCH_SIZE
        assert all(s <= EMBEDDING_BATCH_SIZE for s in batch_sizes)

        # Result covers all 70 inputs
        assert len(result) == 70


# ---------------------------------------------------------------------------
# embed_texts — blank text guard
# ---------------------------------------------------------------------------


class TestEmbedTextsBlankTextRaises:
    def test_embed_texts_blank_text_raises(self):
        """Whitespace-only text → ValueError before any model.encode() call."""
        model = _make_mock_model()
        texts = ["valid text", "   ", "another valid text"]

        with pytest.raises(ValueError):
            embed_texts(model, texts)

        model.encode.assert_not_called()

    def test_embed_texts_empty_string_raises(self):
        """Empty string → ValueError before any model.encode() call."""
        model = _make_mock_model()
        texts = ["valid", ""]

        with pytest.raises(ValueError):
            embed_texts(model, texts)

        model.encode.assert_not_called()

    def test_embed_texts_tabs_only_raises(self):
        """Tab-only string → ValueError before any model.encode() call."""
        model = _make_mock_model()
        texts = ["\t\t\t"]

        with pytest.raises(ValueError):
            embed_texts(model, texts)

        model.encode.assert_not_called()


# ---------------------------------------------------------------------------
# embed_texts — encode() raises RuntimeError
# ---------------------------------------------------------------------------


class TestEmbedTextsEncodeRaises:
    def test_embed_texts_encode_raises(self, capsys):
        """
        When model.encode() raises RuntimeError:
          - the exception propagates to the caller
          - no partial results are returned
          - the failed batch index is logged to stderr
        """
        model = MagicMock()
        model.encode.side_effect = RuntimeError("GPU out of memory")

        texts = ["chunk one", "chunk two", "chunk three"]

        with pytest.raises(RuntimeError, match="GPU out of memory"):
            embed_texts(model, texts)

        # Batch index was logged to stderr
        captured = capsys.readouterr()
        assert "batch index" in captured.err or "batch_idx" in captured.err or "0" in captured.err

    def test_no_partial_results_on_encode_failure(self):
        """Even if encode() fails on a later batch, no partial results leak out."""
        call_count = {"n": 0}

        def _encode_fails_on_second(batch):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("second batch failed")
            return np.zeros((len(batch), 768), dtype=float)

        model = MagicMock()
        model.encode.side_effect = _encode_fails_on_second

        texts = [f"text {i}" for i in range(40)]  # 2 batches: 32 + 8

        with pytest.raises(RuntimeError):
            embed_texts(model, texts)
        # The test just verifies the exception propagates (no partial list returned)


# ---------------------------------------------------------------------------
# make_point_id — determinism
# ---------------------------------------------------------------------------


class TestMakePointIdDeterminism:
    def test_make_point_id_determinism(self):
        """Same (document_id, chunk_index) called twice → identical UUID."""
        uid_a = make_point_id(document_id=5, chunk_index=3)
        uid_b = make_point_id(document_id=5, chunk_index=3)
        assert uid_a == uid_b


# ---------------------------------------------------------------------------
# make_point_id — uniqueness
# ---------------------------------------------------------------------------


class TestMakePointIdUniqueness:
    def test_different_chunk_indices(self):
        """(1, 0) vs (1, 1) → different UUIDs."""
        assert make_point_id(1, 0) != make_point_id(1, 1)

    def test_different_document_ids(self):
        """(1, 0) vs (2, 0) → different UUIDs."""
        assert make_point_id(1, 0) != make_point_id(2, 0)


# ---------------------------------------------------------------------------
# make_point_id — invalid arguments
# ---------------------------------------------------------------------------


class TestMakePointIdInvalidDocumentId:
    def test_make_point_id_invalid_document_id(self):
        """document_id=0 → ValueError."""
        with pytest.raises(ValueError):
            make_point_id(document_id=0, chunk_index=0)

    def test_negative_document_id_raises(self):
        """document_id=-1 → ValueError."""
        with pytest.raises(ValueError):
            make_point_id(document_id=-1, chunk_index=0)


class TestMakePointIdInvalidChunkIndex:
    def test_make_point_id_invalid_chunk_index(self):
        """chunk_index=-1 → ValueError."""
        with pytest.raises(ValueError):
            make_point_id(document_id=1, chunk_index=-1)


# ---------------------------------------------------------------------------
# make_point_id — namespace constant
# ---------------------------------------------------------------------------


class TestMakePointIdNamespaceConstant:
    def test_make_point_id_namespace_constant(self):
        """NAMESPACE_UUID matches the hard-coded value from the spec."""
        assert NAMESPACE_UUID == uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


# ---------------------------------------------------------------------------
# build_points — null fields preserved
# ---------------------------------------------------------------------------


class TestBuildPointsNullFieldsPreserved:
    def test_build_points_null_fields_preserved(self):
        """fiscal_quarter=None in chunk dict → payload has fiscal_quarter=None."""
        chunk = _minimal_chunk(fiscal_quarter=None)
        vector = _dummy_vector(768)

        points = build_points([chunk], [vector])

        assert len(points) == 1
        assert points[0].payload["fiscal_quarter"] is None

    def test_null_document_category_preserved(self):
        """document_category=None → payload has document_category=None."""
        chunk = _minimal_chunk(document_category=None)
        vector = _dummy_vector(768)

        points = build_points([chunk], [vector])

        assert points[0].payload["document_category"] is None


# ---------------------------------------------------------------------------
# build_points — wrong dimension
# ---------------------------------------------------------------------------


class TestBuildPointsWrongDimension:
    def test_build_points_wrong_dimension(self):
        """Vector with 512 dims → ValueError containing document_id, chunk_index, '512'."""
        chunk = _minimal_chunk(document_id=7, chunk_index=2)
        bad_vector = _dummy_vector(512)

        with pytest.raises(ValueError) as exc_info:
            build_points([chunk], [bad_vector])

        msg = str(exc_info.value)
        assert "512" in msg
        assert "7" in msg    # document_id
        assert "2" in msg    # chunk_index


# ---------------------------------------------------------------------------
# build_points — all 11 payload fields present
# ---------------------------------------------------------------------------


class TestBuildPointsAllPayloadFieldsPresent:
    def test_build_points_all_11_payload_fields_present(self):
        """Every PointStruct payload must contain all 11 required keys."""
        chunk = _minimal_chunk()
        vector = _dummy_vector(768)

        points = build_points([chunk], [vector])

        payload_keys = set(points[0].payload.keys())
        for field in _REQUIRED_PAYLOAD_FIELDS:
            assert field in payload_keys, f"Missing required payload field: {field!r}"


# ---------------------------------------------------------------------------
# build_points — no prohibited keys
# ---------------------------------------------------------------------------


class TestBuildPointsNoProhibitedKeys:
    def test_build_points_no_prohibited_keys(self):
        """None of group/distressed/healthy/label/ground_truth appear in payload."""
        # Inject prohibited keys into the input chunk dict — they must not survive
        chunk = _minimal_chunk()
        chunk["group"] = "distressed"
        chunk["distressed"] = True
        chunk["healthy"] = False
        chunk["label"] = 1
        chunk["ground_truth"] = "2019-01-01"

        vector = _dummy_vector(768)
        points = build_points([chunk], [vector])

        payload_keys = set(points[0].payload.keys())
        for key in _PROHIBITED_PAYLOAD_KEYS:
            assert key not in payload_keys, f"Prohibited key found in payload: {key!r}"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestQueryPrefixConstant:
    def test_query_prefix_constant(self):
        assert QUERY_PREFIX == "Represent this sentence for searching relevant passages: "


class TestEmbeddingBatchSizeConstant:
    def test_embedding_batch_size_constant(self):
        assert EMBEDDING_BATCH_SIZE == 32


# ---------------------------------------------------------------------------
# Property 1: Point ID determinism
# ---------------------------------------------------------------------------

from hypothesis import assume, given, settings
from hypothesis import strategies as st


class TestPropertyPointIdDeterminism:
    @given(st.integers(min_value=1), st.integers(min_value=0))
    @settings(max_examples=100)
    def test_property_point_id_determinism(self, document_id, chunk_index):
        """
        Property 1: Point ID determinism
        Feature: embedding-qdrant-pipeline
        Validates: Requirements 10.1, 10.2

        For any (document_id >= 1, chunk_index >= 0), calling make_point_id
        twice with identical inputs returns equal UUIDs.
        """
        uid_a = make_point_id(document_id, chunk_index)
        uid_b = make_point_id(document_id, chunk_index)
        assert uid_a == uid_b


# ---------------------------------------------------------------------------
# Property 2: Point ID uniqueness
# ---------------------------------------------------------------------------


class TestPropertyPointIdUniqueness:
    @given(
        st.tuples(st.integers(min_value=1), st.integers(min_value=0)),
        st.tuples(st.integers(min_value=1), st.integers(min_value=0)),
    )
    @settings(max_examples=100)
    def test_property_point_id_uniqueness(self, pair_a, pair_b):
        """
        Property 2: Point ID uniqueness
        Feature: embedding-qdrant-pipeline
        Validates: Requirement 10.3

        For any two distinct (document_id, chunk_index) pairs (differing in at
        least one field), make_point_id returns different UUID values for each pair.
        """
        assume(pair_a != pair_b)
        uid_a = make_point_id(pair_a[0], pair_a[1])
        uid_b = make_point_id(pair_b[0], pair_b[1])
        assert uid_a != uid_b


# ---------------------------------------------------------------------------
# Property 3: Payload completeness and null preservation
# ---------------------------------------------------------------------------


class TestPropertyPayloadCompleteness:
    @given(
        st.fixed_dictionaries({
            "company_id": st.integers(min_value=1),
            "document_id": st.integers(min_value=1),
            "company_ticker": st.text(min_size=1, max_size=20),
            "chunk_index": st.integers(min_value=0),
            "document_type": st.text(min_size=1, max_size=50),
            "document_category": st.one_of(st.text(min_size=1, max_size=50), st.none()),
            "fiscal_year": st.one_of(st.text(min_size=1, max_size=10), st.none()),
            "fiscal_quarter": st.one_of(st.text(min_size=1, max_size=5), st.none()),
            "source": st.one_of(st.text(min_size=1, max_size=20), st.none()),
            "file_path": st.text(min_size=1, max_size=200),
            "chunk_text": st.text(min_size=1, max_size=500),
        })
    )
    @settings(max_examples=100)
    def test_property_payload_completeness_and_null_preservation(self, chunk):
        """
        Property 3: Payload completeness and null preservation
        Feature: embedding-qdrant-pipeline
        Validates: Requirements 5.3, 5.5

        For any chunk dict with all 11 fields (optional fields set to str or None
        randomly), every PointStruct payload must contain all 11 keys, and any
        field that was None in the input must remain None in the payload.
        """
        _REQUIRED_FIELDS = (
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
        )
        vector = [0.0] * 768
        points = build_points([chunk], [vector])
        assert len(points) == 1
        payload = points[0].payload

        # All 11 fields must be present in the payload
        for field in _REQUIRED_FIELDS:
            assert field in payload, f"Required field {field!r} missing from payload"

        # None values must be preserved — never substituted with a default
        for field in _REQUIRED_FIELDS:
            if chunk.get(field) is None:
                assert payload[field] is None, (
                    f"Field {field!r} was None in input but got {payload[field]!r} in payload"
                )


# ---------------------------------------------------------------------------
# Property 4: No prohibited payload keys
# ---------------------------------------------------------------------------


class TestPropertyNoProhibitedPayloadKeys:
    @given(
        st.fixed_dictionaries({
            "company_id": st.integers(min_value=1),
            "document_id": st.integers(min_value=1),
            "company_ticker": st.text(min_size=1, max_size=20),
            "chunk_index": st.integers(min_value=0),
            "document_type": st.text(min_size=1, max_size=50),
            "document_category": st.one_of(st.text(min_size=1, max_size=50), st.none()),
            "fiscal_year": st.one_of(st.text(min_size=1, max_size=10), st.none()),
            "fiscal_quarter": st.one_of(st.text(min_size=1, max_size=5), st.none()),
            "source": st.one_of(st.text(min_size=1, max_size=20), st.none()),
            "file_path": st.text(min_size=1, max_size=200),
            "chunk_text": st.text(min_size=1, max_size=500),
            # Inject ALL prohibited keys to verify none leak through into the payload
            "group": st.text(),
            "distressed": st.booleans(),
            "healthy": st.booleans(),
            "label": st.integers(),
            "ground_truth": st.one_of(st.text(), st.none()),
        })
    )
    @settings(max_examples=100)
    def test_property_no_prohibited_payload_keys(self, chunk):
        """
        Property 4: No prohibited payload keys
        Feature: embedding-qdrant-pipeline
        Validates: Requirement 5.4

        The payload of any constructed PointStruct never contains keys named
        group, distressed, healthy, label, or ground_truth — even when those
        keys are present in the input chunk dict.
        """
        _PROHIBITED = frozenset({"group", "distressed", "healthy", "label", "ground_truth"})
        vector = [0.0] * 768
        points = build_points([chunk], [vector])
        payload_keys = set(points[0].payload.keys())
        for key in _PROHIBITED:
            assert key not in payload_keys, f"Prohibited key found in payload: {key!r}"
