"""
Unit tests for src/preprocessing/chunker.py

Example-based tests covering:
  - chunk_text: sentence boundary split, whitespace fallback, exact-position
    fallback, and zero-chunk empty input
  - build_chunk_objects: raises ValueError when company_id is None
  - chunk_and_store: directory + file creation, file overwrite, empty file for
    zero chunks

No database is required — chunker.py is pure I/O / pure computation.

Requirements: 4.3, 4.4, 5.6, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from preprocessing.chunker import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    build_chunk_objects,
    chunk_and_store,
    chunk_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(
    *,
    doc_id: int = 1,
    company_id: int | None = 7,
    file_path: str = "data/raw/TEST/test.pdf",
) -> MagicMock:
    """Return a minimal Document-like mock with all required fields set."""
    doc = MagicMock()
    doc.id = doc_id
    doc.company_id = company_id
    doc.file_path = file_path
    doc.document_type = "annual_report"
    doc.document_category = None
    doc.fiscal_year = "FY2019"
    doc.fiscal_quarter = None
    doc.source = "BSE"
    return doc


# ---------------------------------------------------------------------------
# chunk_text — sentence boundary split
# ---------------------------------------------------------------------------

class TestSentenceBoundarySplit:
    """
    Requirement 4.3: prefer to break at [.!?] + whitespace boundaries.
    """

    def test_sentence_boundary_split(self):
        """
        Build a text composed of short sentences (~100 chars each) so that
        multiple sentence boundaries fall within every look-back window.
        The first chunk must therefore split at a sentence boundary (ends with
        '.', '!', or '?') rather than mid-word.

        Strategy: each sentence is ~100 chars and ends with '. '.  Because the
        look-back window is 200 chars and sentences are 100 chars, at least one
        sentence-ending '.' always falls within the look-back window of any
        split point.
        """
        # Short sentence ~90 chars including trailing '. '
        sentence = ("word " * 16).rstrip() + ". "   # 80 + 2 = 82 chars
        # Repeat enough times to produce well over CHUNK_SIZE
        text = sentence * ((CHUNK_SIZE * 2) // len(sentence) + 5)

        assert len(text) > CHUNK_SIZE, "Precondition: text must exceed CHUNK_SIZE"
        assert len(sentence) < 200, "Precondition: sentence shorter than look-back window"

        chunks = chunk_text(text)

        assert len(chunks) >= 2, "Expected at least two chunks for a long text"

        # The very first chunk must end at a sentence boundary because there
        # are always sentence-ending '.' marks within the look-back window.
        assert chunks[0][-1] in ".!?", (
            f"First chunk does not end at sentence boundary: ...{chunks[0][-30:]!r}"
        )

    def test_sentence_boundary_not_mid_word(self):
        """
        Ensure no chunk ends in the middle of a word (contains no trailing
        partial token after the last whitespace-terminated sentence).
        """
        word = "abcdefghij"  # 10-char token, no punctuation
        sentence = (word + " ") * 60 + "end."  # ~670 chars, ends with '.'
        # Repeat to exceed CHUNK_SIZE
        text = (sentence + " ") * 6   # ~4 KB

        chunks = chunk_text(text)
        assert len(chunks) >= 2

        for chunk in chunks:
            # No chunk should end with a partial word that is not sentence-final
            last_char = chunk[-1]
            # Accept ending on '.', '!', '?', or a complete word character.
            # Disallow ending with whitespace (strip guarantees this) — verify.
            assert last_char != " ", f"Chunk ends with whitespace: {chunk[-10:]!r}"


# ---------------------------------------------------------------------------
# chunk_text — whitespace fallback split
# ---------------------------------------------------------------------------

class TestWhitespaceFallbackSplit:
    """
    Requirement 4.4 (first fallback): when no sentence boundary exists, split
    at whitespace.
    """

    def test_whitespace_fallback_split(self):
        """
        Create text with no [.!?] characters but with spaces, longer than
        CHUNK_SIZE.  Assert that chunks are produced and that no chunk exceeds
        CHUNK_SIZE characters.
        """
        # Words joined by spaces, no sentence-ending punctuation.
        word = "alpha"
        text = (" ".join([word] * ((CHUNK_SIZE * 2) // (len(word) + 1) + 10)))

        assert len(text) > CHUNK_SIZE, "Precondition: text must exceed CHUNK_SIZE"
        # Sanity: no sentence-ending punctuation.
        assert not any(c in text for c in ".!?")

        chunks = chunk_text(text)

        assert len(chunks) >= 2, "Expected multiple chunks for a long no-punctuation text"

        for chunk in chunks:
            assert len(chunk) <= CHUNK_SIZE, (
                f"Chunk exceeds CHUNK_SIZE ({len(chunk)} > {CHUNK_SIZE}): {chunk[:40]!r}"
            )

    def test_whitespace_fallback_split_at_space(self):
        """
        The split point for the whitespace fallback must coincide with a
        space in the original text — i.e., no chunk starts or ends mid-word
        due to an arbitrary character cut.
        """
        # Build text: fixed-width tokens separated by single spaces
        token = "X" * 20          # 20-char token, no punctuation
        tokens = [token] * (CHUNK_SIZE // 10)  # enough to exceed CHUNK_SIZE
        text = " ".join(tokens)

        assert len(text) > CHUNK_SIZE

        chunks = chunk_text(text)

        for chunk in chunks:
            # After stripping, no chunk should start or end with a partial token.
            # Each token is exactly 20 'X' chars — check start and end.
            stripped = chunk.strip()
            # First character should be the start of a token ('X')
            assert stripped[0] == "X", f"Chunk starts unexpectedly: {stripped[:5]!r}"
            # Last character should be the last char of a token ('X')
            assert stripped[-1] == "X", f"Chunk ends unexpectedly: {stripped[-5:]!r}"


# ---------------------------------------------------------------------------
# chunk_text — exact position fallback
# ---------------------------------------------------------------------------

class TestExactPositionFallback:
    """
    Requirement 4.4 (second fallback): no whitespace found → split at exact
    CHUNK_SIZE position.
    """

    def test_exact_position_fallback(self):
        """
        A string of a single repeated character has no whitespace and no
        sentence-ending punctuation.  The first chunk must be exactly
        CHUNK_SIZE characters.
        """
        text = "a" * (CHUNK_SIZE * 2 + 100)

        chunks = chunk_text(text)

        assert len(chunks) >= 2, "Expected at least two chunks for a very long no-whitespace text"
        # The first chunk must be exactly CHUNK_SIZE chars (hard cut).
        assert len(chunks[0]) == CHUNK_SIZE, (
            f"First chunk length {len(chunks[0])} != CHUNK_SIZE {CHUNK_SIZE}"
        )

    def test_exact_position_fallback_all_chunks_within_bound(self):
        """
        Even with a hard cut, every chunk must stay within CHUNK_SIZE.
        """
        text = "b" * (CHUNK_SIZE * 3 + 77)
        chunks = chunk_text(text)

        for chunk in chunks:
            assert len(chunk) <= CHUNK_SIZE


# ---------------------------------------------------------------------------
# chunk_text — zero chunks for empty / below-threshold text
# ---------------------------------------------------------------------------

class TestZeroChunksEdgeCases:
    """
    Requirement 4.5: text shorter than 50 stripped chars → empty list.
    """

    def test_zero_chunks_empty_string(self):
        assert chunk_text("") == []

    def test_zero_chunks_whitespace_only(self):
        assert chunk_text("   \t\n  ") == []

    def test_zero_chunks_short_text(self):
        assert chunk_text("short text") == []

    def test_zero_chunks_exactly_49_chars(self):
        assert chunk_text("a" * 49) == []

    def test_one_chunk_at_exactly_50_chars(self):
        text = "a" * 50
        result = chunk_text(text)
        assert len(result) == 1
        assert result[0] == text


# ---------------------------------------------------------------------------
# build_chunk_objects — null company_id raises ValueError
# ---------------------------------------------------------------------------

class TestBuildChunkObjectsValidation:
    """
    Requirement 5.6: non-nullable fields must not be null/empty.
    """

    def test_null_company_id_raises_value_error(self):
        doc = _make_doc(company_id=None)
        # Provide a chunk long enough that chunk_text would return content.
        chunks = ["a" * 100]

        with pytest.raises(ValueError, match="company_id"):
            build_chunk_objects(chunks, doc, company_ticker="TEST")

    def test_null_file_path_raises_value_error(self):
        doc = _make_doc(file_path=None)
        chunks = ["a" * 100]

        with pytest.raises(ValueError, match="file_path"):
            build_chunk_objects(chunks, doc, company_ticker="TEST")

    def test_empty_file_path_raises_value_error(self):
        doc = _make_doc(file_path="")
        chunks = ["a" * 100]

        with pytest.raises(ValueError, match="file_path"):
            build_chunk_objects(chunks, doc, company_ticker="TEST")

    def test_valid_doc_returns_chunk_objects(self):
        doc = _make_doc()
        chunks = ["a" * 100]
        result = build_chunk_objects(chunks, doc, company_ticker="TEST")

        assert len(result) == 1
        assert result[0]["company_id"] == doc.company_id
        assert result[0]["document_id"] == doc.id

    def test_no_prohibited_keys_in_objects(self):
        """Requirement 5.3: no label-leakage keys."""
        doc = _make_doc()
        chunks = ["a" * 100, "b" * 100]
        result = build_chunk_objects(chunks, doc, company_ticker="TEST")

        prohibited = {"group", "distressed", "healthy", "label", "ground_truth"}
        for obj in result:
            assert not prohibited.intersection(obj.keys()), (
                f"Prohibited key found: {prohibited.intersection(obj.keys())}"
            )


# ---------------------------------------------------------------------------
# chunk_and_store — directory and file creation (Requirement 6.3)
# ---------------------------------------------------------------------------

class TestChunkAndStoreCreatesDirectory:
    def test_creates_ticker_subdirectory(self, tmp_path: Path):
        """
        When the ticker subdirectory does not exist, chunk_and_store must
        create it before writing (Requirement 6.3).
        """
        ticker = "NEWCO"
        doc = _make_doc(doc_id=42)
        text = "word " * 700  # well above CHUNK_SIZE threshold

        result = chunk_and_store(doc, text, ticker, processed_dir=tmp_path)

        expected_dir = tmp_path / ticker
        assert expected_dir.is_dir(), "Ticker subdirectory was not created"

    def test_creates_jsonl_file(self, tmp_path: Path):
        """chunk_and_store must create the JSONL file at the expected path."""
        ticker = "NEWCO"
        doc = _make_doc(doc_id=42)
        text = "word " * 700

        chunk_and_store(doc, text, ticker, processed_dir=tmp_path)

        expected_file = tmp_path / ticker / "42_chunks.jsonl"
        assert expected_file.is_file(), "JSONL file was not created"

    def test_jsonl_file_contains_valid_json_lines(self, tmp_path: Path):
        """Each line in the output file must be valid JSON."""
        ticker = "NEWCO"
        doc = _make_doc(doc_id=42)
        text = "word " * 700

        result = chunk_and_store(doc, text, ticker, processed_dir=tmp_path)

        expected_file = tmp_path / ticker / "42_chunks.jsonl"
        lines = expected_file.read_text(encoding="utf-8").splitlines()

        assert len(lines) == result.chunk_count
        for line in lines:
            obj = json.loads(line)   # raises if invalid
            assert "chunk_text" in obj
            assert "chunk_index" in obj


# ---------------------------------------------------------------------------
# chunk_and_store — overwrites existing file (Requirement 6.4)
# ---------------------------------------------------------------------------

class TestChunkAndStoreOverwritesExistingFile:
    def test_overwrites_rather_than_appends(self, tmp_path: Path):
        """
        Run chunk_and_store twice with different texts; the file should
        reflect only the second run's chunks.
        """
        ticker = "OVERCO"
        doc = _make_doc(doc_id=99)
        output_file = tmp_path / ticker / "99_chunks.jsonl"

        # First run
        text_first = "sentence one. " * 300  # produces some chunks
        chunk_and_store(doc, text_first, ticker, processed_dir=tmp_path)
        first_content = output_file.read_text(encoding="utf-8")

        # Second run with different text
        text_second = ("different content altogether. " * 300)
        chunk_and_store(doc, text_second, ticker, processed_dir=tmp_path)
        second_content = output_file.read_text(encoding="utf-8")

        # The file must have changed and must not contain first-run text
        assert first_content != second_content, "File was not overwritten"

        # Verify: every line parses as JSON and contains chunk_text from second input
        for line in second_content.splitlines():
            obj = json.loads(line)
            # chunk_text should come from the second text, not the first
            assert "sentence one" not in obj["chunk_text"], (
                "First-run content leaked into overwritten file"
            )

    def test_chunk_count_reflects_second_run(self, tmp_path: Path):
        """
        chunk_count returned by the second call should match the lines in the
        file (not the sum of both runs).
        """
        ticker = "OVERCO2"
        doc = _make_doc(doc_id=100)
        output_file = tmp_path / ticker / "100_chunks.jsonl"

        chunk_and_store(doc, "word " * 800, ticker, processed_dir=tmp_path)
        result2 = chunk_and_store(doc, "other " * 800, ticker, processed_dir=tmp_path)

        lines = [l for l in output_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == result2.chunk_count


# ---------------------------------------------------------------------------
# chunk_and_store — empty file for zero chunks (Requirement 6.5)
# ---------------------------------------------------------------------------

class TestChunkAndStoreEmptyFileForZeroChunks:
    def test_empty_file_written_for_below_threshold_text(self, tmp_path: Path):
        """
        When the text is below the 50-char threshold, chunk_text returns [].
        chunk_and_store must still write an empty file (not skip creation).
        """
        ticker = "EMPTYCO"
        doc = _make_doc(doc_id=55)
        text = "short"  # well below 50-char threshold

        result = chunk_and_store(doc, text, ticker, processed_dir=tmp_path)

        expected_file = tmp_path / ticker / "55_chunks.jsonl"
        assert expected_file.is_file(), "Empty JSONL file was not created"
        assert expected_file.read_text(encoding="utf-8") == "", (
            "File should be empty for zero chunks"
        )
        assert result.chunk_count == 0

    def test_empty_file_written_for_empty_string(self, tmp_path: Path):
        """Empty string input → empty JSONL file."""
        ticker = "EMPTYCO2"
        doc = _make_doc(doc_id=56)

        result = chunk_and_store(doc, "", ticker, processed_dir=tmp_path)

        expected_file = tmp_path / ticker / "56_chunks.jsonl"
        assert expected_file.is_file()
        assert expected_file.stat().st_size == 0
        assert result.chunk_count == 0


# ===========================================================================
# Property-Based Tests (Hypothesis)
# ===========================================================================

import re as _re  # used by Property 8 (consecutive-whitespace check)

from hypothesis import HealthCheck, given, assume, settings
from hypothesis import strategies as st

# Required keys every chunk object must carry
_REQUIRED_CHUNK_KEYS = frozenset({
    "document_id",
    "company_id",
    "company_ticker",
    "chunk_index",
    "chunk_text",
    "document_type",
    "document_category",
    "fiscal_year",
    "fiscal_quarter",
    "source",
    "file_path",
})

# Keys that must never appear (label-leakage guard)
_PROHIBITED_CHUNK_KEYS = frozenset({"group", "distressed", "healthy", "label", "ground_truth"})


# ---------------------------------------------------------------------------
# Property 1 — Chunk size bound (Task 8.2)
# Validates: Requirements 4.1
# ---------------------------------------------------------------------------

class TestPropertyChunkSizeBound:
    """
    **Property 1: Chunk size bound**
    For any string where ``len(text.strip()) >= 50``, every chunk produced by
    ``chunk_text(text)`` has ``len(chunk) <= CHUNK_SIZE``.
    """

    @given(st.text(min_size=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_no_chunk_exceeds_chunk_size(self, text: str):
        """Validates: Requirements 4.1"""
        assume(len(text.strip()) >= 50)

        chunks = chunk_text(text)

        for chunk in chunks:
            assert len(chunk) <= CHUNK_SIZE, (
                f"Chunk of length {len(chunk)} exceeds CHUNK_SIZE={CHUNK_SIZE}: "
                f"{chunk[:40]!r}"
            )


# ---------------------------------------------------------------------------
# Property 2 — Low-yield guard (Task 8.3)
# Validates: Requirements 4.5, 9.1
# ---------------------------------------------------------------------------

class TestPropertyLowYieldGuard:
    """
    **Property 2: Low-yield guard**
    For any string where ``len(text.strip()) < 50``, ``chunk_text(text)``
    returns ``[]``.
    """

    @given(st.text(max_size=49))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_short_text_returns_empty_list(self, text: str):
        """Validates: Requirements 4.5, 9.1"""
        assume(len(text.strip()) < 50)

        result = chunk_text(text)

        assert result == [], (
            f"Expected [] for text with stripped length "
            f"{len(text.strip())}, got {len(result)} chunk(s)"
        )


# ---------------------------------------------------------------------------
# Property 3 — Single-chunk short text (Task 8.4)
# Validates: Requirements 4.6
# ---------------------------------------------------------------------------

class TestPropertySingleChunkShortText:
    """
    **Property 3: Single-chunk short text**
    For any string where ``50 <= len(text.strip()) <= CHUNK_SIZE``,
    ``chunk_text(text)`` returns a list of exactly one element.
    """

    @given(st.text(min_size=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_text_fitting_in_one_chunk(self, text: str):
        """Validates: Requirements 4.6"""
        stripped_len = len(text.strip())
        assume(50 <= stripped_len <= CHUNK_SIZE)

        result = chunk_text(text)

        assert len(result) == 1, (
            f"Expected exactly 1 chunk for stripped length "
            f"{stripped_len}, got {len(result)}"
        )


# ---------------------------------------------------------------------------
# Property 4 — Coverage: every character is reachable (Task 8.5)
# Validates: Requirements 4.1, 4.2
# ---------------------------------------------------------------------------

class TestPropertyCoverage:
    """
    **Property 4: Coverage — every character is reachable**
    For any string where ``len(text.strip()) >= 50``, every non-whitespace
    character that appears in the stripped input also appears in at least
    one chunk.
    """

    @given(st.text(min_size=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_all_nonwhitespace_chars_reachable(self, text: str):
        """Validates: Requirements 4.1, 4.2"""
        assume(len(text.strip()) >= 50)

        chunks = chunk_text(text)

        # Build the set of non-whitespace characters from all chunks combined
        chars_in_chunks = set(c for c in "".join(chunks) if not c.isspace())

        # Build the set of non-whitespace characters in the stripped source
        chars_in_source = set(c for c in text.strip() if not c.isspace())

        missing = chars_in_source - chars_in_chunks
        assert not missing, (
            f"Characters present in source but absent from all chunks: "
            f"{missing!r}"
        )


# ---------------------------------------------------------------------------
# Property 5 — chunk_index sequence is contiguous from zero (Task 8.6)
# Validates: Requirements 4.7, 5.4
# ---------------------------------------------------------------------------

class TestPropertyChunkIndexSequence:
    """
    **Property 5: chunk_index sequence is contiguous from zero**
    ``chunk_index`` values across all objects from ``build_chunk_objects()``
    form ``[0, 1, ..., n-1]``.
    """

    @given(st.text(min_size=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_chunk_index_contiguous_from_zero(self, text: str):
        """Validates: Requirements 4.7, 5.4"""
        assume(len(text.strip()) >= 50)

        doc = _make_doc()
        chunks = chunk_text(text)
        if not chunks:
            return  # nothing to check

        objects = build_chunk_objects(chunks, doc, company_ticker="TEST")

        indices = [obj["chunk_index"] for obj in objects]
        expected = list(range(len(objects)))

        assert indices == expected, (
            f"chunk_index sequence {indices} is not contiguous from zero"
        )


# ---------------------------------------------------------------------------
# Property 6 — Metadata completeness and no prohibited keys (Task 8.7)
# Validates: Requirements 5.1, 5.3
# ---------------------------------------------------------------------------

class TestPropertyMetadataCompleteness:
    """
    **Property 6: Metadata completeness and no prohibited keys**
    Every chunk object produced by ``build_chunk_objects()`` contains all
    required keys and none of the prohibited keys.
    """

    @given(st.text(min_size=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_required_keys_present_and_no_prohibited_keys(self, text: str):
        """Validates: Requirements 5.1, 5.3"""
        assume(len(text.strip()) >= 50)

        doc = _make_doc()
        chunks = chunk_text(text)
        if not chunks:
            return  # nothing to check

        objects = build_chunk_objects(chunks, doc, company_ticker="TEST")

        for obj in objects:
            missing_keys = _REQUIRED_CHUNK_KEYS - obj.keys()
            assert not missing_keys, (
                f"Chunk object is missing required keys: {missing_keys!r}"
            )

            leaked_keys = _PROHIBITED_CHUNK_KEYS & obj.keys()
            assert not leaked_keys, (
                f"Chunk object contains prohibited (label-leakage) keys: "
                f"{leaked_keys!r}"
            )


# ---------------------------------------------------------------------------
# Property 7 — Idempotency of chunk_text (Task 8.8)
# Validates: Requirements 9.1, 9.3
# ---------------------------------------------------------------------------

class TestPropertyIdempotency:
    """
    **Property 7: Idempotency of chunk_text**
    ``chunk_text(text) == chunk_text(text)`` for any input.
    Running the function twice on the same input always produces the same
    result.
    """

    @given(st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_chunk_text_is_idempotent(self, text: str):
        """Validates: Requirements 9.1, 9.3"""
        result_a = chunk_text(text)
        result_b = chunk_text(text)

        assert result_a == result_b, (
            f"chunk_text returned different results on identical input "
            f"(first={len(result_a)} chunks, second={len(result_b)} chunks)"
        )


# ---------------------------------------------------------------------------
# Property 8 — Internal whitespace collapsed in chunk_text field (Task 8.9)
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------

class TestPropertyWhitespaceCollapsed:
    """
    **Property 8: Internal whitespace collapsed in chunk_text field**
    No ``chunk_text`` value in any chunk object produced by
    ``build_chunk_objects()`` contains two or more consecutive whitespace
    characters.
    """

    @given(st.text(min_size=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_no_consecutive_whitespace_in_chunk_text(self, text: str):
        """Validates: Requirements 5.5"""
        assume(len(text.strip()) >= 50)

        doc = _make_doc()
        chunks = chunk_text(text)
        if not chunks:
            return  # nothing to check

        objects = build_chunk_objects(chunks, doc, company_ticker="TEST")

        for obj in objects:
            chunk_text_value: str = obj["chunk_text"]
            assert not _re.search(r'\s{2,}', chunk_text_value), (
                f"chunk_text contains consecutive whitespace: "
                f"{chunk_text_value[:60]!r}"
            )
