"""
Chunker: pure text splitting and JSONL persistence for the preprocessing pipeline.

Splits extracted text into overlapping, metadata-annotated chunks appropriate for
financial documents, then writes them to a JSONL file under data/processed/.

No labels or ground-truth information is ever attached to any chunk object.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from models.document import Document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BAAI/bge-base-en-v1.5 reads at most 512 tokens and silently truncates the rest.
# Numeric tables tokenise at roughly 2.5 chars/token, so 1200 chars (~480 tokens)
# keeps every chunk fully inside the embedding window.
CHUNK_SIZE: int = 1200       # characters
CHUNK_OVERLAP: int = 150     # characters carried into the next chunk
_LOOKBACK_WINDOW: int = 150  # chars to scan back looking for a sentence boundary

# Prohibited keys — must never appear in any chunk object (label-leakage guard)
_PROHIBITED_KEYS = frozenset({"group", "distressed", "healthy", "label", "ground_truth"})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """Summary returned by chunk_and_store."""
    chunks: list[dict]
    chunk_count: int


# ---------------------------------------------------------------------------
# Pure splitting logic
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    """
    Split *text* into overlapping segments.  Pure function — no I/O, no side effects.

    Returns:
        []              if len(text.strip()) < 50
        [text.strip()]  if 50 <= len(text.strip()) <= CHUNK_SIZE
        list[str]       overlapping sliding-window chunks for longer texts

    Each chunk has its leading/trailing whitespace stripped.
    Internal whitespace is NOT collapsed here; that happens in build_chunk_objects.
    """
    stripped = text.strip()
    return [stripped[s:e] for s, e in chunk_spans(text)]


def chunk_spans(text: str) -> list[tuple[int, int]]:
    """
    Same splitting as chunk_text, but returns (start, end) offsets into
    ``text.strip()`` so callers can map chunks back to pages.
    """
    stripped = text.strip()
    n = len(stripped)

    if n < 50:
        return []

    if n <= CHUNK_SIZE:
        return [(0, n)]

    spans: list[tuple[int, int]] = []
    pos = 0

    while pos < n:
        end = min(pos + CHUNK_SIZE, n)

        if end < n:
            window_start = max(end - _LOOKBACK_WINDOW, pos)
            window = stripped[window_start:end]

            split_in_window = _find_sentence_boundary(window)

            if split_in_window is not None:
                split = window_start + split_in_window
            else:
                ws_pos = _find_last_whitespace(stripped, window_start, end)
                split = ws_pos if ws_pos is not None else end
        else:
            split = end

        # trim whitespace but keep offsets exact
        s0, e0 = pos, split
        while s0 < e0 and stripped[s0].isspace():
            s0 += 1
        while e0 > s0 and stripped[e0 - 1].isspace():
            e0 -= 1
        if e0 > s0:
            spans.append((s0, e0))

        if split >= n:
            break

        pos = max(split - CHUNK_OVERLAP, pos + 1)

    return spans


def _find_sentence_boundary(window: str) -> int | None:
    """
    Return the position *after* the last sentence-ending punctuation ([.!?])
    that is followed by whitespace or end-of-string inside *window*.

    Returns None if no such boundary exists.
    The returned index is relative to *window* (i.e. the split point within
    the window string, not the original text).
    """
    # We want the *last* match, so we scan the whole window and keep the last hit.
    last_pos: int | None = None
    for match in re.finditer(r'[.!?](?=\s|$)', window):
        # The split point is right after the punctuation mark
        last_pos = match.end()
    return last_pos


def _find_last_whitespace(text: str, start: int, end: int) -> int | None:
    """
    Return the index of the last whitespace character in text[start:end],
    or None if there is none.  The returned index points to the whitespace
    character itself (so the split is *at* that character, leaving it as the
    tail of the previous chunk when stripped).
    """
    for i in range(end - 1, start - 1, -1):
        if text[i].isspace():
            return i
    return None


# ---------------------------------------------------------------------------
# Chunk object construction
# ---------------------------------------------------------------------------

def build_chunk_objects(
    chunks: list[str],
    document: Document,
    company_ticker: str,
    page_ranges: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """
    Wrap raw chunk strings in metadata dicts.

    Each dict carries all fields required by Requirements 5.1/6.2, with
    internal whitespace in chunk_text collapsed to a single space.

    Raises:
        ValueError  if company_id, document_id, file_path, or chunk_text is
                    None or empty string after population.

    Never sets keys: group, distressed, healthy, label, ground_truth.
    """
    result: list[dict] = []

    for i, raw_chunk in enumerate(chunks):
        # Collapse internal whitespace
        chunk_text_collapsed = re.sub(r'\s+', ' ', raw_chunk).strip()

        # Build the object (no prohibited keys)
        obj: dict = {
            "document_id":       document.id,
            "company_id":        document.company_id,
            "company_ticker":    company_ticker,
            "chunk_index":       i,
            "chunk_text":        chunk_text_collapsed,
            "document_type":     document.document_type,
            "document_category": document.document_category,
            "fiscal_year":       document.fiscal_year,
            "fiscal_quarter":    document.fiscal_quarter,
            "source":            document.source,
            "file_path":         document.file_path,
            # 1-based PDF page numbers (None when the page map is unknown)
            "period_end":        _iso_date(document.period_end),
            "page_start":        page_ranges[i][0] if page_ranges else None,
            "page_end":          page_ranges[i][1] if page_ranges else None,
        }

        # Validate non-nullable fields
        _validate_chunk_object(obj, document.id)

        # Safety: assert no prohibited key slipped in
        for key in _PROHIBITED_KEYS:
            assert key not in obj, f"Prohibited key '{key}' found in chunk object"

        result.append(obj)

    return result


def _iso_date(value: object) -> str | None:
    """ISO string for a date/datetime (or an already-ISO string); None otherwise."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value if isinstance(value, str) else None


def _validate_chunk_object(obj: dict, document_id: object) -> None:
    """Raise ValueError if any non-nullable chunk field is None or empty."""
    for field in ("company_id", "document_id", "file_path", "chunk_text"):
        value = obj.get(field)
        if value is None or value == "":
            raise ValueError(
                f"Chunk validation failed for document_id={document_id!r}: "
                f"field '{field}' is {value!r} (must be non-null and non-empty)."
            )


# ---------------------------------------------------------------------------
# Orchestration: split → build objects → write JSONL
# ---------------------------------------------------------------------------

def _page_ranges(pages: list[str]) -> tuple[str, list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Join per-page text and return (stripped_text, spans, page_ranges) where
    page_ranges[i] is the 1-based (first_page, last_page) of chunk i.
    """
    import bisect

    full = "\n".join(pages)
    stripped = full.strip()
    lead = len(full) - len(full.lstrip())
    starts, offset = [], 0
    for page in pages:
        starts.append(offset)
        offset += len(page) + 1  # +1 for the joining newline

    def page_of(pos_in_stripped: int) -> int:
        return max(bisect.bisect_right(starts, pos_in_stripped + lead) - 1, 0) + 1

    spans = chunk_spans(full)
    ranges = [(page_of(s0), page_of(max(e0 - 1, s0))) for s0, e0 in spans]
    return stripped, spans, ranges


def chunk_and_store(
    document: Document,
    text: str,
    company_ticker: str,
    processed_dir: Path,
    pages: list[str] | None = None,
) -> ChunkResult:
    """
    Orchestrate chunking -> build_chunk_objects -> JSONL write.

    When *pages* (one string per PDF page) is given, every chunk records the
    1-based page range it came from; otherwise page_start/page_end are None.

    Output path: processed_dir / company_ticker / "{document.id}_chunks.jsonl"

    Writes an empty file when zero chunks are produced (signals that chunking
    was attempted).  Overwrites any pre-existing file at that path.

    Raises:
        OSError     on filesystem write failure (caller handles)
        ValueError  propagated from build_chunk_objects (caller handles)
    """
    page_ranges = None
    if pages:
        stripped, spans, page_ranges = _page_ranges(pages)
        raw_chunks = [stripped[s0:e0] for s0, e0 in spans]
    else:
        raw_chunks = chunk_text(text)

    chunk_objects: list[dict] = []
    if raw_chunks:
        chunk_objects = build_chunk_objects(
            raw_chunks, document, company_ticker, page_ranges=page_ranges
        )

    output_dir = processed_dir / company_ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{document.id}_chunks.jsonl"

    with open(output_path, "w", encoding="utf-8") as fh:
        for obj in chunk_objects:
            fh.write(json.dumps(obj) + "\n")

    return ChunkResult(chunks=chunk_objects, chunk_count=len(chunk_objects))
