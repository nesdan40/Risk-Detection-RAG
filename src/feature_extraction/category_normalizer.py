"""
Category normalizer for the structured feature extraction pipeline.

Resolves a raw ``document_category`` string or filename token to one of the
nine canonical governance signal labels, ``'Corporate Action - Other'``,
``'Credit Rating'``, or ``'unclassified'``, using a three-layer strategy:

    Layer 1 — Exact alias match (case-insensitive, after suffix stripping)
    Layer 2 — Fuzzy match via ``rapidfuzz.fuzz.partial_ratio`` (threshold 80)
    Layer 3 — Keyword match against the first 2000 characters of chunk text

The public entry point is :func:`normalize`, which is a **pure function**: it
performs no I/O and never raises.  The JSONL read for Layer 3 is done by the
caller and passed in as ``chunk_text``.
"""

from __future__ import annotations

import re
import sys
from typing import Final

from rapidfuzz import fuzz as _fuzz


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

CANONICAL_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Board Meeting",
        "Financial Results",
        "AGM / Annual General Meeting Updates",
        "Appointment",
        "Director Change",
        "Resignation",
        "News Clarification",
        "News Verification",
        "Press Release",
    }
)

CORPORATE_ACTION_OTHER: Final[str] = "Corporate Action - Other"

#: All valid output labels (used for type-checking in properties)
_ALL_VALID_LABELS: Final[frozenset[str]] = (
    CANONICAL_LABELS | {CORPORATE_ACTION_OTHER, "Credit Rating", "unclassified"}
)

_VALID_LAYERS: Final[frozenset[str]] = frozenset(
    {"exact", "fuzzy", "keyword", "unclassified"}
)

# Minimum rapidfuzz score to accept a fuzzy match
FUZZY_THRESHOLD: Final[int] = 80


# ---------------------------------------------------------------------------
# Alias dictionary  (lowercase keys → canonical label)
# ---------------------------------------------------------------------------

ALIAS_DICT: dict[str, str] = {
    # Board Meeting
    "outcome of board meeting": "Board Meeting",
    "outcome of bm": "Board Meeting",
    "obm": "Board Meeting",
    "ooutcome of board meeting": "Board Meeting",
    "board meeting": "Board Meeting",
    # Financial Results
    "financial results updates": "Financial Results",
    "financial result updates": "Financial Results",
    "fiancial results updates": "Financial Results",
    "financial reults updates": "Financial Results",
    "financialresults updates": "Financial Results",  # FInancialResults Updates typo
    "financial results": "Financial Results",
    # AGM
    "agm / annual general meeting updates": "AGM / Annual General Meeting Updates",
    "annual general meeting updates": "AGM / Annual General Meeting Updates",
    "agmupdates": "AGM / Annual General Meeting Updates",
    "agm updates": "AGM / Annual General Meeting Updates",
    "annual general meeting": "AGM / Annual General Meeting Updates",
    "agm": "AGM / Annual General Meeting Updates",
    # Appointment
    "appointment": "Appointment",
    "appointmet": "Appointment",
    "directorappointed": "Appointment",
    # Director Change
    "director change": "Director Change",
    "directorchange": "Director Change",
    "change_in_director": "Director Change",
    "cessation": "Director Change",
    "change in director": "Director Change",
    # Resignation
    "resignation": "Resignation",
    # News Clarification
    "news clarification": "News Clarification",
    "newsclarification": "News Clarification",
    "clarification": "News Clarification",
    # News Verification
    "news verification": "News Verification",
    "newsverification": "News Verification",
    # Press Release
    "press release": "Press Release",
    "press relase": "Press Release",
    "press releas": "Press Release",
    "pressrelease": "Press Release",
    "press release updates": "Press Release",
    "press": "Press Release",
    "pr": "Press Release",
    # Corporate Action - Other
    "book closure": CORPORATE_ACTION_OTHER,
    "book clousure": CORPORATE_ACTION_OTHER,  # typo variant
    "bookclosure": CORPORATE_ACTION_OTHER,
    "aquisition": CORPORATE_ACTION_OTHER,
    "acquisition": CORPORATE_ACTION_OTHER,
    "buyback": CORPORATE_ACTION_OTHER,
    "settlement": CORPORATE_ACTION_OTHER,
    "stayorder": CORPORATE_ACTION_OTHER,
    "change in auditors": CORPORATE_ACTION_OTHER,
    # Credit Rating (as filename token — not a governance signal)
    "credit rating": "Credit Rating",
    # Corporate Action - Other as its own key → fixed point for idempotence
    "corporate action - other": CORPORATE_ACTION_OTHER,
}


# ---------------------------------------------------------------------------
# Keyword rules  (applied in priority order, highest first)
# ---------------------------------------------------------------------------

# Each rule is a tuple:
#   (required_keywords, conjunction_keywords_or_None, canonical_label)
#
# Matching logic:
#   required_keywords: ALL must be present (case-insensitive)
#   conjunction_keywords: if not None, at least ONE must also be present
#
# Priority order (index 0 = highest):
#   1. Credit Rating
#   2. Director Change
#   3. Resignation
#   4. Appointment
#   5. AGM / Annual General Meeting Updates
#   6. Board Meeting
KEYWORD_RULES: list[tuple[list[str], list[str] | None, str]] = [
    # Credit Rating (highest priority)
    (["credit rating"], None, "Credit Rating"),
    (["rating"], ["upgrade", "downgrade", "watch", "outlook"], "Credit Rating"),
    # Director Change
    (["director"], ["change", "cessation", "vacate"], "Director Change"),
    # Resignation
    (["resigned"], None, "Resignation"),
    (["resignation"], None, "Resignation"),
    # Appointment
    (["appointed"], None, "Appointment"),
    (["appointment"], None, "Appointment"),
    # AGM
    (["annual general meeting"], None, "AGM / Annual General Meeting Updates"),
    (["agm"], None, "AGM / Annual General Meeting Updates"),
    # Board Meeting (lowest priority)
    (["board meeting"], None, "Board Meeting"),
    (["outcome of board"], None, "Board Meeting"),
]


# ---------------------------------------------------------------------------
# Suffix stripping
# ---------------------------------------------------------------------------

# Patterns stripped from both document_category and filename-derived tokens
# BEFORE alias/fuzzy matching.  NOT applied to ALIAS_DICT keys.
#
# Note: the requirements specify ``_(N)`` (underscore-prefixed), but real
# filenames also use a space-prefixed form ``(N)`` (e.g.
# "Financial Results Updates (2)_SCANNED").  Both forms are stripped here
# to match the ingestion pipeline's ``_DUPLICATE_SUFFIX_RE`` behaviour.
_SUFFIX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"_scannedpdf\s*$", re.IGNORECASE),       # _SCANNEDpdf (check first)
    re.compile(r"_scanneed\s*$", re.IGNORECASE),          # _SCANNEED (typo variant)
    re.compile(r"_scanned\s*$", re.IGNORECASE),           # _SCANNED
    re.compile(r"[\s_]*\(\d+\)\s*$", re.IGNORECASE),     # _(N) or (N) (space or _)
]


def _strip_suffixes(raw: str) -> str:
    """
    Strip trailing ``_(N)``, ``_SCANNED``, ``_SCANNEED``, ``_SCANNEDpdf``
    suffixes case-insensitively.

    Strips are applied iteratively until no more matches are found, so
    combined suffixes like ``_(2)_SCANNED`` are fully removed.

    Parameters
    ----------
    raw:
        Input string (document_category or filename-derived token).

    Returns
    -------
    Cleaned string with trailing suffixes removed and surrounding whitespace
    stripped.
    """
    text = raw.strip()
    changed = True
    while changed:
        changed = False
        for pattern in _SUFFIX_PATTERNS:
            cleaned = pattern.sub("", text)
            if cleaned != text:
                text = cleaned.strip()
                changed = True
    return text


# ---------------------------------------------------------------------------
# Filename token extraction
# ---------------------------------------------------------------------------

# Compiled patterns for the four extraction strategies.
# Each pattern extracts a named group ``token`` which is the category segment.

# Pattern 1: {TICKER}_Ann_{YYYYMMDD}_{Token}[_(N)][_SCANNED][.ext]
_PATTERN_ANN = re.compile(
    r"^[^_]+_Ann_\d{8}_(?P<token>.+?)(?:_\(\d+\))?(?:_SCANNEDPDF|_SCANNEED|_SCANNED)?(?:\.[^.]+)?$",
    re.IGNORECASE,
)

# Pattern 2: {TICKER}_{YYYYMMDD}_{Token}[_SCANNED][.ext]
# Date is exactly 8 digits.  Token follows the date.
_PATTERN_DATE_THEN_TOKEN = re.compile(
    r"^[^_]+_\d{8}_(?P<token>.+?)(?:_\(\d+\))?(?:_SCANNEDPDF|_SCANNEED|_SCANNED)?(?:\.[^.]+)?$",
    re.IGNORECASE,
)

# Pattern 3: {TICKER}_{Token}_{YYYYMMDD}[_SCANNED][.ext]
# Date is exactly 8 digits at the end (before optional suffix/extension).
_PATTERN_TOKEN_THEN_DATE = re.compile(
    r"^[^_]+_(?P<token>.+?)_\d{8}(?:_\(\d+\))?(?:_SCANNEDPDF|_SCANNEED|_SCANNED)?(?:\.[^.]+)?$",
    re.IGNORECASE,
)

# Pattern 4: {TICKER_OR_ABBREV}_{Token}_{DDMMYYYY}[_SCANNED][.ext]
# DDMMYYYY also has 8 digits — same structure as Pattern 3 structurally;
# this is a conceptual alias for documentation purposes.  Both are covered
# by _PATTERN_TOKEN_THEN_DATE since both produce an 8-digit date suffix.


def _extract_token_from_filename(file_name: str) -> str:
    """
    Extract the category token segment from a filename using four patterns.

    Patterns tried in order:
        1. ``{T}_Ann_{YYYYMMDD}_{Token}[_(N)][_SCANNED]``
        2. ``{T}_{YYYYMMDD}_{Token}[_SCANNED]``  (8-digit date, then token)
        3. ``{T}_{Token}_{YYYYMMDD}[_SCANNED]``  (token, then 8-digit date)
        4. ``{T_ABBREV}_{Token}_{DDMMYYYY}[_SCANNED]``  (same as pattern 3)

    Falls back to the full file stem (filename without extension) if no
    pattern matches.

    Parameters
    ----------
    file_name:
        The raw filename string from ``documents.file_name``.

    Returns
    -------
    Raw token string (before suffix stripping and alias lookup).
    """
    # Strip leading directory components if present
    name_only = file_name.split("/")[-1].split("\\")[-1]
    # Remove extension for the stem fallback, but keep it in the string for
    # regex matching (patterns already handle optional extensions)
    stem = re.sub(r"\.[^.]+$", "", name_only)

    # Pattern 1: Ann pattern
    m = _PATTERN_ANN.match(name_only)
    if m:
        return m.group("token")

    # Pattern 2: TICKER_YYYYMMDD_Token  (8-digit date before token)
    m = _PATTERN_DATE_THEN_TOKEN.match(name_only)
    if m:
        return m.group("token")

    # Pattern 3 / 4: TICKER_Token_YYYYMMDD or TICKER_Token_DDMMYYYY
    m = _PATTERN_TOKEN_THEN_DATE.match(name_only)
    if m:
        return m.group("token")

    # Fallback: use the full file stem
    return stem


# ---------------------------------------------------------------------------
# Core normalization function
# ---------------------------------------------------------------------------


def normalize(
    document_category: str | None,
    file_name: str,
    chunk_text: str | None,
) -> tuple[str, str]:
    """
    Resolve a document's category to a canonical label using three layers.

    Layer 1 — Exact alias match:
        Strip ``_(N)`` / ``_SCANNED`` / ``_SCANNEED`` / ``_SCANNEDpdf``
        suffixes from the input token.  Lowercase.  Look up in
        :data:`ALIAS_DICT`.

        Source of the token: if ``document_category`` is non-``None`` and
        non-empty, use it directly.  Otherwise, extract a token from
        ``file_name`` using :func:`_extract_token_from_filename`.

    Layer 2 — Fuzzy match:
        Compare the cleaned token against every key in :data:`ALIAS_DICT`
        using ``rapidfuzz.fuzz.partial_ratio``.  Accept the highest-scoring
        match if the score is ≥ :data:`FUZZY_THRESHOLD` (80).  Ties broken
        by: (a) longer key wins, (b) lexicographically ascending (earlier key
        wins).

    Layer 3 — Keyword match on ``chunk_text[:2000]``:
        Apply :data:`KEYWORD_RULES` in priority order (case-insensitive).
        Skipped if ``chunk_text`` is ``None``.

    Parameters
    ----------
    document_category:
        Value from ``documents.document_category`` (may be ``None``).
    file_name:
        Value from ``documents.file_name``.
    chunk_text:
        First chunk's text content for Layer 3 fallback (may be ``None``).

    Returns
    -------
    ``(canonical_label, resolution_layer)`` where:

    - ``canonical_label`` ∈ :data:`CANONICAL_LABELS` ∪
      ``{CORPORATE_ACTION_OTHER, 'Credit Rating', 'unclassified'}``
    - ``resolution_layer`` ∈ ``{'exact', 'fuzzy', 'keyword', 'unclassified'}``

    Never raises.  Always returns a valid tuple.
    """
    try:
        return _normalize_impl(document_category, file_name, chunk_text)
    except Exception as exc:  # noqa: BLE001 — intentional catch-all
        print(
            f"WARNING: Unexpected exception in category_normalizer.normalize(): {exc!r}",
            file=sys.stderr,
        )
        return "unclassified", "unclassified"


def _normalize_impl(
    document_category: str | None,
    file_name: str,
    chunk_text: str | None,
) -> tuple[str, str]:
    """Internal implementation of :func:`normalize` (may raise)."""

    # -----------------------------------------------------------------------
    # Determine the raw token to work with
    # -----------------------------------------------------------------------
    if document_category and document_category.strip():
        raw_token = document_category
    else:
        raw_token = _extract_token_from_filename(file_name)

    # -----------------------------------------------------------------------
    # Layer 1: Exact alias match
    # -----------------------------------------------------------------------
    cleaned = _strip_suffixes(raw_token)
    key = cleaned.lower()
    result = ALIAS_DICT.get(key)
    if result is not None:
        return result, "exact"

    # -----------------------------------------------------------------------
    # Layer 2: Fuzzy match
    # -----------------------------------------------------------------------
    best_score = -1
    best_key: str | None = None

    for dict_key in ALIAS_DICT:
        score = _fuzz.partial_ratio(key, dict_key)
        if score >= FUZZY_THRESHOLD:
            if score > best_score:
                best_score = score
                best_key = dict_key
            elif score == best_score and best_key is not None:
                # Tie-breaking: prefer longer key; then lexicographically
                # ascending (earlier = smaller string comes first)
                if len(dict_key) > len(best_key):
                    best_key = dict_key
                elif len(dict_key) == len(best_key) and dict_key < best_key:
                    best_key = dict_key

    if best_key is not None:
        return ALIAS_DICT[best_key], "fuzzy"

    # -----------------------------------------------------------------------
    # Layer 3: Keyword match on chunk_text
    # -----------------------------------------------------------------------
    if chunk_text is not None:
        text_sample = chunk_text[:2000].lower()
        for required_keywords, conjunction_keywords, label in KEYWORD_RULES:
            if all(kw in text_sample for kw in required_keywords):
                if conjunction_keywords is None or any(
                    kw in text_sample for kw in conjunction_keywords
                ):
                    return label, "keyword"

    # -----------------------------------------------------------------------
    # All layers failed
    # -----------------------------------------------------------------------
    return "unclassified", "unclassified"
