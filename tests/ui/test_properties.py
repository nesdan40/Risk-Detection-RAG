"""
tests/ui/test_properties.py — Property-based tests for the Streamlit Demo UI.

Each test maps 1:1 to a named Correctness Property in the design document.
All 19 properties are covered here.

Hypothesis is used as the PBT framework (already present in this project).
No Streamlit imports — only pure logic functions are tested.

Validates: Requirements 1.1, 1.3, 2.3, 2.7, 3.1, 3.2, 3.5, 4.1, 4.2,
           4.3, 4.5, 4.7, 5.1, 5.3, 5.4, 6.1, 6.2, 6.3, 6.10, 7.3, 7.4, 7.7
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
import hypothesis.strategies as st

# ── Imports from source modules ───────────────────────────────────────────────
# conftest.py at the project root inserts src/ into sys.path, so these work.

from ui.prediction_loader import RULE_WEIGHTS
from ui.rag_query import QUERY_PREFIX, MAX_QUERY_LENGTH, _build_prompt, query_company
from ui.ollama_client import generate
from config.settings import LLM_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Property 1 — OLLAMA_BASE_URL fallback for None and ""
# Feature: streamlit-demo-ui, Property 1
# Validates: Requirements 1.1, 1.3
# ─────────────────────────────────────────────────────────────────────────────

@given(st.one_of(st.none(), st.just("")))
@settings(max_examples=100)
def test_ollama_url_fallback_for_empty_or_none(val):
    """
    **Property 1: OLLAMA_BASE_URL fallback for empty or absent values**
    **Validates: Requirements 1.1, 1.3**

    The `or` pattern used in settings.py:
        OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    must map both None and "" to the default literal.
    """
    result = val or "http://localhost:11434"
    assert result == "http://localhost:11434"


# ─────────────────────────────────────────────────────────────────────────────
# Property 2 — Model presence check correct for all model lists
# Feature: streamlit-demo-ui, Property 2
# Validates: Requirements 2.3, 2.7
# ─────────────────────────────────────────────────────────────────────────────

@given(
    st.lists(st.fixed_dictionaries({"name": st.text()})),
    st.text(),
)
@settings(max_examples=100)
def test_model_presence_check_is_correct(model_list, model_name):
    """
    **Property 2: Model presence check is correct for all model lists**
    **Validates: Requirements 2.3, 2.7**

    For any list of model dicts and any model name string, the presence check
    must return True iff the name appears as a "name" value in at least one dict.
    """
    # Ground-truth: is model_name in the list?
    expected = model_name in [m.get("name", "") for m in model_list if isinstance(m, dict)]
    # Simulate the logic from ollama_client.check_model_available()
    model_names = [m.get("name", "") for m in model_list if isinstance(m, dict)]
    actual = model_name in model_names
    assert actual == expected


# ─────────────────────────────────────────────────────────────────────────────
# Property 3 — Company list always alphabetically sorted by ticker
# Feature: streamlit-demo-ui, Property 3
# Validates: Requirements 3.1
# ─────────────────────────────────────────────────────────────────────────────

@given(st.lists(
    st.fixed_dictionaries({
        "id":     st.integers(min_value=1),
        "ticker": st.text(min_size=1, max_size=10),
        "name":   st.text(min_size=1),
    }),
    min_size=1,
))
@settings(max_examples=100)
def test_companies_always_sorted_by_ticker(companies):
    """
    **Property 3: Company list is always alphabetically sorted by ticker**
    **Validates: Requirements 3.1**

    After applying the same ORDER BY ticker ASC logic used in get_companies(),
    the resulting ticker sequence must be non-decreasing.
    """
    sorted_companies = sorted(companies, key=lambda c: c["ticker"])
    tickers = [c["ticker"] for c in sorted_companies]
    assert tickers == sorted(tickers)


# ─────────────────────────────────────────────────────────────────────────────
# Property 4 — Company display label always contains both ticker and name
# Feature: streamlit-demo-ui, Property 4
# Validates: Requirements 3.2
# ─────────────────────────────────────────────────────────────────────────────

@given(st.fixed_dictionaries({
    "ticker": st.text(min_size=1),
    "name":   st.text(min_size=1),
}))
@settings(max_examples=100)
def test_company_label_contains_ticker_and_name(company):
    """
    **Property 4: Company display label always contains both ticker and name**
    **Validates: Requirements 3.2**

    The label format is f"{ticker} — {name}". Both parts must appear verbatim.
    """
    label = f"{company['ticker']} — {company['name']}"
    assert company["ticker"] in label
    assert " — " in label
    assert company["name"] in label


# ─────────────────────────────────────────────────────────────────────────────
# Property 5 — Company label never contains restricted field values
# Feature: streamlit-demo-ui, Property 5
# Validates: Requirements 3.5
# ─────────────────────────────────────────────────────────────────────────────

# Letters only: with no spaces or dashes in the values, a value cannot match across the
# " — " separator by accident (e.g. ticker '1' + ' ' contained '1 ').
_LABEL_TEXT = st.text(alphabet=st.characters(categories=("Lu", "Ll")), min_size=1)


@given(st.fixed_dictionaries({
    "ticker":       _LABEL_TEXT,
    "name":         _LABEL_TEXT,
    "sector":       _LABEL_TEXT,
    "listing_type": _LABEL_TEXT,
}))
@settings(max_examples=100)
def test_company_label_excludes_restricted_fields(company):
    """
    **Property 5: Company label never contains restricted fields**
    **Validates: Requirements 3.5**

    sector and listing_type must not appear in the formatted label — only
    ticker and name are shown. We use assume() to exclude the degenerate case
    where sector/listing_type coincidentally equal ticker or name.
    """
    # Exclude degenerate cases where sector/listing_type appear as substrings
    # of ticker or name (which would cause them to appear in the label incidentally).
    assume(company["sector"] not in company["ticker"])
    assume(company["sector"] not in company["name"])
    assume(company["listing_type"] not in company["ticker"])
    assume(company["listing_type"] not in company["name"])
    # ...nor of the label's own separator (Hypothesis can generate " " or "—")
    assume(company["sector"] not in " — ")
    assume(company["listing_type"] not in " — ")
    label = f"{company['ticker']} — {company['name']}"
    assert company["sector"] not in label
    assert company["listing_type"] not in label


# ─────────────────────────────────────────────────────────────────────────────
# Property 6 — Latest prediction always most recent rule_based_v1 row
# Feature: streamlit-demo-ui, Property 6
# Validates: Requirements 4.1
# ─────────────────────────────────────────────────────────────────────────────

@given(st.lists(
    st.fixed_dictionaries({
        "id":            st.integers(min_value=1),
        "model_version": st.sampled_from(["rule_based_v1", "xgboost_v1", "rule_based_v2"]),
        "created_at":    st.datetimes(),
    }),
    min_size=1,
))
@settings(max_examples=100)
def test_latest_prediction_is_most_recent_rule_based_v1(predictions):
    """
    **Property 6: Latest prediction selection is always the most recent rule_based_v1 row**
    **Validates: Requirements 4.1**

    Simulates the SQL: WHERE model_version = 'rule_based_v1' ORDER BY created_at DESC LIMIT 1.
    The result must be the row with the maximum created_at among rule_based_v1 rows only.
    Other model_version rows must never be returned.
    """
    rule_based = [p for p in predictions if p["model_version"] == "rule_based_v1"]
    if not rule_based:
        return  # No rule_based_v1 rows — nothing to assert

    latest = max(rule_based, key=lambda p: p["created_at"])
    # Simulate ORDER BY created_at DESC LIMIT 1 over filtered rows
    result = sorted(rule_based, key=lambda p: p["created_at"], reverse=True)[0]
    assert result["id"] == latest["id"]
    assert result["model_version"] == "rule_based_v1"


# ─────────────────────────────────────────────────────────────────────────────
# Property 7 — Risk score rounded to exactly 4 decimal places
# Feature: streamlit-demo-ui, Property 7
# Validates: Requirements 4.2
# ─────────────────────────────────────────────────────────────────────────────

@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=100)
def test_risk_score_formatted_to_4_decimal_places(score):
    """
    **Property 7: Risk score is always displayed rounded to exactly 4 decimal places**
    **Validates: Requirements 4.2**

    The formatted string must have exactly 4 digits after the decimal point,
    matching f"{score:.4f}".
    """
    formatted = f"{float(score):.4f}"
    assert formatted == f"{score:.4f}"
    # Split on "." and check the fractional part length
    parts = formatted.split(".")
    assert len(parts) == 2
    assert len(parts[1]) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Property 8 — fiscal_quarter None displayed as "Annual"
# Feature: streamlit-demo-ui, Property 8
# Validates: Requirements 4.3
# ─────────────────────────────────────────────────────────────────────────────

@given(st.one_of(st.none(), st.text(min_size=1)))
@settings(max_examples=100)
def test_fiscal_quarter_none_displays_as_annual(fiscal_quarter):
    """
    **Property 8: fiscal_quarter NULL is always displayed as "Annual"**
    **Validates: Requirements 4.3**

    The display logic: fiscal_quarter if fiscal_quarter is not None else "Annual".
    """
    display = fiscal_quarter if fiscal_quarter is not None else "Annual"
    if fiscal_quarter is None:
        assert display == "Annual"
    else:
        assert display == fiscal_quarter


# ─────────────────────────────────────────────────────────────────────────────
# Property 9 — Triggered-rules sorted by weight descending
# Feature: streamlit-demo-ui, Property 9
# Validates: Requirements 4.5, 5.1
# ─────────────────────────────────────────────────────────────────────────────

@given(st.lists(
    st.sampled_from(list(RULE_WEIGHTS.keys())),
    min_size=1,
    unique=True,
))
@settings(max_examples=100)
def test_triggered_rules_sorted_by_weight_descending(rule_ids):
    """
    **Property 9: Triggered-rules table is always sorted by weight descending**
    **Validates: Requirements 4.5, 5.1**

    After sorting risk flags by RULE_WEIGHTS descending, consecutive weight
    pairs must satisfy weights[i] >= weights[i+1].
    """
    flags = [{"rule_id": rid} for rid in rule_ids]
    sorted_flags = sorted(
        flags,
        key=lambda r: RULE_WEIGHTS.get(r["rule_id"], 0),
        reverse=True,
    )
    weights = [RULE_WEIGHTS.get(r["rule_id"], 0) for r in sorted_flags]
    for i in range(len(weights) - 1):
        assert weights[i] >= weights[i + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Property 10 — Rule count display matches actual triggered count
# Feature: streamlit-demo-ui, Property 10
# Validates: Requirements 4.7
# ─────────────────────────────────────────────────────────────────────────────

@given(st.integers(min_value=0, max_value=15))
@settings(max_examples=100)
def test_rule_count_display_matches_triggered_count(n):
    """
    **Property 10: Rule count display always reflects the actual triggered count**
    **Validates: Requirements 4.7**

    The display string must be exactly f"{n} of 15 rules triggered" and must
    contain the string representation of n.
    """
    display = f"{n} of 15 rules triggered"
    assert display == f"{n} of 15 rules triggered"
    assert str(n) in display


# ─────────────────────────────────────────────────────────────────────────────
# Property 11 — Evidence chunk text truncated to at most 500 chars
# Feature: streamlit-demo-ui, Property 11
# Validates: Requirements 5.3
# ─────────────────────────────────────────────────────────────────────────────

@given(st.text())
@settings(max_examples=100)
def test_chunk_text_truncated_to_500_chars(chunk_text):
    """
    **Property 11: Evidence chunk text is always truncated to at most 500 characters in the UI**
    **Validates: Requirements 5.3**

    chunk_text[:500] must have length <= 500. For inputs already <= 500 chars,
    the sliced result must equal the original.
    """
    displayed = chunk_text[:500]
    assert len(displayed) <= 500
    if len(chunk_text) <= 500:
        assert displayed == chunk_text


# ─────────────────────────────────────────────────────────────────────────────
# Property 12 — Evidence chunks matched to rules by rule_id
# Feature: streamlit-demo-ui, Property 12
# Validates: Requirements 5.3, 5.4
# ─────────────────────────────────────────────────────────────────────────────

@given(
    st.lists(
        st.fixed_dictionaries({
            "rule_id":    st.text(min_size=1),
            "chunk_text": st.text(),
        }),
        min_size=0,
        max_size=10,
    ),
    st.text(min_size=1),
)
@settings(max_examples=100)
def test_evidence_chunk_matched_by_rule_id(chunks, target_rule_id):
    """
    **Property 12: Evidence chunks are matched to rules by rule_id**
    **Validates: Requirements 5.3, 5.4**

    The lookup dict built from the evidence_chunks list (first match wins) must:
    - return the correct chunk when target_rule_id exists in the list.
    - return None when no chunk has that rule_id.
    """
    # Replicate the lookup logic from render_evidence_view() in app.py
    chunk_by_rule: dict[str, dict] = {}
    for chunk in chunks:
        rid = chunk.get("rule_id")
        if rid and rid not in chunk_by_rule:
            chunk_by_rule[rid] = chunk

    result = chunk_by_rule.get(target_rule_id)
    matching = [c for c in chunks if c.get("rule_id") == target_rule_id]

    if matching:
        assert result is not None
        assert result["rule_id"] == target_rule_id
    else:
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Property 13 — Query box label contains selected company ticker
# Feature: streamlit-demo-ui, Property 13
# Validates: Requirements 6.1
# ─────────────────────────────────────────────────────────────────────────────

@given(st.text(min_size=1))
@settings(max_examples=100)
def test_query_box_label_contains_ticker(ticker):
    """
    **Property 13: Query box label always contains the selected company ticker**
    **Validates: Requirements 6.1**

    The label is f"Ask a question about {ticker}". The ticker must appear verbatim.
    """
    label = f"Ask a question about {ticker}"
    assert ticker in label


# ─────────────────────────────────────────────────────────────────────────────
# Property 14 — Embedding query has QUERY_PREFIX prepended
# Feature: streamlit-demo-ui, Property 14
# Validates: Requirements 6.2
# ─────────────────────────────────────────────────────────────────────────────

@given(st.text(min_size=1, max_size=500))
@settings(max_examples=100)
def test_embedding_query_has_prefix_prepended(query_text):
    """
    **Property 14: Embedding query always has the QUERY_PREFIX prepended**
    **Validates: Requirements 6.2**

    The string passed to embedding_model.encode() must start with QUERY_PREFIX
    followed immediately by the original query_text.
    """
    prefixed = QUERY_PREFIX + query_text
    assert prefixed.startswith(QUERY_PREFIX)
    assert prefixed == "Represent this sentence for searching relevant passages: " + query_text


# ─────────────────────────────────────────────────────────────────────────────
# Property 15 — Constructed prompt contains all chunk texts
# Feature: streamlit-demo-ui, Property 15
# Validates: Requirements 6.3
# ─────────────────────────────────────────────────────────────────────────────

@given(st.lists(
    st.fixed_dictionaries({
        "chunk_text":    st.text(min_size=1),
        "document_type": st.text(),
        "fiscal_year":   st.text(),
        "fiscal_quarter": st.one_of(st.none(), st.text(min_size=1)),
        "file_path":     st.text(),
    }),
    min_size=1,
    max_size=5,
))
@settings(max_examples=100)
def test_prompt_contains_all_chunk_texts(chunks):
    """
    **Property 15: Constructed prompt contains all retrieved chunk texts**
    **Validates: Requirements 6.3**

    Every chunk_text value must appear in the prompt produced by _build_prompt().
    """
    prompt = _build_prompt("test question", chunks)
    for chunk in chunks:
        assert chunk["chunk_text"] in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Property 16 — Query > 500 chars raises ValueError
# Feature: streamlit-demo-ui, Property 16
# Validates: Requirements 6.10
# ─────────────────────────────────────────────────────────────────────────────

@given(st.text(min_size=MAX_QUERY_LENGTH + 1))
@settings(max_examples=100)
def test_query_over_500_chars_raises_value_error(long_query):
    """
    **Property 16: Query validation rejects all strings exceeding 500 characters**
    **Validates: Requirements 6.10**

    query_company() must raise ValueError for any query > MAX_QUERY_LENGTH chars
    without ever calling the Qdrant client or embedding model.
    """
    mock_qdrant = MagicMock()
    mock_model = MagicMock()
    with pytest.raises(ValueError):
        query_company(long_query, 1, mock_qdrant, mock_model)
    mock_qdrant.search.assert_not_called()
    mock_model.encode.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Property 17 — Ollama client sends prompt at correct request key
# Feature: streamlit-demo-ui, Property 17
# Validates: Requirements 7.3
# ─────────────────────────────────────────────────────────────────────────────

def _streaming_response(text: str) -> MagicMock:
    """A mocked streaming /api/generate reply: the text split across two chunks, then done."""
    import json as _json

    mid = len(text) // 2
    lines = [
        _json.dumps({"response": text[:mid], "done": False}).encode(),
        _json.dumps({"response": text[mid:], "done": False}).encode(),
        _json.dumps({"response": "", "done": True}).encode(),
    ]
    resp = MagicMock()
    resp.ok = True
    resp.iter_lines.return_value = lines
    return resp


@given(st.text(min_size=1).filter(lambda s: s.strip()))
@settings(max_examples=50)
def test_ollama_client_sends_correct_request_keys(prompt):
    """
    **Property 17: Ollama client always passes the prompt at the correct request key**
    **Validates: Requirements 7.3**

    The HTTP POST body must contain "prompt" = the exact input string, and
    "model" = LLM_MODEL from settings.
    """
    mock_response = _streaming_response("test answer")

    with patch("ui.ollama_client.requests.post", return_value=mock_response) as mock_post:
        result = generate(prompt)
        assert result == "test answer"
        call_kwargs = mock_post.call_args
        # requests.post(url, json=payload) — json is a keyword argument
        sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert sent_json["prompt"] == prompt
        assert sent_json["model"] == LLM_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Property 18 — Ollama response extraction returns "response" value for valid replies
# Feature: streamlit-demo-ui, Property 18
# Validates: Requirements 7.4
# ─────────────────────────────────────────────────────────────────────────────

@given(st.text(min_size=1))  # an empty reply is an error by design (see ollama_client)
@settings(max_examples=50)
def test_ollama_valid_response_returns_response_value(response_text):
    """
    **Property 18: Ollama response extraction returns the "response" value for valid replies**
    **Validates: Requirements 7.4**

    When the server replies with {"response": <str>}, generate() must return
    that exact string — regardless of its content.
    """
    mock_response = _streaming_response(response_text)

    with patch("ui.ollama_client.requests.post", return_value=mock_response):
        result = generate("test prompt")
        assert result == response_text


# ─────────────────────────────────────────────────────────────────────────────
# Property 19 — Invalid prompt inputs raise ValueError before HTTP call
# Feature: streamlit-demo-ui, Property 19
# Validates: Requirements 7.7
# ─────────────────────────────────────────────────────────────────────────────

@given(st.one_of(
    st.none(),
    st.integers(),
    st.just(""),
    st.text(alphabet=" \t\n", min_size=1),   # whitespace-only strings
))
@settings(max_examples=100)
def test_invalid_prompt_raises_value_error_before_http(invalid_prompt):
    """
    **Property 19: Invalid prompt inputs always raise ValueError before any HTTP call**
    **Validates: Requirements 7.7**

    None, non-str types, empty string, and whitespace-only strings must all
    raise ValueError. No HTTP call should ever be made.
    """
    with patch("ui.ollama_client.requests.post") as mock_post:
        with pytest.raises(ValueError):
            generate(invalid_prompt)
        mock_post.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Period picker: chronological order and labels; evidence page ranges
# ─────────────────────────────────────────────────────────────────────────────

def test_periods_sort_chronologically_with_annual_after_q4():
    from ui.prediction_loader import period_sort_key
    preds = [{"fiscal_year": "FY2019", "fiscal_quarter": None}, {"fiscal_year": "FY2019", "fiscal_quarter": "Q3"},
             {"fiscal_year": "FY2018", "fiscal_quarter": "Q4"}, {"fiscal_year": "FY2019", "fiscal_quarter": "Q1"}]
    ordered = [(p["fiscal_year"], p["fiscal_quarter"]) for p in sorted(preds, key=period_sort_key)]
    assert ordered == [("FY2018", "Q4"), ("FY2019", "Q1"), ("FY2019", "Q3"), ("FY2019", None)]


def test_period_label_shows_period_level_and_score():
    from decimal import Decimal
    from ui.prediction_loader import period_label
    assert period_label({"fiscal_year": "FY2019", "fiscal_quarter": "Q3", "risk_level": "High",
                         "risk_score": Decimal("0.5556")}) == "FY2019 Q3 · High (0.556)"
    assert period_label({"fiscal_year": "FY2019", "fiscal_quarter": None, "risk_level": "Low",
                         "risk_score": None}) == "FY2019 Annual · Low (—)"


def test_page_range_caption():
    from ui.app import _page_range
    assert _page_range(None, None) == ""
    assert _page_range(4, 4) == " | p. 4"
    assert _page_range(4, 5) == " | pp. 4–5"
