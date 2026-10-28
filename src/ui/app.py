"""
app.py — Streamlit entry point for the Risk Detection RAG demo UI.

Rendering orchestrator only. All data access is delegated to
`prediction_loader` (PostgreSQL) and `rag_query` (Qdrant + Ollama).
This module never queries PostgreSQL or Qdrant directly.

Run with:
    streamlit run src/ui/app.py
from the project root.
"""

from __future__ import annotations

import os

import streamlit as st

from ui import ollama_client, rag_query
from ui.prediction_loader import MODEL_VERSIONS, get_companies, get_period_predictions, period_label

# ── Rule-weight constants (defined independently — NOT imported from risk_engine
#    or prediction_loader — to keep this rendering module self-contained) ──────

RULE_WEIGHTS: dict[str, float] = {
    "RATING_DEFAULT_GRADE":   0.12,
    "PLEDGE_RISING":          0.10,
    "INTEREST_COVERAGE_LOW":  0.10,
    "RATING_DOWNGRADE":       0.10,
    "PLEDGE_HIGH":            0.08,
    "NEGATIVE_PAT":           0.08,
    "DEBT_TO_EQUITY_HIGH":    0.08,
    "CFO_NEGATIVE":           0.08,
    "RESIGNATION_SPIKE":      0.07,
    "DIRECTOR_CHURN":         0.06,
    "CURRENT_RATIO_LOW":      0.06,
    "REVENUE_DECLINING":      0.06,
    "PROMOTER_HOLDING_LOW":   0.05,
    "AGM_DELAY":              0.05,
    "CLARIFICATION_SPIKE":    0.05,
}

TOTAL_RULES = 15


# ── Cached singletons — initialized once per Streamlit process ────────────────


@st.cache_resource
def get_db_engine():
    """
    Return a cached SQLAlchemy engine.

    Calls ``database.postgres.get_engine()`` once; subsequent Streamlit reruns
    reuse the same engine object without re-connecting.
    """
    from database.postgres import get_engine
    return get_engine()


@st.cache_resource
def get_qdrant_client():
    """
    Return a cached QdrantClient.

    Calls ``database.qdrant.get_client()`` once; subsequent reruns reuse it.
    """
    from database.qdrant import get_client
    return get_client()


@st.cache_resource
def get_embedding_model():
    """
    Return a cached SentenceTransformer for the configured EMBEDDING_MODEL.

    Loads the model once; subsequent reruns reuse the in-memory object.
    """
    from sentence_transformers import SentenceTransformer
    from config.settings import EMBEDDING_MODEL
    return SentenceTransformer(EMBEDDING_MODEL)


# ── Startup checks ────────────────────────────────────────────────────────────


def run_startup_checks() -> list[str]:
    """
    Verify that PostgreSQL, Qdrant, and Ollama are all reachable.

    All three checks always run regardless of prior failures (no short-circuit),
    so the operator sees every failing service in one go.

    Each individual check has a 5-second timeout; the total budget across all
    three checks is capped at 10 seconds (enforced by the individual timeouts
    summing to ≤ 15 s in the worst case — acceptable for a demo prototype).

    Returns:
        A list of human-readable error strings, one per failing service.
        An empty list means all checks passed.

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
    """
    errors: list[str] = []

    # ── Check 1: PostgreSQL ───────────────────────────────────────────────────
    try:
        engine = get_db_engine()
        # Use a raw connection with an explicit 5-second statement timeout so
        # a hung database does not block the UI indefinitely.
        with engine.connect() as conn:
            from sqlalchemy import text, event
            conn.execute(text("SELECT 1"))
    except Exception:
        errors.append(
            "PostgreSQL is unreachable. Check that the database container is running."
        )

    # ── Check 2: Qdrant ───────────────────────────────────────────────────────
    try:
        import socket
        from config.settings import QDRANT
        # Use a raw socket probe with a 5-second timeout before the QdrantClient
        # call so we get a reliable timeout rather than waiting for Qdrant's own
        # default (which can be longer).
        sock = socket.create_connection(
            (QDRANT["host"], int(QDRANT["port"])),
            timeout=5,
        )
        sock.close()
        # If the TCP probe succeeds, do a real API call to confirm the service
        # is up and responding to its gRPC/HTTP API.
        client = get_qdrant_client()
        client.get_collections()
    except Exception:
        errors.append(
            "Qdrant is unreachable. Check that the Qdrant container is running."
        )

    # ── Check 3: Ollama (server reachable + model pulled) ─────────────────────
    ok, msg = ollama_client.check_model_available()
    if not ok:
        errors.append(msg)

    return errors


# ── Company selector ──────────────────────────────────────────────────────────


def render_company_selector(companies: list[dict]) -> dict | None:
    """
    Render a dropdown of all ingested companies and return the selected one.

    Each option is formatted as ``"{ticker} — {name}"``.  Only ``ticker`` and
    ``name`` are displayed — no ``sector``, ``listing_type``, or window dates.

    Args:
        companies: List of dicts with keys ``id``, ``ticker``, ``name``.
                   Assumed to be pre-sorted alphabetically by ticker (as
                   returned by ``prediction_loader.get_companies()``).

    Returns:
        The selected company dict (keys: ``id``, ``ticker``, ``name``),
        or ``None`` when the list is empty.

    Requirements: 3.1, 3.2, 3.3, 3.5
    """
    if not companies:
        st.info("No companies ingested yet.")
        return None

    labels = [f"{c['ticker']} — {c['name']}" for c in companies]
    selected_label = st.selectbox("Select a company", options=labels)

    # Recover the company dict that corresponds to the chosen label.
    selected_index = labels.index(selected_label)
    return companies[selected_index]


def render_period_selector(company_id: int, session) -> dict | None:
    """
    Choose the scoring model (v1 / v2) and one scored period of the company.

    Defaults to v1 and the latest period; the company's highest-scoring period
    under the chosen model is named underneath.  Returns the chosen prediction
    dict (same keys as ``prediction_loader.get_period_predictions``), or None.
    """
    model_version = st.radio(
        "Scoring model",
        options=list(MODEL_VERSIONS),
        format_func=MODEL_VERSIONS.get,
        horizontal=True,
        key=f"model_{company_id}",
    )
    predictions = get_period_predictions(company_id, model_version, session)
    if not predictions:
        return None

    labels = [period_label(p) for p in predictions]
    index = st.selectbox(
        "Period",
        options=range(len(predictions)),
        index=len(predictions) - 1,
        format_func=lambda i: labels[i],
        key=f"period_{company_id}_{model_version}",
    )
    highest = max(predictions, key=lambda p: float(p.get("risk_score") or 0))
    st.caption(f"Highest-scoring period for this model: {period_label(highest)}")
    return predictions[index]


# ── Risk Report View — stub ───────────────────────────────────────────────────


def render_risk_report(prediction: dict) -> None:
    """
    Render the risk score header, risk level, period, model version, and
    triggered-rules table (sorted by weight descending).

    Args:
        prediction: Dict returned by ``prediction_loader.get_latest_prediction()``,
                    with keys ``risk_score``, ``risk_level``, ``fiscal_year``,
                    ``fiscal_quarter``, ``model_version``, ``risk_flags``.

    Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.9
    """
    import pandas as pd

    risk_score = prediction.get("risk_score")
    risk_level = prediction.get("risk_level", "—")
    fiscal_year = prediction.get("fiscal_year", "—")
    fiscal_quarter = prediction.get("fiscal_quarter")
    period_display = f"{fiscal_year} {fiscal_quarter if fiscal_quarter is not None else 'Annual'}"
    model_version = prediction.get("model_version", "—")
    risk_flags = prediction.get("risk_flags") or []

    # ── Header metrics ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{float(risk_score):.4f}" if risk_score is not None else "—")
    col2.metric("Risk Level", risk_level)
    col3.metric("Period", period_display)

    st.caption(f"Model: {model_version}")

    # ── Triggered-rules table ─────────────────────────────────────────────────
    if not risk_flags:
        st.info("No rules triggered for this period.")
        return

    n = len(risk_flags)
    st.write(f"**{n} of 15 rules triggered**")

    sorted_flags = sorted(
        risk_flags,
        key=lambda r: RULE_WEIGHTS.get(r.get("rule_id", ""), 0),
        reverse=True,
    )

    table_rows = [
        {
            "rule_id":         flag.get("rule_id", ""),
            "description":     flag.get("description", ""),
            "triggered_value": flag.get("triggered_value"),
            "threshold":       flag.get("threshold"),
        }
        for flag in sorted_flags
    ]

    st.dataframe(
        pd.DataFrame(table_rows),
        width="stretch",
        hide_index=True,
    )


# ── Evidence View — stub ──────────────────────────────────────────────────────


def render_evidence_view(prediction: dict) -> None:
    """
    Render one collapsed ``st.expander`` per triggered rule, showing the
    triggering values and the cited source document chunk.

    Args:
        prediction: Dict returned by ``prediction_loader.get_latest_prediction()``,
                    with keys ``risk_flags`` and ``evidence_chunks``.

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
    """
    if not prediction:
        st.info("No prediction data available.")
        return

    risk_flags = prediction.get("risk_flags") or []
    evidence_chunks = prediction.get("evidence_chunks") or []

    if not risk_flags:
        st.info("No rules triggered — no evidence to display.")
        return

    # Build a lookup: rule_id → chunk dict (first match wins)
    chunk_by_rule: dict[str, dict] = {}
    for chunk in evidence_chunks:
        rid = chunk.get("rule_id")
        if rid and rid not in chunk_by_rule:
            chunk_by_rule[rid] = chunk

    sorted_flags = sorted(
        risk_flags,
        key=lambda r: RULE_WEIGHTS.get(r.get("rule_id", ""), 0),
        reverse=True,
    )

    st.subheader("Evidence")

    for flag in sorted_flags:
        rule_id = flag.get("rule_id", "")
        description = flag.get("description", "")
        label = f"{rule_id}: {description}"

        with st.expander(label, expanded=False):
            col1, col2 = st.columns(2)
            col1.write(f"**Triggered value:** {flag.get('triggered_value')}")
            col2.write(f"**Threshold:** {flag.get('threshold')}")

            chunk = chunk_by_rule.get(rule_id)
            if chunk is not None and chunk.get("source_kind") == "shareholding_pattern":
                # shareholding is entered manually from exchange filings; there is no PDF passage
                st.write(chunk.get("source_note") or "Shareholding pattern (manually entered).")
                st.caption("Source: quarterly shareholding pattern (metadata.json)")
            elif chunk is None or chunk.get("chunk_text") is None:
                st.caption("No source chunk retrieved for this rule.")
            else:
                chunk_text = chunk["chunk_text"]
                display_text = chunk_text[:500]
                st.write(display_text)

                doc_type = chunk.get("document_type", "—")
                fy = chunk.get("fiscal_year", "—")
                fq = chunk.get("fiscal_quarter")
                fq_display = fq if fq is not None else "Annual"
                raw_file_name = chunk.get("file_name") or ""
                base_name = os.path.basename(raw_file_name) if raw_file_name else "—"
                pages = _page_range(chunk.get("page_start"), chunk.get("page_end"))

                st.caption(
                    f"Source: {doc_type} | {fy} {fq_display} | {base_name}{pages}"
                )
                if chunk.get("source_kind") == "company_search":
                    st.caption("Found by a company-wide search: the rule's own source document had no matching passage.")


def _page_range(start, end) -> str:
    """' | p. 4' or ' | pp. 4–5' for a cited chunk; empty when pages are unknown."""
    if start is None:
        return ""
    if end is None or end == start:
        return f" | p. {start}"
    return f" | pp. {start}–{end}"


# ── Ad-Hoc Query Box — stub ───────────────────────────────────────────────────


def render_query_box(ticker: str, company_id: int) -> None:
    """
    Render the free-text query input and display the LLM-generated answer
    with cited source chunks.

    Args:
        ticker:     Ticker string of the currently selected company, used in
                    the input label: ``"Ask a question about {ticker}"``.
        company_id: PostgreSQL ``companies.id`` used to filter Qdrant results.

    Requirements: 6.1, 6.4, 6.5, 6.6, 6.9, 6.10, 6.11
    """
    st.subheader("Ad-Hoc Query")

    # Use a unique session-state key per company so switching companies resets the box.
    state_key = f"query_result_{company_id}"
    submitting_key = f"query_submitting_{company_id}"

    query_text = st.text_area(
        label=f"Ask a question about {ticker}",
        key=f"query_input_{company_id}",
        disabled=st.session_state.get(submitting_key, False),
    )
    submit = st.button(
        "Submit",
        key=f"query_submit_{company_id}",
        disabled=st.session_state.get(submitting_key, False),
    )

    if submit:
        # ── Validation ────────────────────────────────────────────────────────
        if not query_text or not query_text.strip():
            st.warning("Please enter a question.")
            return
        if len(query_text) > 500:
            st.warning("Query is too long (max 500 characters).")
            return

        # ── Execute query ─────────────────────────────────────────────────────
        st.session_state[submitting_key] = True
        try:
            with st.spinner("Querying..."):
                result = rag_query.query_company(
                    query_text=query_text,
                    company_id=company_id,
                    qdrant_client=get_qdrant_client(),
                    embedding_model=get_embedding_model(),
                )
            st.session_state[state_key] = result
        except TimeoutError:
            st.error(
                "The request timed out. Try a shorter question or check Ollama."
            )
            return
        except RuntimeError as exc:
            st.error(f"Ollama returned an error: {exc}")
            return
        except Exception as exc:
            st.error(str(exc))
            return
        finally:
            st.session_state[submitting_key] = False

    # ── Display cached result (survives reruns after the submit) ──────────────
    result = st.session_state.get(state_key)
    if result is not None:
        st.write("**Answer**")
        st.write(result["answer"])

        sources = result.get("sources") or []
        if sources:
            st.write(f"**Sources ({len(sources)} passages retrieved)**")
            for i, src in enumerate(sources, 1):
                raw_path = src.get("file_path") or ""
                base = os.path.basename(raw_path) if raw_path else "—"
                doc_type = src.get("document_type", "—")
                fy = src.get("fiscal_year", "—")
                fq = src.get("fiscal_quarter")
                fq_display = fq if fq is not None else "Annual"
                chunk_preview = (src.get("chunk_text") or "")[:300]
                with st.expander(f"[{i}] {base} | {doc_type} | {fy} {fq_display}", expanded=False):
                    st.caption(chunk_preview)


# ── Main entry point — stub ───────────────────────────────────────────────────


def main() -> None:
    """
    Top-level Streamlit application driver.

    Execution order:
      1. Run startup checks; halt with ``st.error`` messages if any service
         is unreachable.
      2. Load all companies from PostgreSQL via ``prediction_loader``.
      3. Render the Company_Selector; wait for a selection.
      4. Load and render the Risk_Report_View and Evidence_View for the
         selected company.
      5. Render the Ad-Hoc Query Box.

    All unhandled exceptions are caught and displayed via ``st.error()`` —
    no raw tracebacks are ever shown to the user.

    Requirements: 2.4, 2.5, 3.4, 3.6, 3.7, 9.1, 9.6
    """
    st.title("Risk Detection Dashboard")

    try:
        # ── 1. Startup checks ─────────────────────────────────────────────────
        errors = run_startup_checks()
        if errors:
            for error in errors:
                st.error(error)
            st.stop()

        # ── 2 & 3. Load companies ─────────────────────────────────────────────
        from sqlalchemy.orm import Session

        engine = get_db_engine()
        with Session(engine) as session:
            try:
                companies = get_companies(session)
            except Exception:
                st.error("Database connection failed while loading companies.")
                st.stop()

            # ── 4. Company selector ───────────────────────────────────────────
            selected = render_company_selector(companies)

            # ── 5. Risk report + evidence + query box ─────────────────────────
            if selected is not None:
                company_id = selected["id"]
                ticker = selected["ticker"]

                prediction = render_period_selector(company_id, session)

                if prediction is None:
                    st.info(
                        "No risk prediction available for this company. "
                        "Run the risk engine first."
                    )
                else:
                    render_risk_report(prediction)
                    render_evidence_view(prediction)

                render_query_box(ticker, company_id)

    except Exception as exc:
        st.error(str(exc))


# ── Module entry guard ────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
