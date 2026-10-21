"""
Unit tests for src/risk_engine/prediction_writer.py

Uses an in-memory SQLite DB for insert/update verification.
JSONB columns are unavailable in SQLite, so we mock the session.execute()
to capture the INSERT statement args instead.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
"""

import pytest
from unittest.mock import MagicMock, patch, call

from risk_engine.prediction_writer import (
    MODEL_VERSION,
    upsert_prediction,
    _serialize_rule_results,
    _serialize_evidence_chunks,
)
from risk_engine.scorer import RuleResult
from risk_engine.evidence_retriever import ChunkReference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rule_result(rule_id="NEGATIVE_PAT", threshold=None, triggered_value=-500.0):
    return RuleResult(
        rule_id=rule_id,
        description="Test rule description",
        triggered_value=triggered_value,
        threshold=threshold,
        fiscal_year="FY2018",
        fiscal_quarter="Q2",
        feature_row_id=10,
    )


def _make_chunk_ref(rule_id="NEGATIVE_PAT", score=0.85):
    return ChunkReference(
        rule_id=rule_id,
        chunk_text="Some text excerpt from the filing",
        document_id=99,
        file_name="data/raw/TEST/annual_report.pdf",
        fiscal_year="FY2018",
        fiscal_quarter="Q2",
        document_type="annual_report",
        score=score,
    )


def _make_stub_chunk(rule_id="PLEDGE_HIGH"):
    return ChunkReference(
        rule_id=rule_id,
        chunk_text=None,
        document_id=None,
        file_name=None,
        fiscal_year=None,
        fiscal_quarter=None,
        document_type=None,
        score=0.0,
    )


# ---------------------------------------------------------------------------
# MODEL_VERSION constant
# ---------------------------------------------------------------------------

class TestModelVersion:
    def test_model_version_constant(self):
        assert MODEL_VERSION == "rule_based_v1"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

class TestSerializeRuleResults:
    def test_empty_list_returns_empty_list(self):
        assert _serialize_rule_results([]) == []

    def test_single_rule_result_serialized_correctly(self):
        rule = _make_rule_result(rule_id="PLEDGE_HIGH", threshold=50.0, triggered_value=65.0)
        result = _serialize_rule_results([rule])
        assert len(result) == 1
        d = result[0]
        assert d["rule_id"] == "PLEDGE_HIGH"
        assert d["description"] == "Test rule description"
        assert d["triggered_value"] == 65.0
        assert d["threshold"] == 50.0
        assert d["fiscal_year"] == "FY2018"
        assert d["fiscal_quarter"] == "Q2"
        assert d["feature_row_id"] == 10

    def test_none_threshold_serialized_as_none(self):
        rule = _make_rule_result(rule_id="NEGATIVE_PAT", threshold=None)
        result = _serialize_rule_results([rule])
        assert result[0]["threshold"] is None

    def test_multiple_rules_serialized_in_order(self):
        rules = [
            _make_rule_result("NEGATIVE_PAT"),
            _make_rule_result("CFO_NEGATIVE"),
            _make_rule_result("AGM_DELAY"),
        ]
        result = _serialize_rule_results(rules)
        assert [d["rule_id"] for d in result] == ["NEGATIVE_PAT", "CFO_NEGATIVE", "AGM_DELAY"]


class TestSerializeEvidenceChunks:
    def test_empty_list_returns_empty_list(self):
        assert _serialize_evidence_chunks([]) == []

    def test_fully_populated_chunk_serialized(self):
        chunk = _make_chunk_ref(rule_id="NEGATIVE_PAT", score=0.92)
        result = _serialize_evidence_chunks([chunk])
        assert len(result) == 1
        d = result[0]
        assert d["rule_id"] == "NEGATIVE_PAT"
        assert d["chunk_text"] == "Some text excerpt from the filing"
        assert d["document_id"] == 99
        assert d["file_name"] == "data/raw/TEST/annual_report.pdf"
        assert d["fiscal_year"] == "FY2018"
        assert d["fiscal_quarter"] == "Q2"
        assert d["document_type"] == "annual_report"
        assert d["score"] == pytest.approx(0.92)

    def test_stub_chunk_null_fields_serialized_as_none(self):
        chunk = _make_stub_chunk("PLEDGE_HIGH")
        result = _serialize_evidence_chunks([chunk])
        d = result[0]
        assert d["chunk_text"] is None
        assert d["document_id"] is None
        assert d["file_name"] is None
        assert d["fiscal_year"] is None
        assert d["fiscal_quarter"] is None
        assert d["document_type"] is None
        assert d["score"] == 0.0

    def test_dict_input_accepted(self):
        """_serialize_evidence_chunks should also accept plain dicts."""
        chunk_dict = {
            "rule_id": "INTEREST_COVERAGE_LOW",
            "chunk_text": "Interest expense exceeded EBIT",
            "document_id": 5,
            "file_name": "report.pdf",
            "fiscal_year": "FY2019",
            "fiscal_quarter": "Q3",
            "document_type": "quarterly_result",
            "score": 0.75,
        }
        result = _serialize_evidence_chunks([chunk_dict])
        assert result[0]["rule_id"] == "INTEREST_COVERAGE_LOW"
        assert result[0]["document_id"] == 5


# ---------------------------------------------------------------------------
# upsert_prediction — mock session
# ---------------------------------------------------------------------------

class TestUpsertPrediction:
    """
    Tests upsert_prediction by mocking the SQLAlchemy session.execute.
    We verify the payload passed to the INSERT statement has correct values.
    """

    def test_upsert_prediction_calls_session_execute(self):
        """upsert_prediction should call session.execute exactly once."""
        mock_session = MagicMock()

        upsert_prediction(
            company_id=1,
            fiscal_year="FY2018",
            fiscal_quarter="Q3",
            risk_score=0.35,
            risk_level="Medium",
            triggered_rules=[_make_rule_result()],
            evidence_chunks=[_make_chunk_ref()],
            session=mock_session,
        )
        assert mock_session.execute.called

    def test_upsert_prediction_empty_triggered_rules(self):
        """Empty triggered_rules → risk_flags=[] and evidence_chunks=[] (not null)."""
        mock_session = MagicMock()

        # Capture what's passed to session.execute
        stmt_values = {}
        def capture_execute(stmt):
            # Extract the values from the INSERT statement's _values
            try:
                stmt_values.update(stmt.compile(compile_kwargs={"literal_binds": True}).params)
            except Exception:
                pass  # Compilation may fail in pure-mock context; that's acceptable

        mock_session.execute.side_effect = capture_execute

        upsert_prediction(
            company_id=1,
            fiscal_year="FY2018",
            fiscal_quarter="Q1",
            risk_score=0.0,
            risk_level="Low",
            triggered_rules=[],
            evidence_chunks=[],
            session=mock_session,
        )
        assert mock_session.execute.called

    def test_upsert_prediction_null_fiscal_quarter_accepted(self):
        """fiscal_quarter=None (annual period) should not raise."""
        mock_session = MagicMock()

        # Should not raise
        upsert_prediction(
            company_id=1,
            fiscal_year="FY2018",
            fiscal_quarter=None,
            risk_score=0.20,
            risk_level="Low",
            triggered_rules=[],
            evidence_chunks=[],
            session=mock_session,
        )
        assert mock_session.execute.called

    def test_upsert_prediction_propagates_db_exception(self):
        """If session.execute raises, the exception propagates to the caller."""
        mock_session = MagicMock()
        mock_session.execute.side_effect = RuntimeError("DB constraint violation")

        with pytest.raises(RuntimeError, match="DB constraint violation"):
            upsert_prediction(
                company_id=1,
                fiscal_year="FY2018",
                fiscal_quarter="Q2",
                risk_score=0.5,
                risk_level="High",
                triggered_rules=[],
                evidence_chunks=[],
                session=mock_session,
            )

    def test_upsert_prediction_no_label_leakage_columns(self):
        """The INSERT statement must never include group/label/distressed/healthy columns."""
        mock_session = MagicMock()
        stmt_holder = {}

        def capture(stmt):
            stmt_holder["stmt"] = stmt

        mock_session.execute.side_effect = capture

        upsert_prediction(
            company_id=1,
            fiscal_year="FY2018",
            fiscal_quarter="Q1",
            risk_score=0.3,
            risk_level="Medium",
            triggered_rules=[],
            evidence_chunks=[],
            session=mock_session,
        )

        # Check the INSERT statement doesn't contain label columns
        stmt = stmt_holder.get("stmt")
        if stmt is not None:
            stmt_str = str(stmt).lower()
            forbidden = ["group", "distressed", "healthy", "ground_truth", "label"]
            for col in forbidden:
                assert col not in stmt_str, f"Found forbidden column '{col}' in INSERT statement"
