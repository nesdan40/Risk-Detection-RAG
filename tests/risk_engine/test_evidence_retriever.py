"""
Unit tests for src/risk_engine/evidence_retriever.py

Validates: Requirements 5.1, 5.4, 5.5, 5.6, 5.7
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from risk_engine.evidence_retriever import (
    QUERY_PREFIX,
    RULE_QUERY_TEMPLATES,
    ChunkReference,
    retrieve_evidence,
    _make_stub,
    _retrieve_single_evidence,
)
from risk_engine.scorer import RuleResult
from risk_engine.rules import RULES


# ---------------------------------------------------------------------------
# Helpers — build a minimal RuleResult for test use
# ---------------------------------------------------------------------------

def _make_rule_result(rule_id="NEGATIVE_PAT", description="Net loss period") -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        description=description,
        triggered_value=-1000.0,
        threshold=None,
        fiscal_year="FY2018",
        fiscal_quarter="Q2",
        feature_row_id=1,
    )


def _make_mock_qdrant_point(*, chunk_text="Some relevant text", document_id=42,
                             file_path="data/raw/DHFL/annual_reports/DHFL_AR_FY2018.pdf",
                             fiscal_year="FY2018", fiscal_quarter="Q2",
                             document_type="annual_report", score=0.87):
    """Return a mock Qdrant ScoredPoint-like object."""
    point = MagicMock()
    point.score = score
    point.payload = {
        "chunk_text": chunk_text,
        "document_id": document_id,
        "file_path": file_path,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "document_type": document_type,
        "company_id": 1,
    }
    return point


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_query_prefix_constant(self):
        """QUERY_PREFIX is the official BGE v1.5 retrieval instruction."""
        assert QUERY_PREFIX == "Represent this sentence for searching relevant passages: "

    def test_rule_query_templates_completeness(self):
        """RULE_QUERY_TEMPLATES must have entries for all 15 rule_ids."""
        expected_ids = {r.rule_id for r in RULES}
        actual_ids = set(RULE_QUERY_TEMPLATES.keys())
        assert actual_ids == expected_ids, (
            f"Missing templates: {expected_ids - actual_ids}; "
            f"Extra templates: {actual_ids - expected_ids}"
        )

    def test_rule_query_templates_non_empty_values(self):
        """Every template value must be a non-empty string."""
        for rule_id, query in RULE_QUERY_TEMPLATES.items():
            assert isinstance(query, str) and query.strip(), (
                f"Rule {rule_id} has blank query template"
            )


# ---------------------------------------------------------------------------
# retrieve_evidence — top-level function
# ---------------------------------------------------------------------------

class TestRetrieveEvidenceTopLevel:
    def test_retrieve_evidence_zero_rules(self):
        """Empty triggered_rules list → returns [] immediately."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        result = retrieve_evidence(
            triggered_rules=[],
            company_id=1,
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )
        assert result == []
        mock_client.query_points.assert_not_called()

    def test_retrieve_evidence_qdrant_returns_result(self):
        """When Qdrant returns a hit, ChunkReference is fully populated."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)

        qdrant_point = _make_mock_qdrant_point(
            chunk_text="Net loss reported this quarter",
            document_id=99,
            file_path="data/raw/DHFL/ar/report.pdf",
            fiscal_year="FY2018",
            fiscal_quarter="Q2",
            document_type="annual_report",
            score=0.92,
        )
        mock_client.query_points.return_value.points = [qdrant_point]

        rules = [_make_rule_result("NEGATIVE_PAT")]
        result = retrieve_evidence(
            triggered_rules=rules,
            company_id=1,
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )

        assert len(result) == 1
        ref = result[0]
        assert isinstance(ref, ChunkReference)
        assert ref.rule_id == "NEGATIVE_PAT"
        assert ref.chunk_text is not None
        assert ref.document_id == 99
        assert ref.file_name == "data/raw/DHFL/ar/report.pdf"
        assert ref.fiscal_year == "FY2018"
        assert ref.fiscal_quarter == "Q2"
        assert ref.document_type == "annual_report"
        assert ref.score == pytest.approx(0.92)

    def test_retrieve_evidence_chunk_text_truncated_to_500_chars(self):
        """Chunk text longer than 500 chars must be truncated to 500."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.0] * 768)

        long_text = "A" * 1200
        qdrant_point = _make_mock_qdrant_point(chunk_text=long_text)
        mock_client.query_points.return_value.points = [qdrant_point]

        result = retrieve_evidence(
            triggered_rules=[_make_rule_result()],
            company_id=1,
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )
        assert len(result[0].chunk_text) == 500

    def test_retrieve_evidence_qdrant_returns_empty(self):
        """When Qdrant returns [], stub ChunkReference with rule_id only, score=0.0."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.0] * 768)
        mock_client.query_points.return_value.points = []

        result = retrieve_evidence(
            triggered_rules=[_make_rule_result("PLEDGE_HIGH")],
            company_id=1,
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )
        assert len(result) == 1
        stub = result[0]
        assert stub.rule_id == "PLEDGE_HIGH"
        assert stub.chunk_text is None
        assert stub.document_id is None
        assert stub.file_name is None
        assert stub.score == 0.0

    def test_retrieve_evidence_qdrant_raises(self, capsys):
        """When Qdrant raises, stderr error is logged, stub is returned, no re-raise."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.0] * 768)
        mock_client.query_points.side_effect = RuntimeError("Qdrant is down")

        # Must not raise
        result = retrieve_evidence(
            triggered_rules=[_make_rule_result("RATING_DOWNGRADE")],
            company_id=1,
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )

        assert len(result) == 1
        stub = result[0]
        assert stub.rule_id == "RATING_DOWNGRADE"
        assert stub.chunk_text is None
        assert stub.score == 0.0

        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert "RATING_DOWNGRADE" in captured.err

    def test_retrieve_evidence_returns_one_ref_per_rule(self):
        """Number of returned ChunkReferences equals number of triggered rules."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.0] * 768)
        mock_client.query_points.return_value.points = []

        rules = [
            _make_rule_result("NEGATIVE_PAT"),
            _make_rule_result("CFO_NEGATIVE"),
            _make_rule_result("DEBT_TO_EQUITY_HIGH"),
        ]
        result = retrieve_evidence(
            triggered_rules=rules,
            company_id=1,
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )
        assert len(result) == 3
        assert [r.rule_id for r in result] == [
            "NEGATIVE_PAT", "CFO_NEGATIVE", "DEBT_TO_EQUITY_HIGH"
        ]


# ---------------------------------------------------------------------------
# _make_stub
# ---------------------------------------------------------------------------

class TestMakeStub:
    def test_stub_has_correct_rule_id(self):
        stub = _make_stub("AGM_DELAY")
        assert stub.rule_id == "AGM_DELAY"

    def test_stub_all_fields_null_or_zero(self):
        stub = _make_stub("PLEDGE_RISING")
        assert stub.chunk_text is None
        assert stub.document_id is None
        assert stub.file_name is None
        assert stub.fiscal_year is None
        assert stub.fiscal_quarter is None
        assert stub.document_type is None
        assert stub.score == 0.0


# ---------------------------------------------------------------------------
# Query prefix applied during embedding
# ---------------------------------------------------------------------------

class TestQueryPrefixUsage:
    def test_query_prefix_prepended_to_template(self):
        """The embedding call should use QUERY_PREFIX + template text."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        encoded = MagicMock()
        encoded.tolist.return_value = [0.0] * 768
        mock_model.encode.return_value = encoded
        mock_client.query_points.return_value.points = []

        rule_result = _make_rule_result("NEGATIVE_PAT")
        retrieve_evidence(
            triggered_rules=[rule_result],
            company_id=1,
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )

        called_text = mock_model.encode.call_args[0][0]
        assert called_text.startswith(QUERY_PREFIX)
        expected_template = RULE_QUERY_TEMPLATES["NEGATIVE_PAT"]
        assert expected_template in called_text

    def test_fallback_to_description_for_unknown_rule_id(self):
        """For unknown rule_id, description is used as query text with QUERY_PREFIX."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        encoded = MagicMock()
        encoded.tolist.return_value = [0.0] * 768
        mock_model.encode.return_value = encoded
        mock_client.query_points.return_value.points = []

        unknown_rule = RuleResult(
            rule_id="UNKNOWN_RULE",
            description="some unusual signal",
            triggered_value=1.0,
            threshold=None,
            fiscal_year="FY2018",
            fiscal_quarter="Q1",
            feature_row_id=None,
        )
        retrieve_evidence(
            triggered_rules=[unknown_rule],
            company_id=1,
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )

        called_text = mock_model.encode.call_args[0][0]
        assert called_text.startswith(QUERY_PREFIX)
        assert "some unusual signal" in called_text


# ---------------------------------------------------------------------------
# Evidence located by the value's provenance (page + row label) or the fired rating
# ---------------------------------------------------------------------------


def _point(text, page=2, document_id=703, score=0.5):
    p = _make_mock_qdrant_point(chunk_text=text, document_id=document_id, score=score)
    p.payload.update({"page_start": page, "page_end": page})
    p.id = f"pt-{page}-{score}"
    return p


def _client_returning(*batches):
    client = MagicMock()
    client.query_points.side_effect = [MagicMock(points=list(b)) for b in batches]
    return client


def _model():
    m = MagicMock()
    m.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
    return m


class TestProvenanceEvidence:
    def test_value_page_chunk_with_the_row_label_is_cited_and_centred(self):
        """Reliance Capital: REVENUE_DECLINING cited a subsidiary note instead of the revenue row."""
        filler = "Notes on subsidiaries and other matters. " * 30
        row = filler + "Total Revenue from operations  3 780  4 591  5 518  18 323  20 447"
        client = _client_returning([_point("Segment results only", score=0.9), _point(row, score=0.6)])
        ref = _retrieve_single_evidence(
            rule_result=_make_rule_result("REVENUE_DECLINING"), company_id=1, qdrant_client=client,
            embedding_model=_model(),
            scope={"document_id": 703, "page": 2, "label": "Total Revenue from operations"})
        assert "Total Revenue from operations  3 780" in ref.chunk_text
        assert len(ref.chunk_text) == 500
        assert ref.source_kind == "document_chunk"
        page_filter = client.query_points.call_args_list[0].kwargs["query_filter"].must
        assert {c.key for c in page_filter} >= {"document_id", "page_start", "page_end"}

    def test_no_chunk_on_the_value_page_falls_back_to_the_document(self):
        client = _client_returning([], [_point("Net loss for the year", page=5)])
        ref = _retrieve_single_evidence(
            rule_result=_make_rule_result("NEGATIVE_PAT"), company_id=1, qdrant_client=client,
            embedding_model=_model(), scope={"document_id": 703, "page": 2, "label": "Profit after tax"})
        assert ref.chunk_text == "Net loss for the year" and ref.page_start == 5

    def test_credit_letter_chunk_naming_the_fired_rating_is_cited(self):
        """A letter's best semantic match may not name the rating at all."""
        client = _client_returning([
            _point("The rating action reflects the delay in debt servicing.", score=0.9),
            _point("Long Term debt 'CARE D' Revised from 'CARE BB; Stable'", score=0.7),
        ])
        ref = _retrieve_single_evidence(
            rule_result=_make_rule_result("RATING_DEFAULT_GRADE"), company_id=1, qdrant_client=client,
            embedding_model=_model(), scope={"document_id": 703, "rating": "D"})
        assert "CARE D" in ref.chunk_text
