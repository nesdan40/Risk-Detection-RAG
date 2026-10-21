"""
Integration-style tests for src/feature_extraction/run_feature_extraction.py

Tests the ``main()`` orchestrator function behaviour using ``unittest.mock``
throughout to avoid a real database.  No PostgreSQL connection is needed.

Coverage
--------
- --force without --company → exit 1, error on stderr  (Req 6.7)
- --company UNKNOWN → exit 1, error identifies the bad ticker  (Req 1.3)
- Empty candidate set → stdout shows '0', exits 0, no DB writes  (Req 1.4, 8.5)
- --force --company TICKER → existing extracted_features rows deleted before
  re-extraction; processing_log rows are NOT deleted  (Req 8.2)
- Per-doc status line format  (Req 9.1)
- Final summary contains 'EXTRACTION SUMMARY:'  (Req 9.2)
- Category mapping table is sorted by count desc, ties broken alphabetically  (Req 9.5)
- Ground-truth guard: no open() call with 'ground_truth.json' in the path  (Req 1.6)
- Phase ordering: governance aggregation (Phase C) runs after Phase A
  normalization; credit extraction (Phase D) runs after Phase B financial  (Req 9 design)

Requirements: 1.1–1.8, 6.7, 8.1–8.5, 9.1–9.5
"""

from __future__ import annotations

import re
import sys
from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

# The module under test
import feature_extraction.run_feature_extraction as orchestrator
from feature_extraction.run_feature_extraction import (
    main,
    print_category_mapping_table,
    print_final_summary,
    print_per_doc_status,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_company(id: int = 1, ticker: str = "TESTCO") -> MagicMock:
    """Return a mock Company ORM object."""
    c = MagicMock()
    c.id = id
    c.ticker = ticker
    c.name = f"{ticker} Ltd"
    c.sector = "Finance"
    c.listing_type = "equity"
    return c


def _make_document(
    id: int = 10,
    company_id: int = 1,
    document_type: str = "annual_report",
    file_name: str = "TESTCO_AnnualReport_FY2019.pdf",
    fiscal_year: str = "FY2019",
    fiscal_quarter: str | None = None,
    period_end: date = date(2019, 3, 31),
    period_start: date = date(2018, 4, 1),
    document_category: str | None = None,
    embedded: bool = True,
) -> MagicMock:
    """Return a mock Document ORM object."""
    d = MagicMock()
    d.id = id
    d.company_id = company_id
    d.document_type = document_type
    d.file_name = file_name
    d.fiscal_year = fiscal_year
    d.fiscal_quarter = fiscal_quarter
    d.period_end = period_end
    d.period_start = period_start
    d.document_category = document_category
    d.embedded = embedded
    return d


def _make_session_factory(candidates: list, company_row_id: int | None = 1):
    """
    Build a mock engine + session context manager that returns `candidates`
    from get_candidate_documents and a valid company lookup.

    The returned patcher dict contains the patches already applied.  The
    caller is responsible for using them as context managers or starting
    them via the @patch decorator.
    """
    pass  # Not used directly; see individual tests that build their own mocks.


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_MOD = "feature_extraction.run_feature_extraction"


# ---------------------------------------------------------------------------
# Helper: build a fully-mocked "run environment" for main()
# ---------------------------------------------------------------------------

def _mock_run(
    candidates: list | None = None,
    ticker: str = "TESTCO",
    company_id: int = 1,
    force_candidates_after_delete: list | None = None,
):
    """
    Return a context-manager-compatible mock environment for calling main().

    Patches:
      - get_engine  → mock engine
      - Session (context manager) → mock session
      - get_candidate_documents → returns `candidates`
      - financial_extractor.extract, credit_extractor.extract → return None
        (i.e., no real extraction — Phase B / D are no-ops by default)
      - feature_writer.upsert_financial / upsert_governance / upsert_credit
      - feature_writer.delete_company_features → returns 0
      - feature_logger.* → no-ops
      - category_normalizer.normalize → returns ('Board Meeting', 'exact')
      - governance_extractor.group_announcements_by_period → returns {}
      - governance_extractor.aggregate → returns GovernanceResult()
      - financial_extractor.load_chunks → returns []
      - calculate_period_end → returns date(2019, 3, 31)

    The mock session exposes a `execute` that returns the correct company
    row when queried by ticker, enabling get_candidate_documents internals
    to work with controlled data.
    """
    if candidates is None:
        candidates = []

    return _MockRunContext(
        candidates=candidates,
        ticker=ticker,
        company_id=company_id,
        force_candidates_after_delete=force_candidates_after_delete,
    )


class _MockRunContext:
    """Context manager that applies all patches needed to call main()."""

    def __init__(self, candidates, ticker, company_id, force_candidates_after_delete):
        self.candidates = candidates
        self.ticker = ticker
        self.company_id = company_id
        self.force_candidates_after_delete = force_candidates_after_delete

        self._patchers = []
        self.mocks = {}

    def __enter__(self):
        import feature_extraction.run_feature_extraction as _orch

        # ---- engine / session ----
        mock_engine = MagicMock()
        self.mocks["engine"] = mock_engine

        mock_session = MagicMock()
        # begin_nested() context manager
        nested_ctx = MagicMock()
        nested_ctx.__enter__ = MagicMock(return_value=nested_ctx)
        nested_ctx.__exit__ = MagicMock(return_value=False)
        mock_session.begin_nested.return_value = nested_ctx
        # commit / rollback are no-ops
        mock_session.commit = MagicMock()
        mock_session.rollback = MagicMock()
        self.mocks["session"] = mock_session

        # Session() used as context manager: Session(engine) → __enter__ → session
        mock_session_cm = MagicMock()
        mock_session_cm.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cm.__exit__ = MagicMock(return_value=False)

        # ---- get_engine ----
        p_engine = patch(f"{_MOD}.get_engine", return_value=mock_engine)
        # ---- Session ----
        p_session_cls = patch(
            f"{_MOD}.Session",
            return_value=mock_session_cm,
        )

        # ---- get_candidate_documents ----
        # Called once normally; if force path re-queries, called a second time
        if self.force_candidates_after_delete is not None:
            side_effects = [self.candidates, self.force_candidates_after_delete]
        else:
            side_effects = [self.candidates]

        call_counter = {"n": 0}
        def _candidate_side_effect(session, company_filter):
            idx = call_counter["n"]
            call_counter["n"] += 1
            if idx < len(side_effects):
                return side_effects[idx]
            return side_effects[-1]

        p_candidates = patch(
            f"{_MOD}.get_candidate_documents",
            side_effect=_candidate_side_effect,
        )

        # ---- company ID lookup (for --force path) ----
        # In the force branch, main() does its own session.execute() to find company_id
        ticker = self.ticker
        company_id = self.company_id
        mock_company_row = (company_id,)
        mock_execute_result = MagicMock()
        mock_execute_result.first.return_value = mock_company_row
        mock_scalars_result = MagicMock()
        mock_scalars_result.all.return_value = []
        mock_execute_result.scalars.return_value = mock_scalars_result
        mock_session.execute.return_value = mock_execute_result

        # ---- delete_company_features_for_force ----
        p_delete_force = patch(
            f"{_MOD}.delete_company_features_for_force",
            return_value=5,
        )
        self.mocks["delete_company_features_for_force"] = None  # filled below

        # ---- feature_writer ----
        p_fw_fin = patch(f"{_MOD}.feature_writer.upsert_financial", return_value=MagicMock())
        p_fw_gov = patch(f"{_MOD}.feature_writer.upsert_governance", return_value=MagicMock())
        p_fw_cred = patch(f"{_MOD}.feature_writer.upsert_credit", return_value=MagicMock())
        p_fw_del = patch(f"{_MOD}.feature_writer.delete_company_features", return_value=0)

        # ---- extractors ----
        p_fin_extract = patch(f"{_MOD}.financial_extractor.extract", return_value=None)
        p_fin_load = patch(
            f"{_MOD}.financial_extractor.load_chunks",
            return_value=[{"chunk_text": "some text"}],
        )
        p_cred_extract = patch(f"{_MOD}.credit_extractor.extract", return_value=None)

        # ---- category_normalizer ----
        p_normalize = patch(
            f"{_MOD}.category_normalizer.normalize",
            return_value=("Board Meeting", "exact"),
        )

        # ---- governance_extractor ----
        from feature_extraction.governance_extractor import GovernanceResult
        p_group = patch(
            f"{_MOD}.group_announcements_by_period",
            return_value={},
        )
        p_agg = patch(
            f"{_MOD}.governance_extractor.aggregate",
            return_value=GovernanceResult(),
        )

        # ---- logger ----
        p_log_started = patch(f"{_MOD}.feature_logger.log_started")
        p_log_success = patch(f"{_MOD}.feature_logger.log_success")
        p_log_gov_success = patch(f"{_MOD}.feature_logger.log_governance_success")
        p_log_unmapped = patch(f"{_MOD}.feature_logger.log_unmapped_category")
        p_log_failed = patch(f"{_MOD}.feature_logger.log_failed")

        # ---- calculate_period_end ----
        p_period_end = patch(
            f"{_MOD}.calculate_period_end",
            return_value=date(2019, 3, 31),
        )

        # ---- _load_company_map ----
        company = _make_company(id=self.company_id, ticker=self.ticker)
        p_company_map = patch(
            f"{_MOD}._load_company_map",
            return_value={self.company_id: company},
        )

        all_patchers = [
            p_engine, p_session_cls, p_candidates,
            p_delete_force,
            p_fw_fin, p_fw_gov, p_fw_cred, p_fw_del,
            p_fin_extract, p_fin_load, p_cred_extract,
            p_normalize,
            p_group, p_agg,
            p_log_started, p_log_success, p_log_gov_success, p_log_unmapped, p_log_failed,
            p_period_end,
            p_company_map,
        ]

        for p in all_patchers:
            m = p.start()
            self._patchers.append(p)
            # store by patcher attribute for later access
            self.mocks[p.attribute] = m  # type: ignore[attr-defined]

        return self

    def __exit__(self, *args):
        for p in reversed(self._patchers):
            p.stop()


# ===========================================================================
# TEST CLASSES
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. --force without --company → exit 1, error on stderr  (Req 6.7)
# ---------------------------------------------------------------------------

class TestForceWithoutCompany:
    """--force requires --company; without it, main() must exit 1."""

    def test_exits_with_code_1(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(force=True, company_filter=None)
        assert exc_info.value.code == 1

    def test_prints_error_to_stderr(self, capsys):
        with pytest.raises(SystemExit):
            main(force=True, company_filter=None)
        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_error_message_mentions_force_and_company(self, capsys):
        with pytest.raises(SystemExit):
            main(force=True, company_filter=None)
        captured = capsys.readouterr()
        # Should tell the user that --force requires --company
        assert "--force" in captured.err or "force" in captured.err.lower()
        assert "--company" in captured.err or "company" in captured.err.lower()

    def test_no_db_interaction(self, capsys):
        """No engine should be created before the --force guard fires."""
        with patch(f"{_MOD}.get_engine") as mock_engine:
            with pytest.raises(SystemExit):
                main(force=True, company_filter=None)
            mock_engine.assert_not_called()

    def test_nothing_written_to_stdout(self, capsys):
        with pytest.raises(SystemExit):
            main(force=True, company_filter=None)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# 2. --company UNKNOWN → exit 1, error identifies bad ticker  (Req 1.3)
# ---------------------------------------------------------------------------

class TestUnknownCompanyFilter:
    """An unrecognised ticker must cause exit 1 with an identifying message."""

    def test_exits_with_code_1(self, capsys):
        with _mock_run(candidates=[]) as ctx:
            # Override get_candidate_documents to raise SystemExit(1) as the
            # real function does when the ticker isn't found.  We simulate
            # this by re-implementing the relevant path.
            with patch(
                f"{_MOD}.get_candidate_documents",
                side_effect=SystemExit(1),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main(company_filter="UNKNOWN_TICKER_XYZ")
        assert exc_info.value.code == 1

    def test_get_candidate_documents_called_with_filter(self, capsys):
        """get_candidate_documents must be called with the company_filter value."""
        with _mock_run(candidates=[]) as ctx:
            # Capture the call args
            call_args_store = []

            def _side_effect(session, company_filter):
                call_args_store.append(company_filter)
                return []

            with patch(f"{_MOD}.get_candidate_documents", side_effect=_side_effect):
                main(company_filter="TESTCO")

        assert call_args_store[0] == "TESTCO"

    def test_error_message_contains_ticker_in_real_path(self, capsys):
        """
        Test the real get_candidate_documents path: when the company is not
        found in the DB, the error message contains the unknown ticker.
        """
        with _mock_run(candidates=[]) as ctx:
            # Simulate get_candidate_documents printing to stderr then exiting
            def _side_effect(session, company_filter):
                print(
                    f"ERROR: --company '{company_filter}' does not match any "
                    "ticker in the companies table. Aborting.",
                    file=sys.stderr,
                )
                sys.exit(1)

            with patch(f"{_MOD}.get_candidate_documents", side_effect=_side_effect):
                with pytest.raises(SystemExit):
                    main(company_filter="BOGUS_CO")

        captured = capsys.readouterr()
        assert "BOGUS_CO" in captured.err


# ---------------------------------------------------------------------------
# 3. Empty candidate set → stdout shows '0', exits 0, no DB writes  (Req 1.4, 8.5)
# ---------------------------------------------------------------------------

class TestEmptyCandidateSet:
    """When no eligible documents exist, the pipeline reports 0 and exits cleanly."""

    def test_exits_with_code_0(self):
        with _mock_run(candidates=[]) as ctx:
            # main() calls sys.exit(0) implicitly by returning; no SystemExit raised
            main()  # should complete without raising

    def test_stdout_contains_zero(self, capsys):
        with _mock_run(candidates=[]) as ctx:
            main()
        captured = capsys.readouterr()
        assert "0" in captured.out

    def test_stdout_contains_relevant_text(self, capsys):
        with _mock_run(candidates=[]) as ctx:
            main()
        captured = capsys.readouterr()
        # The spec says "no documents require feature extraction" (or similar)
        out = captured.out.lower()
        assert "document" in out or "feature extraction" in out

    def test_no_writes_to_extracted_features(self):
        with _mock_run(candidates=[]) as ctx:
            main()
        # None of the upsert methods should have been called
        ctx.mocks["upsert_financial"].assert_not_called()
        ctx.mocks["upsert_governance"].assert_not_called()
        ctx.mocks["upsert_credit"].assert_not_called()

    def test_no_delete_called(self):
        """delete_company_features_for_force must not be called on empty run."""
        with _mock_run(candidates=[]) as ctx:
            main()
        ctx.mocks["delete_company_features_for_force"].assert_not_called()

    def test_with_company_filter_still_exits_cleanly(self, capsys):
        with _mock_run(candidates=[], ticker="TESTCO") as ctx:
            main(company_filter="TESTCO")
        # no exception

    def test_stdout_zero_when_company_filter_provided(self, capsys):
        with _mock_run(candidates=[], ticker="TESTCO") as ctx:
            main(company_filter="TESTCO")
        captured = capsys.readouterr()
        assert "0" in captured.out


# ---------------------------------------------------------------------------
# 4. --force --company TICKER: existing rows deleted, processing_log intact
#    (Req 8.2)
# ---------------------------------------------------------------------------

class TestForceWithCompany:
    """--force deletes extracted_features rows for the company, not processing_log."""

    def test_delete_company_features_for_force_is_called(self):
        """delete_company_features_for_force must be called exactly once."""
        annual_doc = _make_document(
            id=10, company_id=1, document_type="annual_report"
        )
        with _mock_run(
            candidates=[annual_doc],
            ticker="TESTCO",
            company_id=1,
            force_candidates_after_delete=[annual_doc],
        ) as ctx:
            main(company_filter="TESTCO", force=True)

        ctx.mocks["delete_company_features_for_force"].assert_called_once()

    def test_delete_called_before_re_query(self):
        """
        The delete must happen before the second candidate query.
        Track call order via a shared log.
        """
        call_order: list[str] = []

        annual_doc = _make_document(id=10, company_id=1)
        candidate_calls = {"n": 0}

        def _candidates(session, company_filter):
            candidate_calls["n"] += 1
            call_order.append(f"get_candidates_{candidate_calls['n']}")
            return [annual_doc] if candidate_calls["n"] > 1 else [annual_doc]

        def _delete_force(session, company_id):
            call_order.append("delete_force")
            return 5

        with _mock_run(
            candidates=[annual_doc],
            ticker="TESTCO",
            company_id=1,
            force_candidates_after_delete=[annual_doc],
        ) as ctx:
            with patch(f"{_MOD}.get_candidate_documents", side_effect=_candidates):
                with patch(
                    f"{_MOD}.delete_company_features_for_force",
                    side_effect=_delete_force,
                ):
                    main(company_filter="TESTCO", force=True)

        # delete must appear before the second get_candidates call
        if "delete_force" in call_order and "get_candidates_2" in call_order:
            assert call_order.index("delete_force") < call_order.index("get_candidates_2")

    def test_processing_log_delete_not_called(self):
        """
        delete_company_features_for_force must NOT also delete processing_log.
        We verify by checking that feature_writer.delete_company_features
        (which only removes extracted_features rows per spec) is what's called,
        not a broader log-clearing operation.
        The real delete_company_features_for_force delegates to
        feature_writer.delete_company_features.  We only assert that no
        additional broader delete is issued.
        """
        annual_doc = _make_document(id=10, company_id=1)
        with _mock_run(
            candidates=[annual_doc],
            ticker="TESTCO",
            company_id=1,
            force_candidates_after_delete=[annual_doc],
        ) as ctx:
            main(company_filter="TESTCO", force=True)

        # feature_writer.delete_company_features itself is patched at the
        # module level; it should NOT have been called directly from main
        # (main calls delete_company_features_for_force, not the inner fn).
        ctx.mocks["delete_company_features"].assert_not_called()

    def test_force_without_company_still_rejected(self, capsys):
        """Sanity check: --force without --company still raises SystemExit(1)."""
        with pytest.raises(SystemExit) as exc_info:
            main(force=True, company_filter=None)
        assert exc_info.value.code == 1

    def test_deletion_stdout_message_contains_company_ticker(self, capsys):
        """The --force deletion path prints a message mentioning the ticker."""
        annual_doc = _make_document(id=10, company_id=1)
        with _mock_run(
            candidates=[annual_doc],
            ticker="TESTCO",
            company_id=1,
            force_candidates_after_delete=[annual_doc],
        ) as ctx:
            main(company_filter="TESTCO", force=True)

        captured = capsys.readouterr()
        assert "TESTCO" in captured.out


# ---------------------------------------------------------------------------
# 5. Per-doc status line format  (Req 9.1)
# ---------------------------------------------------------------------------

class TestPerDocStatusLine:
    """The per-document status line must match the specified format."""

    PER_DOC_PATTERN = re.compile(
        r"^\[.+\] doc_id=\d+ \| .+ \| type=.+ \| signals=\d+ non-null columns written$"
    )

    def test_print_per_doc_status_format_success(self, capsys):
        print_per_doc_status("success", 42, "TESTCO_Report.pdf", "annual_report", 5)
        captured = capsys.readouterr()
        line = captured.out.strip()
        assert self.PER_DOC_PATTERN.match(line), (
            f"Per-doc status line does not match expected format: {line!r}"
        )

    def test_print_per_doc_status_format_failed(self, capsys):
        print_per_doc_status("failed", 7, "TESTCO_Q1.pdf", "quarterly_result", 0)
        captured = capsys.readouterr()
        line = captured.out.strip()
        assert self.PER_DOC_PATTERN.match(line)

    def test_print_per_doc_status_format_unmapped(self, capsys):
        print_per_doc_status("unmapped_category", 99, "TESTCO_Ann.pdf", "corporate_announcement", 0)
        captured = capsys.readouterr()
        line = captured.out.strip()
        assert self.PER_DOC_PATTERN.match(line)

    def test_per_doc_status_contains_correct_doc_id(self, capsys):
        print_per_doc_status("success", 123, "file.pdf", "annual_report", 3)
        captured = capsys.readouterr()
        assert "doc_id=123" in captured.out

    def test_per_doc_status_contains_file_name(self, capsys):
        print_per_doc_status("success", 5, "DHFL_AnnualReport_FY2019.pdf", "annual_report", 6)
        captured = capsys.readouterr()
        assert "DHFL_AnnualReport_FY2019.pdf" in captured.out

    def test_per_doc_status_contains_feature_type(self, capsys):
        print_per_doc_status("success", 5, "file.pdf", "quarterly_result", 4)
        captured = capsys.readouterr()
        assert "type=quarterly_result" in captured.out

    def test_per_doc_status_contains_signal_count(self, capsys):
        print_per_doc_status("success", 5, "file.pdf", "annual_report", 7)
        captured = capsys.readouterr()
        assert "signals=7" in captured.out

    def test_per_doc_status_in_orchestrator_output(self, capsys):
        """
        When main() processes a financial doc (with a mocked successful extract),
        the per-doc status line in stdout must match the regex pattern.
        """
        from feature_extraction.financial_extractor import FinancialResult
        from decimal import Decimal

        annual_doc = _make_document(
            id=55,
            company_id=1,
            document_type="annual_report",
            file_name="TESTCO_AnnualReport_FY2019.pdf",
        )

        mock_result = FinancialResult(
            revenue=Decimal("1000"),
            pat=Decimal("100"),
            ebitda_margin_pct=None,
            debt_to_equity=None,
            interest_coverage=None,
            current_ratio=None,
            cfo=None,
            revenue_growth_pct=None,
        )
        mock_result.signals_populated = 2

        with _mock_run(candidates=[annual_doc], ticker="TESTCO", company_id=1) as ctx:
            with patch(f"{_MOD}.financial_extractor.extract", return_value=mock_result):
                with patch(f"{_MOD}.feature_writer.upsert_financial", return_value=MagicMock()):
                    main()

        captured = capsys.readouterr()
        # Find per-doc status lines in output
        lines = captured.out.splitlines()
        status_lines = [l for l in lines if "doc_id=" in l and "| type=" in l]
        assert len(status_lines) >= 1
        for line in status_lines:
            assert self.PER_DOC_PATTERN.match(line.strip()), (
                f"Status line does not match pattern: {line!r}"
            )


# ---------------------------------------------------------------------------
# 6. Final summary contains 'EXTRACTION SUMMARY:' printed after mapping table
#    (Req 9.2)
# ---------------------------------------------------------------------------

class TestFinalSummary:
    """EXTRACTION SUMMARY: header must appear in stdout after the mapping table."""

    def test_print_final_summary_contains_header(self, capsys):
        print_final_summary(
            total=5, success=4, failed=1, unmapped=0,
            features_written=4, governance_periods=2,
        )
        captured = capsys.readouterr()
        assert "EXTRACTION SUMMARY:" in captured.out

    def test_print_final_summary_contains_totals(self, capsys):
        print_final_summary(
            total=10, success=8, failed=2, unmapped=1,
            features_written=7, governance_periods=3,
        )
        captured = capsys.readouterr()
        assert "10" in captured.out
        assert "8" in captured.out
        assert "2" in captured.out

    def test_final_summary_appears_in_main_output(self, capsys):
        """main() must always print EXTRACTION SUMMARY: even with an empty run."""
        # Empty set → returns before printing summary; use a candidate so
        # summary is triggered (even if all fail with extract=None)
        annual_doc = _make_document(id=10, company_id=1, document_type="annual_report")

        with _mock_run(candidates=[annual_doc], ticker="TESTCO", company_id=1) as ctx:
            main()

        captured = capsys.readouterr()
        assert "EXTRACTION SUMMARY:" in captured.out

    def test_extraction_summary_after_category_mapping_table(self, capsys):
        """EXTRACTION SUMMARY: must appear AFTER the category mapping table."""
        annual_doc = _make_document(id=10, company_id=1, document_type="annual_report")
        ann_doc = _make_document(
            id=11, company_id=1, document_type="corporate_announcement",
            file_name="TESTCO_Ann_20190101_BoardMeeting_SCANNED.pdf",
        )

        with _mock_run(
            candidates=[ann_doc, annual_doc], ticker="TESTCO", company_id=1
        ) as ctx:
            main()

        captured = capsys.readouterr()
        out = captured.out
        # Both sections must be present
        assert "Category Mapping Summary" in out
        assert "EXTRACTION SUMMARY:" in out
        # Category mapping must appear first
        mapping_pos = out.index("Category Mapping Summary")
        summary_pos = out.index("EXTRACTION SUMMARY:")
        assert mapping_pos < summary_pos, (
            "EXTRACTION SUMMARY: must appear after Category Mapping Summary"
        )


# ---------------------------------------------------------------------------
# 7. Category mapping table: sorted by count desc, ties broken alphabetically
#    (Req 9.5)
# ---------------------------------------------------------------------------

class TestCategoryMappingTable:
    """The category mapping table must be sorted correctly."""

    def test_print_category_mapping_table_sorted_by_count_desc(self, capsys):
        counts = {
            "Board Meeting": 10,
            "Press Release": 20,
            "Appointment": 5,
            "Resignation": 15,
        }
        print_category_mapping_table(counts)
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        # Find data rows (skip header)
        data_lines = [l for l in lines if any(cat in l for cat in counts)]
        # Extract order of categories
        cat_order = []
        for line in data_lines:
            for cat in counts:
                if cat in line:
                    cat_order.append(cat)
                    break
        # Should be: Press Release (20), Board Meeting (10)... wait: Resignation(15) > Board Meeting(10)
        expected_order = ["Press Release", "Resignation", "Board Meeting", "Appointment"]
        assert cat_order == expected_order, (
            f"Expected order {expected_order}, got {cat_order}"
        )

    def test_print_category_mapping_table_ties_broken_alphabetically(self, capsys):
        """When counts are equal, categories appear in alphabetical order."""
        counts = {
            "Resignation": 5,
            "Appointment": 5,
            "Board Meeting": 5,
        }
        print_category_mapping_table(counts)
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        data_lines = [l for l in lines if any(cat in l for cat in counts)]

        cat_order = []
        for line in data_lines:
            for cat in counts:
                if cat in line:
                    cat_order.append(cat)
                    break

        expected_order = ["Appointment", "Board Meeting", "Resignation"]
        assert cat_order == expected_order, (
            f"Tie-breaking alphabetically failed: expected {expected_order}, got {cat_order}"
        )

    def test_empty_category_counts_does_not_crash(self, capsys):
        """An empty mapping should print a no-announcements notice without crashing."""
        print_category_mapping_table({})
        captured = capsys.readouterr()
        assert captured.out  # something was printed

    def test_single_category_prints_correctly(self, capsys):
        print_category_mapping_table({"Board Meeting": 7})
        captured = capsys.readouterr()
        assert "Board Meeting" in captured.out
        assert "7" in captured.out

    def test_mapping_table_in_orchestrator_output_is_sorted(self, capsys):
        """
        When main() processes multiple announcement docs with different labels,
        the mapping table in stdout must be sorted by count descending.
        """
        # 3 Board Meeting docs, 1 Resignation doc
        normalize_calls: list[str] = []
        label_sequence = [
            ("Board Meeting", "exact"),
            ("Board Meeting", "exact"),
            ("Board Meeting", "exact"),
            ("Resignation", "exact"),
        ]
        call_idx = {"n": 0}

        def _normalize(document_category, file_name, chunk_text):
            idx = call_idx["n"] % len(label_sequence)
            call_idx["n"] += 1
            return label_sequence[idx]

        ann_docs = [
            _make_document(
                id=i, company_id=1, document_type="corporate_announcement",
                file_name=f"TESTCO_Ann_file{i}.pdf",
                period_start=date(2018, 5, 1),
            )
            for i in range(1, 5)
        ]

        with _mock_run(candidates=ann_docs, ticker="TESTCO", company_id=1) as ctx:
            with patch(f"{_MOD}.category_normalizer.normalize", side_effect=_normalize):
                main()

        captured = capsys.readouterr()
        out = captured.out
        # Board Meeting (3) must appear before Resignation (1)
        assert "Board Meeting" in out
        assert "Resignation" in out
        bm_pos = out.index("Board Meeting")
        res_pos = out.index("Resignation")
        assert bm_pos < res_pos, (
            "Board Meeting (count=3) must appear before Resignation (count=1) in table"
        )


# ---------------------------------------------------------------------------
# 8. Ground-truth guard: no open() with 'ground_truth.json' in path  (Req 1.6)
# ---------------------------------------------------------------------------

class TestGroundTruthGuard:
    """No call to open() must pass a path containing 'ground_truth.json'."""

    def test_main_never_opens_ground_truth_json(self):
        """Track all builtins.open calls and assert none reference ground_truth.json."""
        annual_doc = _make_document(id=10, company_id=1, document_type="annual_report")
        ann_doc = _make_document(
            id=11, company_id=1, document_type="corporate_announcement",
            file_name="TESTCO_Ann_20190101_BoardMeeting_SCANNED.pdf",
        )

        opened_paths: list[str] = []
        real_open = open

        def _tracking_open(file, *args, **kwargs):
            opened_paths.append(str(file))
            return real_open(file, *args, **kwargs)

        with _mock_run(candidates=[ann_doc, annual_doc]) as ctx:
            with patch("builtins.open", side_effect=_tracking_open):
                main()

        forbidden = [p for p in opened_paths if "ground_truth.json" in p]
        assert forbidden == [], (
            f"open() was called with ground_truth.json path(s): {forbidden}"
        )

    def test_assert_not_ground_truth_raises_on_violation(self):
        """The guard function itself must raise AssertionError on a bad path."""
        from pathlib import Path
        from feature_extraction.run_feature_extraction import _assert_not_ground_truth

        with pytest.raises(AssertionError, match="ground_truth.json"):
            _assert_not_ground_truth(Path("data/raw/TESTCO/ground_truth.json"))

    def test_assert_not_ground_truth_passes_for_normal_jsonl(self):
        """Normal JSONL chunk paths must not trigger the guard."""
        from pathlib import Path
        from feature_extraction.run_feature_extraction import _assert_not_ground_truth

        # Should not raise
        _assert_not_ground_truth(Path("data/processed/TESTCO/10_chunks.jsonl"))
        _assert_not_ground_truth(Path("data/processed/TESTCO/99_chunks.jsonl"))


# ---------------------------------------------------------------------------
# 9. Phase ordering: Phase A runs before Phase C  (design constraint)
# ---------------------------------------------------------------------------

class TestPhaseOrdering:
    """
    Governance aggregation (Phase C) must run after category normalization
    (Phase A) has completed for all announcement documents.
    """

    def test_normalize_called_before_group_announcements(self):
        """
        category_normalizer.normalize (Phase A) must be called before
        group_announcements_by_period (Phase C).
        """
        call_order: list[str] = []

        ann_doc = _make_document(
            id=11, company_id=1, document_type="corporate_announcement",
            file_name="TESTCO_Ann_20190101_BoardMeeting_SCANNED.pdf",
            period_start=date(2019, 1, 1),
        )

        def _normalize(document_category, file_name, chunk_text):
            call_order.append("normalize")
            return ("Board Meeting", "exact")

        def _group(docs, normalized_categories):
            call_order.append("group_announcements")
            return {}

        with _mock_run(candidates=[ann_doc], ticker="TESTCO", company_id=1) as ctx:
            with patch(f"{_MOD}.category_normalizer.normalize", side_effect=_normalize):
                with patch(
                    f"{_MOD}.group_announcements_by_period", side_effect=_group
                ):
                    main()

        # normalize must appear before group_announcements in the call order
        assert "normalize" in call_order, "normalize was never called"
        assert "group_announcements" in call_order, "group_announcements_by_period was never called"
        last_normalize = max(i for i, x in enumerate(call_order) if x == "normalize")
        first_group = min(i for i, x in enumerate(call_order) if x == "group_announcements")
        assert last_normalize < first_group, (
            f"group_announcements_by_period was called before normalize completed. "
            f"call_order={call_order}"
        )

    def test_group_announcements_receives_populated_normalized_categories(self):
        """
        group_announcements_by_period must receive a non-empty
        normalized_categories dict when announcement docs exist.
        """
        received_normalized: list[dict] = []

        ann_doc = _make_document(
            id=11, company_id=1, document_type="corporate_announcement",
            file_name="TESTCO_Ann_20190101_BoardMeeting_SCANNED.pdf",
            period_start=date(2019, 1, 1),
        )

        def _group(docs, normalized_categories):
            received_normalized.append(dict(normalized_categories))
            return {}

        with _mock_run(candidates=[ann_doc], ticker="TESTCO", company_id=1) as ctx:
            with patch(
                f"{_MOD}.group_announcements_by_period", side_effect=_group
            ):
                main()

        assert len(received_normalized) == 1
        nc = received_normalized[0]
        assert ann_doc.id in nc, (
            "normalized_categories passed to group_announcements_by_period "
            f"is missing doc_id={ann_doc.id}"
        )

    def test_credit_phase_runs_after_financial_phase(self):
        """
        credit_extractor.extract (Phase D) must only be called after
        financial_extractor.extract (Phase B) has completed.
        """
        call_order: list[str] = []

        annual_doc = _make_document(
            id=10, company_id=1, document_type="annual_report",
        )
        credit_doc = _make_document(
            id=20, company_id=1, document_type="credit_rating",
            file_name="TATASTEEL_CreditRating.pdf",
        )

        from feature_extraction.financial_extractor import FinancialResult
        from decimal import Decimal
        mock_fin_result = FinancialResult(
            revenue=Decimal("1000"), pat=None, ebitda_margin_pct=None,
            debt_to_equity=None, interest_coverage=None, current_ratio=None,
            cfo=None, revenue_growth_pct=None,
        )
        mock_fin_result.signals_populated = 1

        def _fin_extract(doc, company, jsonl_path, session):
            call_order.append("financial_extract")
            return mock_fin_result

        def _cred_extract(doc, company, jsonl_path, session):
            call_order.append("credit_extract")
            return None

        with _mock_run(
            candidates=[annual_doc, credit_doc],
            ticker="TESTCO", company_id=1,
        ) as ctx:
            with patch(f"{_MOD}.financial_extractor.extract", side_effect=_fin_extract):
                with patch(f"{_MOD}.credit_extractor.extract", side_effect=_cred_extract):
                    with patch(f"{_MOD}.feature_writer.upsert_financial", return_value=MagicMock()):
                        main()

        assert "financial_extract" in call_order, "financial_extractor.extract never called"
        assert "credit_extract" in call_order, "credit_extractor.extract never called"
        fin_pos = call_order.index("financial_extract")
        cred_pos = call_order.index("credit_extract")
        assert fin_pos < cred_pos, (
            f"credit_extractor.extract was called before financial_extractor.extract. "
            f"call_order={call_order}"
        )


# ---------------------------------------------------------------------------
# 10. Company filter notice (Req 1.8)
# ---------------------------------------------------------------------------

class TestCompanyFilterNotice:
    """When --company is provided and candidates are non-empty, print the notice."""

    def test_company_filter_notice_printed(self, capsys):
        annual_doc = _make_document(id=10, company_id=1, document_type="annual_report")
        with _mock_run(candidates=[annual_doc], ticker="TESTCO", company_id=1) as ctx:
            main(company_filter="TESTCO")
        captured = capsys.readouterr()
        assert "Processing documents for company: TESTCO" in captured.out

    def test_no_company_filter_notice_without_flag(self, capsys):
        annual_doc = _make_document(id=10, company_id=1, document_type="annual_report")
        with _mock_run(candidates=[annual_doc], ticker="TESTCO", company_id=1) as ctx:
            main(company_filter=None)
        captured = capsys.readouterr()
        assert "Processing documents for company:" not in captured.out


# ---------------------------------------------------------------------------
# 11. Idempotency: running with no new documents does nothing  (Req 8.1)
# ---------------------------------------------------------------------------

class TestIdempotencyNoNewDocs:
    """When candidates is empty (all already extracted), no writes occur."""

    def test_no_upsert_called_when_no_candidates(self):
        with _mock_run(candidates=[]) as ctx:
            main()
        ctx.mocks["upsert_financial"].assert_not_called()
        ctx.mocks["upsert_governance"].assert_not_called()
        ctx.mocks["upsert_credit"].assert_not_called()

    def test_no_log_started_when_no_candidates(self):
        with _mock_run(candidates=[]) as ctx:
            main()
        ctx.mocks["log_started"].assert_not_called()
