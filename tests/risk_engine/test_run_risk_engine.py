"""
Unit tests for src/risk_engine/run_risk_engine.py

Unit tests only (no integration tests requiring live PostgreSQL or Qdrant).
Integration tests are omitted — see tasks.md task 9.1.

Validates: Requirements 1.3, 1.6, 3.6, 7.6
"""

import sys
import io
import pytest
from unittest.mock import MagicMock, patch, call

from risk_engine.run_risk_engine import (
    ScoringResult,
    _quarter_sort_key,
    print_final_summary,
    print_period_status,
    PeriodIdentifier,
)


# ---------------------------------------------------------------------------
# _quarter_sort_key
# ---------------------------------------------------------------------------

class TestQuarterSortKey:
    def test_q1_sorts_first(self):
        assert _quarter_sort_key("Q1") < _quarter_sort_key("Q2")

    def test_q2_before_q3(self):
        assert _quarter_sort_key("Q2") < _quarter_sort_key("Q3")

    def test_q3_before_q4(self):
        assert _quarter_sort_key("Q3") < _quarter_sort_key("Q4")

    def test_q4_before_none(self):
        assert _quarter_sort_key("Q4") < _quarter_sort_key(None)

    def test_none_sorts_last(self):
        assert _quarter_sort_key(None) > _quarter_sort_key("Q1")

    def test_quarter_sort_key_ordering(self):
        """Full ordering: Q1 < Q2 < Q3 < Q4 < None."""
        quarters = ["Q4", None, "Q1", "Q3", "Q2"]
        sorted_quarters = sorted(quarters, key=_quarter_sort_key)
        assert sorted_quarters == ["Q1", "Q2", "Q3", "Q4", None]

    def test_all_sort_keys_are_tuples(self):
        for q in ["Q1", "Q2", "Q3", "Q4", None]:
            key = _quarter_sort_key(q)
            assert isinstance(key, tuple)
            assert len(key) == 2


# ---------------------------------------------------------------------------
# print_final_summary
# ---------------------------------------------------------------------------

class TestPrintFinalSummary:
    def _make_result(self, outcome, ticker="X", fy="FY2018", fq="Q1"):
        return ScoringResult(
            company_id=1,
            ticker=ticker,
            fiscal_year=fy,
            fiscal_quarter=fq,
            outcome=outcome,
            risk_level="Low" if outcome == "scored" else None,
            n_triggered=0,
        )

    def test_summary_counts_scored_and_failed(self, capsys):
        """Scored N periods. S succeeded, F failed."""
        results = [
            self._make_result("scored"),
            self._make_result("scored"),
            self._make_result("failed"),
            self._make_result("skipped"),
        ]
        print_final_summary(results)
        captured = capsys.readouterr()
        assert "4" in captured.out   # total N
        assert "2" in captured.out   # succeeded
        assert "1" in captured.out   # failed

    def test_summary_format_contains_required_text(self, capsys):
        results = [self._make_result("scored")]
        print_final_summary(results)
        captured = capsys.readouterr()
        assert "Risk scoring complete" in captured.out
        assert "succeeded" in captured.out
        assert "failed" in captured.out

    def test_summary_zero_results(self, capsys):
        print_final_summary([])
        captured = capsys.readouterr()
        assert "0" in captured.out
        assert "Risk scoring complete" in captured.out

    def test_summary_skipped_count_toward_n_not_s_or_f(self, capsys):
        results = [
            self._make_result("scored"),
            self._make_result("skipped"),
            self._make_result("skipped"),
        ]
        print_final_summary(results)
        captured = capsys.readouterr()
        # N=3, succeeded=1, failed=0
        lines = captured.out
        assert "3" in lines  # total
        assert "1" in lines  # succeeded


# ---------------------------------------------------------------------------
# print_period_status
# ---------------------------------------------------------------------------

class TestPrintPeriodStatus:
    def _make_scored(self, ticker="DHFL", fq="Q3", risk_level="High", n_triggered=5):
        return ScoringResult(
            company_id=1, ticker=ticker,
            fiscal_year="FY2018", fiscal_quarter=fq,
            outcome="scored", risk_level=risk_level, n_triggered=n_triggered,
        )

    def test_scored_line_format(self, capsys):
        result = self._make_scored()
        print_period_status(result)
        captured = capsys.readouterr()
        out = captured.out
        assert "scored" in out
        assert "DHFL" in out
        assert "FY2018" in out
        assert "Q3" in out
        assert "High" in out
        assert "5" in out

    def test_skipped_line_format(self, capsys):
        result = ScoringResult(
            company_id=1, ticker="DHFL",
            fiscal_year="FY2018", fiscal_quarter="Q1",
            outcome="skipped", risk_level=None, n_triggered=0,
        )
        print_period_status(result)
        captured = capsys.readouterr()
        assert "skipped" in captured.out
        assert "no feature data" in captured.out

    def test_failed_line_prints_to_stderr(self, capsys):
        result = ScoringResult(
            company_id=1, ticker="DHFL",
            fiscal_year="FY2018", fiscal_quarter="Q2",
            outcome="failed", risk_level=None, n_triggered=0,
        )
        print_period_status(result)
        captured = capsys.readouterr()
        assert "failed" in captured.out
        assert "WARNING" in captured.err

    def test_annual_period_no_quarter_in_output(self, capsys):
        result = ScoringResult(
            company_id=1, ticker="DHFL",
            fiscal_year="FY2018", fiscal_quarter=None,
            outcome="scored", risk_level="Low", n_triggered=0,
        )
        print_period_status(result)
        captured = capsys.readouterr()
        assert "FY2018" in captured.out


# ---------------------------------------------------------------------------
# _validate_ticker
# ---------------------------------------------------------------------------

class TestValidateTicker:
    def test_validate_ticker_exits_on_unknown(self):
        """Unknown ticker → SystemExit raised."""
        from risk_engine.run_risk_engine import _validate_ticker

        mock_session = MagicMock()
        mock_session.scalar.return_value = 0  # ticker not found

        with pytest.raises(SystemExit) as exc_info:
            _validate_ticker("UNKNOWN_TICKER", mock_session)

        assert exc_info.value.code == 1

    def test_validate_ticker_returns_none_on_known(self):
        """Known ticker → returns None without raising."""
        from risk_engine.run_risk_engine import _validate_ticker

        mock_session = MagicMock()
        mock_session.scalar.return_value = 1  # ticker found

        result = _validate_ticker("DHFL", mock_session)
        assert result is None


# ---------------------------------------------------------------------------
# ScoringResult dataclass
# ---------------------------------------------------------------------------

class TestScoringResultDataclass:
    def test_scoring_result_fields(self):
        result = ScoringResult(
            company_id=1,
            ticker="DHFL",
            fiscal_year="FY2018",
            fiscal_quarter="Q3",
            outcome="scored",
            risk_level="High",
            n_triggered=4,
        )
        assert result.company_id == 1
        assert result.ticker == "DHFL"
        assert result.fiscal_year == "FY2018"
        assert result.fiscal_quarter == "Q3"
        assert result.outcome == "scored"
        assert result.risk_level == "High"
        assert result.n_triggered == 4


# ---------------------------------------------------------------------------
# PeriodIdentifier dataclass
# ---------------------------------------------------------------------------

class TestPeriodIdentifierDataclass:
    def test_period_identifier_fields(self):
        pid = PeriodIdentifier(
            company_id=2, ticker="TEST", fiscal_year="FY2019", fiscal_quarter="Q4"
        )
        assert pid.company_id == 2
        assert pid.ticker == "TEST"
        assert pid.fiscal_year == "FY2019"
        assert pid.fiscal_quarter == "Q4"


# ---------------------------------------------------------------------------
# process_period — skipped on zero features
# ---------------------------------------------------------------------------

class TestProcessPeriodSkippedOnZeroFeatures:
    """
    Validates: Requirements 1.3, 3.6

    When load_period_features returns a PeriodFeatureSet with all empty lists,
    process_period must return outcome='skipped' and must NOT call upsert_prediction.
    """

    def test_process_period_skipped_on_zero_features(self):
        from risk_engine.run_risk_engine import process_period, PeriodIdentifier
        from risk_engine.scorer import PeriodFeatureSet

        period = PeriodIdentifier(
            company_id=1,
            ticker="TESTCO",
            fiscal_year="FY2019",
            fiscal_quarter="Q2",
        )

        empty_feature_set = PeriodFeatureSet(
            company_id=1,
            fiscal_year="FY2019",
            fiscal_quarter="Q2",
            financial_features=[],
            governance_features=[],
            credit_features=[],
            shareholding_patterns=[],
        )

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_qdrant = MagicMock()
        mock_model = MagicMock()
        mock_engine = MagicMock()

        with (
            patch("risk_engine.run_risk_engine.Session", return_value=mock_session),
            patch(
                "risk_engine.run_risk_engine.load_period_features",
                return_value=empty_feature_set,
            ),
            patch("risk_engine.run_risk_engine.upsert_prediction") as mock_upsert,
            patch("risk_engine.run_risk_engine.log_risk_scoring_started"),
            patch("risk_engine.run_risk_engine.log_risk_scoring_failed"),
            patch("risk_engine.run_risk_engine.log_risk_scoring_success"),
        ):
            result = process_period(period, mock_qdrant, mock_model, mock_engine)

        assert result.outcome == "skipped"
        assert result.ticker == "TESTCO"
        assert result.fiscal_year == "FY2019"
        assert result.fiscal_quarter == "Q2"
        assert result.risk_level is None
        assert result.n_triggered == 0
        mock_upsert.assert_not_called()


# ===========================================================================
# Integration tests — task 9.1
# ===========================================================================
#
# These tests exercise the full pipeline:
#   seed data → process_period() → assert DB state
#
# All tests are marked @pytest.mark.integration and are skipped automatically
# when PostgreSQL is unavailable.
#
# Qdrant and the embedding model are always mocked so a live Qdrant is not
# required.  The mock returns an empty result set so evidence_chunks will
# contain stubs (score=0.0) — sufficient for verifying prediction writing
# without a real vector store.
#
# Isolation strategy:
#   Each test receives its own unique ticker (via uuid4 suffix) so tests
#   never collide with each other or with production data.  A fixture tears
#   down the seeded rows after every test.
#
# Validates: Requirements 1.2, 3.6, 6.1, 7.1, 7.2, 8.1, 8.5
# ===========================================================================

import os
import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text, select, delete
from sqlalchemy.orm import Session

from models import (
    Base,
    Company,
    ExtractedFeature,
    ShareholdingPattern,
    Prediction,
    ProcessingLog,
)
from risk_engine.rules import RULES, TOTAL_WEIGHT
from risk_engine.run_risk_engine import (
    PeriodIdentifier,
    process_period,
    get_scoreable_periods,
)

# ---------------------------------------------------------------------------
# Database availability check
# ---------------------------------------------------------------------------

_POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
_POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")
# Integration tests run against a dedicated test database, never the project's
# working database (they insert and delete rows).
_POSTGRES_DB = os.getenv("TEST_POSTGRES_DB", "financial_risk_test_db")
_POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
_POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

_DATABASE_URL = (
    f"postgresql+psycopg2://{_POSTGRES_USER}:{_POSTGRES_PASSWORD}"
    f"@{_POSTGRES_HOST}:{_POSTGRES_PORT}/{_POSTGRES_DB}"
)
_ADMIN_URL = _DATABASE_URL.rsplit("/", 1)[0] + "/postgres"


def _pg_available() -> bool:
    """Ensure the test database and its tables exist; True if usable."""
    try:
        admin = create_engine(_ADMIN_URL, isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": _POSTGRES_DB}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{_POSTGRES_DB}"'))
        admin.dispose()

        from models import Base  # registers every table, including its indexes

        eng = create_engine(_DATABASE_URL, echo=False)
        Base.metadata.create_all(eng)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


_pg_up = _pg_available()

pg_required = pytest.mark.skipif(
    not _pg_up,
    reason="PostgreSQL not available — integration tests skipped",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def pg_engine():
    """Create a SQLAlchemy engine and yield it; dispose after the test."""
    engine = create_engine(_DATABASE_URL, echo=False, future=True)
    yield engine
    engine.dispose()


def _unique_ticker(prefix: str = "TST") -> str:
    """Return a short unique ticker like 'TST_A3F2' to isolate each test."""
    return f"{prefix}_{uuid.uuid4().hex[:4].upper()}"


def _make_mock_qdrant() -> MagicMock:
    """Return a Qdrant client mock that always returns an empty search result."""
    client = MagicMock()
    client.query_points.return_value.points = []
    client.search.return_value = []
    return client


def _make_mock_embedding_model() -> MagicMock:
    """Return an embedding model mock whose encode() returns a 768-dim vector."""
    model = MagicMock()
    vector = MagicMock()
    vector.tolist.return_value = [0.0] * 768
    model.encode.return_value = vector
    return model


def _seed_company(session: Session, ticker: str, name: str | None = None) -> Company:
    """Insert a Company row and return the flushed ORM object (with id)."""
    company = Company(
        ticker=ticker,
        name=name or f"Test Company {ticker}",
        sector="Test",
        listing_type="equity",
    )
    session.add(company)
    session.flush()  # populate .id without committing
    return company


def _seed_financial_feature(
    session: Session,
    company_id: int,
    fiscal_year: str,
    fiscal_quarter: str | None,
    *,
    pat: float | None = None,
    cfo: float | None = None,
    interest_coverage: float | None = None,
    debt_to_equity: float | None = None,
    current_ratio: float | None = None,
    revenue_growth_pct: float | None = None,
) -> ExtractedFeature:
    feat = ExtractedFeature(
        company_id=company_id,
        document_id=None,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        feature_type="financial",
        pat=pat,
        cfo=cfo,
        interest_coverage=interest_coverage,
        debt_to_equity=debt_to_equity,
        current_ratio=current_ratio,
        revenue_growth_pct=revenue_growth_pct,
    )
    session.add(feat)
    session.flush()
    return feat


def _seed_governance_feature(
    session: Session,
    company_id: int,
    fiscal_year: str,
    fiscal_quarter: str | None,
    *,
    resignation_count: int | None = None,
    director_change_count: int | None = None,
    appointment_count: int | None = None,
    agm_delay_flag: bool | None = None,
    clarification_count: int | None = None,
    board_meeting_count: int | None = None,
) -> ExtractedFeature:
    feat = ExtractedFeature(
        company_id=company_id,
        document_id=None,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        feature_type="governance",
        resignation_count=resignation_count,
        director_change_count=director_change_count,
        appointment_count=appointment_count,
        agm_delay_flag=agm_delay_flag,
        clarification_count=clarification_count,
        board_meeting_count=board_meeting_count,
    )
    session.add(feat)
    session.flush()
    return feat


def _seed_shareholding(
    session: Session,
    company_id: int,
    fiscal_year: str,
    fiscal_quarter: str,
    *,
    promoter_holding_pct: float | None = 60.0,
    pledge_pct: float | None = 5.0,
) -> ShareholdingPattern:
    sp = ShareholdingPattern(
        company_id=company_id,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        promoter_holding_pct=promoter_holding_pct,
        pledge_pct=pledge_pct,
    )
    session.add(sp)
    session.flush()
    return sp


def _cleanup_company(engine, company_id: int) -> None:
    """
    Delete all rows seeded for a test company in dependency order.
    Called from the test itself (not a fixture) so it runs even when the
    test body creates its own sessions.

    ProcessingLog has no company_id FK — we delete risk_scoring logs whose
    message contains the company ticker, which is always present (the logger
    writes "Scoring {ticker} …" / "Scored {ticker} …" / "Failed {ticker} …").
    We look up the ticker via the Company row before deleting it.
    """
    with Session(engine) as s:
        company = s.get(Company, company_id)
        ticker = company.ticker if company else None

        s.execute(delete(Prediction).where(Prediction.company_id == company_id))

        # ProcessingLog has no company_id column; identify rows by ticker in message
        if ticker:
            s.execute(
                delete(ProcessingLog).where(
                    ProcessingLog.stage == "risk_scoring",
                    ProcessingLog.message.contains(ticker),
                )
            )

        s.execute(delete(ExtractedFeature).where(ExtractedFeature.company_id == company_id))
        s.execute(delete(ShareholdingPattern).where(ShareholdingPattern.company_id == company_id))
        s.execute(delete(Company).where(Company.id == company_id))
        s.commit()


def _make_period(company_id: int, ticker: str, fiscal_year: str, fiscal_quarter: str | None) -> PeriodIdentifier:
    """Build a PeriodIdentifier from plain primitives (safe after session close)."""
    return PeriodIdentifier(
        company_id=company_id,
        ticker=ticker,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
    )


# ---------------------------------------------------------------------------
# Helper: ProcessingLog doesn't have company_id column — query via subquery
# ---------------------------------------------------------------------------

def _get_processing_logs(session: Session, company_id: int, stage: str = "risk_scoring") -> list[ProcessingLog]:
    """
    Fetch processing_log rows for a company by joining through the ExtractedFeature
    table... but processing_log only links via document_id.  For risk scoring,
    document_id is NULL.  We identify our rows by matching on message content
    that includes the ticker (the logger writes ticker in every message).
    """
    company = session.get(Company, company_id)
    ticker = company.ticker if company else ""
    result = session.execute(
        select(ProcessingLog).where(
            ProcessingLog.stage == stage,
            ProcessingLog.message.contains(ticker),
        )
    )
    return list(result.scalars())


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pg_required
class TestIntegrationFullPipeline:
    """Full pipeline integration tests — require live PostgreSQL; Qdrant is mocked."""

    # -----------------------------------------------------------------------
    # 1. Full pipeline — single period
    # -----------------------------------------------------------------------

    def test_full_pipeline_single_period(self, pg_engine):
        """
        Seed company + features + shareholding → run engine → assert:
          - prediction row written with a valid risk_level
          - risk_flags is a non-empty list (at least one rule fires given the seeded values)
          - processing_log success row exists for the ticker
        """
        ticker = _unique_ticker("FPL")
        with Session(pg_engine) as s:
            company = _seed_company(s, ticker)
            cid = company.id
            # Seed financial features designed to fire: NEGATIVE_PAT + CFO_NEGATIVE
            _seed_financial_feature(
                s, cid, "FY2018", "Q3",
                pat=-1_000_000.0,     # NEGATIVE_PAT fires (pat < 0)
                cfo=-500_000.0,       # CFO_NEGATIVE fires (cfo < 0)
                interest_coverage=2.5,
            )
            _seed_shareholding(s, cid, "FY2018", "Q3", promoter_holding_pct=55.0, pledge_pct=10.0)
            s.commit()

        try:
            period = _make_period(cid, ticker, "FY2018", "Q3")
            result = process_period(
                period=period,
                qdrant_client=_make_mock_qdrant(),
                embedding_model=_make_mock_embedding_model(),
                engine=pg_engine,
            )

            assert result.outcome == "scored", f"Expected scored, got {result.outcome}"
            assert result.risk_level in ("Low", "Medium", "High", "Critical")
            assert result.n_triggered >= 2  # at least NEGATIVE_PAT + CFO_NEGATIVE

            # Verify prediction row in DB
            with Session(pg_engine) as s:
                pred = s.execute(
                    select(Prediction).where(
                        Prediction.company_id == cid,
                        Prediction.fiscal_year == "FY2018",
                        Prediction.fiscal_quarter == "Q3",
                        Prediction.model_version == "rule_based_v1",
                    )
                ).scalar_one_or_none()

                assert pred is not None, "Prediction row not written"
                assert pred.risk_level in ("Low", "Medium", "High", "Critical")
                assert isinstance(pred.risk_flags, list)
                assert len(pred.risk_flags) >= 2, "Expected at least 2 triggered rules in risk_flags"

                # Verify processing_log success row
                logs = _get_processing_logs(s, cid)
                success_logs = [l for l in logs if l.status == "success"]
                assert len(success_logs) >= 1, "No success log row found"
        finally:
            _cleanup_company(pg_engine, cid)

    # -----------------------------------------------------------------------
    # 2. Idempotency — run twice, second run updates in place
    # -----------------------------------------------------------------------

    def test_idempotency(self, pg_engine):
        """
        Run process_period twice on the same period.
        The second run should update the existing row (upsert) — same PK,
        risk_score unchanged, created_at refreshed.
        """
        ticker = _unique_ticker("IDP")
        with Session(pg_engine) as s:
            company = _seed_company(s, ticker)
            cid = company.id
            _seed_financial_feature(s, cid, "FY2018", "Q1", pat=-200.0)
            _seed_shareholding(s, cid, "FY2018", "Q1", pledge_pct=5.0, promoter_holding_pct=60.0)
            s.commit()

        try:
            period = _make_period(cid, ticker, "FY2018", "Q1")
            qdrant = _make_mock_qdrant()
            emb = _make_mock_embedding_model()

            # First run
            r1 = process_period(period=period, qdrant_client=qdrant, embedding_model=emb, engine=pg_engine)
            assert r1.outcome == "scored"

            with Session(pg_engine) as s:
                pred1 = s.execute(
                    select(Prediction).where(
                        Prediction.company_id == cid,
                        Prediction.fiscal_year == "FY2018",
                        Prediction.fiscal_quarter == "Q1",
                        Prediction.model_version == "rule_based_v1",
                    )
                ).scalar_one()
                pred1_id = pred1.id
                pred1_score = float(pred1.risk_score)
                pred1_created = pred1.created_at

            # Second run
            r2 = process_period(period=period, qdrant_client=qdrant, embedding_model=emb, engine=pg_engine)
            assert r2.outcome == "scored"

            with Session(pg_engine) as s:
                # There must be exactly one row (upsert, not insert)
                preds = s.execute(
                    select(Prediction).where(
                        Prediction.company_id == cid,
                        Prediction.fiscal_year == "FY2018",
                        Prediction.fiscal_quarter == "Q1",
                        Prediction.model_version == "rule_based_v1",
                    )
                ).scalars().all()
                assert len(preds) == 1, "Upsert should maintain exactly one row"

                pred2 = preds[0]
                # Same PK (id) — row was updated, not duplicated
                assert pred2.id == pred1_id
                # Risk score is deterministic — same input → same score
                assert float(pred2.risk_score) == pytest.approx(pred1_score)
                # created_at refreshed on conflict update
                assert pred2.created_at >= pred1_created
        finally:
            _cleanup_company(pg_engine, cid)

    # -----------------------------------------------------------------------
    # 3. Empty feature set → no prediction written
    # -----------------------------------------------------------------------

    def test_empty_feature_set_produces_no_prediction(self, pg_engine):
        """
        A period with zero feature rows → outcome='skipped', no prediction row,
        processing_log contains a 'failed' row with 'No feature data found for period'.
        """
        ticker = _unique_ticker("EMP")
        with Session(pg_engine) as s:
            company = _seed_company(s, ticker)
            cid = company.id
            # No features, no shareholding — intentionally empty
            s.commit()

        try:
            period = _make_period(cid, ticker, "FY2018", "Q2")
            result = process_period(
                period=period,
                qdrant_client=_make_mock_qdrant(),
                embedding_model=_make_mock_embedding_model(),
                engine=pg_engine,
            )

            assert result.outcome == "skipped"
            assert result.risk_level is None

            with Session(pg_engine) as s:
                # No prediction row
                pred = s.execute(
                    select(Prediction).where(
                        Prediction.company_id == cid,
                        Prediction.fiscal_year == "FY2018",
                        Prediction.fiscal_quarter == "Q2",
                    )
                ).scalar_one_or_none()
                assert pred is None, "No prediction should be written for empty feature set"

                # processing_log failure row with expected message
                logs = _get_processing_logs(s, cid)
                failed_logs = [l for l in logs if l.status == "failed"]
                assert len(failed_logs) >= 1
                assert any(
                    "No feature data found for period" in (l.message or "")
                    for l in failed_logs
                ), "Expected 'No feature data found for period' in failed log message"
        finally:
            _cleanup_company(pg_engine, cid)

    # -----------------------------------------------------------------------
    # 4. Partial feature set — governance only
    # -----------------------------------------------------------------------

    def test_partial_feature_set_scores_available_rules(self, pg_engine):
        """
        Only governance features seeded (no financial, no shareholding).
        Financial/shareholding rules skip; governance rules may fire.
        Prediction row must be written regardless.
        """
        ticker = _unique_ticker("PFS")
        with Session(pg_engine) as s:
            company = _seed_company(s, ticker)
            cid = company.id
            # Resignation spike: resignation_count=5 ≥ 3 threshold → RESIGNATION_SPIKE fires
            _seed_governance_feature(
                s, cid, "FY2018", "Q3",
                resignation_count=5,
                director_change_count=0,
                agm_delay_flag=False,
                clarification_count=0,
            )
            s.commit()

        try:
            period = _make_period(cid, ticker, "FY2018", "Q3")
            result = process_period(
                period=period,
                qdrant_client=_make_mock_qdrant(),
                embedding_model=_make_mock_embedding_model(),
                engine=pg_engine,
            )

            assert result.outcome == "scored"
            assert result.risk_level in ("Low", "Medium", "High", "Critical")

            with Session(pg_engine) as s:
                pred = s.execute(
                    select(Prediction).where(
                        Prediction.company_id == cid,
                        Prediction.fiscal_year == "FY2018",
                        Prediction.fiscal_quarter == "Q3",
                        Prediction.model_version == "rule_based_v1",
                    )
                ).scalar_one_or_none()
                assert pred is not None, "Prediction row should be written for partial features"

                # RESIGNATION_SPIKE should fire (resignation_count=5 >= 3)
                risk_flags = pred.risk_flags or []
                fired_ids = {f["rule_id"] for f in risk_flags}
                assert "RESIGNATION_SPIKE" in fired_ids, (
                    f"Expected RESIGNATION_SPIKE in risk_flags; got {fired_ids}"
                )
        finally:
            _cleanup_company(pg_engine, cid)

    # -----------------------------------------------------------------------
    # 5. All-null features → Low risk, risk_score=0.0, risk_flags=[]
    # -----------------------------------------------------------------------

    def test_all_nulls_produces_low_risk(self, pg_engine):
        """
        Feature rows with all signal columns NULL → no rules fire →
        risk_score=0.0, risk_level='Low', risk_flags=[].
        """
        ticker = _unique_ticker("NUL")
        with Session(pg_engine) as s:
            company = _seed_company(s, ticker)
            cid = company.id
            # Financial row — all signal columns NULL
            _seed_financial_feature(s, cid, "FY2018", "Q2")
            # Governance row — all signal columns NULL
            _seed_governance_feature(s, cid, "FY2018", "Q2")
            s.commit()

        try:
            period = _make_period(cid, ticker, "FY2018", "Q2")
            result = process_period(
                period=period,
                qdrant_client=_make_mock_qdrant(),
                embedding_model=_make_mock_embedding_model(),
                engine=pg_engine,
            )

            assert result.outcome == "scored"
            assert result.risk_level == "Low"
            assert result.n_triggered == 0

            with Session(pg_engine) as s:
                pred = s.execute(
                    select(Prediction).where(
                        Prediction.company_id == cid,
                        Prediction.fiscal_year == "FY2018",
                        Prediction.fiscal_quarter == "Q2",
                        Prediction.model_version == "rule_based_v1",
                    )
                ).scalar_one()
                assert float(pred.risk_score) == pytest.approx(0.0)
                assert pred.risk_level == "Low"
                assert pred.risk_flags == []
        finally:
            _cleanup_company(pg_engine, cid)

    # -----------------------------------------------------------------------
    # 6. Company filter scopes correctly
    # -----------------------------------------------------------------------

    def test_company_filter_scopes_correctly(self, pg_engine):
        """
        Two companies seeded.  get_scoreable_periods with company_filter=TICKER_A
        returns only TICKER_A periods; TICKER_B periods are not included.
        """
        ticker_a = _unique_ticker("FLA")
        ticker_b = _unique_ticker("FLB")
        with Session(pg_engine) as s:
            co_a = _seed_company(s, ticker_a)
            co_b = _seed_company(s, ticker_b)
            cid_a, cid_b = co_a.id, co_b.id

            _seed_financial_feature(s, cid_a, "FY2018", "Q1", pat=-100.0)
            _seed_financial_feature(s, cid_b, "FY2018", "Q1", pat=-200.0)
            _seed_shareholding(s, cid_a, "FY2018", "Q1")
            _seed_shareholding(s, cid_b, "FY2018", "Q1")
            s.commit()

        try:
            with Session(pg_engine) as s:
                periods = get_scoreable_periods(
                    session=s,
                    company_filter=ticker_a,
                    fiscal_year_filter=None,
                )

            returned_tickers = {p.ticker for p in periods}
            assert ticker_a in returned_tickers, f"{ticker_a} should be in results"
            assert ticker_b not in returned_tickers, f"{ticker_b} should NOT be in results"
        finally:
            _cleanup_company(pg_engine, cid_a)
            _cleanup_company(pg_engine, cid_b)

    # -----------------------------------------------------------------------
    # 7. Multiple rules fire → correct risk_score
    # -----------------------------------------------------------------------

    def test_multiple_rules_fire_correct_score(self, pg_engine):
        """
        Seed features that trigger exactly 3 known rules:
          - NEGATIVE_PAT     (weight 0.08)  — pat < 0
          - CFO_NEGATIVE     (weight 0.08)  — cfo < 0
          - INTEREST_COVERAGE_LOW (weight 0.10) — interest_coverage < 1.5

        Expected risk_score = round((0.08 + 0.08 + 0.10) / TOTAL_WEIGHT, 4)
        len(risk_flags) == 3
        """
        # Weights from rules.py
        w_neg_pat = next(r.weight for r in RULES if r.rule_id == "NEGATIVE_PAT")
        w_cfo_neg = next(r.weight for r in RULES if r.rule_id == "CFO_NEGATIVE")
        w_int_cov = next(r.weight for r in RULES if r.rule_id == "INTEREST_COVERAGE_LOW")
        expected_score = round((w_neg_pat + w_cfo_neg + w_int_cov) / TOTAL_WEIGHT, 4)

        ticker = _unique_ticker("MRF")
        with Session(pg_engine) as s:
            company = _seed_company(s, ticker)
            cid = company.id
            _seed_financial_feature(
                s, cid, "FY2019", "Q2",
                pat=-50_000.0,           # NEGATIVE_PAT fires
                cfo=-30_000.0,           # CFO_NEGATIVE fires
                interest_coverage=1.2,   # INTEREST_COVERAGE_LOW fires (< 1.5)
                debt_to_equity=1.5,      # DEBT_TO_EQUITY_HIGH: 1.5 < 3.0 → no fire
                current_ratio=1.5,       # CURRENT_RATIO_LOW: 1.5 >= 1.0 → no fire
                revenue_growth_pct=5.0,  # REVENUE_DECLINING: 5.0 >= -10.0 → no fire
            )
            # Shareholding: safe values — no shareholding rules should fire
            _seed_shareholding(
                s, cid, "FY2019", "Q2",
                promoter_holding_pct=60.0,  # PROMOTER_HOLDING_LOW: 60 >= 25 → no fire
                pledge_pct=5.0,             # PLEDGE_HIGH: 5.0 < 50.0 → no fire
            )
            s.commit()

        try:
            period = _make_period(cid, ticker, "FY2019", "Q2")
            result = process_period(
                period=period,
                qdrant_client=_make_mock_qdrant(),
                embedding_model=_make_mock_embedding_model(),
                engine=pg_engine,
            )

            assert result.outcome == "scored"
            assert result.n_triggered == 3, (
                f"Expected 3 rules triggered, got {result.n_triggered}"
            )

            with Session(pg_engine) as s:
                pred = s.execute(
                    select(Prediction).where(
                        Prediction.company_id == cid,
                        Prediction.fiscal_year == "FY2019",
                        Prediction.fiscal_quarter == "Q2",
                        Prediction.model_version == "rule_based_v1",
                    )
                ).scalar_one()

                assert len(pred.risk_flags) == 3, (
                    f"Expected 3 risk_flags entries, got {len(pred.risk_flags)}: "
                    f"{[f['rule_id'] for f in pred.risk_flags]}"
                )
                assert float(pred.risk_score) == pytest.approx(expected_score, abs=1e-4), (
                    f"Expected risk_score={expected_score}, got {float(pred.risk_score)}"
                )
        finally:
            _cleanup_company(pg_engine, cid)

    # -----------------------------------------------------------------------
    # 8. Evidence retrieval failure → graceful stub
    # -----------------------------------------------------------------------

    def test_evidence_retrieval_failure_graceful(self, pg_engine):
        """
        Mock Qdrant.search raises an exception.
        process_period should still complete with outcome='scored',
        and evidence_chunks should contain stub entries with score=0.0.
        """
        ticker = _unique_ticker("EVF")
        with Session(pg_engine) as s:
            company = _seed_company(s, ticker)
            cid = company.id
            _seed_financial_feature(s, cid, "FY2018", "Q4", pat=-1_000.0)  # NEGATIVE_PAT fires
            _seed_shareholding(s, cid, "FY2018", "Q4", pledge_pct=5.0, promoter_holding_pct=60.0)
            s.commit()

        try:
            # Qdrant client whose search always raises
            failing_qdrant = MagicMock()
            failing_qdrant.query_points.side_effect = RuntimeError("Qdrant unavailable")

            period = _make_period(cid, ticker, "FY2018", "Q4")
            result = process_period(
                period=period,
                qdrant_client=failing_qdrant,
                embedding_model=_make_mock_embedding_model(),
                engine=pg_engine,
            )

            # Engine must not propagate the Qdrant exception
            assert result.outcome == "scored", (
                f"Expected outcome='scored' even with Qdrant failure, got '{result.outcome}'"
            )

            with Session(pg_engine) as s:
                pred = s.execute(
                    select(Prediction).where(
                        Prediction.company_id == cid,
                        Prediction.fiscal_year == "FY2018",
                        Prediction.fiscal_quarter == "Q4",
                        Prediction.model_version == "rule_based_v1",
                    )
                ).scalar_one_or_none()

                assert pred is not None, "Prediction should be written despite Qdrant failure"

                # evidence_chunks must be a list (stubs) — all stubs have score=0.0
                ev = pred.evidence_chunks or []
                assert isinstance(ev, list)
                # Every entry returned by evidence_retriever on exception is a stub
                for chunk in ev:
                    assert chunk.get("score") == pytest.approx(0.0), (
                        f"Stub chunk should have score=0.0, got {chunk.get('score')}"
                    )
        finally:
            _cleanup_company(pg_engine, cid)


# ---------------------------------------------------------------------------
# build_evidence_scopes: provenance that locates each flag's evidence
# ---------------------------------------------------------------------------


def test_evidence_scope_carries_value_page_label_and_rating():
    from types import SimpleNamespace
    from risk_engine.run_risk_engine import build_evidence_scopes

    fin = SimpleNamespace(id=1, document_id=10, fiscal_year="FY2020", fiscal_quarter="Q4", additional_signals={
        "sources": {"pat": {"page": 2, "label": "Profit/(loss) for the period"},
                    "interest_coverage": {"page": 2, "how": "computed (PBT + finance cost) / finance cost"}}})
    cred = SimpleNamespace(id=2, document_id=11, fiscal_year="FY2020", fiscal_quarter="Q2",
                           credit_rating="D", additional_signals={})
    fs = SimpleNamespace(financial_features=[fin], governance_features=[], credit_features=[cred],
                         shareholding_patterns=[])
    rules = [SimpleNamespace(rule_id="NEGATIVE_PAT", feature_row_id=1),
             SimpleNamespace(rule_id="INTEREST_COVERAGE_LOW", feature_row_id=1),
             SimpleNamespace(rule_id="RATING_DEFAULT_GRADE", feature_row_id=2)]
    scopes = build_evidence_scopes(fs, rules)
    assert scopes["NEGATIVE_PAT"] == {"document_id": 10, "page": 2, "label": "Profit/(loss) for the period"}
    assert scopes["INTEREST_COVERAGE_LOW"] == {"document_id": 10, "page": 2, "label": "finance cost"}
    assert scopes["RATING_DEFAULT_GRADE"] == {"document_id": 11, "rating": "D"}
