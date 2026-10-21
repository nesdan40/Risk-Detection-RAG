"""
Unit and property-based tests for evaluate_rules_v2 in src/risk_engine/scorer.py

v2 keeps v1's rules, thresholds and weights; only aggregation differs:
evaluable-weight normalisation (with a denominator floor), coverage reporting,
and severity floors.
"""

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings

from risk_engine.prediction_writer import MODEL_VERSION_V2, upsert_prediction
from risk_engine.rules import (
    RULES,
    TOTAL_WEIGHT,
    V2_MIN_DENOMINATOR_FRACTION,
)
from risk_engine.scorer import (
    PeriodFeatureSet,
    _rule_is_evaluable,
    evaluate_rules,
    evaluate_rules_v2,
)
from tests.risk_engine.test_scorer import (
    _FakeCreditFeature,
    _FakeFinancialFeature,
    _FakeGovernanceFeature,
    _FakeShareholdingPattern,
    _empty_feature_set,
    period_feature_set_strategy,
)

_WEIGHT = {r.rule_id: r.weight for r in RULES}
_MIN_DENOM = V2_MIN_DENOMINATOR_FRACTION * TOTAL_WEIGHT


def _fs(fin=None, gov=None, cred=None, sp=None, fiscal_quarter="Q1"):
    return PeriodFeatureSet(
        company_id=1,
        fiscal_year="FY2018",
        fiscal_quarter=fiscal_quarter,
        financial_features=[fin] if fin else [],
        governance_features=[gov] if gov else [],
        credit_features=[cred] if cred else [],
        shareholding_patterns=sp or [],
    )


def _full_coverage_fs(**fin_overrides):
    """Every rule evaluable; nothing fires unless overridden."""
    fin_kwargs = dict(
        pat=100.0, cfo=100.0, interest_coverage=5.0, debt_to_equity=0.5,
        current_ratio=2.0, revenue_growth_pct=5.0,
    )
    fin_kwargs.update(fin_overrides)
    return _fs(
        fin=_FakeFinancialFeature(**fin_kwargs),
        gov=_FakeGovernanceFeature(
            resignation_count=0, director_change_count=0,
            agm_delay_flag=False, clarification_count=0,
        ),
        cred=_FakeCreditFeature(credit_rating="AA", rating_change_flag=False),
        sp=[
            _FakeShareholdingPattern(id=1, fiscal_quarter="Q1", promoter_holding_pct=60.0, pledge_pct=0.0),
            _FakeShareholdingPattern(id=2, fiscal_quarter="Q2", promoter_holding_pct=60.0, pledge_pct=0.0),
        ],
    )


class TestCoverage:
    def test_empty_feature_set(self):
        result = evaluate_rules_v2(_empty_feature_set())
        assert result.risk_score == 0.0
        assert result.risk_level == "Low"
        assert result.coverage == 0.0
        assert result.evaluated_rule_ids == []

    def test_full_coverage(self):
        result = evaluate_rules_v2(_full_coverage_fs())
        assert result.coverage == 1.0
        assert len(result.evaluated_rule_ids) == len(RULES)
        assert result.triggered_rules == []

    def test_partial_coverage_financial_only(self):
        result = evaluate_rules_v2(_fs(fin=_FakeFinancialFeature(pat=10.0, interest_coverage=3.0)))
        assert set(result.evaluated_rule_ids) == {"NEGATIVE_PAT", "INTEREST_COVERAGE_LOW"}
        expected = round((_WEIGHT["NEGATIVE_PAT"] + _WEIGHT["INTEREST_COVERAGE_LOW"]) / TOTAL_WEIGHT, 4)
        assert result.coverage == expected

    def test_agm_flag_false_is_evaluable(self):
        result = evaluate_rules_v2(_fs(gov=_FakeGovernanceFeature(agm_delay_flag=False)))
        assert "AGM_DELAY" in result.evaluated_rule_ids
        assert result.triggered_rules == []

    def test_director_churn_evaluable_with_one_count(self):
        result = evaluate_rules_v2(_fs(gov=_FakeGovernanceFeature(director_change_count=1)))
        assert "DIRECTOR_CHURN" in result.evaluated_rule_ids
        assert "RESIGNATION_SPIKE" not in result.evaluated_rule_ids

    def test_pledge_rising_needs_two_values(self):
        one = _fs(sp=[_FakeShareholdingPattern(pledge_pct=5.0)])
        assert "PLEDGE_RISING" not in evaluate_rules_v2(one).evaluated_rule_ids
        assert "PLEDGE_HIGH" in evaluate_rules_v2(one).evaluated_rule_ids

    def test_quarterly_shareholding_uses_scored_quarter(self):
        # Only a Q2 row exists while scoring Q1 → PROMOTER_HOLDING_LOW not evaluable
        fs = _fs(sp=[_FakeShareholdingPattern(fiscal_quarter="Q2", promoter_holding_pct=10.0)])
        assert "PROMOTER_HOLDING_LOW" not in evaluate_rules_v2(fs).evaluated_rule_ids

    def test_credit_rules(self):
        fs = _fs(cred=_FakeCreditFeature(credit_rating="BBB"))
        ids = evaluate_rules_v2(fs).evaluated_rule_ids
        assert "RATING_DEFAULT_GRADE" in ids
        assert "RATING_DOWNGRADE" not in ids


class TestNormalisation:
    def test_full_coverage_matches_v1(self):
        fs = _full_coverage_fs(pat=-5.0, cfo=-5.0)
        v1 = evaluate_rules(fs)
        v2 = evaluate_rules_v2(fs)
        assert v2.risk_score == v1.risk_score

    def test_divides_by_evaluable_weight(self):
        # Financial + governance + shareholding evaluable; credit missing
        fs = _full_coverage_fs(pat=-5.0, cfo=-5.0)
        fs.credit_features = []
        result = evaluate_rules_v2(fs)
        evaluable = TOTAL_WEIGHT - _WEIGHT["RATING_DOWNGRADE"] - _WEIGHT["RATING_DEFAULT_GRADE"]
        assert evaluable > _MIN_DENOM
        expected = round((_WEIGHT["NEGATIVE_PAT"] + _WEIGHT["CFO_NEGATIVE"]) / evaluable, 4)
        assert result.risk_score == expected

    def test_denominator_floor_limits_sparse_periods(self):
        # Only shareholding evaluable; PROMOTER_HOLDING_LOW fires
        fs = _fs(sp=[_FakeShareholdingPattern(fiscal_quarter="Q1", promoter_holding_pct=13.0, pledge_pct=0.0)])
        result = evaluate_rules_v2(fs)
        assert [t.rule_id for t in result.triggered_rules] == ["PROMOTER_HOLDING_LOW"]
        assert result.risk_score == round(_WEIGHT["PROMOTER_HOLDING_LOW"] / _MIN_DENOM, 4)
        assert result.risk_level == "Low"


class TestSeverityFloor:
    def test_floor_raises_low_to_medium(self):
        fs = _full_coverage_fs(pat=-876.39, interest_coverage=0.4)
        result = evaluate_rules_v2(fs)
        assert result.risk_score < 0.25
        assert result.risk_level == "Medium"
        assert result.floor_applied is True

    def test_floor_needs_both_rules(self):
        result = evaluate_rules_v2(_full_coverage_fs(pat=-5.0))
        assert result.risk_level == "Low"
        assert result.floor_applied is False

    def test_floor_never_lowers_level(self):
        fs = _fs(fin=_FakeFinancialFeature(
            pat=-5.0, interest_coverage=0.4, cfo=-5.0, debt_to_equity=5.0,
            current_ratio=0.5, revenue_growth_pct=-50.0,
        ))
        result = evaluate_rules_v2(fs)
        assert result.risk_level in ("High", "Critical")
        assert result.floor_applied is False


@given(features=period_feature_set_strategy())
@settings(max_examples=200)
def test_property_triggered_rules_are_evaluable(features):
    """Every rule that fires must have been counted as evaluable."""
    result = evaluate_rules_v2(features)
    evaluated = set(result.evaluated_rule_ids)
    for t in result.triggered_rules:
        assert t.rule_id in evaluated


@given(features=period_feature_set_strategy())
@settings(max_examples=200)
def test_property_v2_bounds_and_v1_agreement(features):
    """Same rules fire as v1; score in [0, 1]; v2 score >= v1 score; coverage in [0, 1]."""
    v1 = evaluate_rules(features)
    v2 = evaluate_rules_v2(features)
    assert [t.rule_id for t in v2.triggered_rules] == [t.rule_id for t in v1.triggered_rules]
    assert 0.0 <= v2.risk_score <= 1.0
    assert v2.risk_score >= v1.risk_score
    assert 0.0 <= v2.coverage <= 1.0


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.rule_id)
def test_each_rule_has_evaluability_check(rule):
    """Every catalog rule is evaluable under full coverage (no rule silently unmapped)."""
    assert _rule_is_evaluable(rule, _full_coverage_fs())


def test_upsert_prediction_writes_v2_model_version():
    session = MagicMock()
    upsert_prediction(
        company_id=1, fiscal_year="FY2018", fiscal_quarter="Q1",
        risk_score=0.1, risk_level="Low", triggered_rules=[], evidence_chunks=[],
        session=session, model_version=MODEL_VERSION_V2,
    )
    stmt = session.execute.call_args[0][0]
    assert stmt.compile().params["model_version"] == "rule_based_v2"


def test_upsert_prediction_rejects_unknown_model_version():
    with pytest.raises(ValueError):
        upsert_prediction(
            company_id=1, fiscal_year="FY2018", fiscal_quarter="Q1",
            risk_score=0.1, risk_level="Low", triggered_rules=[], evidence_chunks=[],
            session=MagicMock(), model_version="xgboost_v1",
        )
