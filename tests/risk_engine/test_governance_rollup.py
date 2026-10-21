"""
Unit tests for rollup_governance in src/risk_engine/run_risk_engine.py

Annual periods take the fiscal year's worst quarter (max), never a sum, so the
single-quarter governance rules keep their meaning under frozen thresholds.
"""

from models import ExtractedFeature
from risk_engine.run_risk_engine import rollup_governance
from risk_engine.scorer import PeriodFeatureSet, evaluate_rules


def _gov(id, q, resign=None, change=None, clar=None, agm=None):
    return ExtractedFeature(
        id=id, company_id=1, fiscal_year="FY2017", fiscal_quarter=q, feature_type="governance",
        resignation_count=resign, director_change_count=change,
        clarification_count=clar, agm_delay_flag=agm,
    )


def test_empty_returns_none():
    assert rollup_governance([]) is None


def test_counts_are_max_not_sum():
    rows = [_gov(1, "Q1", resign=2, change=0, clar=1), _gov(2, "Q2", resign=2, change=1, clar=4),
            _gov(3, "Q3", resign=1, change=0, clar=2)]
    r = rollup_governance(rows)
    assert r.resignation_count == 2      # a sum would be 5 and fire RESIGNATION_SPIKE (>= 3)
    assert r.clarification_count == 4
    assert r.resignation_count + r.director_change_count == 3  # worst single-quarter churn (Q2)


def test_churn_comes_from_one_quarter():
    """Max resignations in Q1, max director changes in Q2: churn must not combine them."""
    rows = [_gov(1, "Q1", resign=3, change=0), _gov(2, "Q2", resign=0, change=3)]
    r = rollup_governance(rows)
    assert r.resignation_count == 3
    assert r.resignation_count + r.director_change_count == 3  # not 6


def test_row_identity_points_at_worst_quarter():
    rows = [_gov(10, "Q1", resign=0, change=1), _gov(11, "Q3", resign=1, change=2)]
    r = rollup_governance(rows)
    assert r.id == 11 and r.fiscal_quarter == "Q3"
    assert r.additional_signals["source_row_ids"] == [10, 11]


def test_agm_flag_any_true_else_false_else_none():
    assert rollup_governance([_gov(1, "Q1", agm=False), _gov(2, "Q2", agm=True)]).agm_delay_flag is True
    assert rollup_governance([_gov(1, "Q1", agm=False), _gov(2, "Q2")]).agm_delay_flag is False
    assert rollup_governance([_gov(1, "Q1"), _gov(2, "Q2")]).agm_delay_flag is None


def test_all_null_counts_stay_null():
    r = rollup_governance([_gov(1, "Q1"), _gov(2, "Q2")])
    assert r.resignation_count is None and r.director_change_count is None
    assert r.clarification_count is None


def test_rolled_row_fires_rules_like_its_worst_quarter():
    rows = [_gov(1, "Q1", resign=1, change=0), _gov(2, "Q2", resign=3, change=1, agm=True)]
    fs = PeriodFeatureSet(
        company_id=1, fiscal_year="FY2017", fiscal_quarter=None,
        financial_features=[], governance_features=[rollup_governance(rows)],
        credit_features=[], shareholding_patterns=[],
    )
    fired = {t.rule_id for t in evaluate_rules(fs).triggered_rules}
    assert fired == {"RESIGNATION_SPIKE", "DIRECTOR_CHURN", "AGM_DELAY"}
