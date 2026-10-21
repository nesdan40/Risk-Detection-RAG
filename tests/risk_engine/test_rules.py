"""
Unit tests for src/risk_engine/rules.py

Validates: Requirements 2.1, 2.2, 2.3, 2.12, 2.18
"""

import pytest

from risk_engine.rules import RULES, TOTAL_WEIGHT, RuleDefinition


class TestRulesCatalogStructure:
    def test_rules_catalog_count(self):
        """Catalog must contain exactly 15 rules."""
        assert len(RULES) == 15

    def test_rule_ids_unique(self):
        """Every rule_id must be distinct."""
        ids = [r.rule_id for r in RULES]
        assert len(ids) == len(set(ids)), "Duplicate rule_id detected"

    def test_rule_weights_positive(self):
        """Every rule weight must be strictly positive."""
        for rule in RULES:
            assert rule.weight > 0, f"Rule {rule.rule_id} has non-positive weight {rule.weight}"

    def test_total_weight_calculation(self):
        """TOTAL_WEIGHT must equal the sum of all individual weights."""
        assert TOTAL_WEIGHT == pytest.approx(sum(r.weight for r in RULES))

    def test_all_rules_are_rule_definition_instances(self):
        """Every entry in RULES must be a RuleDefinition dataclass instance."""
        for rule in RULES:
            assert isinstance(rule, RuleDefinition), f"{rule!r} is not a RuleDefinition"

    def test_all_rules_have_non_empty_description(self):
        """Every rule must have a non-empty description string."""
        for rule in RULES:
            assert isinstance(rule.description, str) and rule.description.strip(), (
                f"Rule {rule.rule_id} has blank description"
            )


class TestExpectedRuleIds:
    """Verify each of the 15 expected rule_ids is present in the catalog."""

    EXPECTED_IDS = {
        "PLEDGE_RISING",
        "PLEDGE_HIGH",
        "PROMOTER_HOLDING_LOW",
        "INTEREST_COVERAGE_LOW",
        "NEGATIVE_PAT",
        "DEBT_TO_EQUITY_HIGH",
        "CURRENT_RATIO_LOW",
        "CFO_NEGATIVE",
        "REVENUE_DECLINING",
        "RATING_DOWNGRADE",
        "RATING_DEFAULT_GRADE",
        "RESIGNATION_SPIKE",
        "DIRECTOR_CHURN",
        "AGM_DELAY",
        "CLARIFICATION_SPIKE",
    }

    def test_all_expected_ids_present(self):
        actual_ids = {r.rule_id for r in RULES}
        assert actual_ids == self.EXPECTED_IDS


class TestShareholdingRuleThresholds:
    def test_pledge_rising_thresholds(self):
        rule = next(r for r in RULES if r.rule_id == "PLEDGE_RISING")
        assert rule.threshold == pytest.approx(10.0)
        assert rule.min_quarters == 2

    def test_pledge_high_threshold(self):
        rule = next(r for r in RULES if r.rule_id == "PLEDGE_HIGH")
        assert rule.threshold == pytest.approx(50.0)

    def test_promoter_holding_low_threshold(self):
        rule = next(r for r in RULES if r.rule_id == "PROMOTER_HOLDING_LOW")
        assert rule.threshold == pytest.approx(25.0)


class TestFinancialRuleThresholds:
    def test_interest_coverage_low_threshold(self):
        rule = next(r for r in RULES if r.rule_id == "INTEREST_COVERAGE_LOW")
        assert rule.threshold == pytest.approx(1.5)

    def test_debt_to_equity_high_threshold(self):
        rule = next(r for r in RULES if r.rule_id == "DEBT_TO_EQUITY_HIGH")
        assert rule.threshold == pytest.approx(3.0)

    def test_current_ratio_low_threshold(self):
        rule = next(r for r in RULES if r.rule_id == "CURRENT_RATIO_LOW")
        assert rule.threshold == pytest.approx(1.0)

    def test_revenue_declining_threshold(self):
        rule = next(r for r in RULES if r.rule_id == "REVENUE_DECLINING")
        assert rule.threshold == pytest.approx(-10.0)

    def test_negative_pat_no_threshold(self):
        """NEGATIVE_PAT uses an implicit pat < 0 check — no threshold field."""
        rule = next(r for r in RULES if r.rule_id == "NEGATIVE_PAT")
        assert rule.threshold is None

    def test_cfo_negative_no_threshold(self):
        """CFO_NEGATIVE uses an implicit cfo < 0 check — no threshold field."""
        rule = next(r for r in RULES if r.rule_id == "CFO_NEGATIVE")
        assert rule.threshold is None


class TestCreditRuleThresholds:
    def test_rating_default_grade_list(self):
        rule = next(r for r in RULES if r.rule_id == "RATING_DEFAULT_GRADE")
        assert rule.default_grades == ["D", "C", "SD", "NM"]

    def test_rating_downgrade_no_threshold(self):
        rule = next(r for r in RULES if r.rule_id == "RATING_DOWNGRADE")
        assert rule.threshold is None
        assert rule.default_grades == []


class TestGovernanceRuleThresholds:
    def test_resignation_spike_min_count(self):
        rule = next(r for r in RULES if r.rule_id == "RESIGNATION_SPIKE")
        assert rule.min_count == 3

    def test_director_churn_min_count(self):
        rule = next(r for r in RULES if r.rule_id == "DIRECTOR_CHURN")
        assert rule.min_count == 4

    def test_clarification_spike_min_count(self):
        rule = next(r for r in RULES if r.rule_id == "CLARIFICATION_SPIKE")
        assert rule.min_count == 5

    def test_agm_delay_no_threshold(self):
        rule = next(r for r in RULES if r.rule_id == "AGM_DELAY")
        assert rule.threshold is None
        assert rule.min_count is None
