"""
Rule-based scorer for the risk detection engine.

Contains the core data structures and rule evaluation logic:
  - PeriodFeatureSet  — container for all structured data available for a period
  - RuleResult        — a single rule that has fired, with triggering context
  - ScorerResult      — aggregate result of evaluating all rules for a period
  - evaluate_rules()  — public entry point; iterates RULES, computes risk score

NULL semantics: when a feature value required by a rule is NULL (absent, not
zero), the rule is silently skipped — it neither fires nor raises an exception.
"""

from dataclasses import dataclass, field
from typing import Literal

from models import ExtractedFeature, ShareholdingPattern
from risk_engine.rules import (
    RULES,
    TOTAL_WEIGHT,
    V2_MIN_DENOMINATOR_FRACTION,
    V2_SEVERITY_FLOORS,
    RuleDefinition,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PeriodFeatureSet:
    """
    Container for all structured data available for a single (company, period).

    Loaded from PostgreSQL by run_risk_engine.py and passed to evaluate_rules().
    The scorer never issues its own database queries.

    Attributes:
        company_id:           Primary key of the company being scored.
        fiscal_year:          e.g. 'FY2018'.
        fiscal_quarter:       e.g. 'Q3', or None for an annual-level period.
        financial_features:   ExtractedFeature rows with feature_type='financial'.
        governance_features:  ExtractedFeature rows with feature_type='governance'.
        credit_features:      ExtractedFeature rows with feature_type='credit'.
        shareholding_patterns: ShareholdingPattern rows for the same company + FY
                              (all quarters, sorted ascending by fiscal_quarter).
    """

    company_id: int
    fiscal_year: str
    fiscal_quarter: str | None

    financial_features: list[ExtractedFeature]
    governance_features: list[ExtractedFeature]
    credit_features: list[ExtractedFeature]
    shareholding_patterns: list[ShareholdingPattern]


@dataclass
class RuleResult:
    """
    Represents a single rule that has fired for a period.

    Attributes:
        rule_id:          Matches RuleDefinition.rule_id (e.g. 'PLEDGE_HIGH').
        description:      Human-readable description copied from RuleDefinition.
        triggered_value:  The actual feature value that caused the rule to fire.
        threshold:        The threshold from RuleDefinition (None for implicit ones).
        fiscal_year:      Fiscal year of the feature row that triggered the rule.
        fiscal_quarter:   Fiscal quarter of the feature row (or None for annual).
        feature_row_id:   extracted_features.id or shareholding_patterns.id that
                          caused the trigger. None when not traceable to a single row.
    """

    rule_id: str
    description: str
    triggered_value: float | str | bool | None
    threshold: float | str | None
    fiscal_year: str
    fiscal_quarter: str | None
    feature_row_id: int | None


@dataclass
class ScorerResult:
    """
    Aggregate result of evaluating all 15 rules for a single period.

    Attributes:
        risk_score:      Normalized score in [0.0, 1.0], rounded to 4 decimal places.
        risk_level:      'Low' | 'Medium' | 'High' | 'Critical'.
        triggered_rules: List of RuleResult objects for every rule that fired.
        coverage:        v2 only — weight of rules with input data / TOTAL_WEIGHT,
                         rounded to 4 decimal places. None for v1.
        evaluated_rule_ids: v2 only — rule_ids whose input data was present.
        floor_applied:   v2 only — True when a severity floor raised risk_level.
    """

    risk_score: float
    risk_level: Literal["Low", "Medium", "High", "Critical"]
    triggered_rules: list[RuleResult]
    coverage: float | None = None
    evaluated_rule_ids: list[str] = field(default_factory=list)
    floor_applied: bool = False


# ---------------------------------------------------------------------------
# Public evaluation entry point
# ---------------------------------------------------------------------------


def evaluate_rules(features: PeriodFeatureSet) -> ScorerResult:
    """
    Evaluate all 15 rules in the RULES catalog against the provided feature set.

    For each rule:
    - If required feature value(s) are NULL or absent → skip silently (no fire).
    - If rule condition is met → append RuleResult to the triggered list.
    - If rule condition is not met → do nothing.

    After all rules have been evaluated:
    - risk_score = round(sum(triggered weights) / TOTAL_WEIGHT, 4)
    - risk_score = 0.0 when no rules fire
    - risk_level assigned from score thresholds:
        [0.0, 0.25)  → 'Low'
        [0.25, 0.50) → 'Medium'
        [0.50, 0.75) → 'High'
        [0.75, 1.0]  → 'Critical'

    Args:
        features: PeriodFeatureSet containing all available data for the period.

    Returns:
        ScorerResult with risk_score, risk_level, and triggered_rules.

    Raises:
        Never — all NULL handling is a silent skip; no exceptions propagate.
    """
    triggered: list[RuleResult] = []

    for rule in RULES:
        result = _evaluate_single_rule(rule, features)
        if result is not None:
            triggered.append(result)

    # Compute risk score
    if triggered:
        triggered_ids = {t.rule_id for t in triggered}
        sum_triggered_weights = sum(
            r.weight for r in RULES if r.rule_id in triggered_ids
        )
        risk_score = round(sum_triggered_weights / TOTAL_WEIGHT, 4)
    else:
        risk_score = 0.0

    return ScorerResult(
        risk_score=risk_score,
        risk_level=_score_to_level(risk_score),
        triggered_rules=triggered,
    )


def evaluate_rules_v2(features: PeriodFeatureSet) -> ScorerResult:
    """
    rule_based_v2: same 15 rules, thresholds and weights as v1; different aggregation.

    - A rule is *evaluable* when its input data is present (see _rule_is_evaluable).
    - risk_score = sum(triggered weights) / denominator, where
      denominator = max(evaluable weight, V2_MIN_DENOMINATOR_FRACTION * TOTAL_WEIGHT),
      rounded to 4 decimal places (0.0 when nothing fires).
    - coverage = evaluable weight / TOTAL_WEIGHT.
    - risk_level uses the v1 cut-offs, then V2_SEVERITY_FLOORS may raise it.
      The floor never lowers a level and never changes risk_score.

    Raises:
        Never — NULL handling is identical to evaluate_rules().
    """
    triggered: list[RuleResult] = []
    evaluated_ids: list[str] = []

    for rule in RULES:
        if _rule_is_evaluable(rule, features):
            evaluated_ids.append(rule.rule_id)
        result = _evaluate_single_rule(rule, features)
        if result is not None:
            triggered.append(result)

    evaluable_weight = sum(r.weight for r in RULES if r.rule_id in evaluated_ids)
    denominator = max(evaluable_weight, V2_MIN_DENOMINATOR_FRACTION * TOTAL_WEIGHT)

    triggered_ids = {t.rule_id for t in triggered}
    sum_triggered_weights = sum(r.weight for r in RULES if r.rule_id in triggered_ids)
    risk_score = round(min(sum_triggered_weights / denominator, 1.0), 4) if triggered else 0.0

    risk_level = _score_to_level(risk_score)
    floor_applied = False
    for rule_set, floor_level in V2_SEVERITY_FLOORS:
        if rule_set <= triggered_ids and _LEVEL_ORDER[floor_level] > _LEVEL_ORDER[risk_level]:
            risk_level = floor_level  # type: ignore[assignment]
            floor_applied = True

    return ScorerResult(
        risk_score=risk_score,
        risk_level=risk_level,
        triggered_rules=triggered,
        coverage=round(evaluable_weight / TOTAL_WEIGHT, 4),
        evaluated_rule_ids=evaluated_ids,
        floor_applied=floor_applied,
    )


_LEVEL_ORDER: dict[str, int] = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def _score_to_level(risk_score: float) -> Literal["Low", "Medium", "High", "Critical"]:
    """Map a normalized score to a level: <0.25 Low, <0.50 Medium, <0.75 High, else Critical."""
    if risk_score < 0.25:
        return "Low"
    if risk_score < 0.50:
        return "Medium"
    if risk_score < 0.75:
        return "High"
    return "Critical"


def _rule_is_evaluable(rule: RuleDefinition, features: PeriodFeatureSet) -> bool:
    """
    True when the input data a rule needs is present, regardless of whether it fires.

    Mirrors the NULL checks in the _eval_* functions. One deliberate difference:
    AGM_DELAY is evaluable when agm_delay_flag is False (data present, no delay).
    """
    rid = rule.rule_id
    patterns = features.shareholding_patterns

    if rid == "PLEDGE_RISING":
        return sum(1 for sp in patterns if sp.pledge_pct is not None) >= 2
    if rid in ("PLEDGE_HIGH", "PROMOTER_HOLDING_LOW"):
        attr = "pledge_pct" if rid == "PLEDGE_HIGH" else "promoter_holding_pct"
        if features.fiscal_quarter is not None:
            sp = _find_shareholding_for_quarter(patterns, features.fiscal_quarter)
            return sp is not None and getattr(sp, attr) is not None
        return any(getattr(sp, attr) is not None for sp in patterns)

    financial_attr = {
        "INTEREST_COVERAGE_LOW": "interest_coverage",
        "NEGATIVE_PAT": "pat",
        "DEBT_TO_EQUITY_HIGH": "debt_to_equity",
        "CURRENT_RATIO_LOW": "current_ratio",
        "CFO_NEGATIVE": "cfo",
        "REVENUE_DECLINING": "revenue_growth_pct",
    }
    if rid in financial_attr:
        fin = _get_first_financial_feature(features)
        return fin is not None and getattr(fin, financial_attr[rid]) is not None

    if rid in ("RATING_DOWNGRADE", "RATING_DEFAULT_GRADE"):
        cred = _get_first_credit_feature(features)
        attr = "rating_change_flag" if rid == "RATING_DOWNGRADE" else "credit_rating"
        return cred is not None and getattr(cred, attr) is not None

    gov = _get_first_governance_feature(features)
    if gov is None:
        return False
    if rid == "RESIGNATION_SPIKE":
        return gov.resignation_count is not None
    if rid == "DIRECTOR_CHURN":
        return gov.resignation_count is not None or gov.director_change_count is not None
    if rid == "AGM_DELAY":
        return gov.agm_delay_flag is not None
    if rid == "CLARIFICATION_SPIKE":
        return gov.clarification_count is not None
    return False


# ---------------------------------------------------------------------------
# Rule dispatcher
# ---------------------------------------------------------------------------


def _evaluate_single_rule(
    rule: RuleDefinition,
    features: PeriodFeatureSet,
) -> RuleResult | None:
    """
    Dispatch to the rule-specific evaluation function via rule_id.

    Returns:
        RuleResult if the rule fires, None otherwise (including NULL / absent data).
    """
    dispatch: dict[str, object] = {
        "PLEDGE_RISING": _eval_pledge_rising,
        "PLEDGE_HIGH": _eval_pledge_high,
        "PROMOTER_HOLDING_LOW": _eval_promoter_holding_low,
        "INTEREST_COVERAGE_LOW": _eval_interest_coverage_low,
        "NEGATIVE_PAT": _eval_negative_pat,
        "DEBT_TO_EQUITY_HIGH": _eval_debt_to_equity_high,
        "CURRENT_RATIO_LOW": _eval_current_ratio_low,
        "CFO_NEGATIVE": _eval_cfo_negative,
        "REVENUE_DECLINING": _eval_revenue_declining,
        "RATING_DOWNGRADE": _eval_rating_downgrade,
        "RATING_DEFAULT_GRADE": _eval_rating_default_grade,
        "RESIGNATION_SPIKE": _eval_resignation_spike,
        "DIRECTOR_CHURN": _eval_director_churn,
        "AGM_DELAY": _eval_agm_delay,
        "CLARIFICATION_SPIKE": _eval_clarification_spike,
    }
    fn = dispatch.get(rule.rule_id)
    if fn is None:
        # Unknown rule_id — skip silently (defensive guard)
        return None
    return fn(rule, features)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Rule evaluation stubs — full implementation follows in task 3.2
# ---------------------------------------------------------------------------


def _eval_pledge_rising(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    PLEDGE_RISING: promoter pledge percentage increasing over consecutive quarters.

    Fires when at least `min_quarters` consecutive quarter-over-quarter increases
    in pledge_pct are found, each increase >= `threshold` percentage points.
    Skipped when fewer than two non-NULL pledge_pct values are available.
    """
    # Filter to rows with non-NULL pledge_pct and sort by quarter
    patterns = sorted(
        [sp for sp in features.shareholding_patterns if sp.pledge_pct is not None],
        key=lambda sp: _quarter_sort_key(sp.fiscal_quarter),
    )

    if len(patterns) < 2:
        return None

    consecutive_count = 0
    max_increase = 0.0
    triggering_row_id = None

    for i in range(1, len(patterns)):
        increase = float(patterns[i].pledge_pct) - float(patterns[i - 1].pledge_pct)
        if increase >= rule.threshold:
            consecutive_count += 1
            if increase > max_increase:
                max_increase = increase
                triggering_row_id = patterns[i].id
        else:
            # Reset — consecutive streak broken
            consecutive_count = 0
            max_increase = 0.0
            triggering_row_id = None

    if consecutive_count >= rule.min_quarters:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=round(max_increase, 2),
            threshold=rule.threshold,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=triggering_row_id,
        )
    return None


def _eval_pledge_high(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    PLEDGE_HIGH: pledge_pct >= threshold.

    Quarterly period: uses the shareholding_patterns row for the scored quarter.
    Annual period: uses the maximum pledge_pct across all quarters in the FY.
    Skipped when pledge_pct is NULL.
    """
    if features.fiscal_quarter is not None:
        # Quarterly: use the row for the specific quarter
        pattern = _find_shareholding_for_quarter(
            features.shareholding_patterns, features.fiscal_quarter
        )
        if pattern is None or pattern.pledge_pct is None:
            return None
        if float(pattern.pledge_pct) >= rule.threshold:
            return RuleResult(
                rule_id=rule.rule_id,
                description=rule.description,
                triggered_value=float(pattern.pledge_pct),
                threshold=rule.threshold,
                fiscal_year=features.fiscal_year,
                fiscal_quarter=features.fiscal_quarter,
                feature_row_id=pattern.id,
            )
    else:
        # Annual: use the maximum pledge_pct across the FY
        patterns_with_pledge = [
            sp for sp in features.shareholding_patterns if sp.pledge_pct is not None
        ]
        if not patterns_with_pledge:
            return None
        max_pattern = max(patterns_with_pledge, key=lambda sp: float(sp.pledge_pct))
        if float(max_pattern.pledge_pct) >= rule.threshold:
            return RuleResult(
                rule_id=rule.rule_id,
                description=rule.description,
                triggered_value=float(max_pattern.pledge_pct),
                threshold=rule.threshold,
                fiscal_year=features.fiscal_year,
                fiscal_quarter=None,
                feature_row_id=max_pattern.id,
            )
    return None


def _eval_promoter_holding_low(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    PROMOTER_HOLDING_LOW: promoter_holding_pct < threshold.

    Quarterly period: uses the shareholding_patterns row for the scored quarter.
    Annual period: uses the minimum promoter_holding_pct across all quarters in the FY.
    Skipped when promoter_holding_pct is NULL.
    """
    if features.fiscal_quarter is not None:
        # Quarterly: use the row for the specific quarter
        pattern = _find_shareholding_for_quarter(
            features.shareholding_patterns, features.fiscal_quarter
        )
        if pattern is None or pattern.promoter_holding_pct is None:
            return None
        if float(pattern.promoter_holding_pct) < rule.threshold:
            return RuleResult(
                rule_id=rule.rule_id,
                description=rule.description,
                triggered_value=float(pattern.promoter_holding_pct),
                threshold=rule.threshold,
                fiscal_year=features.fiscal_year,
                fiscal_quarter=features.fiscal_quarter,
                feature_row_id=pattern.id,
            )
    else:
        # Annual: use the minimum promoter_holding_pct across the FY
        patterns_with_holding = [
            sp for sp in features.shareholding_patterns if sp.promoter_holding_pct is not None
        ]
        if not patterns_with_holding:
            return None
        min_pattern = min(patterns_with_holding, key=lambda sp: float(sp.promoter_holding_pct))
        if float(min_pattern.promoter_holding_pct) < rule.threshold:
            return RuleResult(
                rule_id=rule.rule_id,
                description=rule.description,
                triggered_value=float(min_pattern.promoter_holding_pct),
                threshold=rule.threshold,
                fiscal_year=features.fiscal_year,
                fiscal_quarter=None,
                feature_row_id=min_pattern.id,
            )
    return None


def _eval_interest_coverage_low(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    INTEREST_COVERAGE_LOW: interest_coverage < threshold.
    Skipped when interest_coverage is NULL.
    """
    fin = _get_first_financial_feature(features)
    if fin is None or fin.interest_coverage is None:
        return None
    if float(fin.interest_coverage) < rule.threshold:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=float(fin.interest_coverage),
            threshold=rule.threshold,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=fin.id,
        )
    return None


def _eval_negative_pat(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    NEGATIVE_PAT: pat < 0.
    Skipped when pat is NULL.
    """
    fin = _get_first_financial_feature(features)
    if fin is None or fin.pat is None:
        return None
    if float(fin.pat) < 0:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=float(fin.pat),
            threshold=None,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=fin.id,
        )
    return None


def _eval_debt_to_equity_high(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    DEBT_TO_EQUITY_HIGH: debt_to_equity > threshold.
    Skipped when debt_to_equity is NULL.
    """
    fin = _get_first_financial_feature(features)
    if fin is None or fin.debt_to_equity is None:
        return None
    if float(fin.debt_to_equity) > rule.threshold:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=float(fin.debt_to_equity),
            threshold=rule.threshold,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=fin.id,
        )
    return None


def _eval_current_ratio_low(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    CURRENT_RATIO_LOW: current_ratio < threshold.
    Skipped when current_ratio is NULL.
    """
    fin = _get_first_financial_feature(features)
    if fin is None or fin.current_ratio is None:
        return None
    if float(fin.current_ratio) < rule.threshold:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=float(fin.current_ratio),
            threshold=rule.threshold,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=fin.id,
        )
    return None


def _eval_cfo_negative(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    CFO_NEGATIVE: cfo < 0.
    Skipped when cfo is NULL.
    """
    fin = _get_first_financial_feature(features)
    if fin is None or fin.cfo is None:
        return None
    if float(fin.cfo) < 0:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=float(fin.cfo),
            threshold=None,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=fin.id,
        )
    return None


def _eval_revenue_declining(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    REVENUE_DECLINING: revenue_growth_pct < threshold.
    Skipped when revenue_growth_pct is NULL.
    """
    fin = _get_first_financial_feature(features)
    if fin is None or fin.revenue_growth_pct is None:
        return None
    if float(fin.revenue_growth_pct) < rule.threshold:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=float(fin.revenue_growth_pct),
            threshold=rule.threshold,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=fin.id,
        )
    return None


def _eval_rating_downgrade(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    RATING_DOWNGRADE: the rating in force was a downgrade versus the same agency's
    previous rating (additional_signals['rating_direction'] == 'downgrade'), or the
    rating changed and the outlook is Negative / Watch.
    Skipped when no credit row, or when change/direction information is unavailable.
    """
    cred = _get_first_credit_feature(features)
    if cred is None or cred.rating_change_flag is None:
        return None
    direction = (getattr(cred, "additional_signals", None) or {}).get("rating_direction")
    negative_outlook = cred.rating_outlook in ("Negative", "Watch")
    if direction == "downgrade" or (cred.rating_change_flag and negative_outlook):
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=direction or True,
            threshold=None,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=cred.id,
        )
    return None


def _eval_rating_default_grade(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    RATING_DEFAULT_GRADE: credit_rating (stripped + uppercased) is in default_grades.
    Skipped when credit_rating is NULL.
    """
    cred = _get_first_credit_feature(features)
    if cred is None or cred.credit_rating is None:
        return None
    normalized = cred.credit_rating.strip().upper()
    if normalized in rule.default_grades:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=cred.credit_rating,
            threshold=None,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=cred.id,
        )
    return None


def _eval_resignation_spike(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    RESIGNATION_SPIKE: resignation_count >= min_count.
    Skipped when resignation_count is NULL.
    """
    gov = _get_first_governance_feature(features)
    if gov is None or gov.resignation_count is None:
        return None
    if gov.resignation_count >= rule.min_count:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=gov.resignation_count,
            threshold=rule.min_count,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=gov.id,
        )
    return None


def _eval_director_churn(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    DIRECTOR_CHURN: (resignation_count + director_change_count) >= min_count.

    NULL handling: treat NULL as 0 only when the other count is non-NULL.
    Skip when both counts are NULL.
    """
    gov = _get_first_governance_feature(features)
    if gov is None:
        return None

    resignation = gov.resignation_count
    director_change = gov.director_change_count

    # Both NULL → skip
    if resignation is None and director_change is None:
        return None

    # Treat NULL as 0 only when the other is non-NULL
    effective_resignation = resignation if resignation is not None else 0
    effective_director_change = director_change if director_change is not None else 0

    total_churn = effective_resignation + effective_director_change
    if total_churn >= rule.min_count:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=total_churn,
            threshold=rule.min_count,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=gov.id,
        )
    return None


def _eval_agm_delay(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    AGM_DELAY: agm_delay_flag is True.
    Skipped when agm_delay_flag is NULL or False.
    """
    gov = _get_first_governance_feature(features)
    if gov is None or gov.agm_delay_flag is None or not gov.agm_delay_flag:
        return None
    return RuleResult(
        rule_id=rule.rule_id,
        description=rule.description,
        triggered_value=True,
        threshold=None,
        fiscal_year=features.fiscal_year,
        fiscal_quarter=features.fiscal_quarter,
        feature_row_id=gov.id,
    )


def _eval_clarification_spike(
    rule: RuleDefinition, features: PeriodFeatureSet
) -> RuleResult | None:
    """
    CLARIFICATION_SPIKE: clarification_count >= min_count.
    Skipped when clarification_count is NULL.
    """
    gov = _get_first_governance_feature(features)
    if gov is None or gov.clarification_count is None:
        return None
    if gov.clarification_count >= rule.min_count:
        return RuleResult(
            rule_id=rule.rule_id,
            description=rule.description,
            triggered_value=gov.clarification_count,
            threshold=rule.min_count,
            fiscal_year=features.fiscal_year,
            fiscal_quarter=features.fiscal_quarter,
            feature_row_id=gov.id,
        )
    return None


# ---------------------------------------------------------------------------
# Helper functions — used by _eval_* implementations in task 3.2
# ---------------------------------------------------------------------------


def _get_first_financial_feature(features: PeriodFeatureSet) -> ExtractedFeature | None:
    """Return the first financial feature row for the period, or None if absent."""
    return features.financial_features[0] if features.financial_features else None


def _get_first_governance_feature(features: PeriodFeatureSet) -> ExtractedFeature | None:
    """Return the first governance feature row for the period, or None if absent."""
    return features.governance_features[0] if features.governance_features else None


def _get_first_credit_feature(features: PeriodFeatureSet) -> ExtractedFeature | None:
    """
    Return the credit row that is in force for the period: the most recent rating
    issued on or before the period end (the loader supplies rows newest-first and
    already excludes later ones).  Rows without a rating are skipped.
    """
    for f in features.credit_features:
        if f.credit_rating is not None:
            return f
    return features.credit_features[0] if features.credit_features else None


def _find_shareholding_for_quarter(
    patterns: list[ShareholdingPattern],
    quarter: str,
) -> ShareholdingPattern | None:
    """Find the ShareholdingPattern row that matches the given fiscal_quarter."""
    for sp in patterns:
        if sp.fiscal_quarter == quarter:
            return sp
    return None


def _quarter_sort_key(quarter: str | None) -> tuple:
    """
    Sort key for fiscal quarters: Q1 < Q2 < Q3 < Q4 < None (annual).

    Returns a 2-tuple so that standard tuple comparison gives the right order:
        Q1 → (0, 0)
        Q2 → (0, 1)
        Q3 → (0, 2)
        Q4 → (0, 3)
        None → (1, 0)   — sorts after all named quarters
    """
    if quarter is None:
        return (1, 0)
    quarter_order = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    return (0, quarter_order.get(quarter, 999))
