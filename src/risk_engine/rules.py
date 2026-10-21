"""
Rule catalog for the rule-based risk engine.

All 15 risk detection rules are defined here as RuleDefinition dataclass
instances. Thresholds and weights are declared in this module only — the
Scorer in scorer.py reads from this catalog and applies rules uniformly
without hardcoding any threshold values.
"""

from dataclasses import dataclass, field


@dataclass
class RuleDefinition:
    """
    Represents a single risk detection rule with its identifier, description,
    weight, and zero or more threshold attributes.

    All threshold fields have default values so that rules not using a
    particular threshold can omit it at instantiation.

    Attributes:
        rule_id:        Unique string identifier for the rule (e.g. 'PLEDGE_HIGH').
        description:    Human-readable description of the risk signal.
        weight:         Contribution to the normalized risk_score when triggered.
        threshold:      Generic numeric threshold (interpretation varies per rule).
        min_quarters:   Minimum number of consecutive quarters (PLEDGE_RISING).
        min_count:      Minimum count value to trigger (governance count rules).
        default_grades: List of credit rating strings that indicate default
                        (RATING_DEFAULT_GRADE).
    """

    rule_id: str
    description: str
    weight: float

    # Optional threshold fields — only relevant to specific rules
    threshold: float | None = None
    min_quarters: int | None = None
    min_count: int | None = None
    default_grades: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule catalog — exactly 15 rules
# ---------------------------------------------------------------------------

RULES: list[RuleDefinition] = [
    # ------------------------------------------------------------------
    # Shareholding signals (3 rules)
    # ------------------------------------------------------------------
    RuleDefinition(
        rule_id="PLEDGE_RISING",
        description="Promoter pledge percentage increasing over consecutive quarters",
        weight=0.10,
        threshold=10.0,    # minimum percentage-point increase per quarter
        min_quarters=2,    # minimum number of consecutive rising quarters required
    ),
    RuleDefinition(
        rule_id="PLEDGE_HIGH",
        description="Promoter pledge percentage at or above critical level",
        weight=0.08,
        threshold=50.0,    # fires when pledge_pct >= 50.0
    ),
    RuleDefinition(
        rule_id="PROMOTER_HOLDING_LOW",
        description="Low promoter ownership suggesting weak insider confidence",
        weight=0.05,
        threshold=25.0,    # fires when promoter_holding_pct < 25.0
    ),

    # ------------------------------------------------------------------
    # Financial signals (6 rules)
    # ------------------------------------------------------------------
    RuleDefinition(
        rule_id="INTEREST_COVERAGE_LOW",
        description="Interest coverage below safe threshold",
        weight=0.10,
        threshold=1.5,     # fires when interest_coverage < 1.5
    ),
    RuleDefinition(
        rule_id="NEGATIVE_PAT",
        description="Loss-making period (negative profit after tax)",
        weight=0.08,
        # Implicit threshold: pat < 0 — no separate threshold field needed
    ),
    RuleDefinition(
        rule_id="DEBT_TO_EQUITY_HIGH",
        description="Excessive leverage",
        weight=0.08,
        threshold=3.0,     # fires when debt_to_equity > 3.0
    ),
    RuleDefinition(
        rule_id="CURRENT_RATIO_LOW",
        description="Inadequate liquidity",
        weight=0.06,
        threshold=1.0,     # fires when current_ratio < 1.0
    ),
    RuleDefinition(
        rule_id="CFO_NEGATIVE",
        description="Negative operating cash flow",
        weight=0.08,
        # Implicit threshold: cfo < 0 — no separate threshold field needed
    ),
    RuleDefinition(
        rule_id="REVENUE_DECLINING",
        description="Significant revenue contraction",
        weight=0.06,
        threshold=-10.0,   # fires when revenue_growth_pct < -10.0
    ),

    # ------------------------------------------------------------------
    # Credit signals (2 rules)
    # ------------------------------------------------------------------
    RuleDefinition(
        rule_id="RATING_DOWNGRADE",
        description="Credit rating downgrade with negative trajectory",
        weight=0.10,
        # No threshold fields — triggers on rating_change_flag=TRUE
        # AND rating_outlook IN ('Negative', 'Watch')
    ),
    RuleDefinition(
        rule_id="RATING_DEFAULT_GRADE",
        description="Credit rating at default or near-default grade",
        weight=0.12,
        default_grades=["D", "C", "SD", "NM"],
    ),

    # ------------------------------------------------------------------
    # Governance signals (4 rules)
    # ------------------------------------------------------------------
    RuleDefinition(
        rule_id="RESIGNATION_SPIKE",
        description="Unusually high resignation count in a single quarter",
        weight=0.07,
        min_count=3,       # fires when resignation_count >= 3
    ),
    RuleDefinition(
        rule_id="DIRECTOR_CHURN",
        description="High aggregate director turnover in a single quarter",
        weight=0.06,
        min_count=4,       # fires when (resignation_count + director_change_count) >= 4
    ),
    RuleDefinition(
        rule_id="AGM_DELAY",
        description="Delayed annual general meeting",
        weight=0.05,
        # Implicit threshold: agm_delay_flag = TRUE
    ),
    RuleDefinition(
        rule_id="CLARIFICATION_SPIKE",
        description="Elevated media scrutiny through high clarification count",
        weight=0.05,
        min_count=5,       # fires when clarification_count >= 5
    ),
]

# Total weight — used by the Scorer to normalize risk_score to [0.0, 1.0]
TOTAL_WEIGHT: float = sum(r.weight for r in RULES)


# ---------------------------------------------------------------------------
# rule_based_v2 aggregation settings
# ---------------------------------------------------------------------------
# v2 keeps every rule threshold and weight above unchanged (frozen baseline);
# it only changes how triggered weights are aggregated into a score.

# v2 normalises by the weight of rules that had input data, but never by less
# than this fraction of TOTAL_WEIGHT, so sparse periods cannot inflate scores.
V2_MIN_DENOMINATOR_FRACTION: float = 0.5

# Severity floors: when every rule in the set fires, risk_level is raised to at
# least the given level (risk_score itself is not changed).
V2_SEVERITY_FLOORS: list[tuple[frozenset[str], str]] = [
    # Loss-making and unable to cover interest from earnings
    (frozenset({"NEGATIVE_PAT", "INTEREST_COVERAGE_LOW"}), "Medium"),
]
