"""
Governance extractor for the structured feature extraction pipeline.

Aggregates governance signal counts per ``(company_id, fiscal_year,
fiscal_quarter)`` group from categorised corporate announcement documents.

All computation in this module is **pure** (no I/O, no DB access):

- :func:`group_announcements_by_period` — buckets documents into fiscal periods.
- :func:`aggregate` — counts canonical labels into a :class:`GovernanceResult`.
- :func:`evaluate_agm_delay_flag` — evaluates the AGM delay flag for a fiscal year.

The orchestrator (``run_feature_extraction.py``) owns the DB session and passes
these helpers pre-resolved category data from :mod:`category_normalizer`.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 2.8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

from feature_extraction.fiscal_utils import derive_fiscal_period

if TYPE_CHECKING:
    from models.document import Document


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CategorizedAnnouncement:
    """
    A corporate announcement document with its resolved canonical category.

    Produced by the orchestrator after running :func:`category_normalizer.normalize`
    on every ``corporate_announcement`` document, then pairing the result with
    the document's metadata.

    Attributes
    ----------
    document_id:       PK from ``documents.id``.
    file_name:         ``documents.file_name`` value.
    company_id:        FK to ``companies.id``.
    period_start:      ``documents.period_start`` — may be None for bad data.
    fiscal_year:       Indian fiscal year label, e.g. ``'FY2019'``.
    fiscal_quarter:    Quarter label, one of ``'Q1'``–``'Q4'``.
    canonical_label:   Resolved canonical category or ``'unclassified'``.
    resolution_layer:  Which normalizer layer resolved the category.
    """

    document_id: int
    file_name: str
    company_id: int
    period_start: date | None
    fiscal_year: str
    fiscal_quarter: str
    canonical_label: str
    resolution_layer: str


@dataclass
class GovernanceResult:
    """
    Aggregated governance signals for one ``(company_id, fiscal_year,
    fiscal_quarter)`` period.

    All count fields default to 0 so the caller can always safely read them.
    ``agm_delay_flag`` is None until explicitly set by the orchestrator from
    :func:`evaluate_agm_delay_flag`.

    Note on ``clarification_count``:
        Per schema design, ``clarification_count`` is a **superset** of
        ``press_release_count`` — it counts Press Release **plus** News
        Clarification **plus** News Verification.  Both columns are always
        populated explicitly.

    Attributes
    ----------
    board_meeting_count:    Count of 'Board Meeting' documents.
    director_change_count:  Count of 'Director Change' documents.
    appointment_count:      Count of 'Appointment' documents.
    resignation_count:      Count of 'Resignation' documents.
    press_release_count:    Count of 'Press Release' documents.
    clarification_count:    Count of Press Release + News Clarification +
                            News Verification documents (superset).
    agm_delay_flag:         True/False/None — set separately by the orchestrator.
    corporate_actions:      List of metadata dicts for 'Corporate Action - Other'
                            documents (excluded from typed counts).
    """

    board_meeting_count: int = 0
    director_change_count: int = 0
    appointment_count: int = 0
    resignation_count: int = 0
    press_release_count: int = 0
    clarification_count: int = 0
    agm_delay_flag: bool | None = None
    corporate_actions: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def group_announcements_by_period(
    documents: list["Document"],
    normalized_categories: dict[int, tuple[str, str]],
) -> dict[tuple[str, str], list[CategorizedAnnouncement]]:
    """
    Group corporate announcement documents into fiscal periods.

    Uses :func:`fiscal_utils.derive_fiscal_period` to map each document's
    ``period_start`` date to a ``(fiscal_year, fiscal_quarter)`` pair.

    Documents with ``NULL period_start`` are silently excluded here — the
    orchestrator logs those separately before calling this function.

    Parameters
    ----------
    documents:
        List of ``corporate_announcement`` :class:`Document` ORM objects.
        Documents with ``NULL period_start`` are excluded from the output.
    normalized_categories:
        Mapping of ``document_id`` → ``(canonical_label, resolution_layer)``
        as returned by :func:`category_normalizer.normalize`.

    Returns
    -------
    Dict mapping ``(fiscal_year, fiscal_quarter)`` →
    list of :class:`CategorizedAnnouncement`.  Each list is ordered by
    document insertion order (ascending ``document_id`` via the caller's
    candidate query).

    Requirements: 4.1, 4.2
    """
    groups: dict[tuple[str, str], list[CategorizedAnnouncement]] = {}

    for doc in documents:
        # Exclude documents with NULL period_start — caller logs these
        if doc.period_start is None:
            continue

        # period_start may be a datetime or date depending on DB driver
        ps: date = (
            doc.period_start
            if isinstance(doc.period_start, date)
            else doc.period_start.date()  # type: ignore[union-attr]
        )

        fiscal_year, fiscal_quarter = derive_fiscal_period(ps)

        # Resolve category; fall back to 'unclassified' if not in the map
        canonical_label, resolution_layer = normalized_categories.get(
            doc.id, ("unclassified", "unclassified")
        )

        announcement = CategorizedAnnouncement(
            document_id=doc.id,
            file_name=doc.file_name or "",
            company_id=doc.company_id,
            period_start=ps,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            canonical_label=canonical_label,
            resolution_layer=resolution_layer,
        )

        key = (fiscal_year, fiscal_quarter)
        if key not in groups:
            groups[key] = []
        groups[key].append(announcement)

    return groups


def aggregate(
    announcements: list[CategorizedAnnouncement],
) -> GovernanceResult:
    """
    Aggregate governance counts for one fiscal period.

    This is a **pure function**: no I/O, no DB access, no side effects.

    Counting rules
    --------------
    - ``board_meeting_count``   — documents with ``canonical_label == 'Board Meeting'``
    - ``director_change_count`` — ``'Director Change'``
    - ``appointment_count``     — ``'Appointment'``
    - ``resignation_count``     — ``'Resignation'``
    - ``press_release_count``   — ``'Press Release'``
    - ``clarification_count``   — ``'Press Release'`` + ``'News Clarification'``
                                  + ``'News Verification'`` (intentional superset)
    - ``'unclassified'``        — excluded from ALL counts
    - ``'Corporate Action - Other'`` — excluded from counts; recorded in
                                  ``corporate_actions`` list
    - ``agm_delay_flag``        — NOT set here; the orchestrator sets it separately
                                  via :func:`evaluate_agm_delay_flag`

    Parameters
    ----------
    announcements:
        All :class:`CategorizedAnnouncement` objects for ONE ``(company_id,
        fiscal_year, fiscal_quarter)`` period.  The caller is responsible for
        pre-filtering to the correct period.

    Returns
    -------
    :class:`GovernanceResult` with all counts populated.
    ``agm_delay_flag`` is always ``None`` on return from this function.

    Requirements: 4.3, 4.7, 4.8, 2.8
    """
    result = GovernanceResult()

    for ann in announcements:
        label = ann.canonical_label

        if label == "Board Meeting":
            result.board_meeting_count += 1

        elif label == "Director Change":
            result.director_change_count += 1

        elif label == "Appointment":
            result.appointment_count += 1

        elif label == "Resignation":
            result.resignation_count += 1

        elif label == "Press Release":
            result.press_release_count += 1
            result.clarification_count += 1  # superset includes press releases

        elif label == "News Clarification":
            result.clarification_count += 1

        elif label == "News Verification":
            result.clarification_count += 1

        elif label == "Financial Results":
            # Financial Results announcements are governance documents but do
            # not map to any of the nine typed governance count columns —
            # they are categorised and silently skipped here.
            pass

        elif label == "AGM / Annual General Meeting Updates":
            # AGM docs contribute only to the agm_delay_flag (evaluated
            # separately); they do not increment any typed count column.
            pass

        elif label == "Corporate Action - Other":
            # Excluded from typed counts; recorded for additional_signals
            period_start_str: str | None = (
                ann.period_start.isoformat() if ann.period_start is not None else None
            )
            result.corporate_actions.append(
                {
                    "document_id": ann.document_id,
                    "file_name": ann.file_name,
                    "date": period_start_str,
                    "resolved_label": label,
                }
            )

        elif label == "Credit Rating":
            # Credit Rating docs are handled by the credit extractor; they
            # have no typed governance column.
            pass

        # 'unclassified' — excluded silently from all counts (Req 4.7)

    return result


def evaluate_agm_delay_flag(
    agm_announcements: list[CategorizedAnnouncement],
    fiscal_year: str | None = None,
) -> bool | None:
    """
    Evaluate whether AGMs announced in a period were held late.

    An AGM belongs to the accounts of the fiscal year that ended on the 31 March
    before the meeting: April-December of year Y -> accounts year ending 31 Mar Y,
    January-March of year Y -> accounts year ending 31 Mar Y-1.  The statutory
    deadline is six months after that year end, i.e. 30 September of the accounts
    year.  So a meeting held on 28 Dec 2016 (accounts FY2016) is late; one held on
    30 Jun 2016 is on time.

    The flag describes the *period the meeting took place in* (it is only knowable
    once the meeting has happened, so it must not be attached to an earlier period).

    Returns True if any AGM in the list was late, False if there were AGMs and all
    were on time, None if there were none (or none with a usable date).
    """
    dated = [a.period_start for a in agm_announcements if a.period_start is not None]
    if not dated:
        return None
    for d in dated:
        accounts_year = d.year if d.month >= 4 else d.year - 1
        if d > date(accounts_year, 9, 30):
            return True
    return False
