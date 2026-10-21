"""
Credit signal extractor for the structured feature extraction pipeline.

Reads the chunk text of ``credit_rating`` documents (and of announcements that
resolved to 'Credit Rating') and extracts the long-term rating, its outlook, and
the rating agency.

Design notes
------------
- A rating only counts when a rating-agency name immediately precedes an
  UPPERCASE rating symbol ("CARE AAA", "[ICRA]AA+", "IND BBB-/RWN").  The earlier
  free-floating pattern matched ordinary words and list markers ("a", "(b)", "C").
- Short-term ratings (A1+, A4, P1 ...) are recorded in ``additional_signals`` but
  never become the long-term ``credit_rating``.
- When the letter announces a revision ("downgraded", "revised rating"), the
  *new* rating is the last one mentioned.  Otherwise the most conservative
  (lowest) rating mentioned is used.
- ``rating_change_flag`` is NOT decided at extraction time (it depended on which
  document happened to be processed first).  ``recompute_rating_changes`` sets it
  after all rows exist, ordered by rating date, per agency.
- Never raises: ``extract()`` returns None on any problem or when no rating is found.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_extraction.financial_extractor import concatenate_chunk_text, load_chunks
from models.company import Company
from models.document import Document
from models.feature import ExtractedFeature

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class CreditResult:
    """
    credit_rating:       long-term rating, e.g. 'AAA', 'BBB-', 'D'
    rating_outlook:      'Stable' | 'Negative' | 'Positive' | 'Watch' | None
    rating_change_flag:  always None here; set by recompute_rating_changes()
    instrument_ratings:  every rating mention in the document (for traceability)
    agency:              agency of the chosen rating
    extras:              extra overflow signals (revision flag, short-term ratings ...)
    """

    credit_rating: str | None = None
    rating_outlook: str | None = None
    rating_change_flag: bool | None = None
    instrument_ratings: list[dict] | None = None
    agency: str | None = None
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rating scale
# ---------------------------------------------------------------------------

# Best -> worst.  Index is used to tell downgrades from upgrades.
SCALE = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
         "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]
_RANK = {r: i for i, r in enumerate(SCALE)}

_AGENCY = (
    r"(?:\[?\s*(?P<agency>ICRA|CRISIL|CARE|IND|IND-RA|IND\s*RA|BWR|BRICKWORK|SMERA|ACUITE|INFOMERICS|FITCH|MOODY'?S)\s*\]?)"
)
# Agency name, optional separator, then an UPPERCASE symbol that is not the start of a word.
RATING_RE = re.compile(
    _AGENCY
    + r"[\s\-:]{0,3}"
    + r"(?P<rating>AAA|AA|A|BBB|BB|B|CCC|CC|C|D|SD)"
    + r"(?:\s?(?P<sign>[+\-]))?"
    + r"(?![A-Za-z0-9])"
    + r"(?:\s*\((?:SO|CE|Triple[^)]{0,20}|Pronounced[^)]{0,30})\))?"
    + r"(?:\s*[/(,;]\s*(?P<outlook>Stable|Negative|Positive|RWN|RWP|Rating\s+Watch[^)\s,;]*|Watch\w*|CreditWatch\w*)\)?)?"
)
# Short-term symbols after an agency name, e.g. "ICRA A1+", "CARE A4"; OCR reads the 1 as l / I
# ("CRISIL Al+"), normalised by _short_term()
SHORT_RE = re.compile(_AGENCY + r"[\s\-:]{0,3}(?P<st>A[1-4lI]\+?|P[1-4]\+?|PR[1-4]\+?)(?![A-Za-z0-9])")
# Fixed-deposit / medium-term scale: "BWR FAAA", "CRISIL FAA+", "ICRA MAA-"
DEPOSIT_RE = re.compile(_AGENCY + r"[\s\-:]{0,3}(?P<scale>[FM])(?P<rating>AAA|AA|A|BBB|BB|B|C|D)(?:\s?(?P<sign>[+\-]))?(?![A-Za-z0-9])")
WATCH_RE = re.compile(r"watch\s+with\s+(negative|positive|developing)\s+implications|rating\s+watch\s+(negative|positive)", re.I)
_LETTER_AGENCY = re.compile(r"\b(ICRA|CRISIL|CARE|BRICKWORK|Brickwork|ACUITE|SMERA|INFOMERICS|India Ratings)\b")
WITHDRAWN_RE = re.compile(r"rating\s+withdrawn|withdrawn\s+(?:the|its)\s+rating|withdrawal\s+of\s+(?:the\s+)?rating", re.I)

OUTLOOK_WORD = re.compile(r"\b(Stable|Negative|Positive)\s+outlook\b|\boutlook\s*(?:is|of|as)?\s*[:\-]?\s*(Stable|Negative|Positive)\b", re.I)
REVISION_RE = re.compile(r"\b(downgrad\w*|upgrad\w*|revis\w+|lowered|revised)\b", re.I)
DIRECTION_DOWN = re.compile(r"\bdowngrad\w*|\blowered\b", re.I)
DIRECTION_UP = re.compile(r"\bupgrad\w*", re.I)


def _norm_outlook(raw: str | None) -> str | None:
    if not raw:
        return None
    low = raw.lower()
    if low in ("rwn", "rwp") or "watch" in low:
        return "Watch"
    return low.capitalize()


def _agency_name(raw: str) -> str:
    a = raw.upper().replace("-", "").replace(" ", "")
    return {"INDRA": "IND", "INDIARATINGS": "IND", "BRICKWORK": "BWR"}.get(a, a)


def find_ratings(text: str) -> list[dict]:
    """All agency-qualified long-term rating mentions, in document order."""
    out = []
    for m in RATING_RE.finditer(text):
        sym = m.group("rating") + (m.group("sign") or "")
        if sym == "SD":
            sym = "D"
        if sym not in _RANK:
            continue
        out.append({
            "agency": _agency_name(m.group("agency")),
            "rating": sym,
            "outlook": _norm_outlook(m.group("outlook")),
            "pos": m.start(),
            "context": text[max(0, m.start() - 40): m.end() + 20].strip(),
        })
    return out


def extract_rating_and_outlook(text: str) -> tuple[str | None, str | None]:
    """Convenience wrapper: (long-term rating, outlook) or (None, None)."""
    picked = _choose(text)
    return (picked["rating"], picked["outlook"]) if picked else (None, None)


def _choose(text: str) -> dict | None:
    ratings = find_ratings(text)
    if not ratings:
        return None
    if REVISION_RE.search(text):
        chosen = ratings[-1]            # the revised rating is listed last
    else:
        chosen = max(ratings, key=lambda r: _RANK[r["rating"]])  # most conservative
    if chosen["outlook"] is None:
        # outlook stated in prose near the chosen rating
        window = text[chosen["pos"]: chosen["pos"] + 400]
        m = OUTLOOK_WORD.search(window)
        if m:
            chosen = {**chosen, "outlook": _norm_outlook(m.group(1) or m.group(2))}
    return chosen


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract(document: Document, company: Company, jsonl_path: Path, session: Session) -> CreditResult | None:
    """Extract the long-term rating from a document's chunk file; None when absent."""
    try:
        return _extract_impl(document, jsonl_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: Unexpected exception in credit_extractor.extract() "
            f"for document_id={document.id}: {exc!r}",
            file=sys.stderr,
        )
        return None


def _extract_impl(document: Document, jsonl_path: Path) -> CreditResult | None:
    try:
        chunks = load_chunks(jsonl_path)
    except ValueError as exc:
        print(f"WARNING: Malformed JSONL for document_id={document.id}: {exc}", file=sys.stderr)
        return None
    if not chunks:
        return None
    text = concatenate_chunk_text(chunks)
    if not text:
        return None

    chosen = _choose(text)
    if chosen is None:
        return _reference_only(text)

    all_ratings = [{k: v for k, v in r.items() if k != "pos"} for r in find_ratings(text)]
    extras: dict = {"agency": chosen["agency"]}
    if REVISION_RE.search(text):
        extras["revision_language"] = True
        if DIRECTION_DOWN.search(text):
            extras["stated_direction"] = "downgrade"
        elif DIRECTION_UP.search(text):
            extras["stated_direction"] = "upgrade"
    short = _short_term(text)
    if short:
        extras["short_term_ratings"] = short

    return CreditResult(
        credit_rating=chosen["rating"],
        rating_outlook=chosen["outlook"],
        rating_change_flag=None,
        instrument_ratings=all_ratings or None,
        agency=chosen["agency"],
        extras=extras,
    )


def _short_term(text: str) -> list[str]:
    """Short-term symbols in the text, OCR 'l' / 'I' read back as 1 ("Al+" -> "A1+")."""
    return sorted({re.sub(r"[lI]", "1", m.group("st")).upper() for m in SHORT_RE.finditer(text)})


def _reference_only(text: str) -> CreditResult | None:
    """
    A rating letter with no long-term rating (commercial-paper ratings, fixed-deposit scale,
    a withdrawal, or a downgrade announced without the new symbol).  Kept as reference
    signals for later models: credit_rating and rating_outlook stay NULL, so the rating rules
    and the rating-change post-pass ignore the row.  None when the text carries none of these.
    """
    extras: dict = {}
    short = _short_term(text)
    if short:
        extras["short_term_ratings"] = short
    deposit = sorted({f"{m.group('scale')}{m.group('rating')}{m.group('sign') or ''}" for m in DEPOSIT_RE.finditer(text)})
    if deposit:
        extras["deposit_ratings"] = deposit
    w = WATCH_RE.search(text)
    if w:
        extras["watch"] = (w.group(1) or w.group(2)).capitalize()
    if WITHDRAWN_RE.search(text):
        extras["rating_withdrawn"] = True
    if DIRECTION_DOWN.search(text):
        extras["stated_direction"] = "downgrade"
    elif DIRECTION_UP.search(text):
        extras["stated_direction"] = "upgrade"
    if not extras:
        return None
    # agency of a matched symbol; else a whole-word agency name ("IND" alone would hit "INDIA")
    sym = SHORT_RE.search(text) or DEPOSIT_RE.search(text)
    named = _LETTER_AGENCY.search(text)
    extras["agency"] = (_agency_name(sym.group("agency")) if sym
                        else _agency_name(named.group(1)) if named else None)
    extras["reference_only"] = True
    return CreditResult(credit_rating=None, rating_outlook=None, rating_change_flag=None,
                        instrument_ratings=None, agency=extras["agency"], extras=extras)


# ---------------------------------------------------------------------------
# Fallback: the company's own rating as disclosed in its annual report
# ---------------------------------------------------------------------------

_AR_RATING_PHRASE = re.compile(r"credit\s+ratings?", re.I)
# the rating is the company's own: facilities, instruments or the company itself
_AR_OWN_CUE = re.compile(
    r"bank\s+(?:loan\s+)?facilit|assigned|reaffirm|the\s+company|debentures?|\bNCDs?\b|commercial\s+paper|term\s+loans?",
    re.I,
)
# ratings of investments, deposits or counterparties are not the company's
_AR_OTHER_CUE = re.compile(r"invest|mutual\s+funds?|counterpart|deposits?\s+with|customers?", re.I)
_AR_WINDOW = 300


def extract_from_annual_report(pages: list[str]) -> CreditResult | None:
    """
    Long-term rating the company discloses in its annual report (Directors' Report /
    Corporate Governance "credit rating" paragraph).  Used only when a company has no
    credit_rating document and no rated announcement.

    Only text within _AR_WINDOW characters after a "credit rating" phrase is read, and only
    when it refers to the company's own borrowings and not to investments or counterparties.
    Returns None when no qualifying rating is found.
    """
    text = re.sub(r"\s+", " ", " ".join(pages))
    windows, last_end = [], -1
    for m in _AR_RATING_PHRASE.finditer(text):
        if m.start() < last_end:
            continue  # inside the previous window
        last_end = m.end() + _AR_WINDOW
        window = text[max(0, m.start() - 60): last_end]
        if _AR_OWN_CUE.search(window) and not _AR_OTHER_CUE.search(window) and find_ratings(window):
            windows.append(window)
    if not windows:
        return None
    joined = " | ".join(windows)
    chosen = _choose(joined)
    if chosen is None:
        return None
    return CreditResult(
        credit_rating=chosen["rating"],
        rating_outlook=chosen["outlook"],
        rating_change_flag=None,
        instrument_ratings=[{k: v for k, v in r.items() if k != "pos"} for r in find_ratings(joined)] or None,
        agency=chosen["agency"],
        extras={"agency": chosen["agency"], "source": "annual_report_disclosure"},
    )


# ---------------------------------------------------------------------------
# Change detection (post-pass)
# ---------------------------------------------------------------------------


def recompute_rating_changes(session: Session, company_id: int) -> int:
    """
    Set rating_change_flag and additional_signals['rating_direction'] on every
    credit row of a company, ordering by rating date (then document id) and
    comparing each rating with the previous one from the same agency (or, if that
    agency has no earlier rating, the previous rating overall).  The first rating
    has no prior and keeps flag NULL.  Returns the number of rows updated.
    """
    rows = session.execute(
        select(ExtractedFeature)
        .where(ExtractedFeature.company_id == company_id, ExtractedFeature.feature_type == "credit")
        .order_by(ExtractedFeature.period_end.asc().nullslast(), ExtractedFeature.document_id.asc())
    ).scalars().all()

    last_by_agency: dict[str | None, str] = {}
    last_any: str | None = None
    updated = 0
    for row in rows:
        rating = (row.credit_rating or "").strip().upper()
        extra = dict(row.additional_signals or {})
        agency = extra.get("agency")
        if rating not in _RANK:
            continue
        prior = last_by_agency.get(agency, last_any)
        if prior is None:
            flag, direction = None, None
        else:
            flag = rating != prior
            direction = None if not flag else ("downgrade" if _RANK[rating] > _RANK[prior] else "upgrade")
        extra["rating_direction"] = direction
        row.rating_change_flag = flag
        row.additional_signals = extra
        last_by_agency[agency] = rating
        last_any = rating
        updated += 1
    return updated
