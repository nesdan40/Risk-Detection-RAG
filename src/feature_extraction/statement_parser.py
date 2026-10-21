"""
Row-based parser for financial statements in per-page, row-preserving text.

Why not whole-document regexes: statements are tables.  The number sits on the
same row as its label, several columns wide (current period first), and the unit
("in lacs", "Rs. in crore", "in millions") appears once in the table header.
This module therefore works row by row:

    label  ->  numbers on the same row  ->  first column (current period)
    unit   ->  nearest header above the row  ->  converted to INR crore

All functions are pure (text in, values out) so they can be tested against real
extracted pages without a database.

Selection rules when a label occurs many times (annual reports repeat figures in
highlights, MD&A and both standalone/consolidated statements):
    1. rows on a page that carries the matching statement header win;
    2. consolidated beats unknown beats standalone;
    3. earlier label in the priority list beats a later one;
    4. earlier page wins.
Every chosen value records its provenance (page, scope, unit, matched label).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

_DATE_RES = [
    re.compile(r"\b\d{1,2}\s?[./-]\s?\d{1,2}\s?[./-]\s?\d{2,4}\b"),   # 31.12.2016
    re.compile(r"\b(?:19|20)\d{2}\s?[-–/]\s?(?:\d{4}|\d{2})\b"),        # 2016-17, 2016/2017
    re.compile(r"\bFY\s?'?\d{2,4}\b", re.I),
    re.compile(r"\b(?:note|notes|ref|refer)\b[^0-9\n]{0,12}\d+[a-z]?", re.I),
]
_NUM_RE = re.compile(r"(?<![\w.])\(?\s?-?\d[\d,]*(?:\.\d+)?\s?\)?(?![\w])")
_LEADING_SERIAL = re.compile(r"^[\s|.\-:;]*(?:\(?[ivx]{1,4}\)|\(?[a-z]\)|\d{1,2}[.)]?(?=\s*\|?\s+[A-Za-z]))[\s|.\-:;]*", re.I)
_YEARISH = re.compile(r"^(?:19|20)\d{2}$")
# "(3-4)", "(1+2)", "(X+XI)": column-formula hints printed next to labels
_FORMULA_RE = re.compile(r"\(\s*[A-Za-z0-9]{1,4}(?:\s*[+\-–−]\s*[A-Za-z0-9]{1,4})+\s*\)")
# bank results write it as bracketed row numbers joined by operators: "(12)-(13)", "(7)-(8)-(9)",
# "(3)·{6)"; read naively, "(12)" is a negative number
_CHAINED_FORMULA_RE = re.compile(r"[({]\s*[A-Za-z0-9]{1,4}\s*[)}](?:\s*[+\-–−·]\s*[({]\s*[A-Za-z0-9]{1,4}\s*[)}])+")
# the same hint after OCR mangled its brackets: "17-8]" for "(7-8)", "(7-Bl" for "(7-8)"
_OCR_FORMULA_RE = re.compile(r"(?<![\d,.])[\[({1lI|]?\d{1,2}[+\-–−][0-9B]{1,2}[\])}lI|](?![\d,.])")
# OCR reads a thousands comma as a point ("8.152" for "8,152"); only fixed in rows that
# also print comma-grouped figures, where a three-decimal value is otherwise implausible
_COMMA_GROUPED = re.compile(r"\d{1,3}(?:,\d{2,3})+")
_POINT_AS_COMMA = re.compile(r"(?<![\d.,])(\d{1,3})\.(\d{3})(?![\d.,])")
# footnote marker glued to the label: "Profit after tax(1)"
_FOOTNOTE_RE = re.compile(r"(?<=[A-Za-z])(?:\(\d{1,2}\))+")
# Ind AS note reference in its own column: "2.18", "2.3.1"
_DOTTED_NOTE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){1,2}$")


def parse_number(token: str) -> Decimal | None:
    """'(1,234.5)' -> -1234.5, '2,36,551' -> 236551, '-12' -> -12."""
    t = token.strip().replace(" ", "")
    neg = t.startswith("(") or t.startswith("-") or t.endswith(")")
    t = t.strip("()-").replace(",", "")
    if not t or t.count(".") > 1:
        return None
    try:
        val = Decimal(t)
    except InvalidOperation:
        return None
    return -val if neg else val


def row_numbers(text: str) -> list[Decimal]:
    """All numeric cells in a row fragment; dates, note refs, formulas and footnote marks removed."""
    for rx in _DATE_RES:
        text = rx.sub(" ", text)
    text = _CHAINED_FORMULA_RE.sub(" ", text)
    text = _FORMULA_RE.sub(" ", text)
    text = _OCR_FORMULA_RE.sub(" ", text)
    text = _FOOTNOTE_RE.sub("", text)
    if _COMMA_GROUPED.search(text):
        text = _POINT_AS_COMMA.sub(r"\1,\2", text)
    out = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0)
        digits = raw.strip("() -").replace(",", "")
        if _YEARISH.match(digits) and "," not in raw:
            continue
        v = parse_number(raw)
        if v is not None:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

# multiplier to convert a figure in this unit to INR crore
_UNIT_TO_CRORE = {
    "crore": Decimal(1), "crores": Decimal(1), "cr": Decimal(1),
    "lakh": Decimal("0.01"), "lakhs": Decimal("0.01"), "lac": Decimal("0.01"), "lacs": Decimal("0.01"),
    "million": Decimal("0.1"), "millions": Decimal("0.1"), "mn": Decimal("0.1"), "mm": Decimal("0.1"),
    "billion": Decimal(100), "billions": Decimal(100), "bn": Decimal(100),
    "thousand": Decimal("0.0001"), "thousands": Decimal("0.0001"),
}
_UNIT_TO_CRORE["rupees"] = Decimal("1E-7")
# 10 lakh crore: above any figure these companies report (a large bank's interest earned is
# ~1 lakh crore); rupees mistaken for crore overshoot it by orders of magnitude
_MAX_PLAUSIBLE_CRORE = Decimal(1000000)
_UNIT_RES = [
    re.compile(r"\bin\s+(?:rs\.?\s*|inr\s*)?(crores?|lakhs?|lacs?|millions?|billions?|thousands?|mn|bn)\b", re.I),
    re.compile(r"[(\[]\s*[^\w\s)\]]{0,3}\s*(?:rs\.?|inr)?\s*(?:in\s+)?(crores?|lakhs?|lacs?|millions?|billions?|mn|bn)\s*[)\]]", re.I),
    re.compile(r"(?:rs\.?|inr|[₹`@])\s*(?:in\s+)?(crores?|lakhs?|lacs?|millions?|billions?|mn|bn)\b", re.I),
    re.compile(r"\b(crores?|lakhs?|lacs?)\b\s*[)\]]", re.I),
    re.compile(r"(?:amount|figures?)\s+(?:are\s+)?in\s+(?:rs\.?\s*)?(crores?|lakhs?|lacs?|millions?)", re.I),
    re.compile(r"[₹`]\s?(?:in\s+)?['‘’]?000\b|\bin\s+['‘’]?000s?\b|\bin\s+thousands?\b", re.I),  # "` in ‘000"
]
# OCR reads the "l" of a unit word as J / I / 1: "in Jacs)", "in Iakhs"
_OCR_LAKH = re.compile(r"\b[JI1](acs?|akhs?)\b")
# Column header row made only of rupee symbols ("`  `  `"): figures are in absolute rupees
_RUPEE_COLUMNS = re.compile(r"^\s*[₹`](?:\s+[₹`]){0,5}\s*$")


def find_unit(lines: list[str], idx: int, lookback: int = 70) -> tuple[str | None, Decimal | None]:
    """Nearest unit declaration at or above line *idx* (then anywhere below within 25 lines)."""
    order = list(range(idx, max(idx - lookback, -1), -1)) + list(range(idx + 1, min(idx + 25, len(lines))))
    for i in order:
        line = _OCR_LAKH.sub(r"l\1", lines[i])
        if _RUPEE_COLUMNS.match(line):
            return "rupees", _UNIT_TO_CRORE["rupees"]
        for rx in _UNIT_RES:
            m = rx.search(line)
            if m:
                word = (m.group(1) if m.lastindex else "thousand").lower()
                if word in _UNIT_TO_CRORE:
                    return word, _UNIT_TO_CRORE[word]
    return None, None


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

# Optional "/(loss)" that follows "profit" in "Profit/(Loss) for the year" style labels.
_LOSS = r"(?:\s*/?\s*\(?\s*loss\s*\)?)?"

# Each metric: ordered list of regexes (earlier = preferred).
LABELS: dict[str, list[re.Pattern]] = {
    "revenue": [re.compile(p, re.I) for p in (
        r"revenue\s+from\s+operations(?!\s*\(?\s*(?:net|%))",
        r"(?:total\s+)?income\s+from\s+operations",
        r"net\s+sales(?:\s*/\s*income\s+from\s+operations)?",
        r"total\s+revenue(?!\s+expenses)",
        r"interest\s+earned",
        r"total\s+income",
    )],
    "pat": [re.compile(p, re.I) for p in (
        r"net\s+profit" + _LOSS + r"\s*after\s+(?:tax|taxes)(?:\s+and\s+minority\s+interest)?",
        r"profit" + _LOSS + r"\s*after\s+(?:tax|taxes)",
        r"net\s+profit" + _LOSS + r"\s*for\s+the\s+(?:period|year|quarter)",
        r"profit" + _LOSS + r"\s*for\s+the\s+(?:period|year|quarter)(?!\s+before)",
        r"net\s+profit\s*/\s*\(\s*loss\s*\)",
        r"net\s+profit(?!\s+before)",
    )],
    "pbt": [re.compile(p, re.I) for p in (
        r"profit" + _LOSS + r"\s*before\s+(?:exceptional\s+items\s+and\s+)?tax",
        r"profit\s+before\s+exceptional\s+items\s+and\s+tax",
        # pre-Ind AS results: "Profit/(Loss) from ordinary activities before tax (7+8)"
        r"profit" + _LOSS + r"\s*from\s+ordinary\s+activities\s+before\s+tax",
        r"profit" + _LOSS + r"\s*before\s*tax",  # OCR drops the space: "beforetax"
        r"loss\s+before\s+(?:exceptional\s+items\s+and\s+)?tax",  # "Loss before tax  (10,298.86)"
    )],
    "finance_cost": [re.compile(p, re.I) for p in (
        r"(?<!before\s)finance\s+costs?(?!\s+income)",  # not "PROFIT BEFORE FINANCE COSTS AND TAX"
        r"interest\s*(?:and|&)\s*finance\s*(?:cost|charges)s?",
        r"interest\s+expense[sd]?",
        r"interest\s+and\s+(?:other\s+)?financ\w+\s+charges",
    )],
    "tax_expense": [re.compile(r"(?:total\s+)?tax\s+expenses?(?!\s+of\s+discontinued)", re.I)],
    "depreciation": [re.compile(p, re.I) for p in (
        r"depreciation\s*(?:,|and|&)\s*amorti[sz]ation(?:\s+expenses?)?",
        r"depreciation(?:\s+and\s+impairment)?(?:\s+expenses?)?",
    )],
    "cfo": [re.compile(p, re.I) for p in (
        r"net\s+cash(?:\s+flows?)?.{0,50}?operat(?:ing|ions)(?:\s+activities)?",
        # after tax paid; must outrank "Cash Generated from Operations" (before tax) printed above it
        r"cash\s+inflow\s*/?\s*\(?\s*outflow\s*\)?\s*from\s+operat(?:ing|ions)(?:\s+activities)?",
        r"cash\s+(?:generated|flows?).{0,40}?operat(?:ing|ions)(?:\s+activities)?",
    )],
    "current_assets": [re.compile(r"total\s+current\s+assets", re.I)],
    "current_liabilities": [re.compile(r"total\s+current\s+liabilities", re.I)],
    "equity": [re.compile(p, re.I) for p in (
        r"total\s+equity(?!\s+and\s+liab)",  # also the misspelt "TOTAL EQUITY AND LIABLITIES"
        r"total\s+shareholders?['’]?s?\s*(?:funds?|equity)",
        r"shareholders?['’]?s?\s+funds?(?!\s*\()",
        r"net\s+worth",
    )],
    # equity components, summed when a balance sheet prints no total-equity row (old Schedule VI)
    "share_capital": [re.compile(r"(?:equity\s+)?share\s+capital", re.I)],
    "reserves": [re.compile(p, re.I) for p in (
        r"reserves\s+(?:and|&)\s+surplus",
        r"other\s+equity",
    )],
    "borrowings": [re.compile(p, re.I) for p in (
        r"(?:long|short)[-\s]term\s+borrowings?",
        r"(?:non[-\s]?current|current)\s+borrowings?",
        r"borrowings?(?:\s*\(other\s+than\s+debt\s+securities\))?",
        r"debt\s+securities",
        r"subordinated\s+liabilities",
        r"current\s+maturities\s+of\s+long[-\s]term\s+(?:debt|borrowings?)",
    )],
}
# ratio rows reported directly (no unit): (regex, plausible_min, plausible_max)
RATIO_LABELS = {
    "debt_to_equity": (re.compile(r"debt[\s\-/]*(?:to[\s\-]*)?equity(?:\s+ratio)?(?:\s*\(times\))?", re.I), Decimal("0"), Decimal("60")),
    # not "debt service coverage": DSCR includes principal repayment, a different ratio
    "interest_coverage": (re.compile(r"interest\s+(?:service\s+)?coverage\s+ratio|interest\s+cover(?:age)?(?:\s+ratio)?", re.I), Decimal("-100"), Decimal("1000")),
    "current_ratio": (re.compile(r"current\s+ratio(?:\s*\(times\))?", re.I), Decimal("0"), Decimal("50")),
}

_SCOPE_RE = re.compile(r"\b(consolidated|standalone|stand[-\s]alone)\b", re.I)
_SCOPE_CONTEXT = re.compile(
    r"(?:consolidated|standalone|stand[-\s]alone)\s+(?:audited\s+|unaudited\s+)?(?:statement|balance|financial|cash|profit|results|total|ind\s*as|summary)",
    re.I,
)
_HEADERS = {
    "pnl": re.compile(r"statements?\s+of\s+(?:consolidated\s+|standalone\s+)?(?:profit|income)|profit\s+and\s+loss\s+account|income\s+statement", re.I),
    "bs": re.compile(r"balance\s+sheet|statement\s+of\s+assets\s+(?:and|&)\s+liabilities", re.I),
    "cf": re.compile(r"cash\s+flows?\s+statement|statements?\s+of\s+cash\s+flows?", re.I),
}
_METRIC_HEADER = {
    "revenue": "pnl", "pat": "pnl", "pbt": "pnl", "finance_cost": "pnl", "depreciation": "pnl",
    "cfo": "cf", "current_assets": "bs", "current_liabilities": "bs", "equity": "bs", "borrowings": "bs",
    "share_capital": "bs", "reserves": "bs",
}
# The liabilities side of a balance sheet: "EQUITY AND LIABILITIES" heading.  Highlights pages and
# Directors' Report summaries mention "balance sheet" too, so debt-to-equity pairs rows only on these.
_BS_LIABILITIES = re.compile(r"equity\s+and\s+liabilities", re.I)
# Form AOC-1 "salient features of the financial statements of subsidiaries"
_SUBSIDIARY_STMT = re.compile(r"\bAOC\s*[-–]?\s*1\b|salient\s+features\s+of\s+the\s+financial\s+statements?\s+of\s+subsidiar", re.I)
# NSE website printout of a results filing (one value per row, figures from the exchange's data)
_NSE_PRINTOUT = re.compile(r"nsearchives\.nseindia\.com/archives/financial_results", re.I)
# Half-yearly balance sheet printed with Q2 / Q4 results
_ASSETS_LIABILITIES_STMT = re.compile(r"statement\s+of\s+assets\s+(?:and|&)\s+liabilities", re.I)
# Evidence that the liability section is complete on the page (needed before reading "no borrowings" as zero)
_BS_LIABILITY_TOTAL = re.compile(r"total\s+(?:non[-\s]?current\s+|current\s+)?liabilities|total\s+equity\s+and\s+liabilities", re.I)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    metric: str
    value: Decimal                 # first column, raw (in the table's own unit)
    second: Decimal | None         # second column (previous period for annual statements)
    page: int                      # 1-based
    line: int
    scope: str                     # consolidated | standalone | unknown
    unit: str | None
    factor: Decimal | None         # to INR crore
    label_rank: int
    on_statement_page: bool
    matched: str = ""
    n_numbers: int = 0             # numeric cells on the row (multi-year summaries have many)
    yoy: Decimal | None = None     # results only: same quarter of the previous year (by header dates)

    def crore(self, v: Decimal | None = None) -> Decimal | None:
        v = self.value if v is None else v
        return None if self.factor is None else (v * self.factor).quantize(Decimal("0.01"))


def _clean(line: str) -> str:
    line = line.replace("|", " ")
    return _LEADING_SERIAL.sub("", line, count=1)


def _scope_at(lines: list[str], li: int) -> str:
    """Scope of a row: nearest scope heading above it on the *same page*.

    Scope is never carried across pages: a heading on an earlier page says nothing
    reliable about a table on a later one (prose mentions flip it by accident).
    """
    for i in range(li, max(li - 45, -1), -1):
        raw = lines[i]
        m = _SCOPE_CONTEXT.search(raw) if len(raw) < 140 else None
        if m:
            # read the scope from this match: glued text ("ofConsolidated statement") has no
            # word boundary, so a separate \b-anchored search can miss it
            return "consolidated" if m.group(0).lower().startswith("consolidated") else "standalone"
    for raw in lines[:8]:  # page title zone
        if len(raw) < 100:
            m = _SCOPE_RE.search(raw)
            if m:
                return "consolidated" if m.group(1).lower() == "consolidated" else "standalone"
    return "unknown"


def _side_by_side(lines: list[str], li: int) -> bool:
    """True when a header just above the row names both scopes: 'Particulars  Standalone  Consolidated'."""
    for i in range(li, max(li - 15, -1), -1):
        low = lines[i].lower()
        if "standalone" in low.replace("stand-alone", "standalone") and "consolidated" in low and len(lines[i]) < 140:
            return True
    return False


# results column headers: "30.06.2016", "31,12.2013" (OCR), "30/09/2014"
_HEADER_DATE = re.compile(r"\b\d{1,2}\s?[.,/-]\s?(\d{1,2})\s?[.,/-]\s?((?:19|20)\d{2})\b")
_HEADER_YEARS = re.compile(r"^[\s|.\-]*(?:(?:19|20)\d{2}[\s|.\-]+){1,9}(?:19|20)\d{2}[\s|.\-]*$")


def _yoy_column(lines: list[str], li: int) -> int | None:
    """Index of the same-quarter-previous-year column for the results row at *li*.

    Uses the nearest column header above the row.  Full dates: the first column after the
    first whose month matches and whose year is one earlier.  Years only (Infosys): the
    standard third column, only when its year is exactly one earlier.  Otherwise None.
    """
    for i in range(li - 1, max(li - 15, -1), -1):
        dates = [(int(m.group(1)), int(m.group(2))) for m in _HEADER_DATE.finditer(lines[i])]
        if len(dates) >= 3:
            month, year = dates[0]
            return next((j for j, d in enumerate(dates[1:], start=1) if d == (month, year - 1)), None)
        if _HEADER_YEARS.match(lines[i]):
            years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", lines[i])]
            if len(years) >= 3:
                return 2 if years[2] == years[0] - 1 else None
    return None


def _scan(pages: list[str], metric: str, min_numbers: int, annual: bool) -> list[Candidate]:
    patterns = LABELS[metric]
    header = _HEADERS[_METRIC_HEADER[metric]] if metric in _METRIC_HEADER else None
    out: list[Candidate] = []
    for pno, page in enumerate(pages, start=1):
        if _SUBSIDIARY_STMT.search(page):
            continue  # Form AOC-1: subsidiaries' figures, not the company's
        lines = page.split("\n")
        on_stmt = (header.search(page) is not None) if (header and annual) else True
        for li, raw in enumerate(lines):
            scope = "unknown"  # resolved lazily, only for rows that match a label
            # strip glued footnote marks ("Profit after tax(1)(2)") before slicing off the label
            line = _FOOTNOTE_RE.sub("", _clean(raw))
            for rank, rx in enumerate(patterns):
                m = rx.search(line)
                if not m or m.start() > 14:
                    continue
                tail = line[m.end():]
                nums = row_numbers(tail)
                if not nums and li + 1 < len(lines):  # numbers wrapped onto the next row
                    nxt = lines[li + 1]
                    if not re.search(r"[A-Za-z]{3,}", nxt):
                        nums = row_numbers(nxt)
                first_tok = tail.split()[0] if tail.split() else ""
                if (annual and len(nums) >= 3 and abs(nums[0]) < 100
                        and nums[0] == nums[0].to_integral_value() and "," not in first_tok):
                    nums = nums[1:]  # note-reference column ("Revenue from operations 25 1,23,456 1,10,000")
                elif (annual and len(nums) >= 3 and _DOTTED_NOTE.match(first_tok)
                        and abs(nums[1]) >= 100 * abs(nums[0])):
                    nums = nums[1:]  # dotted note ref ("Revenue from operations  2.18  68,484  62,441")
                if len(nums) < min_numbers:
                    continue
                unit, factor = find_unit(lines, li)
                # A row printed in absolute rupees ("Net Profit after Tax (`)  8,763,853,100") can sit
                # under a page-level "crore" header from another table; a crore value above
                # Rs 1 lakh crore for these companies means the figures are really rupees.
                if factor is not None and abs(nums[0] * factor) > _MAX_PLAUSIBLE_CRORE                         and abs(nums[0]) >= 10 ** 7:
                    unit, factor = "rupees", _UNIT_TO_CRORE["rupees"]
                scope = _scope_at(lines, li)
                base = dict(metric=metric, page=pno, line=li, unit=unit, factor=factor,
                            label_rank=rank, on_statement_page=on_stmt,
                            matched=m.group(0).strip(), n_numbers=len(nums))
                if not annual:
                    j = _yoy_column(lines, li)
                    base["yoy"] = nums[j] if j is not None and j < len(nums) else None
                if len(nums) == 4 and _side_by_side(lines, li):
                    # [standalone cur, standalone prev, consolidated cur, consolidated prev]
                    base["n_numbers"] = 2  # two columns per scope
                    out.append(Candidate(value=nums[0], second=nums[1], scope="standalone", **base))
                    out.append(Candidate(value=nums[2], second=nums[3], scope="consolidated", **base))
                else:
                    out.append(Candidate(value=nums[0], second=nums[1] if len(nums) > 1 else None,
                                         scope=scope, **base))
                break
    return out


_SCOPE_RANK = {"consolidated": 0, "unknown": 1, "standalone": 2}


def _best(cands: list[Candidate], prefer_page: int | None = None, need_unit: bool = True) -> Candidate | None:
    pool = [c for c in cands if (c.factor is not None or not need_unit)]
    if not pool:
        return None
    return min(pool, key=lambda c: (
        not c.on_statement_page,
        c.n_numbers > 7,                 # ten-year summaries list the oldest year first
        _SCOPE_RANK[c.scope], c.label_rank,
        0 if prefer_page is None else (c.page != prefer_page), c.page, c.line,
    ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ParsedStatements:
    revenue: Decimal | None = None
    revenue_prev: Decimal | None = None          # annual: second column; results: same quarter last year
    pat: Decimal | None = None
    ebitda_margin_pct: Decimal | None = None
    debt_to_equity: Decimal | None = None
    interest_coverage: Decimal | None = None
    current_ratio: Decimal | None = None
    cfo: Decimal | None = None
    additional: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)  # metric -> provenance


def _src(c: Candidate, how: str = "row") -> dict:
    return {"page": c.page, "scope": c.scope, "unit": c.unit, "label": c.matched, "how": how}


def _ratio(pages: list[str], key: str) -> tuple[Decimal | None, dict | None]:
    rx, lo, hi = RATIO_LABELS[key]
    best = None
    for pno, page in enumerate(pages, start=1):
        for raw in page.split("\n"):
            line = _clean(raw)
            m = rx.search(line)
            if not m or m.start() > 14:
                continue
            # a ratio table row carries current and previous year; one number is prose
            # ("... a debt service coverage ratio of 1.25. The Company ...")
            nums = [n for n in row_numbers(line[m.end():]) if lo <= n <= hi]
            if len(nums) >= 2:
                cand = (pno, nums[0], m.group(0).strip())
                if best is None:
                    best = cand
                break
        if best:
            break
    if best is None:
        return None, None
    return best[1], {"page": best[0], "label": best[2], "how": "reported ratio"}


def _computed_debt_to_equity(pages: list[str]) -> tuple[Decimal | None, dict | None]:
    """Borrowings / equity from the liabilities side of one balance sheet page.

    Equity is the total-equity row, or share capital + reserves when the statement has no
    total row (Schedule VI "Shareholders' Funds").  Borrowings are summed from the same page
    and scope.  A complete liabilities section with no borrowings row, on a page titled
    "Balance Sheet", means no debt: 0.  A highlights summary never yields a zero.
    Both figures come from the same table, so the ratio needs no unit.
    """
    bs_pages = {pno for pno, page in enumerate(pages, start=1) if _BS_LIABILITIES.search(page)}
    if not bs_pages:
        return None, None

    def on_bs(metric: str) -> list[Candidate]:
        return [c for c in _scan(pages, metric, 2, True) if c.page in bs_pages]

    options: list[tuple[Candidate, Decimal, str]] = [(c, c.value, "total equity") for c in on_bs("equity")]
    reserves = on_bs("reserves")
    for cap in on_bs("share_capital"):
        # equal column counts: a dotted note ref ("Equity share capital  2.11  2,170  1,088") adds a column
        res = next((r for r in reserves if r.page == cap.page and r.scope == cap.scope and r.line > cap.line
                    and r.n_numbers == cap.n_numbers), None)
        if res is not None:
            options.append((cap, cap.value + res.value, "share capital + reserves"))
    options = [o for o in options if o[1] > 0]
    if not options:
        return None, None
    # a page titled "Balance Sheet" beats a highlights summary that also lists equity and liabilities
    anchor, equity, eq_how = min(options, key=lambda o: (
        not o[0].on_statement_page, _SCOPE_RANK[o[0].scope], o[2] != "total equity", o[0].page, o[0].line))

    borrowings = _scan(pages, "borrowings", 2, True)
    liab_page = anchor.page
    same_page = [c for c in borrowings if c.page == liab_page and c.scope == anchor.scope]
    # liabilities continued overleaf (equity at the foot of one page): use the next page only when
    # this one has neither borrowings nor a liabilities total and the next one carries that total
    if (not same_page and not _BS_LIABILITY_TOTAL.search(pages[liab_page - 1])
            and liab_page < len(pages) and _BS_LIABILITY_TOTAL.search(pages[liab_page])):
        liab_page += 1
        same_page = [c for c in borrowings if c.page == liab_page and c.scope in (anchor.scope, "unknown")]
    # sum distinct label ranks once per row to avoid double counting parent/child rows
    seen_lines, total = set(), Decimal(0)
    for c in sorted(same_page, key=lambda c: c.line):
        if c.line in seen_lines or c.value is None:
            continue
        seen_lines.add(c.line)
        total += c.value

    src = {"page": anchor.page, "scope": anchor.scope}
    if total > 0:
        return (total / equity).quantize(Decimal("0.0001")), {**src, "how": f"computed borrowings / {eq_how}"}
    if not same_page and anchor.on_statement_page and _BS_LIABILITY_TOTAL.search(pages[liab_page - 1]):
        return Decimal("0.0000"), {**src, "how": "balance sheet shows no borrowings"}
    return None, None


# "(2) Current assets", "Current Liabilities" — section headings (not "Other current assets", "Non-current ...")
_CURRENT_SECTION = re.compile(r"^\W*(?:\(\d\)\s*)?current\s+(assets|liabilities)\s*$", re.I)


def _section_subtotal(lines: list[str], start: int) -> Decimal | None:
    """First column of the unlabelled subtotal row closing the section headed at *start*."""
    for raw in lines[start + 1:start + 25]:
        line = _clean(raw)
        if re.match(r"\s*total\b", line, re.I):
            return None  # reached the statement total without a section subtotal
        if re.search(r"[A-Za-z]{3,}", line):
            continue
        nums = row_numbers(line)
        if len(nums) >= 2:
            return nums[0]
    return None


def _current_ratio_from_subtotals(pages: list[str]) -> tuple[Decimal | None, dict | None]:
    """Current assets / current liabilities from section subtotal rows on one balance sheet page.

    Schedule VI and some Ind AS statements print no "Total current assets" row: each section
    ends with an unlabelled subtotal ("99,281,229,489  95,946,280,276").  Both figures come
    from the same table, so the ratio needs no unit.
    """
    found = []
    for pno, page in enumerate(pages, start=1):
        if not (_HEADERS["bs"].search(page) and _BS_LIABILITIES.search(page)):
            continue
        lines = page.split("\n")
        sub: dict[str, Decimal] = {}
        for li, raw in enumerate(lines):
            m = _CURRENT_SECTION.match(raw.strip())
            if m and m.group(1).lower() not in sub:
                v = _section_subtotal(lines, li)
                if v is not None:
                    sub[m.group(1).lower()] = v
        if sub.get("assets") and sub.get("liabilities") and sub["liabilities"] > 0:
            found.append((_SCOPE_RANK[_scope_at(lines, 0)], pno, sub["assets"] / sub["liabilities"],
                          _scope_at(lines, 0)))
    if not found:
        return None, None
    _, pno, v, scope = min(found)
    return v.quantize(Decimal("0.0001")), {"how": "computed current assets / current liabilities subtotals",
                                            "page": pno, "scope": scope}


def _set_ratio(res: ParsedStatements, key: str, v: Decimal, s: dict, nbfc: bool) -> None:
    """Store a ratio as a rule input, or for lenders as reference only.

    Lenders' accounts do not fit the operating-company ratio rules: leverage of 6-10x is their
    business model, interest is their main cost (coverage of 1.1-1.3x is normal), their
    balance sheets are not split current / non-current, and loans disbursed count as operating
    cash outflows (a growing lender's CFO is routinely negative).  The figure is kept in
    additional_signals["{key}_lender"]; the rule input stays NULL (not evaluable).
    """
    if nbfc:
        res.additional[f"{key}_lender"] = str(v)
        res.sources[f"{key}_lender"] = {**(s or {}), "note": "lender: not a rule input"}
    else:
        setattr(res, key, v)
        res.sources[key] = s


def parse_statements(pages: list[str], doc_type: str, nbfc: bool) -> ParsedStatements:
    """Extract all typed financial signals from a document's per-page text."""
    annual = doc_type == "annual_report"
    # NSE website printouts of a filing ("financial_res_TCS_….html") print one value per row
    printout = not annual and any(_NSE_PRINTOUT.search(p) for p in pages)
    # annual: current+previous; results: at least 3 period columns (printouts: one)
    min_nums = 2 if annual else (1 if printout else 3)
    res = ParsedStatements()
    scans = {m: _scan(pages, m, min_nums, annual) for m in
             ("revenue", "pat", "pbt", "finance_cost", "depreciation", "cfo")}

    # Revenue is never negative; a negative first column is an OCR misread of a digit.
    scans["revenue"] = [c for c in scans["revenue"] if c.value is None or c.value > 0]
    # Finance cost is an expense; its sign is presentation only: "(6,027.20)" when expenses
    # are bracketed, or a stray OCR bracket ("27,164)").  Use the magnitude.
    for c in scans["finance_cost"]:
        if c.value is not None:
            c.value = abs(c.value)
    rev = _best(scans["revenue"])
    if rev is not None:
        res.revenue = rev.crore()
        res.sources["revenue"] = _src(rev)
        # previous year only from a plain current/previous row; wider rows are segment or
        # multi-period tables whose second column is something else (SpiceJet: 90,720.72  411.82 ...)
        if annual and rev.second and rev.n_numbers <= 3:
            res.revenue_prev = rev.crore(rev.second)
        elif not annual and rev.yoy and rev.yoy > 0:
            res.revenue_prev = rev.crore(rev.yoy)

    pat = _best(scans["pat"], prefer_page=rev.page if rev else None)
    if pat is not None:
        res.pat = pat.crore()
        res.sources["pat"] = _src(pat)

    cfo = _best(scans["cfo"])
    if cfo is not None:
        # lenders: loans disbursed and deposits taken are operating cash flows, so a growing
        # lender's CFO is routinely negative; kept for reference only
        _set_ratio(res, "cfo", cfo.crore(), _src(cfo), nbfc)

    # ----- EBITDA margin (not meaningful for lenders: interest is their cost of goods)
    pbt = _best(scans["pbt"], prefer_page=rev.page if rev else None)
    fin = _best(scans["finance_cost"], prefer_page=rev.page if rev else None)
    if printout and fin is not None:
        # NSE's FY2018 printout template repeats the tax expense in the finance-cost row
        tax = _best(_scan(pages, "tax_expense", 1, False), prefer_page=fin.page)
        if tax is not None and abs(tax.value) == fin.value:
            res.sources["finance_cost_rejected"] = {"page": fin.page, "value": str(fin.value),
                                                    "why": "equals tax expense (NSE printout defect)"}
            fin = None
    dep = _best(scans["depreciation"], prefer_page=rev.page if rev else None)
    if not nbfc and rev and pbt and fin and rev.factor and pbt.factor and fin.factor:
        revenue = rev.crore()
        ebitda = pbt.crore() + fin.crore() + (dep.crore() if dep and dep.factor else Decimal(0))
        if revenue and revenue > 0:
            res.ebitda_margin_pct = (ebitda / revenue * 100).quantize(Decimal("0.01"))
            res.sources["ebitda_margin_pct"] = {"how": "computed (PBT + finance cost + depreciation) / revenue",
                                                "page": pbt.page, "scope": pbt.scope}

    # ----- ratios: prefer what the company reports, else compute from the statements
    # Interest coverage is computed for results too (first column = current quarter); the
    # reported-ratio search stays annual-only, results rarely print one.
    v, s = _ratio(pages, "interest_coverage") if annual else (None, None)
    if v is None and not nbfc and pbt and fin and fin.factor and pbt.factor and fin.crore():
        v = ((pbt.crore() + fin.crore()) / fin.crore()).quantize(Decimal("0.0001"))
        s = {"how": "computed (PBT + finance cost) / finance cost", "page": pbt.page, "scope": pbt.scope}
    if v is not None:
        _set_ratio(res, "interest_coverage", v, s, nbfc)

    # Balance-sheet ratios: annual reports, and Q2/Q4 results that carry the half-yearly
    # "Statement of Assets and Liabilities".  Reported ratios are read from annual reports only.
    if annual or any(_ASSETS_LIABILITIES_STMT.search(p) for p in pages):
        v, s = _ratio(pages, "current_ratio") if annual else (None, None)
        if v is None and not nbfc:  # lenders' balance sheets aren't split current / non-current
            ca = _scan(pages, "current_assets", 2, True)
            cl = _scan(pages, "current_liabilities", 2, True)
            ca_b = _best(ca, need_unit=False)
            cl_b = _best(cl, prefer_page=ca_b.page if ca_b else None, need_unit=False)
            if ca_b and cl_b and cl_b.value:
                v = (ca_b.value / cl_b.value).quantize(Decimal("0.0001"))
                s = {"how": "computed total current assets / total current liabilities",
                     "page": ca_b.page, "scope": ca_b.scope}
            if v is None:
                v, s = _current_ratio_from_subtotals(pages)
        if v is not None:
            _set_ratio(res, "current_ratio", v, s, nbfc)

        v, s = _ratio(pages, "debt_to_equity") if annual else (None, None)
        if v is None:
            v, s = _computed_debt_to_equity(pages)
        if v is not None and Decimal(0) <= v <= Decimal(60):
            _set_ratio(res, "debt_to_equity", v, s, nbfc)

    # ----- lender extras
    if nbfc:
        for key, rx in (("gnpa_pct", r"gross\s+(?:npa|non[-\s]performing)"), ("net_npa_pct", r"net\s+(?:npa|non[-\s]performing)")):
            for page in pages:
                done = False
                for raw in page.split("\n"):
                    m = re.search(rx, raw, re.I)
                    if m and "%" in raw or (m and "ratio" in raw.lower()):
                        nums = [n for n in row_numbers(raw[m.end():]) if Decimal(0) <= n <= Decimal(100)]
                        if nums:
                            res.additional[key] = str(nums[0])
                            done = True
                            break
                if done:
                    break
    return res
