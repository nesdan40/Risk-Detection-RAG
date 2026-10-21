"""
Normalise PDF and folder names under data/raw to the canonical scheme the
ingestion parser expects.

    Annual report : {TICKER}_AnnualReport_FY{YYYY}.pdf
    Quarterly     : {TICKER}_Q{n}FY{YYYY}[_SCANNED].pdf
    Announcement  : {TICKER}_Ann_{YYYYMMDD}_{Category}[ (N)][_SCANNED].pdf
    Credit rating : {TICKER}_Ann_{YYYYMMDD}_Credit Rating[_{Agency}][ (N)][_SCANNED].pdf

{TICKER} is the ticker in each company's metadata.json (folder names differ).
Dates may be YYYYMMDD or day-first DDMMYYYY.  Files whose date is not a clean
8-digit date are moved to data/quarantine/ rather than guessed.

Default is a dry run.  --apply performs the renames and writes
data/rename_log.csv (old -> new) so the change can be reversed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
QUARANTINE = ROOT / "data" / "quarantine"
LOG = ROOT / "data" / "rename_log.csv"

FOLDERS = {"annual": "annual_reports", "quarterly": "quarterly_results",
           "ann": "corporate_announcements", "credit": "credit_ratings"}

# lowercase alias (underscores -> spaces) -> filename category
_ALIASES = {
    "Outcome Of Board Meeting": ["outcome of board meeting", "outcome of bm", "obm",
                                 "ooutcome of board meeting", "board meeting"],
    "Financial Results Updates": ["financial results updates", "financial result updates",
                                  "fiancial results updates", "financial reults updates",
                                  "financialresults updates", "financial results",
                                  "financial results updates cl"],
    "Annual General Meeting Updates": ["annual general meeting", "annual general meeting updates",
                                       "agm", "agmupdates", "agm updates", "updates"],
    "Appointment": ["appointment", "appointmet", "directorappointed"],
    "Director Change": ["director change", "directorchange", "change in director",
                        "cessation", "change"],
    "Resignation": ["resignation"],
    "News Clarification": ["news clarification", "newsclarification", "clarification"],
    "News Verification": ["news verification", "newsverification"],
    "Press Release": ["press release", "press relase", "press releas", "pressrelease",
                      "press release updates", "press", "pr"],
    "Book Closure": ["book closure", "book clousure", "bookclosure"],
    "Acquisition": ["aquisition", "acquisition"],
    "Buyback": ["buyback"],
    "Settlement": ["settlement"],
    "Stay Order": ["stayorder", "stay order"],
    "Change In Auditors": ["change in auditors"],
}
ALIAS = {a: cat for cat, al in _ALIASES.items() for a in al}

_SCANNED = re.compile(r"[\s_]*SCANN?E+D(?:PDF)?", re.I)
_COPY = re.compile(r"[\s_]*\bcopy\b", re.I)
_DUP = re.compile(r"[\s_]*\((\d+)\)")
_DATE8 = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def _valid(y: int, m: int, d: int) -> bool:
    try:
        date(y, m, d)
        return 1990 <= y <= 2030
    except ValueError:
        return False


def parse_date8(s: str) -> str | None:
    """Return YYYYMMDD if *s* is unambiguously YYYYMMDD or DDMMYYYY, else None."""
    ymd = _valid(int(s[:4]), int(s[4:6]), int(s[6:]))
    dmy = _valid(int(s[4:]), int(s[2:4]), int(s[:2]))
    if ymd == dmy:
        return None
    return s if ymd else s[4:] + s[2:4] + s[:2]


def kind_of(folder: str) -> str | None:
    k = folder.lower()
    return ("annual" if "annual" in k else "quarterly" if "quarter" in k
            else "credit" if "credit" in k else "ann" if "announcement" in k else None)


def canonical(name: str, kind: str, ticker: str) -> str | None:
    """New file name, or None when the date can't be resolved safely."""
    stem = name[:-4]
    scanned = bool(_SCANNED.search(stem))
    if kind == "annual":
        m = re.search(r"FY(\d{4})", stem, re.I)
        return f"{ticker}_AnnualReport_FY{m.group(1)}.pdf" if m else None
    if kind == "quarterly":
        m = re.search(r"Q([1-4])_?FY(\d{2,4})", stem, re.I)
        if not m:
            return None
        fy = m.group(2) if len(m.group(2)) == 4 else "20" + m.group(2)
        return f"{ticker}_Q{m.group(1)}FY{fy}{'_SCANNED' if scanned else ''}.pdf"

    runs = _DATE8.findall(stem)
    if len(runs) != 1 or re.search(r"(?<!\d)(?:\d{6,7}|\d{9,})(?!\d)", stem):
        return None
    ymd = parse_date8(runs[0])
    if ymd is None:
        return None
    dup = _DUP.search(stem)
    rest = _DATE8.sub(" ", stem)
    rest = _SCANNED.sub("", rest)
    rest = _COPY.sub("", rest)
    rest = _DUP.sub("", rest)
    rest = re.sub(r"^[A-Za-z]+(?=[_\-\s])", "", rest, count=1)      # ticker prefix
    rest = re.sub(r"^[_\-\s.]*Ann(?=[_\-\s])", "", rest, flags=re.I)  # optional 'Ann'
    rest = re.sub(r"[_\-\s.]+", " ", rest).strip()
    low = rest.lower()
    if kind == "credit" or low.startswith(("credit rat", "creditrating")):
        agency = re.sub(r"^credit\s*rat(?:ing|ins)\s*", "", rest, flags=re.I).strip()
        cat = "Credit Rating" + (f"_{agency}" if agency else "")
    else:
        cat = ALIAS.get(low)
        if cat is None:
            return None
    suffix = f" ({dup.group(1)})" if dup else ""
    return f"{ticker}_Ann_{ymd}_{cat}{suffix}{'_SCANNED' if scanned else ''}.pdf"


def plan() -> tuple[list, list, list, list]:
    renames, quarantine, unchanged, folder_ops = [], [], 0, []
    used: set[str] = set()
    candidates = []  # (pdf, sub, target_dir, new_name)
    for cdir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        ticker = json.loads((cdir / "metadata.json").read_text(encoding="utf-8"))["ticker"]
        for sub in sorted(p for p in cdir.iterdir() if p.is_dir()):
            kind = kind_of(sub.name)
            if kind is None:
                continue
            target_dir = cdir / FOLDERS[kind]
            if sub.name != FOLDERS[kind]:
                folder_ops.append((sub, target_dir))
            for pdf in sorted(sub.glob("*.pdf")):
                new = canonical(pdf.name, kind, ticker)
                if new is None:
                    if kind in ("ann", "credit") and not re.search(r"[Aa]nnouncement$", pdf.stem):
                        quarantine.append(pdf)
                    else:
                        unchanged += 1
                    continue
                if new.lower() == pdf.name.lower():
                    new = pdf.name  # case-only differences: leave alone
                candidates.append((pdf, sub, target_dir, new))
    # Files that already carry their canonical name keep it; others yield on collision.
    candidates.sort(key=lambda c: c[0].name != c[3])
    for pdf, sub, target_dir, new in candidates:
        if True:
            if True:
                stem, n = new[:-4], 2
                while (str(target_dir / new)).lower() in used:     # collision -> next free (N)
                    base = re.sub(r"(_SCANNED)?$", "", stem)
                    sc = "_SCANNED" if stem.endswith("_SCANNED") else ""
                    new = f"{re.sub(r' \(\d+\)$', '', base)} ({n}){sc}.pdf"
                    n += 1
                used.add(str(target_dir / new).lower())
                if new == pdf.name and sub.name == target_dir.name:
                    unchanged += 1
                else:
                    renames.append((pdf, target_dir / new))
    return renames, quarantine, unchanged, folder_ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    renames, quarantine, unchanged, folder_ops = plan()
    print(f"unchanged: {unchanged} | renames/moves: {len(renames)} | to quarantine: {len(quarantine)} "
          f"| folder renames: {len(folder_ops)}")
    for q in quarantine:
        print("  QUARANTINE:", q.relative_to(RAW))
    for o, n in renames[:400 if not args.apply else 0]:
        if o.name != n.name:
            print(f"  {o.parent.parent.name}: {o.name}  ->  {n.name}")
    if not args.apply:
        print("\n(dry run - nothing changed)")
        return 0
    rows = []
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    for q in quarantine:
        dest = QUARANTINE / q.parent.parent.name / q.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(q), str(dest))
        rows.append((str(q), str(dest), "quarantine"))
    # files first (still in old folders), then folders (two-step so case-only renames work on Windows)
    for o, n in renames:
        n.parent.mkdir(parents=True, exist_ok=True)
        if n.exists() and n.resolve() != o.resolve():
            print("REFUSING to overwrite", n)
            return 1
        if o.resolve() != n.resolve() or o.name != n.name:
            tmp = o.with_name(o.name + ".renaming")
            o.rename(tmp)
            shutil.move(str(tmp), str(n))
            rows.append((str(o), str(n), "rename"))
    for sub, target in folder_ops:
        if sub.name.lower() == target.name.lower():
            # case-only difference: same directory on Windows, so rename via a temp name
            tmp = sub.with_name(sub.name + ".tmp_rename")
            sub.rename(tmp)
            tmp.rename(target)
        elif sub.exists() and not any(sub.iterdir()):
            sub.rmdir()
    with LOG.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([("old", "new", "action"), *rows])
    print(f"applied {len(rows)} changes; log: {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
