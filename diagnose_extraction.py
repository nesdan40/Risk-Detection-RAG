"""
Diagnose why financial features come out NULL.

Run from the project root:
    python diagnose_extraction.py --processed data/processed --ocr-max-pages 30

For every {doc_id}_chunks.jsonl it reports:
  - chunk count / total chars (0 chunks => doc was marked "low_yield")
  - whether the statements that hold the numbers are present in the text
  - PDF page count vs OCR_MAX_PAGES (truncation check)
  - fraction of PDF pages with no text layer (scan check)
  - how many times key labels occur and what follows them (layout check)
Output: diagnosis.csv + console summary.
"""
import argparse, csv, json, re

import pandas as pd

import json, glob
from statement_parser import extract_statement

ok = fail = 0
for f in glob.glob("data/processed/*/*_chunks.jsonl"):
    text = " ".join(json.loads(l)["chunk_text"] for l in open(f, encoding="utf-8") if l.strip())
    r = extract_statement(text)
    if "revenue" in r and "pat" in r: ok += 1
    else:
        fail += 1
        if fail <= 10: print("MISS", f, len(text))
print(ok, "ok /", fail, "missed")

d = pd.read_csv("diagnosis.csv")
print(d.pdf_pages.notna().mean(), "share of rows with PDF stats")
print(d.groupby("doc_type")[["n_chunks","chars","digit_ratio"]].describe().T)
d["no_stmt"] = (d.pnl_hits==0)&(d.balance_sheet_hits==0)
print(d.groupby(["doc_type","no_stmt"]).size())
print(d[d.no_stmt].groupby("ticker").size())
print(d.groupby("doc_type")[["revenue_hits","pat_hits"]].agg(["mean","max"]))
print(d[(d.doc_type=="quarterly_result")].pat_sample.dropna().head(8).to_string())
print(d[(d.doc_type=="annual_report")].pat_sample.dropna().head(5).to_string())

from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

LABELS = {
    "revenue": r"(total income|revenue from operations|total revenue|net interest income)",
    "pat": r"(profit after tax|profit for the (year|period)|net profit)",
    "balance_sheet": r"balance sheet",
    "pnl": r"statement of profit (and|&) loss",
    "cash_flow": r"cash flows? (from|statement)",
}


def load_chunks(path: Path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pdf_stats(pdf_path: str, ocr_max_pages: int):
    if not fitz or not pdf_path or not Path(pdf_path).exists():
        return {}
    doc = fitz.open(pdf_path)
    n = len(doc)
    empty = sum(1 for p in doc if len(p.get_text("text").strip()) < 50)
    return {
        "pdf_pages": n,
        "pages_without_text_layer": empty,
        "scan_fraction": round(empty / n, 2) if n else None,
        "truncated_by_ocr_cap": n > ocr_max_pages and empty / max(n, 1) > 0.5,
    }


def label_context(text: str, pattern: str, width=120):
    """Return the first match plus the 120 chars after it (layout sample)."""
    m = re.search(pattern, text, re.I)
    if not m:
        return None, 0
    count = len(re.findall(pattern, text, re.I))
    snippet = text[m.start(): m.end() + width].replace("\n", " ⏎ ")
    return snippet, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--ocr-max-pages", type=int, default=30)
    ap.add_argument("--out", default="diagnosis.csv")
    args = ap.parse_args()

    rows = []
    for jf in sorted(Path(args.processed).glob("*/*_chunks.jsonl")):
        ticker = jf.parent.name
        doc_id = jf.name.replace("_chunks.jsonl", "")
        chunks = load_chunks(jf)
        text = "\n".join(c.get("chunk_text", "") for c in chunks)
        meta = chunks[0] if chunks else {}
        dtype = meta.get("document_type")
        if dtype not in (None, "annual_report", "quarterly_result"):
            continue  # only financial docs matter here

        row = {
            "ticker": ticker, "doc_id": doc_id, "doc_type": dtype,
            "fy": meta.get("fiscal_year"), "q": meta.get("fiscal_quarter"),
            "n_chunks": len(chunks), "chars": len(text),
            "digit_ratio": round(sum(ch.isdigit() for ch in text) / max(len(text), 1), 3),
            "file_path": meta.get("file_path"),
        }
        row.update(pdf_stats(meta.get("file_path"), args.ocr_max_pages))

        for name, pat in LABELS.items():
            snip, cnt = label_context(text, pat)
            row[f"{name}_hits"] = cnt
            row[f"{name}_sample"] = snip
        rows.append(row)

    if not rows:
        print("No chunk files found - check --processed path.")
        return

    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    empty = [r for r in rows if r["n_chunks"] == 0]
    no_stmt = [r for r in rows if r["n_chunks"] and not (r["pnl_hits"] or r["balance_sheet_hits"])]
    trunc = [r for r in rows if r.get("truncated_by_ocr_cap")]
    multi = [r for r in rows if r["pat_hits"] > 2]

    print(f"Financial docs analysed: {n}")
    print(f"  empty chunk file (low_yield):                 {len(empty)}")
    print(f"  text present but NO P&L / balance-sheet text: {len(no_stmt)}")
    print(f"  scanned & longer than OCR cap (truncated):    {len(trunc)}")
    print(f"  'profit after tax'-type label >2 times:       {len(multi)}  (ambiguity-guard risk)")
    print(f"Details written to {args.out}")


if __name__ == "__main__":
    main()
