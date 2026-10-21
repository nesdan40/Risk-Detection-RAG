"""
Per-page text extraction from PDFs.

Strategy (per page, so mixed digital/scanned documents work):
  1. PyMuPDF words, regrouped into visual rows (keeps table rows on one line).
  2. If a page has no usable text layer, OCR that page with Tesseract.  The page
     is rendered with PyMuPDF itself, so Poppler is not required.
  3. If PyMuPDF cannot open the file at all, fall back to pdfplumber (no OCR).

The result keeps one string per page.  Row structure (newlines between rows,
wide gaps between columns) is deliberate: the financial extractor parses
statement tables from it.  Chunking collapses whitespace separately.

Never re-raises: every failure is returned as outcome="failed" / "low_yield"
with the error text, so the caller can log it and retry on a later run.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from models.document import Document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMPTY_TEXT_THRESHOLD: int = 50      # whole-document minimum (stripped chars)
MIN_PAGE_CHARS: int = 30            # below this a page is treated as having no text layer
OCR_MAX_PAGES: int = int(os.environ.get("OCR_MAX_PAGES", "60"))
OCR_DPI: int = int(os.environ.get("OCR_DPI", "200"))
_GAP_PT: float = 14.0               # horizontal gap (points) that marks a column break
_TABLE_DOC_TYPES = ("annual_report", "quarterly_result")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ExtractionOutcome:
    text: str
    outcome: Literal["extracted", "low_yield", "failed"]
    char_count: int          # len(text.strip())
    error: str | None        # failure reason, or the OCR error behind a low_yield
    extraction_method: Literal["pymupdf", "pdfplumber", "ocr", "mixed", "none"]
    pages: list[str] = field(default_factory=list)   # one string per PDF page
    page_count: int = 0
    ocr_pages: int = 0
    truncated: bool = False  # True if the OCR page budget ran out


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------


def _page_rows(page) -> str:
    """Rebuild visual rows from PyMuPDF words: one line per row, wide gaps for columns."""
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    if not words:
        return ""
    words = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[tuple]] = []
    row_mid = None
    for w in words:
        mid = (w[1] + w[3]) / 2
        height = max(w[3] - w[1], 1.0)
        if row_mid is not None and abs(mid - row_mid) <= height * 0.5:
            rows[-1].append(w)
        else:
            rows.append([w])
            row_mid = mid
    lines = []
    for row in rows:
        row.sort(key=lambda w: w[0])
        parts = [row[0][4]]
        for prev, cur in zip(row, row[1:]):
            parts.append("  " if cur[0] - prev[2] >= _GAP_PT else " ")
            parts.append(cur[4])
        lines.append("".join(parts))
    return "\n".join(lines)


def _tesseract_ready() -> str | None:
    """Locate the tesseract binary; return an error string if it is unavailable."""
    import pytesseract  # noqa: PLC0415

    configured = os.environ.get("TESSERACT_CMD")
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
        return None
    if shutil.which("tesseract"):
        return None
    default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default)
        return None
    return "tesseract executable not found (install Tesseract or set TESSERACT_CMD)"


def _ocr_page(page, doc_type: str | None) -> str:
    """Render one page and OCR it. Raises on failure."""
    import pytesseract  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    pix = page.get_pixmap(dpi=OCR_DPI, colorspace="gray")
    image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    # psm 6 (uniform block) keeps statement-table rows on single lines.
    config = "--psm 6" if doc_type in _TABLE_DOC_TYPES else "--psm 3"
    return pytesseract.image_to_string(image, config=config)


def _extract_pages_pymupdf(abs_path: Path, doc_type: str | None):
    """Return (pages, ocr_pages, truncated, ocr_errors). Raises if the file can't be opened."""
    import fitz  # PyMuPDF  # noqa: PLC0415

    doc = fitz.open(str(abs_path))
    pages: list[str] = []
    ocr_pages, truncated, errors = 0, False, []
    try:
        ocr_unavailable = None
        for page in doc:
            text = _page_rows(page)
            if len(text.strip()) < MIN_PAGE_CHARS:
                if ocr_pages >= OCR_MAX_PAGES:
                    truncated = True
                else:
                    if ocr_unavailable is None:
                        try:
                            ocr_unavailable = _tesseract_ready() or ""
                        except ImportError as exc:
                            ocr_unavailable = f"Missing OCR dependency: {exc}"
                    if ocr_unavailable:
                        if ocr_unavailable not in errors:
                            errors.append(ocr_unavailable)
                    else:
                        try:
                            text = _ocr_page(page, doc_type)
                            ocr_pages += 1
                        except Exception as exc:  # noqa: BLE001
                            msg = f"OCR failed on page {page.number + 1}: {exc}"[:300]
                            if len(errors) < 3:
                                errors.append(msg)
            pages.append(text)
    finally:
        doc.close()
    return pages, ocr_pages, truncated, errors


def _extract_pages_pdfplumber(abs_path: Path) -> list[str]:
    import pdfplumber  # noqa: PLC0415

    with pdfplumber.open(str(abs_path)) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _failed(msg: str) -> ExtractionOutcome:
    return ExtractionOutcome(
        text="", outcome="failed", char_count=0, error=msg[:1000], extraction_method="none"
    )


def extract(document: Document, project_root: Path) -> ExtractionOutcome:
    """
    Extract per-page text from the PDF at project_root / document.file_path.

    outcome:
        "extracted" — at least EMPTY_TEXT_THRESHOLD characters found
        "low_yield" — file opened but (almost) no text, even after OCR; ``error``
                      carries the OCR failure reason when there was one
        "failed"    — file missing or unreadable by every backend
    """
    try:
        abs_path = project_root / Path(document.file_path)
    except Exception as exc:  # noqa: BLE001
        return _failed(str(exc))

    if not abs_path.exists():
        return _failed(f"File not found: {abs_path}")

    doc_type = getattr(document, "document_type", None)
    method: Literal["pymupdf", "pdfplumber", "ocr", "mixed", "none"]
    ocr_pages, truncated, errors = 0, False, []
    try:
        pages, ocr_pages, truncated, errors = _extract_pages_pymupdf(abs_path, doc_type)
        method = "pymupdf"
    except Exception as exc:  # noqa: BLE001
        try:
            pages = _extract_pages_pdfplumber(abs_path)
            method = "pdfplumber"
        except Exception as exc2:  # noqa: BLE001
            return _failed(f"PyMuPDF: {exc}; pdfplumber: {exc2}")

    if ocr_pages:
        method = "ocr" if ocr_pages == len(pages) else "mixed"

    text = "\n".join(pages)
    chars = len(text.strip())
    error_text = "; ".join(errors)[:1000] if errors else None
    if truncated:
        error_text = ((error_text + "; ") if error_text else "") + f"OCR page budget ({OCR_MAX_PAGES}) exhausted"

    return ExtractionOutcome(
        text=text,
        outcome="extracted" if chars >= EMPTY_TEXT_THRESHOLD else "low_yield",
        char_count=chars,
        error=error_text,
        extraction_method=method,
        pages=pages,
        page_count=len(pages),
        ocr_pages=ocr_pages,
        truncated=truncated,
    )
