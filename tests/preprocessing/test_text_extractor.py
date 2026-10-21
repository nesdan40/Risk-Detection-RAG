"""
Tests for src/preprocessing/text_extractor.py

Real (tiny) PDFs are generated with PyMuPDF so row reconstruction and per-page
OCR routing are exercised for real.  Only the OCR engine itself is mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from preprocessing import text_extractor
from preprocessing.text_extractor import EMPTY_TEXT_THRESHOLD, extract

LONG = "Statement of Profit and Loss for the year ended March 31, 2018 " * 2


def _make_pdf(path: Path, pages: list[list[tuple[float, float, str]] | None]) -> None:
    """Write a PDF. Each page is a list of (x, y, text) items; None = blank page."""
    doc = fitz.open()
    for items in pages:
        page = doc.new_page()
        for x, y, text in items or []:
            page.insert_text((x, y), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def _doc(rel: str, doc_type: str = "annual_report") -> MagicMock:
    d = MagicMock()
    d.file_path = rel
    d.document_type = doc_type
    return d


def _run(tmp_path: Path, pages, **kw):
    pdf = tmp_path / "x.pdf"
    _make_pdf(pdf, pages)
    return extract(_doc("x.pdf", kw.get("doc_type", "annual_report")), tmp_path)


# ---------------------------------------------------------------------------
# Digital PDFs
# ---------------------------------------------------------------------------


def test_digital_pdf_is_extracted_per_page(tmp_path):
    out = _run(tmp_path, [[(50, 100, LONG)], [(50, 100, LONG)]])
    assert out.outcome == "extracted"
    assert out.extraction_method == "pymupdf"
    assert out.page_count == 2 and len(out.pages) == 2
    assert out.ocr_pages == 0
    assert out.char_count >= EMPTY_TEXT_THRESHOLD


def test_table_row_stays_on_one_line_with_column_gaps(tmp_path):
    """The financial parser depends on 'label  value  value' staying one row."""
    out = _run(tmp_path, [[
        (50, 100, "Revenue from operations"), (300, 100, "1,234"), (400, 100, "1,000"),
        (50, 120, "Profit after tax"), (300, 120, "(56)"), (400, 120, "78"),
        (50, 60, LONG),
    ]])
    lines = out.pages[0].split("\n")
    assert any(l.startswith("Revenue from operations") and "1,234" in l and "1,000" in l for l in lines)
    assert any(l.startswith("Profit after tax") and "(56)" in l and "78" in l for l in lines)
    assert "  1,234" in out.pages[0]  # wide gap preserved as two spaces


def test_text_property_joins_pages(tmp_path):
    out = _run(tmp_path, [[(50, 100, LONG)], [(50, 100, LONG)]])
    assert out.text == "\n".join(out.pages)


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_missing_file_is_failed(tmp_path):
    out = extract(_doc("nope/ghost.pdf"), tmp_path)
    assert out.outcome == "failed"
    assert "File not found" in (out.error or "")
    assert out.pages == []


def test_corrupt_file_is_failed_not_raised(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"this is not a pdf at all")
    out = extract(_doc("bad.pdf"), tmp_path)
    assert out.outcome == "failed"
    assert out.extraction_method == "none"


# ---------------------------------------------------------------------------
# Scanned pages -> per-page OCR (OCR engine mocked)
# ---------------------------------------------------------------------------


def test_blank_page_is_ocrd(tmp_path):
    with patch.object(text_extractor, "_tesseract_ready", return_value=None), \
         patch.object(text_extractor, "_ocr_page", return_value=LONG) as ocr:
        out = _run(tmp_path, [None])
    assert out.outcome == "extracted"
    assert out.extraction_method == "ocr"
    assert out.ocr_pages == 1
    ocr.assert_called_once()


def test_table_documents_use_table_ocr_mode(tmp_path):
    seen = {}

    def fake_ocr(page, doc_type):
        seen["doc_type"] = doc_type
        return LONG

    with patch.object(text_extractor, "_tesseract_ready", return_value=None), \
         patch.object(text_extractor, "_ocr_page", side_effect=fake_ocr):
        _run(tmp_path, [None], doc_type="quarterly_result")
    assert seen["doc_type"] == "quarterly_result"


def test_mixed_document_only_ocrs_pages_without_text(tmp_path):
    with patch.object(text_extractor, "_tesseract_ready", return_value=None), \
         patch.object(text_extractor, "_ocr_page", return_value=LONG) as ocr:
        out = _run(tmp_path, [[(50, 100, LONG)], None])
    assert out.extraction_method == "mixed"
    assert out.ocr_pages == 1
    assert ocr.call_count == 1  # the digital page was not OCR'd


def test_ocr_unavailable_gives_low_yield_with_reason(tmp_path):
    """Previously a missing OCR engine was swallowed; now the reason is reported."""
    with patch.object(text_extractor, "_tesseract_ready", return_value="tesseract executable not found"):
        out = _run(tmp_path, [None])
    assert out.outcome == "low_yield"
    assert "tesseract" in (out.error or "")


def test_ocr_exception_gives_low_yield_with_reason(tmp_path):
    with patch.object(text_extractor, "_tesseract_ready", return_value=None), \
         patch.object(text_extractor, "_ocr_page", side_effect=RuntimeError("boom")):
        out = _run(tmp_path, [None])
    assert out.outcome == "low_yield"
    assert "boom" in (out.error or "")


def test_ocr_page_budget_marks_truncation(tmp_path):
    with patch.object(text_extractor, "OCR_MAX_PAGES", 1), \
         patch.object(text_extractor, "_tesseract_ready", return_value=None), \
         patch.object(text_extractor, "_ocr_page", return_value=LONG):
        out = _run(tmp_path, [None, None, None])
    assert out.ocr_pages == 1
    assert out.truncated is True
    assert "budget" in (out.error or "")
    assert out.page_count == 3
