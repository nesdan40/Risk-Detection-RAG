# Risk-Detection-RAG

Final-year project: RAG-based financial and governance risk detection for Indian listed companies.
Spec: `.kiro/steering/PRD.md` (requirements), `.kiro/steering/schema.md` (DB). README.md has detail; do not duplicate it here.

## Stack
PostgreSQL (SQLAlchemy 2, Alembic 0001-0004) · Qdrant `financial_documents` (768-d, Cosine, BAAI/bge-base-en-v1.5) · rule-based engine · Streamlit UI · Ollama `qwen2.5:3b` (set in `.env`). Windows; Tesseract needed for OCR (Poppler no longer used).

## Pipeline (run with `PYTHONPATH=src`, one company at a time via `--company TICKER`)
1. `ingestion/run_ingestion.py` takes the **folder name** (`Infosys`, `JAYPEE`)
2. `preprocessing/run_extraction.py` text/OCR + chunks → `data/processed`, `data/extracted/{ticker}/{doc_id}_pages.json`
3. `embedding/run_embedding.py`
4. `feature_extraction/run_feature_extraction.py` (`--force` to redo financials; without it existing rows are skipped)
5. `risk_engine/run_risk_engine.py`
6. `streamlit run src/ui/app.py`

Later stages take the **ticker**. Folder `Infosys` = `INFY`, folder `JAYPEE` = `JPINFRATEC`.

## Key facts
- File names follow `tools/normalize_filenames.py` conventions (`_SCANNED` suffix = OCR). Do not rename files ad hoc.
- Financial parsing: `feature_extraction/statement_parser.py` (consolidated preferred; units converted to INR crore; provenance in `additional_signals["sources"]`).
- Rule thresholds in `risk_engine/rules.py` are the PRD values and are **frozen as the Semester 8 ML baseline**.
- `metadata.json` must never contain labels. `ground_truth.json` is for evaluation only.
- Known accepted limitation: some scanned-PDF digits are misread by OCR (see README).
- Stale and unused: `src/config/constants.py`, root `check_*.py`, `test_*.py`, `diagnose_extraction.py`.

## Workflow guardrails
- **Think before coding.** Read the relevant code and state the plan first. If the cause is unclear, find it before changing anything.
- **Ask, don't guess,** on decisions that are the user's: scope, thresholds, model choice, data deletion.
- **Stay in scope.** No unrelated refactors, renames, formatting sweeps or dependency changes. Mention other problems; do not fix them silently.
- **Smallest change** that solves the problem. Match surrounding style, naming and comment density.
- **No destructive actions without explicit approval:** deleting or overwriting data, resetting the DB or Qdrant, `--force`, moving files. Look at the target first.
- **Run only the companies requested.** Never run the full 14-company pipeline unprompted.
- **Never tune rules or extractors to the evaluation labels.** Fix bugs, not scores. Version any real rule change.
- **No label leakage.** Keep ground truth out of metadata, features and prompts.
- **Verify before claiming done.** Check DB rows, Qdrant counts or test output. Report failures and skipped steps plainly; do not hedge on verified results.
- **Background jobs:** unbuffered output to a log under `logs/`; no sleep-polling. Stop a runaway job rather than waiting on it.
- **Edit regex or escape-heavy code with the Edit/Write tools, not shell one-liners** (backslashes get corrupted).
- **Use a test database** (`financial_risk_test_db`); tests must never write to the real DB.
- Run `pytest` (about 830 tests, no external services needed) before finishing code changes. Update tests when a contract changes on purpose.
