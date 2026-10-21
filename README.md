# RAG-Based Financial & Governance Risk Detection System

> An academic Final-Year Engineering Project — Semester 7 (core pipeline) through Semester 8 (benchmarking & ML evaluation)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Database Schema](#3-database-schema)
4. [Data Collection Methodology](#4-data-collection-methodology)
5. [Pipeline Components](#5-pipeline-components)
   - [Spec 1 — Document Ingestion Pipeline](#spec-1--document-ingestion-pipeline)
   - [Spec 2 — PDF Text Extraction & Chunking](#spec-2--pdf-text-extraction--chunking)
   - [Spec 3 — Embedding & Qdrant Vector Storage](#spec-3--embedding--qdrant-vector-storage)
   - [Spec 4 — Structured Feature Extraction](#spec-4--structured-feature-extraction)
   - [Spec 5 — Rule-Based Risk Engine](#spec-5--rule-based-risk-engine)
   - [Spec 6 — Streamlit Demo UI](#spec-6--streamlit-demo-ui)
6. [Technology Stack & Justification](#6-technology-stack--justification)
7. [Governance & Financial Signal Definitions](#7-governance--financial-signal-definitions)
8. [Rule-Based Risk Engine — Deep Dive](#8-rule-based-risk-engine--deep-dive)
9. [Known Challenges & How They Were Solved](#9-known-challenges--how-they-were-solved)
10. [Project Folder Structure](#10-project-folder-structure)
11. [How to Run the Pipeline](#11-how-to-run-the-pipeline)
12. [Semester 8 Roadmap](#12-semester-8-roadmap)

---

## 1. Project Overview

### What This Project Is

This system is a **Retrieval-Augmented Generation (RAG) based financial risk detection platform** that analyzes publicly available financial disclosures of Indian listed companies to identify early warning signals of financial distress. It combines structured signal extraction from unstructured filings (PDFs), an expert-defined rule-based risk scoring engine, and an evidence-backed reporting interface where every risk claim is traceable to a specific source document and passage.

The system was designed around a core principle: **every risk flag must be explainable and citable**. There are no black-box scores — each output points back to the exact filing that supports it.

### The Problem It Solves

Financial distress in listed companies — defaults, credit downgrades, governance failures — is almost never a sudden event. The signals are there months in advance, scattered across annual reports, quarterly filings, board meeting announcements, and shareholding data. The challenge is that these signals are:

- **Unstructured**: buried in long PDF documents with inconsistent formatting
- **Dispersed**: spread across four different document types from multiple sources
- **Rarely synthesized**: no single tool correlates financial deterioration with governance red flags and credit rating trends simultaneously

This system retrieves, extracts, and reasons over these disclosures to produce a transparent, cited risk assessment — designed to flag distress well before it becomes public knowledge through news articles or default announcements.

### Two-Semester Scope

**Semester 7 — Core Pipeline (current phase)**
Build the complete end-to-end pipeline: document ingestion, text extraction, chunking, vector embedding, structured feature extraction, rule-based risk scoring, evidence retrieval, and a functional web interface. Validated on 14 currently available companies (7 distressed, 7 healthy).

**Semester 8 — Benchmarking & ML Evaluation**
Expand the benchmark dataset to 30 companies (15 distressed, 15 healthy). Run historical backtesting using only pre-event data for distressed companies. Train an XGBoost classifier on the structured feature store. Produce a quantitative comparison of the rule-based engine vs. the ML model across Recall, Precision, F1 Score, False Positive Rate, and Lead Time.

---

## 2. Architecture Overview

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FINANCIAL DISCLOSURES (PDFs)                       │
│  Annual Reports │ Quarterly Results │ Corp. Announcements │ Ratings     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SPEC 1 — DOCUMENT INGESTION                                             │
│  • Reads metadata.json per company (NO ground truth ever read here)      │
│  • Registers companies, documents, shareholding patterns in PostgreSQL   │
│  • Sets text_extracted=FALSE, chunked=FALSE, embedded=FALSE              │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SPEC 2 — TEXT EXTRACTION & CHUNKING                                     │
│  • PyMuPDF → pdfplumber → Tesseract OCR  (3-level cascade per PDF)      │
│  • Splits text into overlapping ~800-token chunks (3200 chars / 400 overlap) │
│  • Outputs JSONL files: data/processed/{TICKER}/{doc_id}_chunks.jsonl   │
│  • Updates text_extracted=TRUE, chunked=TRUE in PostgreSQL               │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                       ┌───────┴───────┐
                       │               │
                       ▼               ▼
   ┌───────────────────────────┐  ┌──────────────────────────────────────┐
   │  SPEC 3 — EMBEDDING       │  │  SPEC 4 — FEATURE EXTRACTION         │
   │  • BAAI/bge-base-en-v1.5  │  │  • Phase A: Normalize announcement   │
   │  • 768-dim embeddings      │  │    categories (3-layer approach)     │
   │  • Batch size 32           │  │  • Phase B: Financial ratios from    │
   │  • Upsert into Qdrant      │  │    annual/quarterly PDFs             │
   │    (collection:            │  │  • Phase C: Governance counts from   │
   │    "financial_documents")  │  │    announcements                     │
   │  • Sets embedded=TRUE      │  │  • Phase D: Credit ratings           │
   └───────────────┬───────────┘  │  • Writes typed rows to PostgreSQL   │
                   │              │    extracted_features table           │
                   │              └──────────────────┬───────────────────┘
                   │                                 │
                   │              ┌──────────────────▼───────────────────┐
                   │              │  SPEC 5 — RULE-BASED RISK ENGINE      │
                   │              │  • Evaluates 15 expert-defined rules  │
                   │              │  • Computes risk_score (0.0–1.0)      │
                   │              │  • Assigns risk_level (Low/Med/High/  │
                   │              │    Critical)                          │
                   │              │  • Retrieves evidence from Qdrant     │
                   │              │  • Writes predictions to PostgreSQL   │
                   └──────────────▼───────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SPEC 6 — STREAMLIT WEB INTERFACE                                        │
│  • Company selection → Risk report dashboard                             │
│  • Triggered rules with cited source documents                           │
│  • Ad-hoc Q&A via RAG: embed → Qdrant retrieval → Ollama (Qwen2.5) LLM  │
└──────────────────────────────────────────────────────────────────────────┘

Data Stores:
  PostgreSQL  — structured metadata, features, predictions, processing logs
  Qdrant      — vector embeddings of document chunks (semantic retrieval)
  Filesystem  — original PDFs (data/raw/) and chunk JSONL files (data/processed/)
```

### Why This Architecture Was Chosen

**Why separate PostgreSQL from Qdrant?**
These two stores solve fundamentally different problems. PostgreSQL holds *structured, typed* data — financial ratios, governance counts, risk scores — that the rule engine reads directly using SQL queries (filtering by company, fiscal period, feature type). Qdrant holds *unstructured text embeddings* that support semantic search: "find passages about cash flow problems for company X." Merging them into a single store would mean either losing the typing and query expressiveness of a relational database, or losing the approximate nearest-neighbor search capability of a vector store. Keeping them separate lets each do what it does best.

**Why a rule-based engine before an ML model?**
At the dataset scale available in Semester 7 (14 companies), there is not enough data to train a reliable ML classifier. More importantly, a rule-based system produces *explainable outputs by construction* — each triggered rule maps to a specific threshold on a specific extracted feature, with a cited source document. This explainability is a core project requirement and a meaningful academic contribution in its own right. The ML layer (Semester 8) will be evaluated against this rule-based baseline, not replace it.

**Why a local LLM via Ollama?**
The project runs entirely on local infrastructure (no cloud dependency). Sending financial disclosure text to an external API raises data privacy concerns and incurs cost. Ollama allows running a capable local model (Qwen2.5) for the RAG Q&A component without any network dependency. For an academic prototype, the quality tradeoff is acceptable.

**Why Streamlit for the UI?**
The project's non-functional requirements explicitly exclude production deployment, multi-user access control, and scalability. Streamlit delivers a functional, interactive demo in far less engineering time than a React/FastAPI stack would require, which is the right tradeoff for a Semester 7 deadline.

---

## 3. Database Schema

The schema is defined in `src/models/` as SQLAlchemy ORM models with three Alembic migrations. All six tables are explained below in plain language alongside their technical purpose.

### Label-Leakage Prevention — The Core Design Constraint

This is the most important design decision in the entire schema. Every table here contains **zero information about whether a company is distressed or healthy**. That information lives exclusively in a separate file (`data/raw/{TICKER}/ground_truth.json`) which is never read by any ingestion, extraction, or prediction pipeline stage — only by the Semester 8 evaluation scripts.

Why does this matter? If the prediction pipeline had access to the distress label (even indirectly, through a column that correlates with it), any performance metrics measured in backtesting would be inflated — the model would have "seen the answer" during training or prediction. Enforcing this separation at the schema level means it is structurally impossible to leak the label, not just a matter of being careful.

The distinction between the two metadata files is:
- `metadata.json` — company name, sector, listing type, document paths, shareholding percentages. **This is the only file the pipeline reads.**
- `ground_truth.json` — distress/healthy label and the date of the distress event. **Never read by any code in `src/`.**

This is enforced at multiple layers: `metadata_reader.py` warns and strips any prohibited field names (`group`, `distressed`, `healthy`, `label`, `ground_truth`) even if someone accidentally added them to `metadata.json`. Feature writers and the RAG query module also have explicit guards against these field names entering the database or LLM prompts.

---

### Table: `companies`

**Plain-language purpose**: One row per company. The starting point of the entire system. Stores who the company is, what kind of entity it is, and the time window over which documents were collected.

The `listing_type` column distinguishes equity-listed companies (the primary target — analyzed by all rules) from `debt_only` entities, which are non-equity listed but sometimes included because their financial filings are informative and publicly available via the MCA portal.

The `collection_window_start` and `collection_window_end` dates define the observation window — for distressed companies in Semester 8, the window ends at the ground-truth distress date to prevent using post-event documents.

```
id                       Primary key
ticker                   Unique exchange ticker (e.g., 'DHFL', 'RCOM')
name                     Full company name
sector                   Sector (e.g., 'Housing Finance', 'Telecom')
listing_type             'equity' or 'debt_only'
collection_window_start  First date documents were collected from
collection_window_end    Last date documents were collected from
created_at               Row creation timestamp
```

---

### Table: `documents`

**Plain-language purpose**: One row per PDF file. Tracks what the document is, where it came from, what it covers, and which processing stages have been applied to it. The three boolean flags (`text_extracted`, `chunked`, `embedded`) act as a pipeline state machine — each pipeline stage queries for documents where its own flag is `FALSE`, does its work, and sets the flag to `TRUE`. This makes every stage re-entrant: a crashed run can be restarted and will pick up exactly where it left off.

The `document_category` field is specific to corporate announcements, where BSE/NSE filings carry a category label (Board Meeting, Director Change, Resignation, etc.). These labels are often inconsistent in practice and require normalization — see Spec 4 for how that is handled.

```
id                  Primary key
company_id          Foreign key to companies
file_name           Original PDF filename
file_path           Relative path from project root
document_type       'annual_report', 'quarterly_result', 'corporate_announcement', 'credit_rating'
document_category   Announcement sub-type (Board Meeting, Director Change, etc.) — NULL for non-announcements
fiscal_year         'FY2017', 'FY2018', etc.
fiscal_quarter      'Q1'–'Q4' — NULL for annual documents
period_start        Calendar start date of the covered period
period_end          Calendar end date of the covered period
source              'BSE', 'NSE', 'company_ir', 'MCA'
text_extracted      Has text been extracted from this PDF? (pipeline flag)
chunked             Has the extracted text been split into chunks? (pipeline flag)
embedded            Have the chunks been embedded into Qdrant? (pipeline flag)
created_at          Row creation timestamp
```

---

### Table: `processing_log`

**Plain-language purpose**: A full audit trail of every processing event across the pipeline. While the boolean flags on `documents` tell you the *current state* (is this document chunked or not?), the processing log tells you the *full history* (when was chunking started, did it succeed, was there an error, was it retried?). This separation is essential for debugging — when a document fails text extraction or produces zero chunks, the log entry contains the exact error message, allowing targeted diagnosis without having to re-run the pipeline.

Every pipeline stage (ingestion, text extraction, chunking, embedding, feature extraction, risk scoring) writes `started`, `success`, or `failed` events here. Log inserts use database savepoints so a logging failure never aborts the underlying processing work.

```
id            Primary key
document_id   Foreign key to documents (nullable — some events are company-level, not doc-level)
stage         'text_extraction', 'chunking', 'embedding', 'feature_extraction', 'risk_scoring'
status        'started', 'success', 'failed'
message       Error detail when failed; progress detail when successful
created_at    Event timestamp
```

---

### Table: `shareholding_patterns`

**Plain-language purpose**: Structured shareholding data, one row per company per fiscal quarter. This table is kept separate from `extracted_features` because shareholding data comes from manually verified values in `metadata.json`, not from PDF text extraction. Keeping the data source separation clean matters: if shareholding data were mixed into `extracted_features`, it would be ambiguous whether a given row came from reliable manual entry or potentially noisy PDF parsing.

The two key metrics — `promoter_holding_pct` (how much of the company the founding promoters own) and `pledge_pct` (what percentage of that holding has been pledged as loan collateral) — are primary distress indicators. Rising pledge percentages signal financial pressure on promoters; falling promoter holdings can signal loss of confidence.

A deliberate convention: `pledge_pct = 0.0` when the source data shows `"-"` (meaning the company has disclosed no pledges). It is stored as `NULL` only when the data is genuinely missing. This distinction matters for the rule engine, which treats `NULL` as "data not available" and skips the rule, rather than misinterpreting zero pledges as normal.

```
id                    Primary key
company_id            Foreign key to companies
fiscal_year           'FY2017', etc.
fiscal_quarter        'Q1'–'Q4'
period_end            Quarter end date
promoter_holding_pct  Promoter ownership percentage
pledge_pct            Percentage of promoter holding pledged as collateral
created_at            Row creation timestamp
```

---

### Table: `extracted_features`

**Plain-language purpose**: The central feature store for the entire project — both the rule-based engine in Semester 7 and the ML model in Semester 8 read from here. Every quantitative signal extracted from company filings ends up in this table.

The design uses typed columns for all known signals (so downstream SQL queries and ML training can read them directly without text parsing), plus a `JSONB` overflow column (`additional_signals`) for signals discovered later that don't fit the fixed schema — for example, NBFC-specific financial metrics that don't apply to regular companies.

The `document_id` column is nullable by design. Financial and credit features are document-specific (tied to one annual report or one credit rating document). Governance features are period aggregates — they represent counts across *all* corporate announcements in a quarter and are not tied to any single document, so `document_id` is `NULL` for governance rows.

Three partial unique indexes enforce idempotent upserts: running feature extraction twice produces the same database state, not duplicate rows.

```
id                  Primary key
company_id          Foreign key to companies
document_id         Foreign key to documents — NULL for period-level governance aggregates
fiscal_year         The fiscal year this feature covers
fiscal_quarter      NULL for annual-level features, 'Q1'–'Q4' for quarterly
period_end          End date of the covered period
feature_type        'financial', 'governance', 'credit', 'shareholding'

-- Financial signals (NULL when not found in the document)
revenue, revenue_growth_pct, pat, ebitda_margin_pct,
debt_to_equity, interest_coverage, current_ratio, cfo

-- Governance signals (counts per period)
board_meeting_count, director_change_count, appointment_count,
resignation_count, agm_delay_flag, clarification_count, press_release_count

-- Credit signals
credit_rating, rating_outlook, rating_change_flag

-- Overflow
additional_signals  JSONB — anything that doesn't fit the typed columns above

created_at          Row creation timestamp
```

---

### Table: `predictions`

**Plain-language purpose**: The output of the risk scoring pipeline. One row per company per fiscal period per model version. The `model_version` column (`'rule_based_v1'`, later `'xgboost_v1'`) allows both the rule-based and ML outputs to coexist in the same table without any schema changes in Semester 8.

`risk_flags` and `evidence_chunks` are stored as JSONB because their structure varies per prediction — a company with three triggered rules has a different flags structure than one with eight. Keeping them in JSONB means the fixed relational schema doesn't need to accommodate a variable number of rules.

```
id                  Primary key
company_id          Foreign key to companies
fiscal_year         The scored fiscal year
fiscal_quarter      The scored fiscal quarter
risk_score          Normalized score from 0.0 (no risk) to 1.0 (maximum risk)
risk_level          'Low', 'Medium', 'High', or 'Critical'
risk_flags          JSONB — which rules fired, with triggered values and thresholds
evidence_chunks     JSONB — Qdrant chunk references supporting each triggered rule
model_version       'rule_based_v1' (Sem 7), 'xgboost_v1' (Sem 8), etc.
created_at          Prediction timestamp
```

---

## 4. Data Collection Methodology

### Document Types and Why These Four

**Annual Reports** — 3 per company (one per fiscal year). The primary source for financial statements: revenue, profit after tax, EBITDA, debt levels, cash flow from operations. Annual reports also contain auditor observations, notes to accounts, and management commentary — all useful for identifying qualitative risk signals. The Business Responsibility Report (BRR) section is excluded from collection as it is not informative for financial distress signals.

**Quarterly Results** — up to 12 per company (one per fiscal quarter, three fiscal years). Quarterly data catches deterioration that annual aggregates hide. A company could have flat annual revenue while showing a clear quarterly downtrend. Consolidated quarterly results are preferred over standalone (subsidiary-only) results because consolidated figures reflect the full economic group.

**Corporate Announcements** — variable count, sourced from the stock exchange disclosure portal. This is the governance signal source. The system collects nine announcement categories: Board Meeting, Financial Results, AGM Updates, Press Release, Director Change, Appointment, Resignation, News Clarification, and News Verification. Collectively these capture board activity, leadership stability, and how actively a company is responding to media scrutiny — all signals that often deteriorate before financial distress becomes apparent.

**Shareholding Patterns** — manually extracted quarterly data on promoter holding percentage and pledge percentage. These are sourced from `metadata.json` rather than PDFs because shareholding disclosures are highly structured and better entered manually than parsed from PDFs.

### Source Hierarchy and Why

**Primary: BSE / NSE** — The stock exchange disclosure portals are the most complete and authoritative source. Every listed company is required to file corporate announcements, quarterly results, and shareholding patterns here. BSE is preferred where both exchanges carry the same document (to avoid duplicates).

**Fallback: Company Investor Relations websites** — Some annual reports are only available on the company's own IR page. This is a secondary fallback, not a primary source, because IR pages are sometimes incomplete, behind login walls, or have documents removed after distress events.

**Fallback: MCA Portal** — Used only for entities that are debt-listed (bonds/debentures) but not equity-listed. These companies are not on BSE/NSE's equity disclosure portal, so the Ministry of Corporate Affairs filing portal is the only available source.

### What Was Explicitly Excluded and Why

**SEBI orders** — SEBI investigation orders and enforcement actions are powerful indicators of distress, but they are published retrospectively (often well after the distress event) and their availability is uneven across companies. Including them would create an inconsistent feature set and risk introducing post-event information for distressed companies.

**Rating agency websites** (CRISIL, ICRA, CARE, India Ratings) — Credit rating reports are collected only when they appear as corporate announcements on BSE/NSE, where the filing date is verifiable. The rating agency websites do not provide reliable filing dates, making it difficult to ensure the information is genuinely pre-event for distressed companies. Absence of credit rating data is explicitly documented as acceptable.

**Business Responsibility Reports (BRR)** — The BRR section within annual reports covers ESG and sustainability topics that are not relevant to financial distress prediction. Excluding it reduces noise without losing signal.

---

## 5. Pipeline Components

---

### Spec 1 — Document Ingestion Pipeline

**What it does**

The ingestion pipeline is the entry point for all company data. It scans `data/raw/{TICKER}/`, reads the company's `metadata.json`, and registers everything in PostgreSQL: the company record, each PDF document, and the quarterly shareholding pattern data. No PDF content is read at this stage — ingestion is purely about building the metadata registry that all downstream stages depend on.

**Entry point**: `python -m ingestion.run_ingestion [--company TICKER]`

**Key design decisions**

*Idempotency as a hard requirement.* Running ingestion twice on the same dataset produces the same database state, not duplicates. Documents are checked by `file_path` before insertion; companies are upserted on `ticker`. This means ingestion is safe to re-run after adding new documents or correcting metadata.

*Fiscal calendar awareness.* Indian fiscal years run April to March, not January to December. The pipeline implements this correctly: FY2019 is April 2018 to March 2019. The fiscal quarter mapping is Q1 = April–June, Q2 = July–September, Q3 = October–December, Q4 = January–March. This affects `period_start` and `period_end` date calculations for every document type.

*`metadata.json` vs `ground_truth.json` separation.* The ingestion pipeline reads only `metadata.json`. It actively checks for prohibited field names (`group`, `distressed`, `healthy`, `label`, `ground_truth`) and logs a warning if any are found — a safety net against accidental label leakage through the metadata file. The file `ground_truth.json` is never opened by any code in `src/`.

*Source normalization.* The `source` field on documents is derived from the exchange the document was sourced from. `BSE+NSE` is normalized to `BSE` (the exchange, not the combination), `MCA` maps to the `debt_only` listing type.

**Known limitations**

- The `document_category` field parsed from filenames is the raw string from the filename. Category normalization (mapping inconsistent strings to canonical values) is handled in Spec 4, not here, so documents in this table may have non-canonical category values until feature extraction runs.
- Shareholding data is sourced entirely from `metadata.json`. If the `metadata.json` has gaps in quarterly data, those quarters will simply not appear in `shareholding_patterns`.

**How to verify it is working**

After running ingestion, query PostgreSQL:

```sql
SELECT ticker, name, listing_type FROM companies ORDER BY ticker;
SELECT document_type, COUNT(*) FROM documents GROUP BY document_type;
SELECT COUNT(*) FROM shareholding_patterns;
```

All expected companies should appear, with document type counts matching the PDFs in `data/raw/`. The `text_extracted`, `chunked`, and `embedded` flags should all be `FALSE` for newly ingested documents.

---

### Spec 2 — PDF Text Extraction & Chunking

**What it does**

For every document registered in PostgreSQL with `text_extracted = FALSE`, this stage extracts the text content from the PDF and splits it into overlapping chunks. The chunks are written as JSONL files to `data/processed/{TICKER}/{doc_id}_chunks.jsonl`. Each chunk carries full metadata (company, document type, fiscal period, source path) so it can be embedded and retrieved in isolation without needing to re-query the database.

**Entry point**: `python -m preprocessing.run_extraction [--company TICKER]`

**Key design decisions**

*Three-level extraction cascade.* No single PDF parsing library handles all real-world documents correctly. The pipeline tries three approaches in order:

1. **PyMuPDF** — Fast, reliable for digitally-born PDFs with a native text layer. Tried first because it is the most common case and the fastest.
2. **pdfplumber** — Slower but better at extracting text from PDFs with complex table layouts. Used as a fallback if PyMuPDF returns fewer than 50 characters (indicating the PDF has no text layer, or the text is locked).
3. **pytesseract OCR** — Used as the final fallback for scanned PDFs, where the document is essentially a photograph with no underlying text. OCR renders each page with PyMuPDF (200 DPI greyscale by default, `OCR_DPI` to change it) then passes the images to Tesseract. This is slow (minutes per document) but necessary — some quarterly results and older annual reports are scanned.

The threshold of 50 characters to trigger a fallback was chosen deliberately: a legitimate text extraction never returns less than 50 characters from a financial document (even a title page has more). Below 50 is a reliable indicator of a failed or absent text layer.

*Overlapping chunks.* Chunks are 3200 characters (~800 tokens) with 400-character overlap between adjacent chunks. The overlap prevents a key sentence from being cut in half at a chunk boundary and missed by retrieval. Boundary preference order: sentence boundary (period/question mark/exclamation + whitespace) → word boundary → hard character cut. This means the split point is chosen to avoid cutting in the middle of a sentence whenever possible.

*Metadata preserved in every chunk.* Each JSONL line contains `company_id`, `document_id`, `company_ticker`, `document_type`, `document_category`, `fiscal_year`, `fiscal_quarter`, `source`, and `file_path`. This means Qdrant can filter chunks by any of these dimensions without any database join — the metadata is self-contained in the vector payload.

*Two-pass re-entry design.* Pass 1 processes documents with `text_extracted = FALSE`. Pass 2 catches documents where text extraction succeeded (`text_extracted = TRUE`) but chunking failed (`chunked = FALSE`) — for example, a run that was interrupted between extraction and chunk writing. This prevents the need to re-extract text (which is expensive) just to re-run chunking.

*Label-leakage enforcement.* The chunker explicitly never writes keys named `group`, `distressed`, `healthy`, `label`, or `ground_truth` into any chunk payload. This is enforced by code, not convention.

**Known limitations**

- OCR quality varies significantly with scan resolution. Older documents (pre-2015 scans) sometimes produce garbled text. These are logged as `low_yield` in `processing_log` for manual review.
- The 50-character threshold can occasionally misclassify a password-protected PDF (which returns zero or garbage characters) as a scanned document, triggering OCR unnecessarily. This wastes time but produces the right outcome: the OCR attempt will also fail on a protected PDF, resulting in a `failed` log entry.
- OCR requires the Tesseract binary (on `PATH`, at `C:\Program Files\Tesseract-OCR\tesseract.exe`, or set via `TESSERACT_CMD`); see [How to Run the Pipeline](#11-how-to-run-the-pipeline).

**How to verify it is working**

```bash
# Check extraction status flags
python -c "from src.database.postgres import get_engine; ..."
# Or query PostgreSQL directly:
```
```sql
SELECT text_extracted, chunked, COUNT(*) FROM documents GROUP BY text_extracted, chunked;
```
Check `data/processed/{TICKER}/` for JSONL files. Spot-check a file:
```bash
Get-Content data/processed/DHFL/42_chunks.jsonl | Select-Object -First 1 | ConvertFrom-Json
```

---

### Spec 3 — Embedding & Qdrant Vector Storage

**What it does**

For each document with `chunked = TRUE` and `embedded = FALSE`, this stage loads the chunk JSONL, generates a 768-dimensional embedding for each chunk using the BAAI/bge-base-en-v1.5 sentence transformer model, and upserts the embeddings (with their metadata payloads) into Qdrant. Documents with zero chunks (e.g., extraction produced no text) are marked `embedded = TRUE` without any Qdrant write — they simply have nothing to embed.

**Entry point**: `python -m embedding.run_embedding [--company TICKER]`

**Key design decisions**

*Model choice: BAAI/bge-base-en-v1.5.* This model produces 768-dimensional embeddings and is purpose-built for retrieval tasks (the `bge` in the name stands for BAAI General Embeddings, specifically optimized for text matching). It outperforms general-purpose sentence transformers on financial and factual retrieval tasks, which is critical for the evidence-retrieval step of the risk engine.

*Query-time prefix asymmetry.* The `bge` model family uses an asymmetric retrieval scheme: at embedding time, passages are stored as-is. At query time, the query string is prefixed with `"Represent this sentence: "` before embedding. This instruction tuning significantly improves retrieval recall. The prefix is a module-level constant (`QUERY_PREFIX`) documented in both `embedder.py` and `rag_query.py` to ensure it is consistently applied at both stages.

*Deterministic point IDs.* Every chunk's Qdrant point ID is computed as `uuid5(NAMESPACE_UUID, f"{document_id}:{chunk_index}")`. This means the same chunk always maps to the same point ID regardless of how many times embedding is run. Upserts are truly idempotent — re-running embedding on already-embedded documents updates the existing points rather than creating duplicates.

*Batching.* Embeddings are generated in batches of 32 chunks. This balances memory use against throughput. The `sentence-transformers` library handles GPU/CPU automatically; on CPU it is slower but the batch size keeps memory bounded.

*Collection consistency enforcement.* On startup, the pipeline checks if the `financial_documents` collection exists in Qdrant. If not, it creates it with cosine distance and 768 dimensions. If it exists but with mismatched dimensions or distance metric, it raises an error and halts — it never silently drops and recreates the collection, which would lose existing embeddings.

**Known limitations**

- The collection name `"financial_documents"` used by the embedding pipeline and `rag_query.py` differs from the stale `"financial_risk_chunks"` value in `src/database/qdrant.py`. The active pipelines use `"financial_documents"` consistently; the old value is vestigial and unused.
- First run downloads the BAAI/bge-base-en-v1.5 model from HuggingFace (~400MB). Subsequent runs use the local cache.

**How to verify it is working**

```sql
SELECT embedded, COUNT(*) FROM documents GROUP BY embedded;
```
Check Qdrant collection counts via the REST API:
```
GET http://localhost:6333/collections/financial_documents
```
The `vectors_count` should match the total chunks written in Spec 2.

---

### Spec 4 — Structured Feature Extraction

**What it does**

Feature extraction is the most complex pipeline stage. It reads the chunk JSONL files produced by Spec 2 and extracts structured, typed signals from the unstructured text. The output is rows in the `extracted_features` PostgreSQL table — one row per document for financial and credit features, one row per fiscal quarter (across all documents) for governance features.

The stage runs as four sequential phases per company:

- **Phase A (Announcement normalization)** — Normalizes the `document_category` field for all corporate announcement documents, since BSE/NSE categories are inconsistent in practice.
- **Phase B (Financial extraction)** — Extracts financial ratios from annual report and quarterly result documents.
- **Phase C (Governance aggregation)** — Counts governance signals from normalized announcement documents, aggregated per fiscal quarter.
- **Phase D (Credit extraction)** — Extracts credit ratings and outlook from credit rating documents and announcements.

**Entry point**: `python -m feature_extraction.run_feature_extraction [--company TICKER] [--force]`

**Key design decisions**

*Three-layer category normalization (Phase A).* The category labels on corporate announcements from BSE/NSE are inconsistent. The same "Board Meeting" announcement might appear as `"Board Meeting"`, `"BOARD MEETING"`, `"BM"`, `"Board Meeting (N)"` (the `(N)` suffix denotes a late filing), or be labeled inconsistently due to data entry errors. Three layers are applied in order to handle this:

1. **Exact dictionary lookup** — Strip trailing suffixes (`_(N)`, `_SCANNED`), lowercase, and check against a known alias dictionary with hundreds of variant spellings mapped to nine canonical values.
2. **Fuzzy matching** — Use `rapidfuzz.fuzz.partial_ratio` with a threshold of 80 to catch near-matches not in the dictionary.
3. **Keyword matching** — If the first 2000 characters of the document text contain recognizable keywords (e.g., "resignation", "board composition"), assign the category from the text itself.

Documents that fail all three layers are labeled `unclassified` and logged with `status='unmapped_category'` rather than `failed` — this makes them auditable (visible in the processing log) without blocking processing of other documents.

*Decimal arithmetic for financial figures.* Financial values use Python's `Decimal` type rather than `float` throughout extraction and storage. This avoids floating-point rounding errors that would corrupt figures like ₹3,455.78 crores when doing unit conversion (e.g., from lakhs to crores). The conversion is: 1 crore = 100 lakhs. Values are stored in crores in the database.

*NULL semantics are explicit.* A missing value (e.g., no interest coverage ratio found in the document) is stored as `NULL`, never as `0`. This is critical because `NULL` in the risk engine means "data not available, skip this rule," while `0` means "the ratio is actually zero," which are very different signals.

*NBFC handling.* Non-Banking Financial Companies (NBFCs) and housing finance companies use different financial metrics than regular companies (e.g., Net Interest Margin instead of EBITDA Margin). The `financial_extractor` detects NBFC entities by checking `listing_type == 'debt_only'` or sector keywords (`'NBFC'`, `'Housing Finance'`), then writes NBFC-specific metrics into the `additional_signals` JSONB field rather than forcing them into columns that don't semantically apply.

*Governance period aggregation (Phase C).* Governance features are quarter-level aggregates, not document-level. The governance extractor groups all announcements for a company by fiscal quarter (using `fiscal_utils.derive_fiscal_period()`), counts each category, and writes a single governance feature row per quarter with `document_id = NULL`. This is why the `extracted_features` table has a nullable `document_id` — it is a deliberate design choice, not an oversight.

*AGM delay detection.* The Annual General Meeting is required to be held by September of each year in India (i.e., within Q2 of the Indian fiscal year). If no AGM announcement is found in Q2 or Q3 for a given fiscal year, `agm_delay_flag` is set to `TRUE`. This is a governance red flag that has historically appeared before distress.

**Known limitations**

- Financial figure extraction relies on regex patterns over unstructured text. Some annual reports use highly non-standard table layouts that confuse the extractor. Where no statement row can be found the value is `NULL`.
- **Scanned-PDF OCR errors (accepted).** Digits in scanned quarterly results are sometimes misread by Tesseract (e.g. `1,19,638` read as `4,19,836`) or whole tables come back jumbled. A value that parses cleanly but is wrong cannot be detected without a heuristic, so a small number of figures are wrong or missing. Known cases on the two test companies (JPINFRATEC): Q4 FY2016 revenue (4,198 cr vs ~1,196 true; yields a spurious +410% growth), Q2 FY2017 (nothing extracted), FY2015 annual report PAT (not found). Obvious impossibilities are rejected (negative revenue; rupee figures mistaken for crore). Provenance (page, unit, label, scope) is stored in `additional_signals['sources']` so any value can be audited.
- Fiscal-year convention: `FY2019` means April 2018 – March 2019 (named by the year it ends; `Q1 FY2019` = April–June 2018), as computed by `feature_extraction/fiscal_utils.py`. Some source files were named by the year the fiscal year starts; their labels were corrected from the period each document states (IDEA and HDFCBANK: all annual reports, quarterly results and shareholding keys moved +1 year; JSWSTEEL: quarterly results +1 year). Ground-truth periods must be read in the same convention at evaluation time.
- Misfiled documents are quarantined in `data/quarantine/` rather than relabelled when they hold a period outside the collection window or duplicate another file: TCS `Q3FY2017` (held Q3 FY2016) and `Q4FY2018` (held Q4 FY2017); JPINFRATEC `Q2FY2017` and `Q3FY2017` (held Q2/Q3 FY2018). Those quarters have no financial results.
- Credit ratings are sparse where no rating documents were collected (the Infosys and JAYPEE `credit_ratings` folders are empty); the only fallback is a rating the company discloses in its annual report, which covers few years.
- Debt-to-equity, interest coverage and current ratio are not rule inputs for lenders (NBFCs, housing finance, banks): leverage of 6-10x is their business model, interest is their main cost (coverage of 1.1-1.3x is normal) and their balance sheets are not split current / non-current, so the frozen thresholds cannot separate healthy from distressed lenders. Any figure found, computed or reported, is kept in `additional_signals["<ratio>_lender"]` for reference, and `DEBT_TO_EQUITY_HIGH`, `INTEREST_COVERAGE_LOW` and `CURRENT_RATIO_LOW` are not evaluable for them.
- The rule thresholds are the PRD values and are deliberately frozen as the Semester 8 baseline. They are not tuned on the evaluation labels; known false positives (e.g. `PROMOTER_HOLDING_LOW` on professionally managed Infosys) are reported as findings, not patched.
- Credit ratings for healthy companies are often absent entirely — no credit rating announcements were filed on BSE/NSE. This is expected and documented as acceptable; the rating rules simply won't fire for those companies.
- The `--force` flag deletes and re-extracts all features for the specified company. Use with care on large datasets.

**How to verify it is working**

```sql
SELECT feature_type, COUNT(*) FROM extracted_features GROUP BY feature_type;
SELECT company_id, fiscal_year, fiscal_quarter, revenue, pat, debt_to_equity
FROM extracted_features
WHERE feature_type = 'financial'
ORDER BY company_id, fiscal_year, fiscal_quarter;
```
Check `processing_log` for any `unmapped_category` entries:
```sql
SELECT d.file_name, pl.message FROM processing_log pl
JOIN documents d ON d.id = pl.document_id
WHERE pl.status = 'unmapped_category';
```

---

### Spec 5 — Rule-Based Risk Engine

**What it does**

The risk engine reads structured features from PostgreSQL, evaluates 15 expert-defined rules, retrieves supporting evidence from Qdrant, and writes a risk prediction for each company-period combination to the `predictions` table. Each triggered rule contributes a weighted amount to the overall `risk_score`, and each rule's evidence is linked to a specific document chunk retrieved by semantic search.

**Entry point**: `python -m risk_engine.run_risk_engine [--company TICKER] [--fiscal-year FY2019]`

For a complete explanation of how the risk engine works, see [Section 8 — Rule-Based Risk Engine Deep Dive](#8-rule-based-risk-engine--deep-dive).

**How to verify it is working**

```sql
SELECT c.ticker, p.fiscal_year, p.fiscal_quarter, p.risk_score, p.risk_level
FROM predictions p JOIN companies c ON c.id = p.company_id
WHERE p.model_version = 'rule_based_v1'
ORDER BY p.risk_score DESC;
```

---

### Spec 6 — Streamlit Demo UI

**What it does**

The web interface is the presentation layer of the system. It provides two capabilities:
1. **Risk report view** — select a company, see its risk score, risk level, which rules fired, and the source document passages that support each rule.
2. **Ad-hoc Q&A** — ask any free-text question about a company's filings; the system embeds the question, retrieves the top-5 most relevant document chunks from Qdrant, passes them to a local LLM (Ollama/Qwen2.5), and returns a cited answer.

**Entry point**: `streamlit run src/ui/app.py`

**Key design decisions**

*Startup health checks.* On load, the app checks all three infrastructure dependencies simultaneously: PostgreSQL connectivity, Qdrant connectivity, and whether the Ollama model is pulled and ready. If any check fails, all failures are shown at once so the operator can fix everything in one go rather than discovering failures one at a time.

*Singleton caching.* The Sentence Transformer model, Qdrant client, and PostgreSQL engine are expensive to initialize. Streamlit's `@st.cache_resource` ensures each is created exactly once per server process and reused across all reruns and user interactions, keeping the UI responsive.

*Company selector shows only scored companies.* The company dropdown is populated only from companies that already have `rule_based_v1` predictions in the database. This prevents showing a blank risk report for a company whose documents haven't been scored yet.

*RAG with numbered citations.* The LLM prompt explicitly instructs the model to cite passages by number (e.g., `[1]`, `[2]`). The system prompt states: "Answer using ONLY the context passages provided... do not invent information." This grounds the model's responses in actual retrieved text and makes the Q&A output verifiable.

*Label leakage guard in the query pipeline.* The `rag_query.py` module maintains a `_FORBIDDEN_FIELDS` frozenset containing `group`, `label`, `distressed`, `healthy`, `ground_truth`. Before returning sources to the UI, any of these keys are explicitly `pop()`'d from the chunk payload dictionaries. This is a final defense-in-depth check.

*Legacy Qdrant API compatibility.* Some environments may have an older version of `qdrant-client` installed. The retrieval code first tries the modern `query_points()` API and falls back to the legacy `search()` API on `AttributeError`. This ensures the UI works without forcing an exact version pin.

**How to verify it is working**

Open `http://localhost:8501` in a browser. Select a company. The risk report should display with a score, level, and at least one triggered rule with a cited source passage. Type a question in the Q&A box and confirm the answer includes citation numbers like `[1]`.

---

## 6. Technology Stack & Justification

| Technology | Version | Role | Why Chosen |
|---|---|---|---|
| **Python** | 3.11+ | Primary language | Strong ecosystem for data, ML, and PDF processing; no viable alternative for this stack |
| **PostgreSQL** | 17 | Structured data store | Industry-standard relational DB; supports partial indexes needed for upserts; JSONB for overflow signals; Alembic for schema migrations |
| **Qdrant** | Latest | Vector store | Purpose-built for approximate nearest-neighbor search; supports payload filtering (company_id, fiscal_year) natively; simple Docker deployment; better Rust-native performance than FAISS at this scale |
| **SQLAlchemy** | 2.0+ | ORM & query layer | Modern 2.0 API with typed `Mapped` annotations; Alembic integration for migrations; avoids raw SQL string construction (security and maintainability) |
| **Alembic** | 1.13+ | Schema migrations | The standard SQLAlchemy migration tool; migration files are version-controlled so schema changes are reproducible and reversible |
| **BAAI/bge-base-en-v1.5** | via `sentence-transformers` | Embedding model | Asymmetric retrieval design specifically improves passage retrieval; 768 dimensions balances quality and storage; outperforms generic models on factual/financial text matching |
| **Sentence Transformers** | 5.0+ | Embedding inference | Standard library for running pre-trained embedding models locally; GPU-optional |
| **PyMuPDF (fitz)** | 1.26+ | PDF text extraction (primary) | Fastest and most reliable for digitally-born PDFs; C-based, very fast |
| **pdfplumber** | 0.11+ | PDF text extraction (fallback) | Better at complex table layouts than PyMuPDF; used as second-tier fallback |
| **pytesseract** | 0.3+ | OCR (final fallback) | Industry-standard OCR for scanned PDFs; pages are rendered by PyMuPDF, so only the Tesseract binary is needed |
| **Ollama + Qwen2.5:3b** | 0.5+ | Local LLM for RAG Q&A | Enables fully local inference with no cloud API calls; Qwen2.5 is competitive with larger models for document Q&A at this scale |
| **Streamlit** | 1.47+ | Web interface | Rapid UI development without a frontend framework; sufficient for an academic demo; `@st.cache_resource` handles expensive singleton initialization cleanly |
| **XGBoost** | 3.0+ | ML model (Semester 8) | Standard gradient boosting library; well-suited for tabular feature data at small-to-medium dataset sizes; interpretable via SHAP |
| **SHAP** | 0.48+ | ML explainability (Semester 8) | Provides per-prediction feature importance for the XGBoost model; needed to compare explainability of ML vs. rule-based outputs |
| **rapidfuzz** | Latest | Fuzzy string matching | Used in category normalization for announcement types; faster than `fuzzywuzzy` and pure-Python `difflib` |
| **Hypothesis** | 6.100+ | Property-based testing | Automated generation of adversarial test inputs; used throughout to test pure functions (chunker, scorer, filename parser) against invariants like "chunk text is never empty" and "risk score is always in [0, 1]" |
| **Docker Compose** | 3.9 | Infrastructure | Reproducible local environment for PostgreSQL and Qdrant; avoids host-level installation conflicts |

---

## 7. Governance & Financial Signal Definitions

### Financial Signals

| Signal | Column | What It Measures | Why It's a Distress Indicator |
|---|---|---|---|
| Revenue | `revenue` | Total operating revenue in crores (INR) | Absolute baseline; context for all other financial ratios |
| Revenue Growth | `revenue_growth_pct` | Year-over-year (or quarter-over-quarter) revenue change (%) | Sustained revenue contraction (< −10%) signals deteriorating business; often precedes cash flow problems |
| Profit After Tax (PAT) | `pat` | Net profit after all taxes in crores | Negative PAT indicates the company is loss-making; sustained losses deplete reserves and force borrowing |
| EBITDA Margin | `ebitda_margin_pct` | Earnings before interest/tax/depreciation/amortization as % of revenue | Measures core operational profitability; declining EBITDA margin means the business model is weakening even before financing costs |
| Debt-to-Equity | `debt_to_equity` | Total debt divided by shareholders' equity | High leverage (> 3.0) means most assets are financed by debt; small revenue shocks can trigger default. Not a rule input for lenders (see Known limitations) |
| Interest Coverage | `interest_coverage` | EBIT divided by interest expense, computed as (PBT + finance cost) / finance cost when not reported | Measures how many times the company can pay its interest from operating earnings; below 1.5 means the interest burden is dangerously close to EBIT |
| Current Ratio | `current_ratio` | Current assets divided by current liabilities | Measures short-term liquidity; below 1.0 means the company may not be able to meet obligations maturing in the next 12 months |
| Cash Flow from Operations (CFO) | `cfo` | Operating cash generated, excluding investing/financing activities | A company can report accounting profits while running negative operating cash flow; negative CFO is one of the strongest early distress signals |

### Governance Signals

| Signal | Column | What It Measures | Why It's a Distress Indicator |
|---|---|---|---|
| Board Meeting Count | `board_meeting_count` | Number of board meetings in the period | Very few board meetings may suggest governance vacuum; unusually many may suggest crisis management |
| Director Change Count | `director_change_count` | Directors added or removed in the period | High churn in board composition often precedes or accompanies governance failures |
| Appointment Count | `appointment_count` | New director appointments in the period | Context for interpreting director change — are changes restructuring or replacement? |
| Resignation Count | `resignation_count` | Director resignations in the period | Resignations by independent directors are a strong signal — they often occur when the directors have concerns they cannot resolve |
| AGM Delay Flag | `agm_delay_flag` | Whether the AGM was held later than required | Indian law requires the AGM by September (Q2). A delayed AGM indicates the company could not present its audited accounts on time — a red flag for financial problems |
| Clarification Count | `clarification_count` | Press Releases + News Clarifications + News Verifications in the period | Companies issue clarifications in response to news articles or exchange queries. High counts signal elevated media scrutiny and potential information concerns |
| Press Release Count | `press_release_count` | Standalone press releases in the period | Distinguishes proactive communication from reactive clarifications |

### Credit Signals

| Signal | Column | What It Measures | Why It's a Distress Indicator |
|---|---|---|---|
| Credit Rating | `credit_rating` | Assigned rating (AAA, AA+, A, BBB, BB, B, D, etc.) | A low rating directly signals assessed default risk; D-grade instruments are already in default |
| Rating Outlook | `rating_outlook` | Outlook assigned by the rating agency (Stable, Negative, Watch) | Negative outlook or Credit Watch precedes a formal downgrade; often the earliest public signal of deteriorating credit assessment |
| Rating Change Flag | `rating_change_flag` | Whether the rating changed from the prior period | A downgrade, especially combined with Negative outlook, is a direct predictor of near-term financial stress |

### Shareholding Signals

| Signal | Column | What It Measures | Why It's a Distress Indicator |
|---|---|---|---|
| Promoter Holding | `promoter_holding_pct` | Percentage of total shares held by founding promoters | Very low promoter holding (< 25%) reduces incentive for promoters to rescue a distressed company; can indicate gradual exit |
| Pledge Percentage | `pledge_pct` | Percentage of promoter holding pledged as loan collateral | High pledge (≥ 50%) means promoters borrowed money using their own shares as collateral; if the stock price falls, lenders can force-sell the shares, accelerating distress in a spiral |

---

## 8. Rule-Based Risk Engine — Deep Dive

### How the Engine Works Conceptually

The engine is built on a **rules-as-data** philosophy. All 15 rules are defined as dataclass instances in `src/risk_engine/rules.py`. Each rule has a `rule_id`, a human-readable `description`, a `weight` (its contribution to the overall risk score), and type-specific threshold fields. The scoring logic in `scorer.py` reads from this catalog — no threshold values are hardcoded in the evaluation functions.

This separation means that changing a threshold (e.g., adjusting the pledge percentage cutoff based on backtesting results in Semester 8) requires editing only the rule catalog, not the scoring logic.

### The 15 Rules

**Shareholding rules (3)**
- `PLEDGE_RISING` (weight 0.10) — Promoter pledge percentage has increased by at least 10 percentage points per quarter for at least 2 consecutive quarters. A rising trend is more informative than a single high reading.
- `PLEDGE_HIGH` (weight 0.08) — Pledge percentage is at or above 50%, indicating a critically high portion of promoter holding is encumbered.
- `PROMOTER_HOLDING_LOW` (weight 0.05) — Promoter holding is below 25%, signaling weak insider confidence or possible gradual disinvestment.

**Financial rules (6)**
- `INTEREST_COVERAGE_LOW` (weight 0.10) — Interest coverage ratio below 1.5; the company earns less than one and a half times its interest expense from operations.
- `NEGATIVE_PAT` (weight 0.08) — Profit after tax is negative; the company is loss-making this period.
- `DEBT_TO_EQUITY_HIGH` (weight 0.08) — Debt-to-equity exceeds 3.0; the company is heavily leveraged. Not evaluable for lenders.
- `CURRENT_RATIO_LOW` (weight 0.06) — Current ratio below 1.0; short-term liquidity is under strain.
- `CFO_NEGATIVE` (weight 0.08) — Cash flow from operations is negative; the business consumes more cash than it generates.
- `REVENUE_DECLINING` (weight 0.06) — Revenue growth is below −10%; the business is contracting materially.

**Credit rules (2)**
- `RATING_DOWNGRADE` (weight 0.10) — The rating in force is a downgrade versus the same agency's previous rating, or the rating changed and the outlook is Negative or Watch.
- `RATING_DEFAULT_GRADE` (weight 0.12) — The current rating is `D`, `C`, `SD` (Selective Default) or `NM` (Not Meaningful). The highest single-rule weight in the system, reflecting that an instrument in default is an extremely strong distress signal.

**Governance rules (4)**
- `RESIGNATION_SPIKE` (weight 0.07) — Three or more director resignations in a single quarter.
- `DIRECTOR_CHURN` (weight 0.06) — Resignations plus other director changes of 4 or more in a single quarter.
- `AGM_DELAY` (weight 0.05) — The `agm_delay_flag` is `TRUE`; the flag sits on the quarter in which the AGM was held.
- `CLARIFICATION_SPIKE` (weight 0.05) — Five or more news clarifications/press releases in a single quarter, indicating elevated external scrutiny.

Governance rows exist per quarter only. An annual period uses the year's worst quarter (maximum resignations and clarifications, the largest single-quarter churn, and the AGM flag if any quarter has it), never a sum, so the single-quarter rules keep their meaning.

### How risk_score and risk_level Are Derived

```
risk_score = (sum of weights of all triggered rules) / (sum of all 15 rule weights)
```

The total weight of all 15 rules sums to a fixed constant (`TOTAL_WEIGHT`). Dividing by this constant normalizes the score to the range [0.0, 1.0] regardless of which rules fire. A company that triggers every single rule scores 1.0; a company with no triggered rules scores 0.0.

Risk levels are assigned by threshold bands:

| risk_score Range | risk_level |
|---|---|
| 0.00 – 0.24 | Low |
| 0.25 – 0.49 | Medium |
| 0.50 – 0.74 | High |
| 0.75 – 1.00 | Critical |

**`rule_based_v2` aggregation.** v1 (above) divides by the weight of all 15 rules, so rules that cannot see any data still dilute the score. v2 keeps every rule, threshold and weight unchanged and only changes aggregation:

- the denominator is the weight of the rules whose input data is present, but never less than 0.5 × `TOTAL_WEIGHT`, so a sparse period cannot inflate its score;
- `coverage` (evaluable weight / `TOTAL_WEIGHT`) is reported per period;
- severity floor: if `NEGATIVE_PAT` and `INTEREST_COVERAGE_LOW` both fire, the level is at least Medium (the score itself is unchanged).

Both versions are written to `predictions` side by side (`model_version` = `rule_based_v1` / `rule_based_v2`); run v2 with `run_risk_engine.py --model-version rule_based_v2`. The UI shows v1.

**NULL-safety.** Rules silently skip when the required feature column is `NULL`. This is critical: a company without any credit rating data should not be penalized for "failing" the credit rules — it simply doesn't trigger them. The system distinguishes "data not available" (`NULL`) from "value is genuinely zero" (`0`).

### How Evidence Is Tied Back to Source Documents

For each triggered rule, the engine performs a semantic search against Qdrant using a natural-language query template specific to that rule. For example:

- `PLEDGE_HIGH` → queries for `"promoter pledging of shares collateral loan"`
- `NEGATIVE_PAT` → queries for `"net loss profit after tax negative"`
- `RESIGNATION_SPIKE` → queries for `"director resignation board composition change"`

The query is embedded using the same model as the documents (with the `"Represent this sentence: "` prefix applied at query time, as required by the `bge` model's asymmetric design), and the top-1 most semantically similar chunk for that company is returned. This chunk — the actual passage from an actual filing — is stored in `evidence_chunks` in the `predictions` table and displayed in the UI alongside the triggered rule.

This is the system's explainability mechanism: every risk flag has a direct citation.

---

## 9. Known Challenges & How They Were Solved

### Document Category Inconsistency

**The problem**: BSE/NSE corporate announcement categories are entered by company secretaries using free text, leading to dozens of variant spellings and formats for the same category. "Board Meeting", "BOARD MEETING", "Board Meeting (N)", "Bd. Meeting", "BM" all refer to the same category. Without normalization, the governance extractor would not be able to count how many board meetings occurred in a quarter.

**The solution**: A three-layer normalization approach (described in detail in Spec 4):
1. An alias dictionary with hundreds of known variants mapped to nine canonical values.
2. Fuzzy matching (rapidfuzz, threshold 80) for near-misses not in the dictionary.
3. Keyword extraction from the document text itself as a last resort.

Documents that fail all three layers are logged as `unmapped_category` for manual review, not silently dropped.

### Scanned PDF Handling

**The problem**: Some filings — particularly older annual reports and quarterly results submitted as image scans — have no underlying text layer. Standard PDF text extraction returns nothing or garbage.

**The solution**: The three-level extraction cascade detects this condition when the extracted text is below the 50-character threshold. For these documents, PyMuPDF renders each page to a greyscale image (200 DPI by default) and passes it to Tesseract OCR. No Poppler or `pdf2image` is needed; the Tesseract binary is found on `PATH`, at the default Windows install path, or via the `TESSERACT_CMD` environment variable.

OCR results are slower (minutes per document) and lower quality than native text extraction, but are necessary to avoid a large gap in the dataset. Problematic documents are logged with the `extraction_method = 'ocr'` field in `processing_log` so they can be identified if OCR quality is a concern.

### Non-Standard Listing Status

**The problem**: Some companies in the benchmark dataset are not equity-listed on BSE/NSE. They may be listed on the debt market (bonds/debentures traded on BSE) without having an equity listing. This affects what data is available (no quarterly shareholding filings on BSE) and what financial metrics apply (NBFC/housing finance companies use different metrics than regular companies).

**The solution**:
- A `listing_type` column on the `companies` table distinguishes `'equity'` (standard) from `'debt_only'`.
- `debt_only` companies are sourced via MCA for annual filings.
- The `financial_extractor` checks `listing_type` and sector keywords to identify NBFC entities, then writes NBFC-specific metrics into `additional_signals` rather than forcing them into columns designed for regular companies.
- Rules that require shareholding data (e.g., `PLEDGE_RISING`) simply won't fire for `debt_only` companies because no shareholding rows will exist — they are skipped via NULL-safety, not by special-casing.

### Fiscal Year Calendar Mismatch

**The problem**: Standard date libraries use January–December years. Indian fiscal years run April–March. An announcement dated January 15, 2019 belongs to FY2019 (April 2018–March 2019), not FY2018. Getting this wrong would misattribute signals to the wrong fiscal period.

**The solution**: A dedicated `fiscal_utils.derive_fiscal_period()` function centralizes this logic. All components that need to map a calendar date to a fiscal period call this function rather than implementing their own logic. The Indian fiscal year and quarter boundaries are defined in one place.

---

## 10. Project Folder Structure

```
d:\Risk-Detection-RAG\
│
├── .env                          # Environment variables (DB credentials, model names, paths)
├── .gitignore
├── alembic.ini                   # Alembic migration config — points to src/models/base.py
├── pytest.ini                    # Test runner config (rootdir, test paths)
├── conftest.py                   # Pytest fixtures shared across all test modules
├── requirements.txt              # All Python dependencies (pinned version ranges)
│
├── docker/
│   └── docker-compose.yml        # PostgreSQL 17 + Qdrant — infrastructure definition
│
├── migrations/
│   └── versions/
│       ├── 0001_initial_schema.py              # All 6 tables
│       ├── 0002_extracted_features_upsert_constraints.py  # Partial unique indexes for upserts
│       └── 0003_predictions_upsert_constraint.py          # Predictions upsert index
│
├── tools/
│   └── normalize_filenames.py    # Canonical PDF file naming (_SCANNED suffix = OCR)
│
├── data/
│   ├── raw/
│   │   └── {TICKER}/
│   │       ├── metadata.json             # Company info + shareholding data (NO ground truth)
│   │       ├── ground_truth.json         # Distress label + event date — NEVER READ BY src/
│   │       ├── annual_reports/           # {TICKER}_AnnualReport_FY{YYYY}.pdf
│   │       ├── quarterly_results/        # {TICKER}_Q{n}FY{YYYY}[_SCANNED].pdf
│   │       ├── corporate_announcements/  # {TICKER}_Ann_{YYYYMMDD}_{Category}[_(N)].pdf
│   │       └── credit_ratings/           # Subset of announcements: _Credit Rating.pdf
│   ├── processed/
│   │   └── {TICKER}/
│   │       └── {doc_id}_chunks.jsonl     # Chunk JSONL output from Spec 2
│   └── benchmark/                # Reserved for Semester 8 evaluation scripts
│
├── src/
│   ├── config/
│   │   ├── settings.py           # Central config: all env vars, path constants, service coords
│   │   └── constants.py          # Supported extensions, document types, risk levels
│   │
│   ├── database/
│   │   ├── postgres.py           # SQLAlchemy engine factory + connection test
│   │   └── qdrant.py             # Qdrant client factory + connection test
│   │
│   ├── models/
│   │   ├── base.py               # SQLAlchemy DeclarativeBase
│   │   ├── company.py            # companies table ORM
│   │   ├── document.py           # documents table ORM
│   │   ├── processing_log.py     # processing_log table ORM
│   │   ├── shareholding_pattern.py  # shareholding_patterns table ORM
│   │   ├── feature.py            # extracted_features table ORM (with upsert indexes)
│   │   └── prediction.py         # predictions table ORM
│   │
│   ├── ingestion/                # SPEC 1 — Document Ingestion
│   │   ├── run_ingestion.py      # Entry point: discover companies → process each
│   │   ├── metadata_reader.py    # Reads metadata.json; prohibits ground-truth fields
│   │   ├── company_registrar.py  # Upsert company rows; maps exchange → listing_type
│   │   ├── document_registrar.py # Registers PDFs; calls FilenameParser
│   │   ├── filename_parser.py    # Parses filename → fiscal period, category, source
│   │   ├── shareholding_registrar.py  # Upserts shareholding_patterns rows
│   │   └── ingestion_logger.py   # Savepoint-protected log writes
│   │
│   ├── preprocessing/            # SPEC 2 — Text Extraction & Chunking
│   │   ├── run_extraction.py     # Entry point: two-pass re-entrant design
│   │   ├── text_extractor.py     # PyMuPDF → pdfplumber → OCR cascade
│   │   ├── chunker.py            # Overlapping chunks (3200 chars / 400 overlap)
│   │   └── extraction_logger.py  # Savepoint-protected log writes
│   │
│   ├── embedding/                # SPEC 3 — Embedding & Vector Storage
│   │   ├── run_embedding.py      # Entry point: load JSONL → embed → upsert Qdrant
│   │   ├── embedder.py           # BAAI/bge-base-en-v1.5; deterministic point IDs
│   │   ├── qdrant_client_factory.py  # Creates/validates "financial_documents" collection
│   │   └── embedding_logger.py   # Savepoint-protected log writes
│   │
│   ├── feature_extraction/       # SPEC 4 — Structured Feature Extraction
│   │   ├── run_feature_extraction.py   # Entry point: 4-phase pipeline per company
│   │   ├── category_normalizer.py      # 3-layer announcement category normalization
│   │   ├── financial_extractor.py      # Revenue, PAT, EBITDA, leverage, liquidity, CFO
│   │   ├── governance_extractor.py     # Board counts, resignations, AGM delay
│   │   ├── credit_extractor.py         # Rating, outlook, change flag
│   │   ├── fiscal_utils.py             # Indian fiscal calendar utilities
│   │   ├── feature_writer.py           # Upsert logic for extracted_features rows
│   │   └── feature_logger.py           # Savepoint-protected log writes
│   │
│   ├── risk_engine/              # SPEC 5 — Rule-Based Risk Engine
│   │   ├── run_risk_engine.py    # Entry point: load features → score → evidence → write
│   │   ├── rules.py              # All 15 RuleDefinition instances (thresholds, weights)
│   │   ├── scorer.py             # evaluate_rules() — pure function; produces risk_score
│   │   ├── evidence_retriever.py # Semantic Qdrant search per triggered rule
│   │   ├── prediction_writer.py  # Upsert predictions (model_version='rule_based_v1')
│   │   └── risk_logger.py        # Savepoint-protected log writes
│   │
│   ├── ui/                       # SPEC 6 — Streamlit Demo UI
│   │   ├── app.py                # Main Streamlit app (4 sections: health, selector, report, Q&A)
│   │   ├── prediction_loader.py  # DB queries: companies list, latest prediction, features
│   │   ├── rag_query.py          # RAG pipeline: embed → Qdrant → prompt → Ollama → response
│   │   └── ollama_client.py      # HTTP client for Ollama streaming API
│   │
│   └── main.py                   # Infrastructure connectivity test (postgres + qdrant)
│
├── tests/
│   ├── ingestion/                # Unit + property tests for all ingestion modules
│   ├── preprocessing/            # 103 tests incl. Hypothesis property tests for chunker
│   ├── embedding/                # Unit tests for embedder + Qdrant factory
│   ├── feature_extraction/       # Unit + property tests for all extractors
│   ├── risk_engine/              # Unit + Hypothesis property tests for scorer + rules
│   └── ui/                       # Unit tests for ollama_client, prediction_loader, rag_query
│
├── notebooks/                    # Jupyter notebooks for exploratory analysis
│
└── .kiro/
    ├── specs/                    # All 6 spec definitions (requirements.md, design.md, tasks.md)
    │   ├── document-ingestion-pipeline/
    │   ├── pdf-text-extraction-chunking/
    │   ├── embedding-qdrant-pipeline/
    │   ├── structured-feature-extraction/
    │   ├── rule-based-risk-engine/
    │   └── streamlit-demo-ui/
    └── steering/                 # Project-wide rules injected into every AI session
        ├── prd.md                # Product Requirements Document (authoritative)
        └── schema.md             # Database schema (authoritative technical contract)
```

---

## 11. How to Run the Pipeline

### Prerequisites

- Python 3.11+
- Docker Desktop (for PostgreSQL and Qdrant)
- Tesseract OCR installed on the system (for scanned PDF fallback; Poppler is not needed)
- Ollama installed and running at `http://localhost:11434`

### Step 1 — Start Infrastructure

```bash
cd docker
docker compose up -d
```

This starts:
- PostgreSQL 17 on port `5433` (host) → `5432` (container)
- Qdrant on port `6333` (HTTP REST) and `6334` (gRPC)

Verify both are running:
```bash
docker compose ps
```

### Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure Environment

Copy the example values or set these in your `.env` file at the project root:

```
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5433
POSTGRES_DB=financial_risk_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

QDRANT_HOST=localhost
QDRANT_PORT=6333

EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
LLM_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434

# Optional — only if tesseract is not on PATH or in C:\Program Files\Tesseract-OCR
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Step 4 — Apply Database Migrations

```bash
alembic upgrade head
```

This runs all three migration files and creates the six tables in PostgreSQL.

Verify:
```bash
# Connect to PostgreSQL and list tables
docker exec -it financial-risk-postgres psql -U postgres -d financial_risk_db -c "\dt"
```

### Step 5 — Pull the Ollama Model

```bash
ollama pull qwen2.5:3b
```

### Step 6 — Verify Infrastructure Connectivity

```bash
python -m src.main
```

Should print successful connection messages for both PostgreSQL and Qdrant.

### Step 7 — Run the Pipeline (in order)

Each command can be run for a specific company using `--company TICKER` or for all companies by omitting it.

```bash
# Stage 1: Ingest documents from data/raw/ into PostgreSQL
python -m ingestion.run_ingestion

# Stage 2: Extract text from PDFs and produce chunk JSONL files
python -m preprocessing.run_extraction

# Stage 3: Embed chunks and upsert into Qdrant
python -m embedding.run_embedding

# Stage 4: Extract structured financial/governance/credit features
python -m feature_extraction.run_feature_extraction

# Stage 5: Run rule-based risk scoring and write predictions
python -m risk_engine.run_risk_engine

# Stage 6: Launch the Streamlit UI
streamlit run src/ui/app.py
```

All commands are idempotent — they can be re-run safely after adding new documents or correcting data. Use `--force` on Stage 4 to re-extract features for a specific company from scratch.

### Running Tests

```bash
# Run all tests (fast — no external services required)
pytest

# Run a specific module's tests
pytest tests/risk_engine/

# Run with verbose output
pytest -v tests/preprocessing/test_chunker.py
```

---

## 12. Semester 8 Roadmap

The pipeline built in Semester 7 is the foundation for the full evaluation in Semester 8. Here is what comes next.

### Phase 2 — Dataset Expansion (Early Semester 8)

Expand the benchmark from 14 to 30 companies — 15 distressed and 15 healthy. The same ingestion, extraction, and feature extraction pipeline runs on the new companies without modification. Distressed companies in this dataset will have a `collection_window_end` equal to their `ground_truth.json` distress date, ensuring only pre-event documents are collected.

### Phase 3 — Historical Backtesting (Mid Semester 8)

For each distressed company, the prediction pipeline will use only documents filed before the ground-truth distress date (stored in `ground_truth.json`, loaded only at this stage for the first time). The rule-based engine's predictions for those pre-event periods will be compared against the known outcomes. Performance metrics to be computed:

- **Recall** — what fraction of actually distressed companies did the system flag as High or Critical risk?
- **False Positive Rate** — what fraction of healthy companies were incorrectly flagged?
- **Lead Time** — how many months before the distress event did the system first predict High or Critical risk?
- **Citation Accuracy** — for a manually reviewed sample of triggered rules, do the cited source passages genuinely support the claim?

Targets: Recall ≥ 70%, FPR < 25%, Lead Time 3–6 months, Citation Accuracy ≥ 90%.

### Phase 4 — XGBoost Training & Comparative Evaluation (Late Semester 8)

Train an XGBoost classifier using the rows from `extracted_features` and `shareholding_patterns` as the feature matrix, with labels loaded from `ground_truth.json` at training time only. Evaluate against the same backtesting split as the rule-based engine. Compute the full comparison across Recall, Precision, F1 Score, False Positive Rate, and Lead Time for both approaches.

An optional Random Forest model will serve as an additional benchmark to characterize how much XGBoost's gradient boosting specifically contributes over a simpler ensemble.

The final deliverable of Phase 4 is a documented quantitative comparison with a clear conclusion: does an ML approach meaningfully improve early financial risk detection over an expert-designed rule-based system, and at what cost in explainability?

---

*This README was generated to support academic project review presentations. All technical details are sourced from the actual codebase, schema, and product requirements document.*
