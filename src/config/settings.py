from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

# -------------------------
# Project Paths
# -------------------------

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTRACTED_DIR = DATA_DIR / "extracted"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# -------------------------
# PostgreSQL
# -------------------------

POSTGRES = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT")),
    "database": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

# -------------------------
# Qdrant
# -------------------------

QDRANT = {
    "host": os.getenv("QDRANT_HOST"),
    "port": int(os.getenv("QDRANT_PORT")),
    "collection": "financial_risk_chunks"
}

# -------------------------
# Models
# -------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"