from config.settings import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    POSTGRES,
    QDRANT,
    EMBEDDING_MODEL,
    LLM_MODEL
)

from database.postgres import test_connection as test_postgres
from database.qdrant import test_connection as test_qdrant

print("\n===== DATABASE CONNECTIVITY TEST =====\n")

test_postgres()

test_qdrant()

# print("=== PROJECT CONFIGURATION ===\n")

# print(f"Project Root      : {PROJECT_ROOT}")
# print(f"Raw Data Folder   : {RAW_DATA_DIR}")

# print("\nPostgreSQL Config")
# print(POSTGRES)

# print("\nQdrant Config")
# print(QDRANT)

# print("\nModels")
# print(f"Embedding Model : {EMBEDDING_MODEL}")
# print(f"LLM Model       : {LLM_MODEL}")