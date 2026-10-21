"""
db_setup.py — create all tables in financial_risk_db.

Two modes:
  1. Direct (default) — calls Base.metadata.create_all(), fastest for first-time setup.
  2. Alembic            — runs `alembic upgrade head` via subprocess, preferred when
                          you want the alembic_version table populated for future migrations.

Usage (from project root, with venv active):
    # Direct mode
    python src/db_setup.py

    # Alembic mode
    python src/db_setup.py --alembic
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Make src/ importable when this script is run directly ───────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from database.postgres import get_engine  # noqa: E402
from models import Base  # noqa: E402  — imports all models, registers metadata


def create_tables_direct() -> None:
    """Drop nothing; create any table that doesn't exist yet (safe to re-run)."""
    engine = get_engine()
    print("Creating tables via SQLAlchemy create_all …")
    Base.metadata.create_all(engine)
    created = sorted(Base.metadata.tables.keys())
    print(f"✓ Done. Tables registered: {created}")


def create_tables_alembic() -> None:
    """Run 'alembic upgrade head' from the project root."""
    alembic_exe = PROJECT_ROOT / ".venv" / "Scripts" / "alembic.exe"
    if not alembic_exe.exists():
        # Fall back to PATH-installed alembic
        alembic_exe = "alembic"

    print("Running alembic upgrade head …")
    result = subprocess.run(
        [str(alembic_exe), "upgrade", "head"],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print("✗ alembic upgrade head failed — see output above.")
        sys.exit(result.returncode)
    print("✓ Alembic upgrade head completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create all schema tables.")
    parser.add_argument(
        "--alembic",
        action="store_true",
        help="Use 'alembic upgrade head' instead of SQLAlchemy create_all",
    )
    args = parser.parse_args()

    if args.alembic:
        create_tables_alembic()
    else:
        create_tables_direct()


if __name__ == "__main__":
    main()
