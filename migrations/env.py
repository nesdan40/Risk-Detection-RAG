"""
Alembic migration environment for the Risk Detection RAG project.

- Reads database connection settings from .env via config.settings.
- Points target_metadata at all ORM models so autogenerate works correctly.
- Supports both offline (SQL script) and online (live connection) migration modes.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# ── Make src/ importable ────────────────────────────────────────────────────
# Alembic runs from the project root; we need src/ on the path so that
# `from models import ...` and `from config.settings import ...` resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Load .env before importing settings ─────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

# ── Import all ORM models so autogenerate sees every table ──────────────────
from models import Base  # noqa: E402  (imports all models transitively)

# ── Alembic config ───────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# ── Override the DB URL from environment variables ───────────────────────────
# This ensures .env is always the single source of truth for the connection.
_db_url = (
    f"postgresql+psycopg2://"
    f"{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@"
    f"{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/"
    f"{os.environ['POSTGRES_DB']}"
)
config.set_main_option("sqlalchemy.url", _db_url)


def run_migrations_offline() -> None:
    """Emit migration SQL to stdout without a live DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
