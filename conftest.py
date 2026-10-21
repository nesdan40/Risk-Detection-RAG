"""
Root conftest.py — adds src/ to sys.path so all test modules can import
from the source packages (models, ingestion, config, etc.) without
installing the project as a package.
"""

import sys
from pathlib import Path

# Ensure src/ is importable from any test file
sys.path.insert(0, str(Path(__file__).parent / "src"))
