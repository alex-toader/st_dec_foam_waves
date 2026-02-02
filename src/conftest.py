"""
Pytest Configuration
====================

Automatically loaded by pytest. Adds src/ to sys.path
so all modules are importable without installation.

Portable - works wherever the project is cloned.

Usage:
    cd ST_8/src
    pytest tests/ -v
"""

import sys
from pathlib import Path


def pytest_configure(config):
    """Add src/ to path before any imports happen."""
    src_root = Path(__file__).parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


# Also do it at module level for non-pytest usage
src_root = Path(__file__).parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
