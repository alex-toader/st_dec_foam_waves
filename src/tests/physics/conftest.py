"""Pytest configuration for physics tests."""
import sys
from pathlib import Path


def pytest_configure(config):
    """Add src/ to path before any test imports."""
    src_root = Path(__file__).parent.parent.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


# Also do it at module level for import ordering
src_root = Path(__file__).parent.parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
