#!/usr/bin/env python3
"""
Issue 4 Debug Script: Bath → P_L Investigation
==============================================

This script runs the diagnostic/exploration functions for Issue 4
(local bath produces P_L via Schur complement).

The actual diagnostic functions (run_*, main) are in
tests/physics/test_bath_internals.py for historical reasons.
This wrapper provides a clean entry point.

Usage:
    cd ST_8/src
    python scripts/issue4_debug.py

Jan 2026
"""

import sys
from pathlib import Path

# Add src/ to path
src_root = Path(__file__).parent.parent.resolve()
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

# Import the main function from test file
# Note: numeric prefix requires importlib
import importlib
bath_tests = importlib.import_module("tests.physics.01_test_bath_internals")
main = bath_tests.main


if __name__ == "__main__":
    results = main()
