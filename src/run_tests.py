#!/usr/bin/env python3
"""
Run All Tests
=============

Portable test runner for ST_8. Works from any location:
    python3 ST_8/src/run_tests.py

Or from within src/:
    python3 run_tests.py

The script automatically sets up the Python path.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run all tests."""
    # Get absolute path to src/
    src_root = Path(__file__).parent.resolve()

    # Set PYTHONPATH to include src/
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    if pythonpath:
        env['PYTHONPATH'] = f"{src_root}:{pythonpath}"
    else:
        env['PYTHONPATH'] = str(src_root)

    # Run pytest from src directory
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
        cwd=src_root,
        env=env,
    )

    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
