"""
ST_8 Source Code
================

Modules:
    core_math_v2 - Discrete Exterior Calculus (DEC) library
    physics      - Physics layer (Bloch, bath, phonons)
    tests        - Test suite

Requirements:
    Python >= 3.9
    numpy >= 1.20
    scipy >= 1.11
"""

import sys

# Python version check
if sys.version_info < (3, 9):
    raise ImportError(f"ST_8 requires Python >= 3.9, got {sys.version}")

# scipy version check (critical for Voronoi stability)
import scipy
_scipy_version = tuple(int(p) for p in scipy.__version__.split('.')[:2] if p.isdigit())
if _scipy_version < (1, 11):
    raise ImportError(f"ST_8 requires scipy >= 1.11, got {scipy.__version__}")

# numpy version check
import numpy as np
_numpy_version = tuple(int(p) for p in np.__version__.split('.')[:2] if p.isdigit())
if _numpy_version < (1, 20):
    raise ImportError(f"ST_8 requires numpy >= 1.20, got {np.__version__}")
