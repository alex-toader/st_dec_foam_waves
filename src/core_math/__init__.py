"""
CORE_MATH_V2 - Pure combinatorial/discrete geometry
====================================================

NO physics. NO wip imports. NO plotting.

Structure:
    builders/   - Geometry construction (Kelvin, multicell, etc.)
    operators/  - DEC operators (d0, d1, Hodge, parity)
    spec/       - Constants and mesh contract

All builders return a MESH DICT with:
    - V, E, F (geometry)
    - complex_type: "surface" or "foam"
    - faces_per_edge: 2 or 3
    - metadata

Jan 2026
"""

from . import builders
from . import operators
from . import spec
