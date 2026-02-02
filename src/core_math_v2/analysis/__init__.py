"""
Analysis functions - depend on operators layer.

These functions compute derived quantities using DEC operators.
Separated from builders to maintain clean layering:
    builders → operators → spec
    analysis → operators → spec

Includes:
- kappa: κ computation for polyhedra
- verify_topology: SC/FCC structure verification
- weaire_phelan_kappa: WP κ computation
- random_foam: Voronoi foam statistics (Koide n tests)
"""

from .kappa import compute_kappa_for_polyhedron
from .verify_topology import verify_sc_solid_structure, verify_fcc_structure
from .weaire_phelan_kappa import compute_wp_kappa, compare_wp_kelvin

# Random foam analysis (Voronoi statistics, Koide n)
from .random_foam import (
    analyze_foam,
    analyze_foam_ghost,
    compute_statistics,
    quick_test_poisson,
    quick_test_jammed,
    koide_n_from_F,
    FoamStatistics,
)
