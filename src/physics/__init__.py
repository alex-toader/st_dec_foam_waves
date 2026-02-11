"""
Physics Layer for ST_8
======================

Built on top of core_math (pure geometry/DEC).

Modules:
    constants - Numerical thresholds and defaults
    bloch     - Bloch wave formulation for periodic structures
    bath      - k-space bath and Schur complement
    hodge     - Voronoi dual Hodge stars for DEC

Classes:
    DisplacementBloch - Main class for phonon band structure (recommended)
    BlochComplex      - DEC-based formulation (deprecated, has known issues)

Hodge Stars:
    build_hodge_stars_uniform  - Simplified uniform cubic lattice (a²·I)
    build_hodge_stars_voronoi  - Correct DEC using Voronoi dual geometry

Jan 2026
"""

# Constants (import first, used by other modules)
from .constants import (
    ZERO_K_THRESHOLD,
    ZERO_EIGENVALUE_THRESHOLD,
    COEFFICIENT_THRESHOLD,
    PSEUDOINVERSE_CUTOFF,
    REGULARIZATION_DEFAULT,  # deprecated alias for PSEUDOINVERSE_CUTOFF
    DISPERSION_K_MIN,
    get_kelvin_builder,
)

# Bloch module
from .bloch import (
    # Main class (recommended)
    DisplacementBloch,
    # Deprecated class (has known exactness bug for k≠0)
    BlochComplex,
    # Utility functions
    compute_edge_crossings,
    compute_edge_geometry,
    build_edge_lookup,
    build_d0_bloch,
    build_d1_bloch,
    build_hodge_stars_uniform,
    build_L_elastic,
)

# Bath module
from .bath import (
    # Discrete operators
    build_vertex_laplacian_bloch,
    build_divergence_operator_bloch,
    compute_discrete_schur,
    build_PL_full,
    # Continuum reference functions
    continuum_P_L,
    bath_schur_continuum,
)

# Gauge Bloch module (elastic-EM bridge)
from .gauge_bloch import (
    compare_gauge_elastic,
    extract_gauge_speeds,
    compute_anisotropy,
    pearson_correlation,
)

# Hodge module (Voronoi dual)
from .hodge import (
    # Generic builder
    build_foam_with_dual_info,
    # Specific lattice builders
    build_c15_with_dual_info,
    build_kelvin_with_dual_info,
    build_wp_with_dual_info,
    # Hodge star computation
    build_hodge_stars_voronoi,
    # Verification
    verify_plateau_structure,
    verify_voronoi_property,
    # Utilities
    wrap_delta,
    get_c15_points,
    get_bcc_points,
    get_a15_points,
)

# Convenience alias
build_kelvin_supercell_periodic = get_kelvin_builder()
