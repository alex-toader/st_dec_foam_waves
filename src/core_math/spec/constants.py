"""
Global constants for core_math
=================================

All tolerances and magic numbers in ONE place.
"""

# Numerical tolerances
EPS_ZERO = 1e-12       # For "is this zero?"
EPS_CLOSE = 1e-10      # For "are these equal?" (exact combinatorics, integer-derived)
EPS_ORTHO = 1e-8       # For orthogonality checks

# Geometry tolerances (less strict than EPS_CLOSE, for floating-point geometry)
PLANAR_TOL = 1e-8      # SVD plane deviation tests (faces must be planar within this)
EDGE_TOL = 1e-3        # abs(d - min_dist) < EDGE_TOL for WP Type A axis-axis edges
EDGE_TOL_POLY = 1e-3   # Edge detection in polyhedra.py (truncated cube)
FACE_TOL_POLY = 0.01   # Face membership in polyhedra.py (vertex on face plane)

# Periodic builder coordinate precision
WRAP_DECIMALS = 6      # wrap_position() rounds to this many decimals
# REQUIREMENT: Periodic builders (BCC, SC, FCC) only accept geometries with
# coordinates on a discrete grid or with vertex separation > 10^(-WRAP_DECIMALS).
# Jittered/floating coordinates may collapse vertices under wrap_position().

# Weaire-Phelan Type B rounding precision
WP_ROUND = 10          # round(..., WP_ROUND) for vertex coordinates in build_wp_type_b
# Must match between raw_vertices creation and v_to_idx lookup

# Default random seed (for reproducibility)
DEFAULT_SEED = 42

# Geometry constants (derived, not arbitrary)
SQRT2 = 1.4142135623730951
SQRT3 = 1.7320508075688772

# Complex type constants
COMPLEX_SURFACE = "surface"  # 2-manifold boundary, 2 faces/edge
COMPLEX_FOAM = "foam"        # 3D Plateau structure, 3 faces/edge (soap films)
COMPLEX_TILING = "tiling"    # Crystal tiling, 3 faces/edge (FCC rhombic dodecahedra)
COMPLEX_SOLID = "solid"      # SC lattice, 4 faces/edge

# Expected faces per edge by type
# NOTE: FOAM and TILING both have k=3, but different physics:
#   - FOAM: Plateau borders, soap film minimization, κ-locking
#   - TILING: Crystal packing, no surface tension physics
# Math tests (trace identities) are identical; physics gates differ.
FACES_PER_EDGE = {
    COMPLEX_SURFACE: 2,
    COMPLEX_FOAM: 3,
    COMPLEX_TILING: 3,  # Same k as foam, different physics
    COMPLEX_SOLID: 4,
}

# =============================================================================
# DEC OPERATOR CONVENTIONS
# =============================================================================
#
# BOUNDARY OPERATORS (exterior derivative):
#   d₀: C⁰ → C¹  (gradient, V → E)     shape: (E, V)
#   d₁: C¹ → C²  (curl, E → F)         shape: (F, E)
#   d₂: C² → C³  (divergence, F → C)   shape: (C, F)
#
# EXACTNESS (discrete analog of ∂² = 0):
#   d₁ d₀ = 0   (curl of gradient is zero)
#   d₂ d₁ = 0   (divergence of curl is zero)
#
# d₂ CONVENTION:
#   d₂[c, f] = +1 if face f is on boundary of cell c with OUTWARD normal
#   d₂[c, f] = -1 if face f is on boundary of cell c with INWARD normal
#   d₂[c, f] = 0  if face f is not on boundary of cell c
#
#   This matches the cell_face_incidence convention in multicell_periodic.py:
#   cell_face_incidence[c] = [(face_idx, orientation), ...]
#   where orientation = +1 means outward, -1 means inward.
#
# PROOF OF d₂d₁ = 0:
#   Each edge e appears in exactly k faces (Plateau structure).
#   For foam (k=3): each edge bounds 3 faces, which belong to exactly 2 cells.
#   The alternating signs around the edge cancel in the product.
#
