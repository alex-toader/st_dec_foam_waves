"""
Physics Layer Constants
=======================

Numerical thresholds and defaults used across physics modules.
Centralizing these prevents magic number proliferation.

Jan 2026
"""

# ---------------------------------------------------------------------
# NUMERICAL THRESHOLDS
# ---------------------------------------------------------------------

# Zero detection for wave vector magnitude |k|
# Used to check if we're at Γ point (k=0) where P_L is undefined
ZERO_K_THRESHOLD = 1e-12

# Zero detection for eigenvalues and norms
# Used for detecting zero modes, null spaces, small coefficients
ZERO_EIGENVALUE_THRESHOLD = 1e-10

# Threshold for skipping negligible matrix coefficients
# Used in dynamical matrix construction
COEFFICIENT_THRESHOLD = 1e-15

# ---------------------------------------------------------------------
# ALGORITHM DEFAULTS
# ---------------------------------------------------------------------

# Cutoff threshold for pseudoinverse computation in Schur complement.
# Eigenvalues with |λ| < cutoff are treated as zero (hard threshold).
# NOTE: This is a thresholded pseudoinverse, NOT Tikhonov/ridge regularization.
# Ridge would use 1/(λ+α) for all λ; we use 1/λ if |λ|>cutoff, else 0.
PSEUDOINVERSE_CUTOFF = 1e-10

# Backward compatibility alias (deprecated name)
REGULARIZATION_DEFAULT = PSEUDOINVERSE_CUTOFF

# Minimum k magnitude for dispersion analysis
# Avoids numerical issues at exact k=0
DISPERSION_K_MIN = 1e-3

# ---------------------------------------------------------------------
# PERIOD CONVENTIONS (L values for each builder)
# ---------------------------------------------------------------------
#
# CRITICAL: The period L passed to DisplacementBloch must match the
# actual periodic cell size. Mismatch causes incorrect Bloch phases.
#
# Builder                      | Period L for N cells
# -----------------------------|---------------------
# build_sc_supercell_periodic  | L = 2 * N
# build_fcc_supercell_periodic | L = 2 * N
# build_bcc_supercell_periodic | L = 4 * N  (Kelvin cell unit = 4)
#
# Example usage:
#   v, e, f, _ = build_sc_supercell_periodic(N=3)
#   L = 2.0 * 3  # = 6.0
#   db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)
#
#   v, e, f, _ = build_bcc_supercell_periodic(N=2)  # Kelvin
#   L = 4.0 * 2  # = 8.0
#   db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)
#
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# COMPATIBILITY ALIASES
# ---------------------------------------------------------------------

# Kelvin foam = BCC foam (same structure, different historical names)
# Import in one place to avoid scattered aliases
def get_kelvin_builder():
    """Get the Kelvin supercell builder (alias for BCC)."""
    from core_math.builders import build_bcc_supercell_periodic
    return build_bcc_supercell_periodic
