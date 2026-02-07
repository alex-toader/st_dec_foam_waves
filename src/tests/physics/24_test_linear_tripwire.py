#!/usr/bin/env python3
"""
T5: Linear Term Tripwire — verify a₁ = 0 in phase velocity dispersion

Tests that the leading correction in PHASE VELOCITY is quadratic in |k|,
not linear. Fits v_phase = ω/|k| = v₀ + a₁|k| + a₂|k|² and requires a₁ ≈ 0.

If a₁ = 0: ω = v₀|k| + a₂|k|³ + ... (standard acoustic, correction is O(k³))
If a₁ ≠ 0: ω = v₀|k| + a₁|k|² + ... (anomalous linear correction in v_phase)

Methodological notes:
  - Phase velocity (not group velocity) is the natural observable for this test
  - Branch ordering verified via classify_modes: omega[0]=T1 for all structures
  - Fit window stability verified: a₁ stays < 10⁻⁵ across k-windows [0.002,0.01]
    through [0.005,0.10] — not a fit artifact

Tests run per structure (SC, FCC, WP, C15) independently.
All use N=1 (except SC which needs N=3).

Feb 2026

OUTPUT:
=======================================================
T5 TESTS (per structure, classify_modes branch tracking)
=======================================================

--- SC (exact benchmark) ---
     Dir    Branch          v₀            a₁     |a₁|/v₀
-------------------------------------------------------
   [100]        T1      2.0000   -2.5480e-06    1.27e-06
                T2      2.0000   -2.5539e-06    1.28e-06
                 L      3.4641   -4.4286e-06    1.28e-06
   [110]        T1      2.0000   -6.4219e-07    3.21e-07
                T2      2.8284   -9.0756e-07    3.21e-07
                 L      2.8284   -9.0390e-07    3.20e-07
   [111]        T1      2.5820   -3.6918e-07    1.43e-07
                T2      2.5820   -3.7023e-07    1.43e-07
                 L      2.5820   -3.6732e-07    1.42e-07
T5.1 PASS: SC max |a₁|/v₀ = 1.28e-06

T5.2 PASS: FCC max |a₁|/v₀ = 2.39e-06
T5.3 PASS: WP N=1 max |a₁|/v₀ = 3.35e-07
T5.3b PASS: WP N=2 max |a₁|/v₀ = 3.16e-07
T5.4 PASS: C15 max |a₁|/v₀ = 2.43e-07

Summary:
| Structure | max |a₁|/v₀ | Threshold  |
|-----------|-------------|------------|
| SC        | 1.28e-06    | < 5e-06    |
| FCC       | 2.39e-06    | < 1e-04    |
| WP N=1    | 3.35e-07    | < 1e-05    |
| WP N=2    | 3.16e-07    | < 5e-05    |
| C15       | 2.43e-07    | < 1e-05    |

T5.5 PASS: fit window stability, max |a₁|/v₀ = 1.60e-06 (< 1e-4)
T5.6 PASS: branch ordering stable across k-grid (WP, C15)

All < 10⁻⁵. Linear term confirmed zero. Dispersion is ω = v₀|k| + O(|k|³).
Branch tracking via classify_modes ensures consistent T1/T2/L identity.

ALL T5 TESTS PASS (7/7)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_math_v2.builders.solids_periodic import (
    build_sc_supercell_periodic,
    build_fcc_supercell_periodic,
)
from core_math_v2.builders.weaire_phelan_periodic import build_wp_supercell_periodic
from core_math_v2.builders.c15_periodic import build_c15_supercell_periodic
from physics.bloch import DisplacementBloch


# =============================================================================
# ANALYSIS
# =============================================================================

def fit_dispersion(bloch, direction, k_values):
    """
    Fit v_phase = ω/|k| = v₀ + a₁|k| + a₂|k|² for each acoustic branch.

    Uses classify_modes to track branches as (T1, T2, L) consistently
    across the k-grid, avoiding mode-crossing artifacts.

    Returns list of (v0, a1, a2) for branches [T1, T2, L].
    """
    k_hat = np.array(direction, dtype=float)
    k_hat = k_hat / np.linalg.norm(k_hat)

    # Collect phase velocities with proper branch identification
    v_T1, v_T2, v_L = [], [], []
    for kappa in k_values:
        k = kappa * k_hat
        omega_T, omega_L, f_L = bloch.classify_modes(k)
        v_T1.append(min(omega_T) / kappa)
        v_T2.append(max(omega_T) / kappa)
        v_L.append(omega_L[0] / kappa)

    results = []
    for branch_data in [v_T1, v_T2, v_L]:
        y = np.array(branch_data)
        A = np.vstack([np.ones_like(k_values), k_values, k_values**2]).T
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
        v0, a1, a2 = coeffs
        results.append((v0, a1, a2))

    return results


BRANCH_LABELS = ["T1", "T2", "L"]


def _test_structure(bloch, name, directions, k_values):
    """
    Test a₁ ≈ 0 for one structure across given directions.

    Uses classify_modes for consistent branch tracking (T1, T2, L).
    Returns max |a₁|/v₀ found.
    """
    max_ratio = 0

    print(f"{'Dir':>8}  {'Branch':>8}  {'v₀':>10}  {'a₁':>12}  {'|a₁|/v₀':>10}")
    print("-" * 55)

    for label, direction in directions:
        fits = fit_dispersion(bloch, direction, k_values)
        for i, (v0, a1, a2) in enumerate(fits):
            ratio = abs(a1) / v0 if v0 > 0.01 else 0
            max_ratio = max(max_ratio, ratio)
            print(f"{label if i==0 else '':>8}  {BRANCH_LABELS[i]:>8}  {v0:>10.4f}  "
                  f"{a1:>12.4e}  {ratio:>10.2e}")

    return max_ratio


# =============================================================================
# BUILDERS (each builds ONE structure)
# =============================================================================

def build_sc(k_L=3.0, k_T=1.0):
    V, E, F, _ = build_sc_supercell_periodic(3)
    return DisplacementBloch(V, E, 6.0, k_L=k_L, k_T=k_T)


def build_fcc(k_L=3.0, k_T=1.0):
    V, E, F, _ = build_fcc_supercell_periodic(1)
    return DisplacementBloch(V, E, 4.0, k_L=k_L, k_T=k_T)


def build_wp(k_L=3.0, k_T=1.0):
    V, E, F = build_wp_supercell_periodic(1)
    return DisplacementBloch(V, E, 4.0, k_L=k_L, k_T=k_T)


def build_c15(k_L=3.0, k_T=1.0):
    V, E, F, _ = build_c15_supercell_periodic(1)
    return DisplacementBloch(V, E, 4.0, k_L=k_L, k_T=k_T)


# =============================================================================
# TESTS (independent per structure)
# =============================================================================

DIRECTIONS = [("[100]", [1, 0, 0]), ("[110]", [1, 1, 0]), ("[111]", [1, 1, 1])]
K_VALUES = np.linspace(0.005, 0.05, 8)


def test_sc_a1_zero():
    """SC: a₁ should be exactly 0 (by symmetry)."""
    bloch = build_sc()
    print("--- SC (exact benchmark) ---")
    max_r = _test_structure(bloch, "SC", DIRECTIONS, K_VALUES)
    assert max_r < 5e-6, f"SC a₁/v₀ should be ~0, got {max_r:.2e}"
    print(f"T5.1 PASS: SC max |a₁|/v₀ = {max_r:.2e}")
    print()


def test_fcc_a1_small():
    """FCC: a₁ should be small."""
    bloch = build_fcc()
    print("--- FCC ---")
    max_r = _test_structure(bloch, "FCC", DIRECTIONS, K_VALUES)
    assert max_r < 1e-4, f"FCC |a₁|/v₀ should be < 1e-4, got {max_r:.2e}"
    print(f"T5.2 PASS: FCC max |a₁|/v₀ = {max_r:.2e}")
    print()


def test_wp_a1_small():
    """WP (N=1): a₁ should be small."""
    bloch = build_wp()
    print("--- WP N=1 ---")
    max_r = _test_structure(bloch, "WP", DIRECTIONS, K_VALUES)
    assert max_r < 1e-5, f"WP |a₁|/v₀ should be < 1e-5, got {max_r:.2e}"
    print(f"T5.3 PASS: WP N=1 max |a₁|/v₀ = {max_r:.2e}")
    print()


def test_wp_n2_cross_check():
    """WP N=2: verify N=1 result is not small-N artifact."""
    V, E, F = build_wp_supercell_periodic(2)
    bloch = DisplacementBloch(V, E, 8.0, k_L=3.0, k_T=1.0)
    # Fewer directions and k-points for speed (D=1104)
    dirs_short = [("[100]", [1, 0, 0]), ("[111]", [1, 1, 1])]
    kv_short = np.linspace(0.005, 0.05, 6)
    print("--- WP N=2 (cross-check) ---")
    max_r = _test_structure(bloch, "WP N=2", dirs_short, kv_short)
    assert max_r < 5e-5, f"WP N=2 |a₁|/v₀ should be < 5e-5, got {max_r:.2e}"
    print(f"T5.3b PASS: WP N=2 max |a₁|/v₀ = {max_r:.2e}")
    print()


def test_c15_a1_small():
    """C15 (N=1): a₁ should be small."""
    bloch = build_c15()
    print("--- C15 ---")
    max_r = _test_structure(bloch, "C15", DIRECTIONS, K_VALUES)
    assert max_r < 1e-5, f"C15 |a₁|/v₀ should be < 1e-5, got {max_r:.2e}"
    print(f"T5.4 PASS: C15 max |a₁|/v₀ = {max_r:.2e}")
    print()


def test_fit_window_stability():
    """a₁ should stay small across different k-windows (not a fit artifact)."""
    bloch = build_c15()
    windows = [
        ("narrow", np.linspace(0.002, 0.01, 8)),
        ("medium", np.linspace(0.002, 0.02, 8)),
        ("wide",   np.linspace(0.005, 0.05, 8)),
        ("wider",  np.linspace(0.005, 0.10, 8)),
    ]

    max_across_windows = 0
    for label, kv in windows:
        fits = fit_dispersion(bloch, [1, 1, 1], kv)
        max_r = max(abs(a1) / v0 for v0, a1, a2 in fits if v0 > 0.01)
        max_across_windows = max(max_across_windows, max_r)

    assert max_across_windows < 1e-4, \
        f"a₁ should be stable across k-windows, max = {max_across_windows:.2e}"
    print(f"T5.5 PASS: fit window stability, max |a₁|/v₀ = {max_across_windows:.2e} (< 1e-4)")
    print()


def test_branch_ordering_stable():
    """classify_modes should give consistent branch identity across k-grid."""
    for name, builder in [("WP", build_wp), ("C15", build_c15)]:
        bloch = builder()
        for label, direction in DIRECTIONS:
            k_hat = np.array(direction, dtype=float)
            k_hat /= np.linalg.norm(k_hat)

            prev_vT1, prev_vT2, prev_vL = None, None, None
            for kappa in K_VALUES:
                k = kappa * k_hat
                omega_T, omega_L, f_L = bloch.classify_modes(k)
                vT1 = min(omega_T) / kappa
                vT2 = max(omega_T) / kappa
                vL = omega_L[0] / kappa

                if prev_vT1 is not None:
                    # Velocities should be stable (< 1% change between k-points)
                    for v_now, v_prev, bname in [(vT1, prev_vT1, "T1"),
                                                  (vT2, prev_vT2, "T2"),
                                                  (vL, prev_vL, "L")]:
                        jump = abs(v_now - v_prev) / v_prev
                        assert jump < 0.01, \
                            f"{name} {label} {bname}: velocity jumped {jump:.1%} between k-points"

                prev_vT1, prev_vT2, prev_vL = vT1, vT2, vL

    print(f"T5.6 PASS: branch ordering stable across k-grid (WP, C15)")
    print()


def run_tests():
    print()
    print("=" * 55)
    print("T5 TESTS (per structure)")
    print("=" * 55)
    print()

    test_sc_a1_zero()
    test_fcc_a1_small()
    test_wp_a1_small()
    test_wp_n2_cross_check()
    test_c15_a1_small()
    test_fit_window_stability()
    test_branch_ordering_stable()

    print("ALL T5 TESTS PASS (7/7)")
    print()


if __name__ == '__main__':
    if '--test' in sys.argv:
        run_tests()
    else:
        run_tests()
