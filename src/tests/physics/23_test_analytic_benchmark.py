#!/usr/bin/env python3
"""
T4: Analytic Benchmark — SC lattice acoustic velocities

Compares DisplacementBloch output against exact Born-von Kármán formulas
for SC lattice with tensorial springs K_e = k_T I + (k_L - k_T)(ê⊗ê).

SC has bonds along cubic axes only → dynamical matrix is DIAGONAL for all k.
This gives exact formulas for acoustic velocities:

  [100]: v_L = a√(k_L/m),  v_T = a√(k_T/m)  (T degenerate)
  [110]: v_xy = a√((k_L+k_T)/(2m)),  v_z = a√(k_T/m)
  [111]: v = a√((k_L+2k_T)/(3m))  (all 3 degenerate)

where a = lattice constant (= 2 in our builder), m = mass.

PURPOSE: Eliminates risk of subtle bug in phase factors / edge crossings.
If DisplacementBloch matches exact formulas → implementation correct.

Feb 2026

OUTPUT:
======================================================================
T4: ANALYTIC BENCHMARK — SC LATTICE
======================================================================

SC lattice: N=3, V=27, E=81
Parameters: k_L=3.0, k_T=1.0, a=2.0, m=1.0, Period L=6.0

 Direction    Branch       Exact     Numeric         Fit    Error(fit)
----------------------------------------------------------------------
     [100]        v1      2.0000      2.0000      1.9994      3.02e-04
                  v2      2.0000      2.0000      1.9994      3.02e-04
                  v3      3.4641      3.4640      3.4631      3.02e-04
     [110]        v1      2.0000      2.0000      1.9997      1.51e-04
                  v2      2.8284      2.8284      2.8280      1.51e-04
                  v3      2.8284      2.8284      2.8280      1.51e-04
     [111]        v1      2.5820      2.5820      2.5817      1.01e-04
                  v2      2.5820      2.5820      2.5817      1.01e-04
                  v3      2.5820      2.5820      2.5817      1.01e-04
Maximum error (fit): 3.02e-04

Degeneracies:
  [100] T degeneracy: v1/v2 - 1 = 1.10e-12  (exact)
  [111] triple degeneracy: max split = 1.05e-12  (exact)

Hermiticity: ||D - D†|| = 0.00e+00

k_L/k_T ratio scan (1 to 10): all match exact formulas (err = 3.02e-04)

CONCLUSION:
  DisplacementBloch MATCHES exact Born-von Kármán formulas.
  No bug in phase factors / edge crossings.

TESTS:
T4.1 PASS: [100] velocities match exact (max err < 0.5%)
T4.2 PASS: [110] velocities match exact (max err < 0.5%)
T4.3 PASS: [111] velocities match exact, degenerate (spread=1.05e-12)
T4.4 PASS: [100] T degeneracy (split=1.10e-12)
T4.5 PASS: Hermiticity < 1e-12 for all k
T4.6 PASS: Isotropic (k_L=k_T) → all modes degenerate
T4.7 PASS: acoustic gap ω₃/ω₂ > 10 for all directions
ALL T4 TESTS PASS (7/7)

NOTE on D(k) diagonality: the docstring states D(k) is diagonal for SC.
This is true for the PRIMITIVE cell (1 atom, 3×3 matrix). Our builder
uses N=3 supercell (27 atoms, 81×81 matrix) where atoms within the
supercell are coupled → D(k) has off-diagonal blocks. The velocity
match to exact formulas IS the validation that the supercell Bloch
construction correctly reproduces the primitive-cell physics.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core_math.builders.solids_periodic import build_sc_supercell_periodic
from physics.bloch import DisplacementBloch


# =============================================================================
# EXACT FORMULAS
# =============================================================================

def sc_velocity_100(k_L, k_T, a, m):
    """Exact acoustic velocities along [100] for SC."""
    v_L = a * np.sqrt(k_L / m)
    v_T = a * np.sqrt(k_T / m)  # doubly degenerate
    return sorted([v_T, v_T, v_L])


def sc_velocity_110(k_L, k_T, a, m):
    """Exact acoustic velocities along [110] for SC.

    D_xx = D_yy = (2/m)(k_L+k_T)(1-cos(ka/√2))  → v = a√((k_L+k_T)/(2m))
    D_zz = (2/m)(2k_T)(1-cos(ka/√2))              → v = a√(k_T/m)

    The two xy modes are NOT L/T separated (both have same speed).
    """
    v_xy = a * np.sqrt((k_L + k_T) / (2 * m))
    v_z = a * np.sqrt(k_T / m)
    return sorted([v_z, v_xy, v_xy])


def sc_velocity_111(k_L, k_T, a, m):
    """Exact acoustic velocities along [111] for SC.

    D_xx = D_yy = D_zz = (2/m)(k_L + 2k_T)(1-cos(ka/√3))
    All 3 modes degenerate: v = a√((k_L + 2k_T)/(3m))
    """
    v = a * np.sqrt((k_L + 2 * k_T) / (3 * m))
    return [v, v, v]


# =============================================================================
# NUMERICAL MEASUREMENT
# =============================================================================

def measure_velocities(bloch, direction, k_small=0.01):
    """
    Measure acoustic velocities from DisplacementBloch at small k.

    Returns sorted [v1, v2, v3] where v = ω/|k|.
    """
    k_hat = np.array(direction, dtype=float)
    k_hat = k_hat / np.linalg.norm(k_hat)

    k = k_small * k_hat
    omega = bloch.frequencies(k)

    # First 3 non-zero frequencies are acoustic
    # At small k they should be ≈ v * |k|
    v = omega[:3] / k_small

    return sorted(v)


def measure_velocities_fit(bloch, direction, n_points=5, k_max=0.05):
    """
    Measure acoustic velocities by linear fit ω vs |k|.

    More accurate than single-point: fits v = dω/dk from multiple k.
    """
    k_hat = np.array(direction, dtype=float)
    k_hat = k_hat / np.linalg.norm(k_hat)

    k_values = np.linspace(0.005, k_max, n_points)
    omega_all = []

    for kappa in k_values:
        k = kappa * k_hat
        omega = bloch.frequencies(k)[:3]
        omega_all.append(omega)

    omega_all = np.array(omega_all)

    # Fit ω = v * |k| for each branch
    velocities = []
    for branch in range(3):
        # Linear fit through origin: v = Σ(ω·k) / Σ(k²)
        v = np.sum(omega_all[:, branch] * k_values) / np.sum(k_values**2)
        velocities.append(v)

    return sorted(velocities)


# =============================================================================
# MAIN
# =============================================================================

def run_analysis():
    print("=" * 70)
    print("T4: ANALYTIC BENCHMARK — SC LATTICE")
    print("=" * 70)
    print()

    # Build SC periodic lattice
    N = 3  # minimum for SC periodic
    k_L = 3.0
    k_T = 1.0
    m = 1.0

    vertices, edges, faces, _ = build_sc_supercell_periodic(N)
    L = 2.0 * N  # period
    a = 2.0  # lattice constant (SC cell side)

    bloch = DisplacementBloch(vertices, edges, L, k_L=k_L, k_T=k_T, mass=m)

    print(f"SC lattice: N={N}, V={len(vertices)}, E={len(edges)}")
    print(f"Parameters: k_L={k_L}, k_T={k_T}, a={a}, m={m}")
    print(f"Period L={L}")
    print()

    # --- Test directions ---
    directions = [
        ("[100]", [1, 0, 0], sc_velocity_100),
        ("[110]", [1, 1, 0], sc_velocity_110),
        ("[111]", [1, 1, 1], sc_velocity_111),
    ]

    print(f"{'Direction':>10}  {'Branch':>8}  {'Exact':>10}  {'Numeric':>10}  {'Fit':>10}  {'Error(fit)':>12}")
    print("-" * 70)

    max_error = 0

    for label, direction, formula in directions:
        v_exact = formula(k_L, k_T, a, m)
        v_numeric = measure_velocities(bloch, direction, k_small=0.01)
        v_fit = measure_velocities_fit(bloch, direction, n_points=5, k_max=0.05)

        for i in range(3):
            err = abs(v_fit[i] - v_exact[i]) / v_exact[i]
            max_error = max(max_error, err)
            print(f"{label if i==0 else '':>10}  {'v'+str(i+1):>8}  {v_exact[i]:>10.4f}  "
                  f"{v_numeric[i]:>10.4f}  {v_fit[i]:>10.4f}  {err:>12.2e}")

    print("-" * 70)
    print(f"Maximum error (fit): {max_error:.2e}")
    print()

    # --- Degeneracies ---
    print("--- Degeneracies ---")
    print()

    v100 = measure_velocities_fit(bloch, [1, 0, 0])
    v110 = measure_velocities_fit(bloch, [1, 1, 0])
    v111 = measure_velocities_fit(bloch, [1, 1, 1])

    degen_100 = abs(v100[0] - v100[1]) / v100[0]
    degen_111 = max(abs(v111[i] - v111[j]) / v111[0]
                    for i in range(3) for j in range(i+1, 3))

    print(f"[100] T degeneracy: v1/v2 - 1 = {degen_100:.2e}  (expect 0)")
    print(f"[111] triple degeneracy: max split = {degen_111:.2e}  (expect 0)")
    print()

    # --- Hermiticity check ---
    print("--- Hermiticity ---")
    k_test = 0.1 * np.array([1, 1, 1]) / np.sqrt(3)
    herm = bloch.check_hermitian(k_test)
    print(f"||D - D†|| at k=[111]*0.1: {herm:.2e}")
    print()

    # --- Different k_L/k_T ratios ---
    print("--- k_L/k_T ratio scan ---")
    print()
    print(f"{'k_L/k_T':>10}  {'v_L[100]':>10}  {'v_T[100]':>10}  {'Exact_L':>10}  {'Exact_T':>10}  {'err_L':>10}  {'err_T':>10}")
    print("-" * 75)

    for ratio in [1.0, 2.0, 3.0, 5.0, 10.0]:
        kL = ratio
        kT = 1.0
        b = DisplacementBloch(vertices, edges, L, k_L=kL, k_T=kT, mass=m)
        v = measure_velocities_fit(b, [1, 0, 0])

        v_L_exact = a * np.sqrt(kL / m)
        v_T_exact = a * np.sqrt(kT / m)

        err_L = abs(v[2] - v_L_exact) / v_L_exact
        err_T = abs(v[0] - v_T_exact) / v_T_exact

        print(f"{ratio:>10.1f}  {v[2]:>10.4f}  {v[0]:>10.4f}  {v_L_exact:>10.4f}  {v_T_exact:>10.4f}  {err_L:>10.2e}  {err_T:>10.2e}")

    print()
    print("=" * 70)
    print("CONCLUSION:")
    print(f"  Max velocity error: {max_error:.2e}")
    if max_error < 0.01:
        print("  DisplacementBloch MATCHES exact Born-von Kármán formulas.")
        print("  No bug in phase factors / edge crossings for SC lattice.")
    else:
        print("  WARNING: Error exceeds 1%. Investigate.")
    print("=" * 70)


# =============================================================================
# TESTS
# =============================================================================

def _build_sc_bloch(k_L=3.0, k_T=1.0):
    """Helper: build SC DisplacementBloch."""
    N = 3
    vertices, edges, faces, _ = build_sc_supercell_periodic(N)
    L = 2.0 * N
    return DisplacementBloch(vertices, edges, L, k_L=k_L, k_T=k_T, mass=1.0)


def test_velocity_100():
    """Acoustic velocities along [100] match exact formula."""
    bloch = _build_sc_bloch()
    a, m, k_L, k_T = 2.0, 1.0, 3.0, 1.0

    v = measure_velocities_fit(bloch, [1, 0, 0])
    v_exact = sc_velocity_100(k_L, k_T, a, m)

    for i in range(3):
        err = abs(v[i] - v_exact[i]) / v_exact[i]
        assert err < 0.005, f"[100] branch {i}: err={err:.4f}, v={v[i]:.4f}, exact={v_exact[i]:.4f}"

    print(f"T4.1 PASS: [100] velocities match exact (max err < 0.5%)")


def test_velocity_110():
    """Acoustic velocities along [110] match exact formula."""
    bloch = _build_sc_bloch()
    a, m, k_L, k_T = 2.0, 1.0, 3.0, 1.0

    v = measure_velocities_fit(bloch, [1, 1, 0])
    v_exact = sc_velocity_110(k_L, k_T, a, m)

    for i in range(3):
        err = abs(v[i] - v_exact[i]) / v_exact[i]
        assert err < 0.005, f"[110] branch {i}: err={err:.4f}, v={v[i]:.4f}, exact={v_exact[i]:.4f}"

    print(f"T4.2 PASS: [110] velocities match exact (max err < 0.5%)")


def test_velocity_111():
    """Acoustic velocities along [111] match exact — all 3 degenerate."""
    bloch = _build_sc_bloch()
    a, m, k_L, k_T = 2.0, 1.0, 3.0, 1.0

    v = measure_velocities_fit(bloch, [1, 1, 1])
    v_exact = sc_velocity_111(k_L, k_T, a, m)

    for i in range(3):
        err = abs(v[i] - v_exact[i]) / v_exact[i]
        assert err < 0.005, f"[111] branch {i}: err={err:.4f}"

    # Check degeneracy
    spread = (max(v) - min(v)) / np.mean(v)
    assert spread < 0.01, f"[111] modes should be degenerate, spread={spread:.4f}"

    print(f"T4.3 PASS: [111] velocities match exact, degenerate (spread={spread:.2e})")


def test_degeneracy_100():
    """[100] transverse modes are degenerate."""
    bloch = _build_sc_bloch()
    v = measure_velocities_fit(bloch, [1, 0, 0])

    # v[0] and v[1] should be equal (both transverse)
    split = abs(v[0] - v[1]) / v[0]
    assert split < 0.005, f"[100] T modes should be degenerate, split={split:.4f}"

    print(f"T4.4 PASS: [100] T degeneracy (split={split:.2e})")


def test_hermiticity():
    """Dynamical matrix is Hermitian."""
    bloch = _build_sc_bloch()
    for direction in [[1, 0, 0], [1, 1, 0], [1, 1, 1], [0.3, 0.7, 0.5]]:
        k = 0.1 * np.array(direction) / np.linalg.norm(direction)
        h = bloch.check_hermitian(k)
        assert h < 1e-12, f"||D-D†||={h:.2e} for k∝{direction}"

    print(f"T4.5 PASS: Hermiticity < 1e-12 for all k")


def test_isotropic_limit():
    """When k_L = k_T, all 3 modes degenerate everywhere."""
    bloch = _build_sc_bloch(k_L=1.0, k_T=1.0)

    for direction in [[1, 0, 0], [1, 1, 0], [1, 1, 1]]:
        v = measure_velocities_fit(bloch, direction)
        spread = (max(v) - min(v)) / np.mean(v)
        assert spread < 0.005, f"Isotropic: modes not degenerate along {direction}, spread={spread:.4f}"

    print(f"T4.6 PASS: Isotropic (k_L=k_T) → all modes degenerate")


def test_acoustic_gap():
    """First 3 modes are acoustic (ω ~ O(k)), with large gap to optical."""
    bloch = _build_sc_bloch()
    for label, direction in [("[100]", [1, 0, 0]), ("[110]", [1, 1, 0]), ("[111]", [1, 1, 1])]:
        k_hat = np.array(direction, dtype=float)
        k_hat /= np.linalg.norm(k_hat)
        k = 0.01 * k_hat
        omega = bloch.frequencies(k)

        # Acoustic: omega[0:3] should scale as v*|k| (small)
        # Optical: omega[3] should have a gap (large compared to acoustic)
        gap_ratio = omega[3] / omega[2]
        assert gap_ratio > 10, f"{label}: acoustic gap too small, ω₃/ω₂ = {gap_ratio:.1f}"

    print(f"T4.7 PASS: acoustic gap ω₃/ω₂ > 10 for all directions")


def run_tests():
    print()
    print("=" * 50)
    print("T4 TESTS")
    print("=" * 50)
    print()

    test_velocity_100()
    test_velocity_110()
    test_velocity_111()
    test_degeneracy_100()
    test_hermiticity()
    test_isotropic_limit()
    test_acoustic_gap()

    print()
    print("ALL T4 TESTS PASS (7/7)")
    print()


if __name__ == '__main__':
    if '--test' in sys.argv:
        run_tests()
    else:
        run_analysis()
        run_tests()
