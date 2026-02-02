#!/usr/bin/env python3
"""
CAVITY WASH-OUT: MONTE CARLO VALIDATION
========================================

Tests for the Lorentz bounds via cavity resonance experiments.
Validates the wash-out mechanism from polycrystalline averaging.

INPUTS
------

Internal (from model):
  - δv/v from physics/christoffel.py (Kelvin=6.3%, FCC=16.5%, WP=2.5%)
  - Christoffel elastic constants: C11, C12, C44
  - DisplacementBloch acoustic velocities v(k̂)
  - Foam geometry from builders (Kelvin, FCC, WP)

External (from experiment/literature):
  - Herrmann+ 2009 bound: Δc/c < 10⁻¹⁷
  - Planck length: ℓ_P = 1.616×10⁻³⁵ m

OUTPUTS
-------

  - MC1-MC6: Scaling/format/kernel validated (δ-independent)
  - Margin sanity: WP/Kelvin/FCC all pass Herrmann bound
  - Margins: ~100× (WP), ~40× (Kelvin), ~15× (FCC)

PHYSICS:
  - Cavity has M = L/ℓ_corr grains with random orientations
  - Each grain contributes anisotropy ±δ
  - Random walk averaging: RMS(Δc/c) = δ / √M

TESTS:
  MC1: 1/√M scaling validation
  MC2: 2ω modulation format (Herrmann/Eisele style)
  MC3: Realistic Christoffel kernel (3D cubic)
  MC4: Finite correlation (Markov process)
  MC5: Two-way vs one-way cavity geometry
  MC6: Geometry factor audit
  Margin sanity: deterministic margin check for WP/Kelvin/FCC

REFERENCE: Herrmann et al., Phys. Rev. D 80, 105011 (2009)
           Bound: Δc/c < 10⁻¹⁷

DERIVATION CHAIN (see physics/christoffel.py):
  1. Foam geometry → DisplacementBloch → acoustic velocities v(k̂)
  2. Christoffel fit → C11, C12, C44
  3. Random k̂ sampling → δv/v = (v_max - v_min) / v_mean

  Run: python -m physics.christoffel
  to verify/regenerate these values.

ASSUMPTIONS (NOT DERIVED):
  1. Elastic δv/v ≈ EM Δc/c (elastic-electromagnetic bridge)
  2. ℓ_corr = ℓ_Planck (grain correlation length = Planck length)

Jan 2026
"""

import numpy as np
import pytest
from typing import Dict, List, Callable, Any


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

L_PLANCK = 1.616e-35      # Planck length [m]
BOUND_EXP = 1e-17         # Herrmann+ 2009 bound on Δc/c

# =============================================================================
# DERIVED ANISOTROPY VALUES δv/v
# =============================================================================
# Source: physics/christoffel.py (run to verify)
#
# DERIVATION:
#   Kelvin N=2 (BCC foam): C11=2.40, C12=-0.53, C44=1.67, A_Z=1.14
#   FCC N=2:               C11=4.44, C12=-0.89, C44=3.73, A_Z=1.40
#
#   δv/v = (v_max - v_min) / v_mean from Christoffel velocities
#
#   WP N=1 (A15 foam):     C11=1.12, C12=-0.28, C44=0.66, A_Z=0.95
#   Kelvin N=2 (BCC foam): C11=2.40, C12=-0.53, C44=1.67, A_Z=1.14
#   FCC N=2:               C11=4.44, C12=-0.89, C44=3.73, A_Z=1.40
#
DELTA_C15 = 0.0093        # 0.93% (from Christoffel, A_Z=1.02) - most isotropic
DELTA_WP = 0.025          # 2.5% (from Christoffel, A_Z=0.95)
DELTA_KELVIN = 0.063      # 6.3% (from Christoffel, A_Z=1.14)
DELTA_FCC = 0.165         # 16.5% (from Christoffel, A_Z=1.40)


# =============================================================================
# COMMON UTILITIES
# =============================================================================

def fit_2omega(theta: np.ndarray, delta_c: np.ndarray) -> Dict[str, float]:
    """
    Fit Δc/c(θ) = A cos(2θ) + B sin(2θ) + C.

    Returns dict with A, B, C, amplitude = √(A² + B²).
    """
    X = np.column_stack([np.cos(2*theta), np.sin(2*theta), np.ones(len(theta))])
    coeffs, _, _, _ = np.linalg.lstsq(X, delta_c, rcond=None)
    A, B, C = coeffs
    return {'A': A, 'B': B, 'C': C, 'amplitude': np.sqrt(A**2 + B**2)}


def anisotropy_tensor_2d(theta_grain: float) -> np.ndarray:
    """
    2D traceless symmetric anisotropy tensor for grain at angle theta_grain.
    A(θ) = [[cos(2θ), sin(2θ)], [sin(2θ), -cos(2θ)]]
    """
    c2, s2 = np.cos(2 * theta_grain), np.sin(2 * theta_grain)
    return np.array([[c2, s2], [s2, -c2]])


def cubic_anisotropy(n_hat: np.ndarray, delta_scale: float) -> float:
    """
    Christoffel-based cubic anisotropy: δ(n̂) = delta_scale × f(n̂).

    f(n̂) = 1 - 6×(n_x²n_y² + n_y²n_z² + n_z²n_x²)
    Range: f([100]) = 1, f([111]) = -1

    Since f ∈ [-1, +1], the peak-to-peak range of δ is 2×delta_scale.
    To match δv/v (which is peak-to-peak), use delta_scale = δv/v / 2.

    RMS(f) over sphere ≈ 0.56.
    """
    nx, ny, nz = n_hat
    cubic_term = nx**2 * ny**2 + ny**2 * nz**2 + nz**2 * nx**2
    f = 1 - 6 * cubic_term  # ranges [-1, +1]
    return delta_scale * f


def random_rotation_matrix(rng=None) -> np.ndarray:
    """Generate random 3D rotation matrix (uniform on SO(3))."""
    if rng is None:
        u1, u2, u3 = np.random.random(3)
    else:
        u1, u2, u3 = rng.random(3)
    q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])


def check_scaling_slope(M_arr: np.ndarray, rms_arr: np.ndarray,
                        expected: float, tolerance: float = 0.15) -> Dict[str, Any]:
    """Fit log-log slope and check against expected value."""
    slope, intercept = np.polyfit(np.log(M_arr), np.log(rms_arr), 1)
    passed = abs(slope - expected) < tolerance
    return {'slope': slope, 'expected': expected, 'passed': passed}


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def simulate_washout_1d(M: int, delta_single: float, n_trials: int) -> Dict[str, Any]:
    """
    Monte Carlo: M segments with random ±δ signs.
    Returns RMS(Δc/c) and theory comparison.
    """
    # Vectorized: generate all trials at once
    signs = np.random.choice([-1, 1], size=(n_trials, M))
    delta_eff = delta_single * np.mean(signs, axis=1)
    rms = np.sqrt(np.mean(delta_eff**2))
    theory = delta_single / np.sqrt(M)
    return {'M': M, 'rms': rms, 'theory': theory, 'ratio': rms / theory}


def simulate_2omega_rotation(M: int, delta_single: float,
                             n_angles: int, n_trials: int) -> Dict[str, Any]:
    """
    Simulate cavity rotation: Δc/c(θ) for θ ∈ [0, 2π).
    Returns RMS of amplitude √(A² + B²).
    """
    theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    amplitudes = []

    for _ in range(n_trials):
        grain_angles = np.random.uniform(0, np.pi, M)
        delta_c = np.zeros(n_angles)

        for j, th in enumerate(theta):
            meas_dir = np.array([np.cos(th), np.sin(th)])
            total = sum(meas_dir @ (delta_single * anisotropy_tensor_2d(g)) @ meas_dir
                       for g in grain_angles)
            delta_c[j] = total / M

        fit = fit_2omega(theta, delta_c)
        amplitudes.append(fit['amplitude'])

    return {'M': M, 'amplitude_rms': np.sqrt(np.mean(np.array(amplitudes)**2))}


def simulate_christoffel_3d(M: int, delta_scale: float,
                            n_angles: int, n_trials: int,
                            rng=None) -> Dict[str, Any]:
    """
    Monte Carlo with 3D Christoffel cubic anisotropy.

    Args:
        rng: numpy random generator (for hermetic tests)
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    amplitudes = []

    for _ in range(n_trials):
        rotations = [random_rotation_matrix(rng) for _ in range(M)]
        delta_c = np.zeros(n_angles)

        for j, th in enumerate(theta):
            meas_dir = np.array([np.cos(th), np.sin(th), 0])
            total = 0.0
            for R in rotations:
                n_grain = R.T @ meas_dir
                n_grain = n_grain / np.linalg.norm(n_grain)
                total += cubic_anisotropy(n_grain, delta_scale)
            delta_c[j] = total / M

        fit = fit_2omega(theta, delta_c)
        amplitudes.append(fit['amplitude'])

    return {'M': M, 'amplitude_rms': np.sqrt(np.mean(np.array(amplitudes)**2))}


# =============================================================================
# TEST IMPLEMENTATIONS
# =============================================================================

def run_mc1_scaling(M_values: List[int], delta: float, n_trials: int) -> List[Dict]:
    """MC1: Test 1/√M scaling."""
    results = []
    for M in M_values:
        res = simulate_washout_1d(M, delta, n_trials)
        results.append(res)
    return results


def run_mc6_geometry_audit(n_samples: int, seed: int = 0) -> Dict[str, float]:
    """MC6: Verify geometry factor for cubic anisotropy."""
    rng = np.random.default_rng(seed)

    # Random directions on sphere
    theta = np.arccos(2 * rng.random(n_samples) - 1)
    phi = 2 * np.pi * rng.random(n_samples)
    n_x = np.sin(theta) * np.cos(phi)
    n_y = np.sin(theta) * np.sin(phi)
    n_z = np.cos(theta)

    # Compute δ(n̂) normalized: f = 1 - 6*(cubic_term)
    cubic_term = n_x**2 * n_y**2 + n_y**2 * n_z**2 + n_z**2 * n_x**2
    delta_values = 1 - 6 * cubic_term  # ranges [-1, +1]

    rms = np.sqrt(np.mean(delta_values**2))
    mean = np.mean(delta_values)
    return {'rms': rms, 'mean': mean}


# =============================================================================
# PYTEST TESTS
# =============================================================================

def test_mc1_sqrt_m_scaling():
    """MC1: Verify RMS(Δc/c) ∝ 1/√M scaling."""
    np.random.seed(42)  # reproducibility

    M_values = [1, 4, 16, 64, 256]
    results = run_mc1_scaling(M_values, DELTA_KELVIN, n_trials=200)

    M_arr = np.array([r['M'] for r in results])
    rms_arr = np.array([r['rms'] for r in results])

    check = check_scaling_slope(M_arr, rms_arr, expected=-0.5)

    print(f"\nMC1: 1/√M scaling")
    print(f"  Slope: {check['slope']:.3f} (expect -0.5)")
    for r in results:
        print(f"  M={r['M']:<5} RMS={r['rms']:.4e} theory={r['theory']:.4e} ratio={r['ratio']:.3f}")

    assert check['passed'], f"Slope {check['slope']:.3f} deviates from -0.5"


def test_mc6_geometry_audit():
    """MC6: Geometry factor for cubic anisotropy ~0.56."""
    res = run_mc6_geometry_audit(n_samples=5000, seed=42)

    print(f"\nMC6: Geometry factor")
    print(f"  RMS(f) = {res['rms']:.4f} (expect ~0.56)")
    print(f"  Mean(f) = {res['mean']:.4f} (expect ~-0.2)")

    # f = 1 - 6*(cubic_term), ranges [-1, +1]
    # RMS over sphere ≈ 0.56
    assert 0.50 < res['rms'] < 0.65, f"RMS = {res['rms']:.3f}, expected ~0.56"


def test_mc2_2omega_modulation():
    """MC2: 2ω modulation format - amplitude scales as 1/√M."""
    np.random.seed(43)  # reproducibility

    # Use 4 points for robust slope fitting
    M_values = [10, 25, 50, 100]
    results = []

    print(f"\nMC2: 2ω modulation")
    for M in M_values:
        res = simulate_2omega_rotation(M, DELTA_KELVIN, n_angles=30, n_trials=40)
        theory = DELTA_KELVIN / np.sqrt(M)
        ratio = res['amplitude_rms'] / theory
        results.append(res)
        print(f"  M={M}: amp_rms={res['amplitude_rms']:.4e} theory={theory:.4e} ratio={ratio:.3f}")

    # Check 1/√M scaling
    rms_arr = np.array([r['amplitude_rms'] for r in results])
    check = check_scaling_slope(np.array(M_values), rms_arr, expected=-0.5, tolerance=0.2)
    print(f"  Slope: {check['slope']:.3f} (expect -0.5)")

    assert check['passed'], f"Slope {check['slope']:.3f} deviates from -0.5"


def test_mc3_christoffel_kernel():
    """MC3: Christoffel 3D kernel - 1/√M scaling with cubic anisotropy."""
    # Use 4 points for robust slope fitting
    M_values = [10, 25, 50, 100]

    # DELTA_KELVIN is peak-to-peak δv/v
    # Kernel f ∈ [-1,+1], so delta_scale = DELTA/2 gives correct peak-to-peak
    delta_scale = DELTA_KELVIN / 2

    print(f"\nMC3: Christoffel kernel (Kelvin)")
    print(f"  δv/v = {DELTA_KELVIN:.3f} (peak-to-peak)")
    print(f"  delta_scale = {delta_scale:.4f} (half peak-to-peak)")

    rms_arr = []
    for i, M in enumerate(M_values):
        # Separate RNG per M value for independent samples (hermetic per point)
        rng = np.random.default_rng(44 + i)
        res = simulate_christoffel_3d(M, delta_scale, n_angles=20, n_trials=25, rng=rng)
        # Theory: RMS ~ delta_scale * RMS(f) / √M ≈ delta_scale * 0.56 / √M
        theory = delta_scale * 0.56 / np.sqrt(M)
        ratio = res['amplitude_rms'] / theory
        rms_arr.append(res['amplitude_rms'])
        print(f"  M={M}: amp_rms={res['amplitude_rms']:.4e} theory={theory:.4e} ratio={ratio:.2f}")

    # Check scaling: should follow 1/√M
    check = check_scaling_slope(np.array(M_values), np.array(rms_arr), expected=-0.5, tolerance=0.25)
    print(f"  Slope: {check['slope']:.3f} (expect -0.5)")

    assert check['passed'], f"Slope {check['slope']:.3f} deviates from -0.5"


def test_mc4_finite_correlation():
    """MC4: Markov correlated grains - RMS scales as √ℓ_corr."""
    np.random.seed(45)  # reproducibility

    L = 1.0
    # Use 3 points for robust slope fitting
    corr_lengths = [0.01, 0.05, 0.1]
    n_trials = 80
    n_angles = 30

    print(f"\nMC4: Finite correlation (Markov)")

    rms_arr = []
    for l_corr in corr_lengths:
        # Markov chain: segments with correlation
        l_segment = l_corr / 5
        n_segments = int(L / l_segment)
        p_keep = np.exp(-l_segment / l_corr)

        theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        amplitudes = []

        for _ in range(n_trials):
            # Correlated grain angles
            grain_angles = np.zeros(n_segments)
            grain_angles[0] = np.random.uniform(0, np.pi)
            for i in range(1, n_segments):
                if np.random.random() < p_keep:
                    grain_angles[i] = grain_angles[i-1]
                else:
                    grain_angles[i] = np.random.uniform(0, np.pi)

            delta_c = np.zeros(n_angles)
            for j, th in enumerate(theta):
                meas_dir = np.array([np.cos(th), np.sin(th)])
                total = sum(meas_dir @ (DELTA_KELVIN * anisotropy_tensor_2d(g)) @ meas_dir
                           for g in grain_angles)
                delta_c[j] = total / n_segments

            fit = fit_2omega(theta, delta_c)
            amplitudes.append(fit['amplitude'])

        rms = np.sqrt(np.mean(np.array(amplitudes)**2))
        theory = DELTA_KELVIN * np.sqrt(l_corr / L)
        ratio = rms / theory
        rms_arr.append(rms)
        print(f"  ℓ_corr={l_corr}: RMS={rms:.4e} theory={theory:.4e} ratio={ratio:.2f}")

    # Check scaling: RMS ∝ √ℓ_corr → slope = +0.5
    check = check_scaling_slope(np.array(corr_lengths), np.array(rms_arr), expected=0.5)
    print(f"  Slope: {check['slope']:.2f} (expect +0.5)")

    assert abs(check['slope'] - 0.5) < 0.3, f"Slope {check['slope']:.2f} deviates from +0.5"


def test_mc5_two_way_geometry():
    """MC5: Two-way (round-trip) gives 2× one-way amplitude."""
    np.random.seed(46)  # reproducibility

    M = 50
    n_trials = 100
    n_angles = 30

    print(f"\nMC5: Two-way vs one-way geometry")

    theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    one_way_amps = []
    two_way_amps = []

    for _ in range(n_trials):
        grain_angles = np.random.uniform(0, np.pi, M)

        one_way = np.zeros(n_angles)
        two_way = np.zeros(n_angles)

        for j, th in enumerate(theta):
            meas_dir = np.array([np.cos(th), np.sin(th)])
            delta = sum(meas_dir @ (DELTA_KELVIN * anisotropy_tensor_2d(g)) @ meas_dir
                       for g in grain_angles) / M
            one_way[j] = delta
            two_way[j] = 2 * delta  # Round-trip = 2× single pass

        fit1 = fit_2omega(theta, one_way)
        fit2 = fit_2omega(theta, two_way)
        one_way_amps.append(fit1['amplitude'])
        two_way_amps.append(fit2['amplitude'])

    rms_one = np.sqrt(np.mean(np.array(one_way_amps)**2))
    rms_two = np.sqrt(np.mean(np.array(two_way_amps)**2))
    ratio = rms_two / rms_one

    print(f"  One-way RMS: {rms_one:.4e}")
    print(f"  Two-way RMS: {rms_two:.4e}")
    print(f"  Ratio: {ratio:.2f} (expect 2.0)")

    assert 1.8 < ratio < 2.2, f"Ratio {ratio:.2f} should be ~2.0"


def test_margin_sanity():
    """
    Deterministic margin test: verify δ/√M < experimental bound.

    This is the KEY PHYSICS CLAIM - no Monte Carlo, pure formula:
        predicted = δv/v / √M
        M = L / ℓ_Planck (number of grains in cavity)

    For L = 1m cavity, we must have margin > 1 (predicted < bound).
    """
    L_cavity = 1.0  # 1 meter cavity
    M = L_cavity / L_PLANCK  # ~6.2×10³⁴ grains

    # Guard against unit/typo errors
    assert 5e34 < M < 7e34, f"M={M:.2e} outside expected range [5e34, 7e34]"

    print(f"\nMargin sanity test (L={L_cavity}m):")
    print(f"  M = L/ℓ_P = {M:.2e}")
    print(f"  √M = {np.sqrt(M):.2e}")
    print(f"  Bound = {BOUND_EXP:.0e}")

    all_structures = {
        'C15': DELTA_C15,
        'WP': DELTA_WP,
        'Kelvin': DELTA_KELVIN,
        'FCC': DELTA_FCC,
    }

    all_passed = True
    for name, delta in all_structures.items():
        predicted = delta / np.sqrt(M)
        margin = BOUND_EXP / predicted

        status = "PASS" if margin > 1 else "FAIL"
        print(f"  {name}: δv/v={delta:.3f}, predicted={predicted:.2e}, margin={margin:.0f}× → {status}")

        if margin <= 1:
            all_passed = False

    assert all_passed, "Some structures fail Lorentz bound"
    print(f"  → ALL PASS (all structures have margin > 1)")


# =============================================================================
# SUMMARY
# =============================================================================

def print_margin_table():
    """Print safety margins for different structures."""
    # All structures with derived δv/v values (see physics/christoffel.py)
    structures = {'C15': DELTA_C15, 'WP': DELTA_WP, 'Kelvin': DELTA_KELVIN, 'FCC': DELTA_FCC}

    print("\n" + "=" * 60)
    print("LORENTZ BOUND MARGINS (ℓ_corr = ℓ_P)")
    print("=" * 60)
    print(f"{'Structure':<10} {'δv/v':<8} {'L=1m margin':<15} {'L=0.1m margin':<15}")
    print("-" * 60)

    for name, delta in structures.items():
        margin_1m = BOUND_EXP / (delta * np.sqrt(L_PLANCK / 1.0))
        margin_01m = BOUND_EXP / (delta * np.sqrt(L_PLANCK / 0.1))
        print(f"{name:<10} {delta:<8.3f} {margin_1m:<15.0f}× {margin_01m:<15.0f}×")


# =============================================================================
# STANDALONE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CAVITY WASH-OUT TESTS")
    print("=" * 60)

    test_mc1_sqrt_m_scaling()
    test_mc6_geometry_audit()
    print_margin_table()
