#!/usr/bin/env python3
"""
R2-1: CAVITY PERTURBATION - END-TO-END WITH FOAM KERNEL
========================================================

Tests the full pipeline: Foam geometry → DisplacementBloch → v(k̂) → cavity 2ω

This complements 03_test_cavity_lorentz.py which uses ANALYTIC cubic kernels.
Here we use REAL foam-derived kernels from Bloch diagonalization.

=============================================================================
IMPORTANT: MODEL PARAMETERS AND WHAT THEY AFFECT
=============================================================================

The tests below use specific numerical values. Here's what matters and what doesn't:

ARBITRARY PARAMETERS (chosen for convenience, DON'T affect test validity):
  - k_L, k_T: Spring constants. Ratio k_L/k_T determines δv/v magnitude.
              We use k_L=3.0, k_T=1.0 (ratio 3:1) to get visible anisotropy (~6%).
              ANY k_L ≠ k_T would work - tests verify STRUCTURE not magnitude.
  - L_CELL:   Cell size. Results are scale-invariant. L=8.0 is arbitrary.
  - beta:     k-NN interpolation smoothing. Numerical parameter, not physics.
              We test sensitivity to this in test_knn_sensitivity.
  - n_directions: Sphere sampling density. Numerical parameter.
              We test sensitivity in test_interpolation_sensitivity.

WHAT THE TESTS ACTUALLY VERIFY (independent of above parameters):
  1. Slope = -0.5    → From CLT averaging, works for ANY k_L/k_T
  2. Format = 2ω     → From spin-2 tensor structure, works for ANY foam
  3. E[C] = 0        → From traceless tensor, works for ANY k_L/k_T
  4. Phase ~ exp(-2iφ) → From spin-2 rotation, works for ANY kernel

WHAT WE DON'T TEST (requires bridge assumption):
  - Absolute value of δc/c (depends on k_L/k_T AND bridge δv/v ↔ δn/n)
  - That foam IS the vacuum (hypothesis, not testable here)

Analogy: Testing a = F/m doesn't require specific F, m values.
         We verify the LAW (slope=1 in log-log), not particular numbers.

=============================================================================

INPUTS
------

Internal (from model):
  - Foam geometry (Kelvin/WP/FCC) from builders
  - DisplacementBloch → v(k̂) for many directions
  - δv(k̂) = v(k̂) - v_mean (local anisotropy)

External:
  - Cavity perturbation formula: Δν/ν ~ weighted average of δn/n
  - Bridge assumption: δn/n ~ δv/v (elastic-EM correspondence)

OUTPUTS
-------

  - 2ω format: Δν/ν(θ) = A cos(2θ) + B sin(2θ) + C
  - C ≈ 0 (traceless tensor)
  - Amplitude √(A²+B²) scales as δv/v / √M
  - End-to-end: foam geometry → observable format

TESTS
-----

  Fast (CI gate):
    test_cavity_perturbation: 1/√M scaling + E[C]≈0 with real foam kernel (~20s)
    test_isotropic_limit: k_L=k_T → zero 2ω (proves signal = anisotropy) (~15s)

  Slow (full validation):
    test_t2_verification: Both transverse modes give consistent results
    test_synthetic_kernel: Pipeline works with analytic kernel
    test_spin2_foam_kernel: Phase transforms as exp(-2iφ) under rotation
    test_c_distribution: C is Gaussian with E[C]=0 (traceless proof)
    test_interpolation_sensitivity: Invariants stable across n_directions
    test_knn_sensitivity: Invariants stable across k-NN parameters
    test_multi_structure: C15, WP, Kelvin, FCC all work

Run fast test:
  pytest src/tests/physics/08_test_cavity_perturbation.py -v

Run slow tests:
  pytest src/tests/physics/08_test_cavity_perturbation.py -v -m slow

Run all:
  python src/tests/physics/08_test_cavity_perturbation.py --all

Jan 2026
"""

import numpy as np
from typing import Dict, Tuple
import pytest

import sys
from pathlib import Path

# =============================================================================
# MODEL PARAMETERS (arbitrary but fixed - see docstring for explanation)
# =============================================================================

# Spring constants: ratio determines anisotropy magnitude, not test validity
K_L = 3.0   # Longitudinal spring constant
K_T = 1.0   # Transverse spring constant
# Note: k_L/k_T = 3 gives δv/v ~ 6%. Any ratio ≠ 1 works for testing structure.

# Cell size: results are scale-invariant, this is just a convenient value
L_CELL = 8.0

# k-NN interpolation: numerical smoothing, tested for sensitivity
KNN_K = 5        # Number of neighbors
KNN_BETA = 20.0  # Exponential weight decay

# Test tolerances (unified across tests)
SLOPE_TOL = 0.15      # Tolerance for -0.5 slope (CLT)
TRACELESS_TOL = 0.10  # |E[C]|/amp tolerance (traceless)

def _find_src():
    """Find src/ by looking for physics/ subdirectory."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = current / 'src'
        if (candidate / 'physics').is_dir():
            return candidate
        # Also check if current IS src
        if (current / 'physics').is_dir():
            return current
        current = current.parent
    raise RuntimeError("Cannot find src/physics directory")

_src = str(_find_src())
if _src not in sys.path:
    sys.path.insert(0, _src)

from physics.bloch import DisplacementBloch
from physics.christoffel import measure_velocities, golden_spiral
from core_math_v2.builders.multicell_periodic import build_bcc_supercell_periodic


# =============================================================================
# UTILITIES
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


def interpolate_kernel(directions: np.ndarray, values: np.ndarray,
                       target: np.ndarray, k: int = None, beta: float = None) -> float:
    """
    Weighted interpolation using k nearest neighbors.

    Uses exponential weights: w_i ∝ exp(beta × dot_i)
    This reduces discretization artifacts compared to nearest-neighbor.

    Defaults to module constants KNN_K, KNN_BETA if not specified.
    """
    if k is None:
        k = KNN_K
    if beta is None:
        beta = KNN_BETA

    assert target.shape == (3,), f"target must be (3,), got {target.shape}"

    # Normalize target (defensive, in case of numerical drift)
    target = target / np.linalg.norm(target)

    k = min(k, len(directions))

    dots = directions @ target
    top_k_idx = np.argpartition(dots, -k)[-k:]
    top_k_dots = dots[top_k_idx]
    top_k_values = values[top_k_idx]

    weights = np.exp(beta * (top_k_dots - np.max(top_k_dots)))
    weights /= np.sum(weights)

    return np.dot(weights, top_k_values)


# =============================================================================
# KERNEL BUILDERS
# =============================================================================

def build_velocity_kernel_kelvin(n_directions: int = 500, mode: str = 'T1') -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build velocity kernel from Kelvin (BCC) foam.

    Note: This builds Kelvin specifically. For multi-structure tests,
    see test_multi_structure() which builds each structure explicitly.

    Uses module constants K_L, K_T, L_CELL.

    Returns:
        directions: (n, 3) sampled k̂ directions
        delta_v_normalized: (n,) normalized δv/v at each direction
        v_mean: mean transverse velocity
        delta_v_over_v: peak-to-peak δv/v
    """
    V, E, F, _ = build_bcc_supercell_periodic(2)
    db = DisplacementBloch(V, E, L_CELL, k_L=K_L, k_T=K_T)

    directions, v_squared = measure_velocities(db, L_CELL, n_directions=n_directions)

    if mode == 'T1':
        v_T = np.sqrt(v_squared[:, 0])
    elif mode == 'T2':
        v_T = np.sqrt(v_squared[:, 1])
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'T1' or 'T2'.")

    v_mean = np.mean(v_T)
    delta_v_normalized = (v_T - v_mean) / v_mean
    delta_v_over_v = (np.max(v_T) - np.min(v_T)) / v_mean

    return directions, delta_v_normalized, v_mean, delta_v_over_v


def compute_effective_2omega_rms(directions: np.ndarray, delta_v: np.ndarray,
                                  n_samples: int = 2000, seed: int = 42) -> float:
    """
    Compute effective 2ω RMS for single grain in xy-plane measurement.

    The xy-plane projection reduces the effective RMS by ~0.73 compared to 3D.
    Uses module constants KNN_K, KNN_BETA for interpolation.
    """
    rng = np.random.default_rng(seed)
    single_grain_amps = []
    theta = np.linspace(0, 2*np.pi, 30, endpoint=False)

    for _ in range(n_samples):
        R = random_rotation_matrix(rng)

        delta_c = np.zeros(30)
        for j, th in enumerate(theta):
            n_hat = np.array([np.cos(th), np.sin(th), 0.0])
            n_grain = R.T @ n_hat
            delta_c[j] = interpolate_kernel(directions, delta_v, n_grain,
                                            k=KNN_K, beta=KNN_BETA)

        fit = fit_2omega(theta, delta_c)
        single_grain_amps.append(fit['amplitude'])

    return np.sqrt(np.mean(np.array(single_grain_amps)**2))


def simulate_cavity(directions: np.ndarray, delta_v: np.ndarray,
                    M: int, n_angles: int = 30, rng=None) -> Dict:
    """
    Simulate cavity with M grains using foam-derived kernel.

    Uses module constants KNN_K, KNN_BETA for interpolation.
    """
    if rng is None:
        rng = np.random.default_rng()

    rotations = [random_rotation_matrix(rng) for _ in range(M)]

    theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    delta_c = np.zeros(n_angles)

    for j, th in enumerate(theta):
        n_hat = np.array([np.cos(th), np.sin(th), 0.0])

        total = 0.0
        for R in rotations:
            n_grain = R.T @ n_hat
            total += interpolate_kernel(directions, delta_v, n_grain,
                                        k=KNN_K, beta=KNN_BETA)

        delta_c[j] = total / M

    fit = fit_2omega(theta, delta_c)

    return {
        'theta': theta,
        'delta_c': delta_c,
        'A': fit['A'],
        'B': fit['B'],
        'C': fit['C'],
        'amplitude': fit['amplitude'],
    }


# =============================================================================
# FAST CI TEST
# =============================================================================

def test_cavity_perturbation():
    """
    Fast pytest gate for CI (~25s).

    Uses parameters: n_trials=15, M_values=[16,32,64,128,256], n_directions=200

    Validates:
    1. 1/√M scaling (slope ≈ -0.5) - physics: CLT averaging
    2. E[C] ≈ 0 (traceless in expectation) - physics: spin-2 tensor
    3. Theory ratio ≈ 1 (0.7 to 1.3) - INTERNAL CONSISTENCY only

    Note on ratio test: Theory uses same k-NN interpolation as simulation,
    so ratio≈1 validates estimator consistency, NOT absolute physics
    normalization. The physics content is in (1) and (2).

    Note on parameters: k_L, k_T, L are arbitrary (see module docstring).
    Tests verify STRUCTURE (slope, traceless) not MAGNITUDE.
    """
    V, E, F, _ = build_bcc_supercell_periodic(2)
    db = DisplacementBloch(V, E, L_CELL, k_L=K_L, k_T=K_T)
    directions, v_squared = measure_velocities(db, L_CELL, n_directions=200)

    v_T1 = np.sqrt(v_squared[:, 0])
    v_mean = np.mean(v_T1)
    delta_v = (v_T1 - v_mean) / v_mean

    effective_rms = compute_effective_2omega_rms(directions, delta_v, n_samples=500, seed=42)

    # 5 points for robust slope fit (powers of 2 for clean spacing)
    M_values = [16, 32, 64, 128, 256]
    n_trials = 15  # Reduced trials since we have more M points
    seed = 100

    all_C_values = []  # Collect all C values for traceless check
    results = []
    for M in M_values:
        amplitudes = []
        C_values = []

        for trial in range(n_trials):
            rng = np.random.default_rng(seed + M*1000 + trial)
            res = simulate_cavity(directions, delta_v, M, n_angles=20, rng=rng)
            amplitudes.append(res['amplitude'])
            C_values.append(res['C'])
            all_C_values.append(res['C'])

        amp_rms = np.sqrt(np.mean(np.array(amplitudes)**2))
        theory = effective_rms / np.sqrt(M)
        ratio = amp_rms / theory if theory > 0 else np.nan

        results.append({
            'M': M,
            'amp_rms': amp_rms,
            'theory': theory,
            'ratio': ratio,
        })

    # Check scaling: slope should be -0.5 (CLT)
    M_arr = np.array([r['M'] for r in results])
    amp_arr = np.array([r['amp_rms'] for r in results])
    slope = np.polyfit(np.log(M_arr), np.log(amp_arr), 1)[0]

    assert abs(slope + 0.5) < SLOPE_TOL, f"Slope {slope:.3f} deviates from -0.5"

    # Check traceless: E[C] ≈ 0 (not |C|/amp)
    # This is the correct physics: traceless tensor → zero bias in expectation
    all_C = np.array(all_C_values)
    C_bias = np.mean(all_C)  # Should be ~0
    amp_mean = np.mean([r['amp_rms'] for r in results])

    assert abs(C_bias) < TRACELESS_TOL * amp_mean, \
        f"|E[C]| = {abs(C_bias):.2e} too large vs amp = {amp_mean:.2e}"

    # Check theory ratio (internal consistency, see docstring)
    ratio_mean = np.mean([r['ratio'] for r in results])
    assert 0.7 < ratio_mean < 1.3, f"Theory ratio {ratio_mean:.3f} not in (0.7, 1.3)"


# =============================================================================
# ISOTROPIC LIMIT TEST (fast, deterministic)
# =============================================================================

def test_isotropic_limit():
    """
    Cement test: k_L = k_T → isotropic → zero 2ω amplitude.

    This proves the 2ω signal comes from ANISOTROPY, not pipeline artifacts.
    If k_L = k_T, the elastic medium is isotropic and v(k̂) = constant,
    so δv/v = 0 everywhere → no 2ω signal.

    Fast and deterministic (no MC needed for this check).
    """
    V, E, F, _ = build_bcc_supercell_periodic(2)

    # Isotropic case: k_L = k_T (both = K_T for simplicity)
    db_iso = DisplacementBloch(V, E, L_CELL, k_L=K_T, k_T=K_T)
    directions, v_squared_iso = measure_velocities(db_iso, L_CELL, n_directions=200)
    v_T1_iso = np.sqrt(v_squared_iso[:, 0])
    v_mean_iso = np.mean(v_T1_iso)
    delta_v_iso = (v_T1_iso - v_mean_iso) / v_mean_iso

    # δv/v should be near zero (numerical precision)
    delta_v_rms_iso = np.sqrt(np.mean(delta_v_iso**2))
    assert delta_v_rms_iso < 1e-6, f"Isotropic δv/v RMS = {delta_v_rms_iso:.2e} (should be ~0)"

    # Effective 2ω RMS should be near zero
    eff_rms_iso = compute_effective_2omega_rms(directions, delta_v_iso, n_samples=100, seed=42)
    assert eff_rms_iso < 1e-6, f"Isotropic 2ω RMS = {eff_rms_iso:.2e} (should be ~0)"

    # Compare with anisotropic case (using K_L, K_T from constants)
    db_aniso = DisplacementBloch(V, E, L_CELL, k_L=K_L, k_T=K_T)
    _, v_squared_aniso = measure_velocities(db_aniso, L_CELL, n_directions=200)
    v_T1_aniso = np.sqrt(v_squared_aniso[:, 0])
    v_mean_aniso = np.mean(v_T1_aniso)
    delta_v_aniso = (v_T1_aniso - v_mean_aniso) / v_mean_aniso
    delta_v_rms_aniso = np.sqrt(np.mean(delta_v_aniso**2))

    # Anisotropic should have non-zero signal (at least 100× larger)
    assert delta_v_rms_aniso > 100 * delta_v_rms_iso, \
        f"Anisotropic δv/v RMS = {delta_v_rms_aniso:.2e} not >> isotropic"


# =============================================================================
# SLOW TESTS (run with pytest -m slow)
# =============================================================================

@pytest.mark.slow
def test_t2_verification():
    """
    Verifies that T2 branch gives similar results to T1.
    Both transverse modes should produce consistent 2ω behavior.
    """
    dirs_T1, delta_v_T1, v_mean_T1, dv_T1 = build_velocity_kernel_kelvin(n_directions=300, mode='T1')
    dirs_T2, delta_v_T2, v_mean_T2, dv_T2 = build_velocity_kernel_kelvin(n_directions=300, mode='T2')

    eff_rms_T1 = compute_effective_2omega_rms(dirs_T1, delta_v_T1, n_samples=500, seed=42)
    eff_rms_T2 = compute_effective_2omega_rms(dirs_T2, delta_v_T2, n_samples=500, seed=42)

    M = 50
    n_trials = 30
    seed = 400

    results_T1 = []
    results_T2 = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        res_T1 = simulate_cavity(dirs_T1, delta_v_T1, M, n_angles=30, rng=rng)
        results_T1.append(res_T1['amplitude'])

        rng = np.random.default_rng(seed + trial)
        res_T2 = simulate_cavity(dirs_T2, delta_v_T2, M, n_angles=30, rng=rng)
        results_T2.append(res_T2['amplitude'])

    amp_rms_T1 = np.sqrt(np.mean(np.array(results_T1)**2))
    amp_rms_T2 = np.sqrt(np.mean(np.array(results_T2)**2))

    ratio_T1 = amp_rms_T1 / (eff_rms_T1 / np.sqrt(M))
    ratio_T2 = amp_rms_T2 / (eff_rms_T2 / np.sqrt(M))

    assert 0.7 < ratio_T1 < 1.3, f"T1 ratio {ratio_T1:.3f} out of range"
    assert 0.7 < ratio_T2 < 1.3, f"T2 ratio {ratio_T2:.3f} out of range"

    rms_ratio = max(eff_rms_T1, eff_rms_T2) / min(eff_rms_T1, eff_rms_T2)
    assert rms_ratio < 2.0, f"T1/T2 differ by {rms_ratio:.2f}x (too large)"


@pytest.mark.slow
def test_synthetic_kernel():
    """
    Uses analytic cubic kernel instead of foam.
    Proves cavity simulation pipeline works independently of Bloch.

    The cubic kernel f(n) = 1 - 6(nx²ny² + ny²nz² + nz²nx²) has known
    analytic properties, so this tests the pipeline without foam complexity.
    """
    def synthetic_kernel(n_hat: np.ndarray) -> float:
        nx, ny, nz = n_hat
        return 1 - 6 * (nx**2 * ny**2 + ny**2 * nz**2 + nz**2 * nx**2)

    directions = golden_spiral(500)
    raw_values = np.array([synthetic_kernel(d) for d in directions])
    delta_v = 0.03 * (raw_values - np.mean(raw_values))

    rng = np.random.default_rng(42)
    single_grain_amps = []
    theta = np.linspace(0, 2*np.pi, 30, endpoint=False)

    for _ in range(1000):
        R = random_rotation_matrix(rng)
        delta_c = np.zeros(30)
        for j, th in enumerate(theta):
            n_hat = np.array([np.cos(th), np.sin(th), 0.0])
            n_grain = R.T @ n_hat
            delta_c[j] = interpolate_kernel(directions, delta_v, n_grain)
        fit = fit_2omega(theta, delta_c)
        single_grain_amps.append(fit['amplitude'])

    effective_rms = np.sqrt(np.mean(np.array(single_grain_amps)**2))

    # 5 points for robust slope fit
    M_values = [16, 32, 64, 128, 256]
    n_trials = 20
    seed = 200

    all_C_values = []
    results = []
    for M in M_values:
        amplitudes = []

        for trial in range(n_trials):
            rng = np.random.default_rng(seed + M*1000 + trial)
            res = simulate_cavity(directions, delta_v, M, n_angles=30, rng=rng)
            amplitudes.append(res['amplitude'])
            all_C_values.append(res['C'])

        amp_rms = np.sqrt(np.mean(np.array(amplitudes)**2))
        results.append({'M': M, 'amp_rms': amp_rms})

    M_arr = np.array([r['M'] for r in results])
    amp_arr = np.array([r['amp_rms'] for r in results])
    slope = np.polyfit(np.log(M_arr), np.log(amp_arr), 1)[0]

    # Check slope (unified tolerance)
    assert abs(slope + 0.5) < SLOPE_TOL, f"Synthetic slope {slope:.3f} too far from -0.5"

    # Check traceless: E[C] ≈ 0 (unified with other tests)
    C_bias = np.mean(all_C_values)
    amp_mean = np.mean(amp_arr)
    assert abs(C_bias) < TRACELESS_TOL * amp_mean, \
        f"Synthetic |E[C]| = {abs(C_bias):.2e} too large vs amp = {amp_mean:.2e}"


@pytest.mark.slow
def test_spin2_foam_kernel():
    """
    Verifies spin-2 transformation on real foam kernel.

    If we rotate measurement directions by φ in the lab frame but fit against
    the original angles, the (A,B) coefficients transform as:
        A' + iB' = (A + iB) × exp(-2iφ)

    The negative sign arises because measuring at θ+φ but recording at θ
    shifts the signal phase backward by 2φ.

    This confirms the spin-2 tensor structure required by SME formalism.

    Test strategy: Check that phase_diff vs φ has slope ≈ -2 (spin-2).
    Individual phase measurements are noisy due to interpolation, but the
    overall trend must show spin-2 scaling.
    """
    # Build foam kernel (using module constants)
    V, E, F, _ = build_bcc_supercell_periodic(2)
    db = DisplacementBloch(V, E, L_CELL, k_L=K_L, k_T=K_T)
    directions, v_squared = measure_velocities(db, L_CELL, n_directions=300)
    v_T1 = np.sqrt(v_squared[:, 0])
    v_mean = np.mean(v_T1)
    delta_v = (v_T1 - v_mean) / v_mean

    # Fixed grain configuration
    rng = np.random.default_rng(42)
    M = 30
    rotations = [random_rotation_matrix(rng) for _ in range(M)]

    # Test rotation angles
    test_angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]

    # Get baseline (φ=0)
    theta_base = np.linspace(0, 2*np.pi, 30, endpoint=False)

    # Collect results for all angles
    results = []
    for phi in test_angles:
        delta_c = np.zeros(30)
        for j, th in enumerate(theta_base):
            th_rot = th + phi
            n_hat = np.array([np.cos(th_rot), np.sin(th_rot), 0.0])
            total = 0.0
            for R in rotations:
                n_grain = R.T @ n_hat
                total += interpolate_kernel(directions, delta_v, n_grain)
            delta_c[j] = total / M

        fit = fit_2omega(theta_base, delta_c)
        A, B = fit['A'], fit['B']
        z = A + 1j * B
        results.append({'phi': phi, 'z': z, 'mag': abs(z), 'phase': np.angle(z)})

    # Check magnitude roughly preserved (±20% due to interpolation noise)
    mags = [r['mag'] for r in results]
    mag_spread = (max(mags) - min(mags)) / np.mean(mags)
    assert mag_spread < 0.25, f"Magnitude spread {mag_spread:.1%} > 25%"

    # Check spin-2 phase scaling: phase_diff = slope × φ, expect slope ≈ -2
    # Use unwrap to avoid ±2π branch cut issues
    phis = np.array([r['phi'] for r in results])  # Include φ=0
    phases = np.array([r['phase'] for r in results])
    phases_unwrapped = np.unwrap(phases)
    phase_diffs = phases_unwrapped - phases_unwrapped[0]

    # Fit slope using φ > 0 only
    slope = np.polyfit(phis[1:], phase_diffs[1:], 1)[0]
    assert abs(slope + 2.0) < 0.2, f"Phase slope {slope:.3f} not close to -2 (spin-2)"


@pytest.mark.slow
def test_c_distribution():
    """
    Verifies C distribution is Gaussian with mean ≈ 0 (traceless in expectation).

    For a traceless spin-2 tensor averaged over random orientations:
    - E[C] = 0 (no systematic bias)
    - C follows Gaussian distribution (CLT)
    - mean(|C|) / std(C) ≈ √(2/π) ≈ 0.798 (half-normal ratio)

    This "kills" the objection "C is not zero" by showing it's zero in expectation
    with Gaussian fluctuations.
    """
    V, E, F, _ = build_bcc_supercell_periodic(2)
    db = DisplacementBloch(V, E, L_CELL, k_L=K_L, k_T=K_T)
    directions, v_squared = measure_velocities(db, L_CELL, n_directions=300)
    v_T1 = np.sqrt(v_squared[:, 0])
    v_mean = np.mean(v_T1)
    delta_v = (v_T1 - v_mean) / v_mean

    # Collect many C samples
    M = 50
    n_trials = 200
    seed = 700

    C_samples = []
    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        res = simulate_cavity(directions, delta_v, M, n_angles=30, rng=rng)
        C_samples.append(res['C'])

    C_arr = np.array(C_samples)
    C_mean = np.mean(C_arr)
    C_std = np.std(C_arr)
    C_abs_mean = np.mean(np.abs(C_arr))

    # Test 1: E[C] ≈ 0 (traceless)
    # |mean(C)| should be << std(C), use 3σ criterion
    assert abs(C_mean) < 3 * C_std / np.sqrt(n_trials), \
        f"|E[C]| = {abs(C_mean):.2e} >> expected σ/√n = {C_std/np.sqrt(n_trials):.2e}"

    # Test 2: Half-normal ratio (Gaussian signature)
    # For Gaussian: mean(|X|) / std(X) = √(2/π) ≈ 0.798
    half_normal_ratio = np.sqrt(2 / np.pi)  # ≈ 0.798
    observed_ratio = C_abs_mean / C_std
    assert abs(observed_ratio - half_normal_ratio) < 0.15, \
        f"Half-normal ratio {observed_ratio:.3f} != {half_normal_ratio:.3f} (not Gaussian?)"


@pytest.mark.slow
def test_interpolation_sensitivity():
    """
    Verifies that invariants (slope, E[C]) are stable across n_directions.

    Tests: n_directions = 100, 200, 400
    This cuts the "insufficient sampling" attack.
    """
    V, E, F, _ = build_bcc_supercell_periodic(2)

    n_dir_values = [100, 200, 400]
    results = []

    for n_dir in n_dir_values:
        db = DisplacementBloch(V, E, L_CELL, k_L=K_L, k_T=K_T)
        directions, v_squared = measure_velocities(db, L_CELL, n_directions=n_dir)
        v_T1 = np.sqrt(v_squared[:, 0])
        v_mean = np.mean(v_T1)
        delta_v = (v_T1 - v_mean) / v_mean

        # Run mini cavity simulation with 5 M points
        M_values = [16, 32, 64, 128, 256]
        n_trials = 12
        seed = 600

        all_C_values = []
        amp_arr = []
        for M in M_values:
            amplitudes = []
            for trial in range(n_trials):
                rng = np.random.default_rng(seed + M*1000 + trial)
                res = simulate_cavity(directions, delta_v, M, n_angles=20, rng=rng)
                amplitudes.append(res['amplitude'])
                all_C_values.append(res['C'])
            amp_arr.append(np.sqrt(np.mean(np.array(amplitudes)**2)))

        slope = np.polyfit(np.log(M_values), np.log(amp_arr), 1)[0]
        C_bias = np.mean(all_C_values)
        amp_mean = np.mean(amp_arr)
        results.append({'n_dir': n_dir, 'slope': slope, 'C_bias': C_bias, 'amp_mean': amp_mean})

    # All should have slope ~ -0.5 (unified tolerance)
    for r in results:
        assert abs(r['slope'] + 0.5) < SLOPE_TOL, \
            f"n_dir={r['n_dir']}: slope={r['slope']:.3f} deviates from -0.5"
        # Check traceless (unified)
        assert abs(r['C_bias']) < TRACELESS_TOL * r['amp_mean'], \
            f"n_dir={r['n_dir']}: |E[C]|={abs(r['C_bias']):.2e} too large"

    # Slopes should be consistent across n_directions
    slopes = [r['slope'] for r in results]
    slope_spread = max(slopes) - min(slopes)
    assert slope_spread < 0.15, \
        f"Slope spread {slope_spread:.3f} too large across n_directions"


@pytest.mark.slow
def test_knn_sensitivity():
    """
    Verifies that invariants (slope, E[C]) are stable across kNN params.

    Tests: (k=1), (k=5, beta=20), (k=10, beta=10)
    This cuts the "interpolation artifacts" attack.
    """
    # Build kernel (using module constants)
    V, E, F, _ = build_bcc_supercell_periodic(2)
    db = DisplacementBloch(V, E, L_CELL, k_L=K_L, k_T=K_T)
    directions, v_squared = measure_velocities(db, L_CELL, n_directions=200)
    v_T1 = np.sqrt(v_squared[:, 0])
    v_mean = np.mean(v_T1)
    delta_v = (v_T1 - v_mean) / v_mean

    # kNN parameter sets to test
    knn_params = [
        (1, 100.0),   # Effectively nearest-neighbor
        (5, 20.0),    # Default (same as KNN_K, KNN_BETA)
        (10, 10.0),   # Smoother
    ]

    results = []
    for k, beta in knn_params:
        # Custom interpolation for this test
        def interp_custom(target):
            return interpolate_kernel(directions, delta_v, target, k=k, beta=beta)

        # Run mini cavity simulation with 5 M points
        M_values = [16, 32, 64, 128, 256]
        n_trials = 12
        seed = 500

        all_C_values = []
        amp_arr = []
        for M in M_values:
            amplitudes = []
            for trial in range(n_trials):
                rng = np.random.default_rng(seed + M*1000 + trial)
                rotations = [random_rotation_matrix(rng) for _ in range(M)]
                theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
                delta_c = np.zeros(20)
                for j, th in enumerate(theta):
                    n_hat = np.array([np.cos(th), np.sin(th), 0.0])
                    total = sum(interp_custom(R.T @ n_hat) for R in rotations)
                    delta_c[j] = total / M
                fit = fit_2omega(theta, delta_c)
                amplitudes.append(fit['amplitude'])
                all_C_values.append(fit['C'])
            amp_arr.append(np.sqrt(np.mean(np.array(amplitudes)**2)))

        slope = np.polyfit(np.log(M_values), np.log(amp_arr), 1)[0]
        C_bias = np.mean(all_C_values)
        amp_mean = np.mean(amp_arr)
        results.append({'k': k, 'beta': beta, 'slope': slope, 'C_bias': C_bias, 'amp_mean': amp_mean})

    # All should have slope ~ -0.5 (unified tolerance)
    for r in results:
        assert abs(r['slope'] + 0.5) < SLOPE_TOL, \
            f"k={r['k']}: slope={r['slope']:.3f} deviates from -0.5"
        # Check traceless (unified)
        assert abs(r['C_bias']) < TRACELESS_TOL * r['amp_mean'], \
            f"k={r['k']}: |E[C]|={abs(r['C_bias']):.2e} too large"


@pytest.mark.slow
def test_multi_structure():
    """
    Verifies that the bridge works for C15, WP, Kelvin, and FCC.

    All structures should give:
    - Slope ≈ -0.5 (CLT)
    - E[C] ≈ 0 (traceless)

    Note: Different structures have different δv/v magnitudes, but all
    should satisfy the STRUCTURAL invariants tested here.
    """
    from core_math_v2.builders import build_fcc_supercell_periodic, build_wp_supercell_periodic
    from core_math_v2.builders.c15_periodic import build_c15_supercell_periodic

    structures = {
        'C15': lambda: (*build_c15_supercell_periodic(1, L_cell=4.0)[:2], 4.0),
        'WP': lambda: (*build_wp_supercell_periodic(1, L_cell=4.0)[:2], 4.0),
        'Kelvin': lambda: (*build_bcc_supercell_periodic(2)[:2], L_CELL),
        'FCC': lambda: (*build_fcc_supercell_periodic(2)[:2], L_CELL),
    }

    n_trials = 40
    seed = 300

    failures = []
    for name, builder in structures.items():
        V, E, L = builder()
        db = DisplacementBloch(V, E, L, k_L=K_L, k_T=K_T)
        directions, v_squared = measure_velocities(db, L, n_directions=300)
        v_T1 = np.sqrt(v_squared[:, 0])
        v_mean = np.mean(v_T1)
        delta_v = (v_T1 - v_mean) / v_mean

        # 5 M points for robust slope
        M_values = [16, 32, 64, 128, 256]
        all_C_values = []
        amp_arr = []
        for M_test in M_values:
            amplitudes = []
            for trial in range(n_trials):
                rng = np.random.default_rng(seed + M_test*1000 + trial)
                res = simulate_cavity(directions, delta_v, M_test, n_angles=30, rng=rng)
                amplitudes.append(res['amplitude'])
                all_C_values.append(res['C'])
            amp_arr.append(np.sqrt(np.mean(np.array(amplitudes)**2)))

        slope = np.polyfit(np.log(M_values), np.log(amp_arr), 1)[0]
        C_bias = np.mean(all_C_values)
        amp_mean = np.mean(amp_arr)

        # Check unified tolerances
        if abs(slope + 0.5) >= SLOPE_TOL:
            failures.append(f"{name}: slope={slope:.3f}")
        if abs(C_bias) >= TRACELESS_TOL * amp_mean:
            failures.append(f"{name}: |E[C]|={abs(C_bias):.2e}")

    assert len(failures) == 0, f"Multi-structure failures: {failures}"


# =============================================================================
# MAIN
# =============================================================================

def run_full_test(n_trials: int = 100):
    """Run full validation with verbose output."""
    print("=" * 70)
    print("R2-1: CAVITY PERTURBATION FORMULA (END-TO-END)")
    print("=" * 70)
    print()

    print("Building velocity kernel from Kelvin foam...")
    directions, delta_v, v_mean, delta_v_over_v = build_velocity_kernel_kelvin(n_directions=500)
    print(f"  δv/v (peak-to-peak) = {delta_v_over_v*100:.2f}%")
    print()

    print("Computing effective 2ω RMS...")
    effective_rms = compute_effective_2omega_rms(directions, delta_v, n_samples=2000, seed=42)
    print(f"  Effective 2ω RMS = {effective_rms*100:.4f}%")
    print()

    M_values = [10, 20, 40, 80, 160, 320]
    results = []

    print("-" * 70)
    print(f"{'M':>6} | {'Amp (RMS)':>12} | {'Theory':>12} | {'Ratio':>8}")
    print("-" * 70)

    for M in M_values:
        amplitudes = []
        for trial in range(n_trials):
            rng = np.random.default_rng(100 + M*1000 + trial)
            res = simulate_cavity(directions, delta_v, M, n_angles=30, rng=rng)
            amplitudes.append(res['amplitude'])

        amp_rms = np.sqrt(np.mean(np.array(amplitudes)**2))
        theory = effective_rms / np.sqrt(M)
        ratio = amp_rms / theory

        results.append({'M': M, 'amp_rms': amp_rms, 'ratio': ratio})
        print(f"{M:>6} | {amp_rms:>12.4e} | {theory:>12.4e} | {ratio:>8.3f}")

    print("-" * 70)

    M_arr = np.array([r['M'] for r in results])
    amp_arr = np.array([r['amp_rms'] for r in results])
    slope = np.polyfit(np.log(M_arr), np.log(amp_arr), 1)[0]

    print()
    print(f"Scaling slope: {slope:.3f} (expect -0.5)")
    print(f"Theory ratio: {np.mean([r['ratio'] for r in results]):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="R2-1: Cavity perturbation end-to-end test")
    parser.add_argument("--all", action="store_true", help="Run full validation")
    parser.add_argument("--trials", type=int, default=100, help="Number of MC trials")
    args = parser.parse_args()

    if args.all:
        run_full_test(n_trials=args.trials)
    else:
        run_full_test(n_trials=args.trials)
