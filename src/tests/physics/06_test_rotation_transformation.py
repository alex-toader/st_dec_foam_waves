#!/usr/bin/env python3
"""
TEST 1A-5: Phase/Sign Transformation Under Rotations
=====================================================

Validates that the 2ω signal transforms correctly under coordinate rotations.

INPUTS
------

Internal (from model):
  - 2D grain orientations (random uniform in [0, π])
  - Single-grain anisotropy δ = 0.063 (typical cubic value)
  - Traceless anisotropy tensor A(g) = [[cos2g, sin2g], [sin2g, -cos2g]]

External:
  - None (pure symmetry test, no experimental data)

OUTPUTS
-------

  - Rotation transformation: (A,B) transform as R(2φ) under sample rotation
  - Amplitude invariance: √(A²+B²) preserved under rotation
  - C = 0 for traceless tensor (mean signal = 0)
  - Spin-2 structure confirmed (quadrupole, same as SME photon sector)
  - 3D cubic kernel: 4ω harmonic in xy-plane, transforms as R(4α)

PHYSICS
-------

The cavity signal has form: Δν/ν(θ) = A cos(2θ) + B sin(2θ) + C

Under a rotation of the sample (all grains) by angle φ:
    A' = A cos(2φ) - B sin(2φ)
    B' = A sin(2φ) + B cos(2φ)
    C' = C  (invariant)
    amplitude' = √(A'² + B'²) = √(A² + B²) = amplitude  (invariant)

This is the transformation law for a spin-2 (quadrupole) tensor component.
The matrix R(2φ) = [[cos,-sin],[sin,cos]] is a standard 2D rotation.
It's the same structure as SME photon sector coefficients.

TESTS IN THIS FILE
------------------

    1. test_rotation_transformation: (A,B,C) transform correctly under φ
    2. test_analytic_coefficients: A = δ×mean(cos 2g), B = δ×mean(sin 2g), C = 0
    3. test_group_composition: R(φ1)R(φ2) = R(φ1+φ2)
    4. test_linear_scaling: 2×delta → 2×(A,B,amplitude)
    5. test_signal_shift_equivalence: signal(θ, grains+φ) = signal(θ-φ, grains)
    6. test_multiple_seeds: robustness across random configurations
    A. test_periodicity_and_parity: tripwire tests (φ+π, -φ inverse, special case)
    C. test_cubic_kernel_rotation: 3D cubic kernel, 4ω harmonic, R(4α) symmetry
    D. test_mean_signal_zero: traceless tensor sanity (mean Δc/c = 0)

WHY THIS MATTERS
----------------

  - Shows our δ(n̂) → Δν/ν(θ) mapping respects the correct symmetry
  - Connects elastic anisotropy to SME formalism
  - Without this, the bridge is "just numerics" not "EFT-compatible"

REFERENCE
---------

  - Kostelecký & Mewes (2002), SME photon sector
  - Müller (2005), cavity + crystal effects

Jan 2026
"""

import numpy as np
from typing import Dict, Tuple, List

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


def anisotropy_tensor_2d(theta_grain: float) -> np.ndarray:
    """
    2D traceless symmetric anisotropy tensor for grain at angle theta_grain.
    A(θ) = [[cos(2θ), sin(2θ)], [sin(2θ), -cos(2θ)]]

    This tensor is traceless: Tr(A) = cos(2θ) - cos(2θ) = 0
    """
    c2, s2 = np.cos(2 * theta_grain), np.sin(2 * theta_grain)
    return np.array([[c2, s2], [s2, -c2]])


def rotate_coefficients(A: float, B: float, phi: float) -> Tuple[float, float]:
    """
    Transform (A, B) under rotation of sample by angle φ.

    Physics:
        Rotating all grains by +φ is equivalent to rotating the
        measurement frame by -φ. The signal f(θ) = A cos(2θ) + B sin(2θ)
        becomes f'(θ) = f(θ - φ).

    Derivation:
        f(θ-φ) = A cos(2(θ-φ)) + B sin(2(θ-φ))
               = A [cos(2θ)cos(2φ) + sin(2θ)sin(2φ)]
               + B [sin(2θ)cos(2φ) - cos(2θ)sin(2φ)]
               = [A cos(2φ) - B sin(2φ)] cos(2θ)
               + [A sin(2φ) + B cos(2φ)] sin(2θ)

    Therefore:
        A' = A cos(2φ) - B sin(2φ)
        B' = A sin(2φ) + B cos(2φ)

    This is the standard rotation matrix R(+2φ) acting on (A, B).
    The factor of 2 reflects the quadrupole (spin-2) nature.
    """
    c2phi = np.cos(2 * phi)
    s2phi = np.sin(2 * phi)
    A_prime = A * c2phi - B * s2phi
    B_prime = A * s2phi + B * c2phi
    return A_prime, B_prime


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_cavity_signal(grain_angles: np.ndarray,
                           delta_single: float,
                           n_angles: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Δc/c(θ) signal for given grain orientations.

    Args:
        grain_angles: Array of grain orientation angles [rad]
        delta_single: Single-grain anisotropy amplitude
        n_angles: Number of measurement angles in [0, 2π)

    Returns:
        theta: Measurement angles
        delta_c: Signal Δc/c at each angle
    """
    M = len(grain_angles)
    theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    delta_c = np.zeros(n_angles)

    for j, th in enumerate(theta):
        meas_dir = np.array([np.cos(th), np.sin(th)])
        total = 0.0
        for g in grain_angles:
            A_grain = delta_single * anisotropy_tensor_2d(g)
            total += meas_dir @ A_grain @ meas_dir
        delta_c[j] = total / M

    return theta, delta_c


def generate_rotated_signal(grain_angles: np.ndarray,
                            phi: float,
                            delta_single: float,
                            n_angles: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate signal after rotating ALL grains by angle φ.

    Physically: rotate the entire sample (or equivalently,
    rotate the measurement frame in the opposite direction).
    """
    rotated_grains = grain_angles + phi
    return generate_cavity_signal(rotated_grains, delta_single, n_angles)


# =============================================================================
# TEST 1: ROTATION TRANSFORMATION (A, B, C)
# =============================================================================

def test_rotation_transformation(seed: int = 42,
                                 M: int = 50,
                                 delta: float = 0.063,
                                 verbose: bool = True) -> Dict:
    """
    Test that (A, B, C) transforms correctly under rotations.

    Verifies:
        - A' = A cos(2φ) - B sin(2φ)
        - B' = A sin(2φ) + B cos(2φ)
        - C' = C (invariant)
        - amplitude = √(A² + B²) is invariant
    """
    rng = np.random.default_rng(seed)
    grain_angles = rng.uniform(0, np.pi, M)

    # Original signal
    theta, delta_c = generate_cavity_signal(grain_angles, delta)
    fit_orig = fit_2omega(theta, delta_c)
    A_orig, B_orig, C_orig = fit_orig['A'], fit_orig['B'], fit_orig['C']
    amp_orig = fit_orig['amplitude']

    if verbose:
        print("=" * 70)
        print("TEST 1: Rotation Transformation (A, B, C)")
        print("=" * 70)
        print(f"\nSetup: M={M} grains, δ={delta:.3f}, seed={seed}")
        print(f"\nOriginal fit:")
        print(f"  A = {A_orig:+.6f}")
        print(f"  B = {B_orig:+.6f}")
        print(f"  C = {C_orig:+.6f}")
        print(f"  amplitude = {amp_orig:.6f}")

    # Test rotation angles (including negative)
    test_angles = [np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, -np.pi/4]

    results = []
    all_passed = True

    if verbose:
        print(f"\n{'φ (deg)':<10} {'A_meas':>10} {'A_th':>10} {'B_meas':>10} {'B_th':>10} {'C':>8} {'amp':>8} {'status':<6}")
        print("-" * 82)

    for phi in test_angles:
        theta_rot, delta_c_rot = generate_rotated_signal(grain_angles, phi, delta)
        fit_rot = fit_2omega(theta_rot, delta_c_rot)
        A_meas, B_meas, C_meas = fit_rot['A'], fit_rot['B'], fit_rot['C']
        amp_meas = fit_rot['amplitude']

        A_theory, B_theory = rotate_coefficients(A_orig, B_orig, phi)

        # Errors
        err_A = abs(A_meas - A_theory)
        err_B = abs(B_meas - B_theory)
        err_C = abs(C_meas - C_orig)  # C should be invariant
        err_amp = abs(amp_meas - amp_orig)

        # Tolerance: 1e-10 allows for BLAS/LAPACK floating point variation
        tol = 1e-10

        passed = (err_A < tol) and (err_B < tol) and (err_C < tol) and (err_amp < tol)
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        results.append({
            'phi_deg': np.degrees(phi),
            'A_meas': A_meas, 'A_theory': A_theory,
            'B_meas': B_meas, 'B_theory': B_theory,
            'C_meas': C_meas, 'C_orig': C_orig,
            'amp_meas': amp_meas,
            'passed': passed
        })

        if verbose:
            print(f"{np.degrees(phi):<10.1f} {A_meas:>+10.5f} {A_theory:>+10.5f} "
                  f"{B_meas:>+10.5f} {B_theory:>+10.5f} {C_meas:>+8.5f} {amp_meas:>8.5f} {status:<6}")

    # Check C invariance separately for clear reporting
    all_C_ok = all(abs(r['C_meas'] - C_orig) < tol for r in results)

    if verbose:
        print("-" * 82)
        print(f"C invariance: C_orig={C_orig:+.6f}, all |C_meas - C_orig| < tol: {all_C_ok}")
        print(f"\nRESULT: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 70)

    return {'original': fit_orig, 'rotations': results, 'all_passed': all_passed}


# =============================================================================
# TEST 2: ANALYTIC COEFFICIENTS (A = δ×mean(cos 2g), B = δ×mean(sin 2g))
# =============================================================================

def test_analytic_coefficients(seed: int = 42,
                               M: int = 50,
                               delta: float = 0.063,
                               verbose: bool = True) -> Dict:
    """
    Test that fit_2omega recovers the EXACT analytic coefficients.

    For signal Δ(θ) = δ × (1/M) × Σ_g [meas @ A(g) @ meas]:

    The tensor contribution is:
        meas @ A(g) @ meas = cos(2θ - 2g) = cos(2θ)cos(2g) + sin(2θ)sin(2g)

    Summing over grains:
        Δ(θ) = δ × [mean(cos 2g) × cos(2θ) + mean(sin 2g) × sin(2θ)]

    Therefore:
        A_true = δ × mean(cos 2g)
        B_true = δ × mean(sin 2g)
        C_true = 0 (traceless tensor → no constant term)

    This test verifies fit_2omega extracts the physical coefficient, not just
    "fits the data".
    """
    rng = np.random.default_rng(seed)
    grain_angles = rng.uniform(0, np.pi, M)

    # Analytic predictions
    A_true = delta * np.mean(np.cos(2 * grain_angles))
    B_true = delta * np.mean(np.sin(2 * grain_angles))
    C_true = 0.0

    # Fit from signal
    theta, delta_c = generate_cavity_signal(grain_angles, delta)
    fit = fit_2omega(theta, delta_c)
    A_fit, B_fit, C_fit = fit['A'], fit['B'], fit['C']

    # Errors
    err_A = abs(A_fit - A_true)
    err_B = abs(B_fit - B_true)
    err_C = abs(C_fit - C_true)

    tol = 1e-10
    passed = (err_A < tol) and (err_B < tol) and (err_C < tol)

    if verbose:
        print("\n" + "=" * 70)
        print("TEST 2: Analytic Coefficients")
        print("=" * 70)
        print(f"\nSetup: M={M} grains, δ={delta:.3f}, seed={seed}")
        print(f"\nAnalytic formulas:")
        print(f"  A_true = δ × mean(cos 2g) = {delta:.3f} × {np.mean(np.cos(2*grain_angles)):+.6f} = {A_true:+.6f}")
        print(f"  B_true = δ × mean(sin 2g) = {delta:.3f} × {np.mean(np.sin(2*grain_angles)):+.6f} = {B_true:+.6f}")
        print(f"  C_true = 0 (traceless tensor)")
        print(f"\nFit results:")
        print(f"  A_fit = {A_fit:+.6f}  (err = {err_A:.2e})")
        print(f"  B_fit = {B_fit:+.6f}  (err = {err_B:.2e})")
        print(f"  C_fit = {C_fit:+.6f}  (err = {err_C:.2e})")
        print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
        print("=" * 70)

    return {
        'analytic': {'A': A_true, 'B': B_true, 'C': C_true},
        'fit': {'A': A_fit, 'B': B_fit, 'C': C_fit},
        'errors': {'A': err_A, 'B': err_B, 'C': err_C},
        'passed': passed
    }


# =============================================================================
# TEST 3: GROUP COMPOSITION R(φ1)R(φ2) = R(φ1+φ2)
# =============================================================================

def test_group_composition(seed: int = 42,
                           M: int = 50,
                           delta: float = 0.063,
                           verbose: bool = True) -> Dict:
    """
    Test group composition: rotating by φ1 then φ2 equals rotating by φ1+φ2.

    This verifies the representation is correct (closure property).

    R(φ1) R(φ2) = R(φ1 + φ2)

    In terms of coefficients: applying φ1 then φ2 to (A,B) should give
    the same result as applying φ1+φ2 directly.
    """
    rng = np.random.default_rng(seed)
    grain_angles = rng.uniform(0, np.pi, M)

    # Test angle pairs
    angle_pairs = [
        (np.pi/6, np.pi/4),   # 30° + 45° = 75°
        (np.pi/3, np.pi/6),   # 60° + 30° = 90°
        (np.pi/4, -np.pi/4),  # 45° - 45° = 0°
        (np.pi/2, np.pi/3),   # 90° + 60° = 150°
    ]

    # Original signal
    theta, delta_c = generate_cavity_signal(grain_angles, delta)
    fit_orig = fit_2omega(theta, delta_c)
    A_orig, B_orig = fit_orig['A'], fit_orig['B']

    results = []
    all_passed = True

    if verbose:
        print("\n" + "=" * 70)
        print("TEST 3: Group Composition R(φ1)R(φ2) = R(φ1+φ2)")
        print("=" * 70)
        print(f"\nOriginal: A={A_orig:+.6f}, B={B_orig:+.6f}")
        print(f"\n{'φ1 (deg)':<10} {'φ2 (deg)':<10} {'A_seq':>10} {'A_dir':>10} {'B_seq':>10} {'B_dir':>10} {'status':<6}")
        print("-" * 70)

    for phi1, phi2 in angle_pairs:
        # Method 1: Sequential rotation (φ1 then φ2)
        A_after_1, B_after_1 = rotate_coefficients(A_orig, B_orig, phi1)
        A_seq, B_seq = rotate_coefficients(A_after_1, B_after_1, phi2)

        # Method 2: Direct rotation by φ1+φ2
        A_dir, B_dir = rotate_coefficients(A_orig, B_orig, phi1 + phi2)

        # Check
        err_A = abs(A_seq - A_dir)
        err_B = abs(B_seq - B_dir)
        tol = 1e-10

        passed = (err_A < tol) and (err_B < tol)
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        results.append({
            'phi1_deg': np.degrees(phi1),
            'phi2_deg': np.degrees(phi2),
            'A_seq': A_seq, 'A_dir': A_dir,
            'B_seq': B_seq, 'B_dir': B_dir,
            'passed': passed
        })

        if verbose:
            print(f"{np.degrees(phi1):<10.1f} {np.degrees(phi2):<10.1f} "
                  f"{A_seq:>+10.6f} {A_dir:>+10.6f} {B_seq:>+10.6f} {B_dir:>+10.6f} {status:<6}")

    if verbose:
        print("-" * 70)
        print(f"\nRESULT: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 70)

    return {'results': results, 'all_passed': all_passed}


# =============================================================================
# TEST 4: LINEAR SCALING (2×delta → 2×amplitude)
# =============================================================================

def test_linear_scaling(seed: int = 42,
                        M: int = 50,
                        delta_base: float = 0.063,
                        verbose: bool = True) -> Dict:
    """
    Test that coefficients scale linearly with delta.

    If delta → k×delta, then:
        A → k×A
        B → k×B
        C → k×C (which is 0)
        amplitude → k×amplitude

    This tests the "bridge linearity": small perturbation → linear response.
    """
    rng = np.random.default_rng(seed)
    grain_angles = rng.uniform(0, np.pi, M)

    # Scale factors to test
    scale_factors = [0.5, 1.0, 2.0, 3.0]

    # Reference at scale=1
    theta, delta_c_ref = generate_cavity_signal(grain_angles, delta_base)
    fit_ref = fit_2omega(theta, delta_c_ref)
    A_ref, B_ref, amp_ref = fit_ref['A'], fit_ref['B'], fit_ref['amplitude']

    results = []
    all_passed = True

    if verbose:
        print("\n" + "=" * 70)
        print("TEST 4: Linear Scaling (k×delta → k×amplitude)")
        print("=" * 70)
        print(f"\nReference (k=1): δ={delta_base:.3f}, A={A_ref:+.6f}, B={B_ref:+.6f}, amp={amp_ref:.6f}")
        print(f"\n{'k':<6} {'delta':<8} {'A_meas':>10} {'A_pred':>10} {'amp_meas':>10} {'amp_pred':>10} {'status':<6}")
        print("-" * 70)

    for k in scale_factors:
        delta_k = k * delta_base
        theta, delta_c = generate_cavity_signal(grain_angles, delta_k)
        fit = fit_2omega(theta, delta_c)
        A_meas, B_meas, amp_meas = fit['A'], fit['B'], fit['amplitude']

        A_pred = k * A_ref
        B_pred = k * B_ref
        amp_pred = k * amp_ref

        err_A = abs(A_meas - A_pred)
        err_B = abs(B_meas - B_pred)
        err_amp = abs(amp_meas - amp_pred)

        # Tolerance: 1e-10 allows for BLAS/LAPACK floating point variation
        tol = 1e-10

        passed = (err_A < tol) and (err_B < tol) and (err_amp < tol)
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        results.append({
            'k': k,
            'A_meas': A_meas, 'A_pred': A_pred,
            'B_meas': B_meas, 'B_pred': B_pred,
            'amp_meas': amp_meas, 'amp_pred': amp_pred,
            'passed': passed
        })

        if verbose:
            print(f"{k:<6.1f} {delta_k:<8.4f} {A_meas:>+10.6f} {A_pred:>+10.6f} "
                  f"{amp_meas:>10.6f} {amp_pred:>10.6f} {status:<6}")

    if verbose:
        print("-" * 70)
        print(f"\nRESULT: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 70)

    return {'results': results, 'all_passed': all_passed}


# =============================================================================
# TEST 5: SIGNAL SHIFT EQUIVALENCE
# =============================================================================

def test_signal_shift_equivalence(seed: int = 42,
                                  M: int = 50,
                                  delta: float = 0.063,
                                  verbose: bool = True) -> Dict:
    """
    Test that rotating grains by φ is equivalent to shifting θ by -φ.

    signal(θ, grains + φ) = signal(θ - φ, grains)

    This is the direct physical definition of the transformation.
    """
    rng = np.random.default_rng(seed)
    grain_angles = rng.uniform(0, np.pi, M)

    n_angles = 60
    theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    # Test angles - must be multiples of 2π/n_angles = 6° for exact grid alignment
    # 30° = 5×6°, 60° = 10×6°, 90° = 15×6°
    test_phis = [np.pi/6, np.pi/3, np.pi/2]  # 30°, 60°, 90°

    results = []
    all_passed = True

    if verbose:
        print("\n" + "=" * 70)
        print("TEST 5: Signal Shift Equivalence")
        print("=" * 70)
        print(f"\nVerifying: signal(θ, grains+φ) = signal(θ-φ, grains)")

    for phi in test_phis:
        # Method 1: Rotate grains, measure at θ (use dedicated function)
        _, signal_rotated = generate_rotated_signal(grain_angles, phi, delta, n_angles)

        # Method 2: Original grains, shifted measurement grid
        # signal(θ, grains+φ) = signal(θ-φ, grains)
        # np.roll(arr, +k)[i] = arr[i-k], so +shift gives signal at θ - φ
        # Note: φ chosen as multiple of Δθ = 2π/n_angles, so roll is exact
        _, signal_original = generate_cavity_signal(grain_angles, delta, n_angles)

        shift_amount = int(round(phi / (2*np.pi) * n_angles)) % n_angles  # modulo for safety
        signal_shifted = np.roll(signal_original, +shift_amount)  # +shift is correct

        # Compare
        max_err = np.max(np.abs(signal_rotated - signal_shifted))
        mean_err = np.mean(np.abs(signal_rotated - signal_shifted))

        # Tolerance: strict numeric (grid-aligned, exact match expected)
        tol = 1e-10
        passed = max_err < tol

        if not passed:
            all_passed = False

        results.append({
            'phi_deg': np.degrees(phi),
            'max_err': max_err,
            'mean_err': mean_err,
            'passed': passed
        })

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"\n  φ = {np.degrees(phi):.1f}°: max_err = {max_err:.2e}, mean_err = {mean_err:.2e} → {status}")

    if verbose:
        print(f"\nRESULT: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 70)

    return {'results': results, 'all_passed': all_passed}


# =============================================================================
# TEST 6: MULTIPLE SEEDS (ROBUSTNESS)
# =============================================================================

def test_multiple_seeds(seeds: List[int] = [0, 1, 2, 3, 4],
                        M: int = 50,
                        delta: float = 0.063,
                        verbose: bool = True) -> Dict:
    """
    Run rotation test across multiple random seeds for robustness.

    This catches edge cases where A or B might be very small
    (near-isotropic configuration) and relative tolerances behave oddly.
    """
    results = []
    all_passed = True

    if verbose:
        print("\n" + "=" * 70)
        print("TEST 6: Multiple Seeds (Robustness)")
        print("=" * 70)
        print(f"\nTesting {len(seeds)} different random grain configurations...")
        print(f"\n{'seed':<6} {'A':>10} {'B':>10} {'amp':>10} {'rotation_test':<15}")
        print("-" * 55)

    for seed in seeds:
        # Run rotation test (non-verbose)
        result = test_rotation_transformation(seed=seed, M=M, delta=delta, verbose=False)

        A = result['original']['A']
        B = result['original']['B']
        amp = result['original']['amplitude']
        rot_passed = result['all_passed']

        if not rot_passed:
            all_passed = False

        results.append({
            'seed': seed,
            'A': A, 'B': B, 'amplitude': amp,
            'rotation_passed': rot_passed
        })

        if verbose:
            status = "PASS" if rot_passed else "FAIL"
            print(f"{seed:<6} {A:>+10.6f} {B:>+10.6f} {amp:>10.6f} {status:<15}")

    if verbose:
        print("-" * 55)
        print(f"\nAll seeds passed: {all_passed}")
        print(f"\nRESULT: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 70)

    return {'results': results, 'all_passed': all_passed}


# =============================================================================
# TEST A: PERIODICITY AND PARITY (TRIPWIRE TESTS)
# =============================================================================

def test_periodicity_and_parity(verbose: bool = True) -> Dict:
    """
    Tripwire tests that catch sign/factor errors instantly.

    Properties of the rotation R(2φ):

    1. PERIODICITY: R(φ + π) = R(φ)
       Because cos(2(φ+π)) = cos(2φ+2π) = cos(2φ)
       and    sin(2(φ+π)) = sin(2φ+2π) = sin(2φ)

    2. PARITY (INVERSE): R(-φ) = R(φ)^(-1)
       Rotating by -φ inverts the transformation.
       If (A,B) → (A',B') under R(φ), then (A',B') → (A,B) under R(-φ).

    3. SPECIAL CASE: (A,B) = (0,1) at φ = π/4
       A' = 0·cos(π/2) - 1·sin(π/2) = -1
       B' = 0·sin(π/2) + 1·cos(π/2) = 0
       So (0,1) → (-1,0) at φ = π/4.

    These are "tripwire" tests: if any of these fail, there's a fundamental
    error in the rotation formula (wrong sign, missing factor of 2, etc.).
    """
    results = {}
    all_passed = True

    if verbose:
        print("\n" + "=" * 70)
        print("TEST A: Periodicity and Parity (Tripwire Tests)")
        print("=" * 70)

    # Test values
    A, B = 0.3, 0.4
    phi = np.pi / 5

    tol = 1e-14  # Very strict (pure math, no numerics)

    # -------------------------------------------------------------------------
    # Test A.1: Periodicity R(φ+π) = R(φ)
    # -------------------------------------------------------------------------
    A1, B1 = rotate_coefficients(A, B, phi)
    A2, B2 = rotate_coefficients(A, B, phi + np.pi)

    err_periodicity = max(abs(A1 - A2), abs(B1 - B2))
    passed_period = err_periodicity < tol
    results['periodicity'] = {'err': err_periodicity, 'passed': passed_period}

    if verbose:
        print(f"\n  A.1 PERIODICITY: R(φ+π) = R(φ)")
        print(f"      R(φ)   : ({A1:+.10f}, {B1:+.10f})")
        print(f"      R(φ+π) : ({A2:+.10f}, {B2:+.10f})")
        print(f"      err = {err_periodicity:.2e} → {'PASS' if passed_period else 'FAIL'}")

    if not passed_period:
        all_passed = False

    # -------------------------------------------------------------------------
    # Test A.2: Parity R(-φ) = R(φ)^(-1)
    # Apply R(φ) then R(-φ) should give back original
    # -------------------------------------------------------------------------
    A_rot, B_rot = rotate_coefficients(A, B, phi)
    A_back, B_back = rotate_coefficients(A_rot, B_rot, -phi)

    err_parity = max(abs(A_back - A), abs(B_back - B))
    passed_parity = err_parity < tol
    results['parity'] = {'err': err_parity, 'passed': passed_parity}

    if verbose:
        print(f"\n  A.2 PARITY: R(-φ) inverts R(φ)")
        print(f"      Original: ({A:+.10f}, {B:+.10f})")
        print(f"      R(φ):     ({A_rot:+.10f}, {B_rot:+.10f})")
        print(f"      R(-φ):    ({A_back:+.10f}, {B_back:+.10f})")
        print(f"      err = {err_parity:.2e} → {'PASS' if passed_parity else 'FAIL'}")

    if not passed_parity:
        all_passed = False

    # -------------------------------------------------------------------------
    # Test A.3: Special case (0,1) at φ=π/4 → (-1,0)
    # -------------------------------------------------------------------------
    A_sp, B_sp = 0.0, 1.0
    phi_sp = np.pi / 4
    A_sp_rot, B_sp_rot = rotate_coefficients(A_sp, B_sp, phi_sp)

    # Expected: A' = -1, B' = 0
    err_special_A = abs(A_sp_rot - (-1.0))
    err_special_B = abs(B_sp_rot - 0.0)
    err_special = max(err_special_A, err_special_B)
    passed_special = err_special < tol
    results['special'] = {'err': err_special, 'passed': passed_special}

    if verbose:
        print(f"\n  A.3 SPECIAL CASE: (0,1) at φ=π/4 → (-1,0)")
        print(f"      Input:    (A,B) = (0, 1)")
        print(f"      Output:   (A',B') = ({A_sp_rot:+.10f}, {B_sp_rot:+.10f})")
        print(f"      Expected: (-1, 0)")
        print(f"      err = {err_special:.2e} → {'PASS' if passed_special else 'FAIL'}")

    if not passed_special:
        all_passed = False

    # Summary
    if verbose:
        print(f"\n  RESULT: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 70)

    results['all_passed'] = all_passed
    return results


# =============================================================================
# TEST C: 3D CUBIC KERNEL WITH Rz ROTATION
# =============================================================================

def cubic_anisotropy_kernel(nx: float, ny: float, nz: float) -> float:
    """
    Cubic anisotropy kernel: f(n̂) = 1 - 6×(n_x²n_y² + n_y²n_z² + n_z²n_x²).

    This is the standard form for cubic symmetry anisotropy.
    For isotropic material: f = 1 everywhere.
    For cubic material: f varies with direction.

    Properties:
    - f([1,0,0]) = 1 (along cube axis)
    - f([1,1,0]/√2) = 1 - 6×(1/4) = -1/2 (face diagonal)
    - f([1,1,1]/√3) = 1 - 6×3×(1/9) = -1 (body diagonal)
    """
    return 1.0 - 6.0 * (nx**2 * ny**2 + ny**2 * nz**2 + nz**2 * nx**2)


def rotation_matrix_z(alpha: float) -> np.ndarray:
    """Rotation matrix Rz(α) for rotation around z-axis by angle α."""
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def test_cubic_kernel_rotation(verbose: bool = True) -> Dict:
    """
    Test that 3D cubic kernel transforms correctly under Rz rotation.

    This links the 2D rotation test to the actual 3D foam/Bloch model.

    Physics:
        The cubic anisotropy kernel f(n̂) defines direction-dependent
        elastic properties. Rotating the coordinate frame by Rz(α)
        transforms n̂ → Rz^(-1) · n̂.

        Equivalently: measuring f at direction n̂ after rotating the
        crystal by α is the same as measuring at Rz^T · n̂ originally.

    Test setup:
        - Sample f(n̂) in the xy-plane: n̂ = (cos φ, sin φ, 0)
        - In-plane, f(φ) = 1/4 + (3/4)cos(4φ) has 4ω character
        - Rotating by α should shift the pattern: f'(φ) = f(φ - α)
        - This verifies the kernel + rotation machinery is correct

    Why this matters:
        - Connects to MC3 test (3D Christoffel kernel)
        - Validates that DisplacementBloch would give correct results
        - Shows the link between abstract 2D test and real 3D foam geometry
    """
    results = {}
    all_passed = True

    if verbose:
        print("\n" + "=" * 70)
        print("TEST C: 3D Cubic Kernel with Rz Rotation")
        print("=" * 70)

    # -------------------------------------------------------------------------
    # C.1: Verify special directions
    # -------------------------------------------------------------------------
    test_dirs = [
        ([1, 0, 0], 1.0, "cube axis [100]"),
        ([1/np.sqrt(2), 1/np.sqrt(2), 0], -0.5, "face diagonal [110]"),
        ([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], -1.0, "body diagonal [111]"),
    ]

    if verbose:
        print(f"\n  C.1 SPECIAL DIRECTIONS:")

    special_passed = True
    tol = 1e-12

    for n_vec, f_expected, label in test_dirs:
        f_computed = cubic_anisotropy_kernel(*n_vec)
        err = abs(f_computed - f_expected)
        passed = err < tol

        if not passed:
            special_passed = False
            all_passed = False

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      f({label}) = {f_computed:+.6f}, expected = {f_expected:+.6f}, err = {err:.2e} → {status}")

    results['special_directions'] = {'passed': special_passed}

    # -------------------------------------------------------------------------
    # C.2: In-plane angular dependence matches analytic formula
    # f(φ) = 1/4 + (3/4)cos(4φ) for n̂ = (cos φ, sin φ, 0)
    # -------------------------------------------------------------------------
    n_phi = 100
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    f_computed = np.array([
        cubic_anisotropy_kernel(np.cos(phi), np.sin(phi), 0)
        for phi in phis
    ])
    f_analytic = 0.25 + 0.75 * np.cos(4 * phis)

    err_inplane = np.max(np.abs(f_computed - f_analytic))
    passed_inplane = err_inplane < tol
    results['inplane_formula'] = {'max_err': err_inplane, 'passed': passed_inplane}

    if not passed_inplane:
        all_passed = False

    if verbose:
        print(f"\n  C.2 IN-PLANE FORMULA: f(φ) = 1/4 + (3/4)cos(4φ)")
        print(f"      max |f_computed - f_analytic| = {err_inplane:.2e} → {'PASS' if passed_inplane else 'FAIL'}")

    # -------------------------------------------------------------------------
    # C.3: Rz rotation shifts the pattern
    # Rotating crystal by α: f_rotated(φ) = f(φ - α)
    # -------------------------------------------------------------------------
    test_alphas = [np.pi/8, np.pi/4, np.pi/3]  # Various rotation angles
    rotation_passed = True

    if verbose:
        print(f"\n  C.3 Rz ROTATION SHIFT:")

    for alpha in test_alphas:
        Rz = rotation_matrix_z(alpha)
        Rz_inv = Rz.T  # For orthogonal matrices, inverse = transpose

        # Method 1: Rotate each direction, then compute kernel
        f_rotated = np.array([
            cubic_anisotropy_kernel(
                *(Rz_inv @ np.array([np.cos(phi), np.sin(phi), 0]))
            )
            for phi in phis
        ])

        # Method 2: Shift the angle (f(φ - α))
        f_shifted = 0.25 + 0.75 * np.cos(4 * (phis - alpha))

        err_rotation = np.max(np.abs(f_rotated - f_shifted))
        passed = err_rotation < tol

        if not passed:
            rotation_passed = False
            all_passed = False

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      α = {np.degrees(alpha):5.1f}°: max err = {err_rotation:.2e} → {status}")

    results['rotation_shift'] = {'passed': rotation_passed}

    # -------------------------------------------------------------------------
    # C.4: Fit 4ω component and verify rotation transforms coefficients
    # f(φ) = C + A cos(4φ) + B sin(4φ)
    # Under Rz(α): A' = A cos(4α) - B sin(4α), B' = A sin(4α) + B cos(4α)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n  C.4 4ω COEFFICIENT TRANSFORMATION:")

    # Fit original
    X = np.column_stack([np.cos(4*phis), np.sin(4*phis), np.ones(n_phi)])
    coeffs_orig, _, _, _ = np.linalg.lstsq(X, f_computed, rcond=None)
    A_orig, B_orig, C_orig = coeffs_orig

    coeff_passed = True
    alpha_test = np.pi / 6  # 30°

    # Rotate and fit
    Rz = rotation_matrix_z(alpha_test)
    Rz_inv = Rz.T
    f_rotated = np.array([
        cubic_anisotropy_kernel(
            *(Rz_inv @ np.array([np.cos(phi), np.sin(phi), 0]))
        )
        for phi in phis
    ])
    coeffs_rot, _, _, _ = np.linalg.lstsq(X, f_rotated, rcond=None)
    A_meas, B_meas, C_meas = coeffs_rot

    # Predict: same as 2ω rotation but with 4α instead of 2φ
    c4a = np.cos(4 * alpha_test)
    s4a = np.sin(4 * alpha_test)
    A_pred = A_orig * c4a - B_orig * s4a
    B_pred = A_orig * s4a + B_orig * c4a

    err_A = abs(A_meas - A_pred)
    err_B = abs(B_meas - B_pred)
    err_C = abs(C_meas - C_orig)  # C invariant

    passed_coeff = (err_A < tol) and (err_B < tol) and (err_C < tol)
    if not passed_coeff:
        coeff_passed = False
        all_passed = False

    results['coefficient_transform'] = {
        'A_orig': A_orig, 'B_orig': B_orig, 'C_orig': C_orig,
        'A_pred': A_pred, 'B_pred': B_pred,
        'A_meas': A_meas, 'B_meas': B_meas, 'C_meas': C_meas,
        'passed': passed_coeff
    }

    if verbose:
        print(f"      Original: A={A_orig:+.6f}, B={B_orig:+.6f}, C={C_orig:+.6f}")
        print(f"      Rotate by α={np.degrees(alpha_test):.1f}°:")
        print(f"        A_pred = {A_pred:+.6f}, A_meas = {A_meas:+.6f}, err = {err_A:.2e}")
        print(f"        B_pred = {B_pred:+.6f}, B_meas = {B_meas:+.6f}, err = {err_B:.2e}")
        print(f"        C_pred = {C_orig:+.6f}, C_meas = {C_meas:+.6f}, err = {err_C:.2e}")
        print(f"      → {'PASS' if passed_coeff else 'FAIL'}")

    # Summary
    if verbose:
        print(f"\n  RESULT: {'PASS' if all_passed else 'FAIL'}")
        if all_passed:
            print(f"\n  Link to 3D model validated:")
            print(f"    - Cubic kernel matches analytic form")
            print(f"    - Rz rotation shifts angular pattern correctly")
            print(f"    - Cubic kernel produces 4ω harmonic in xy-plane")
            print(f"    - Coefficients transform as R(4α) — symmetry consistency for 3D kernel")
        print("=" * 70)

    results['all_passed'] = all_passed
    return results


# =============================================================================
# TEST D: MEAN SIGNAL = 0 (TRACELESS TENSOR SANITY CHECK)
# =============================================================================

def test_mean_signal_zero(seed: int = 42,
                          M: int = 50,
                          delta: float = 0.063,
                          verbose: bool = True) -> Dict:
    """
    Test that mean(Δc/c) = 0 for any grain configuration.

    Physics:
        The anisotropy tensor A(g) is traceless: Tr(A) = 0.
        Therefore the mean of meas @ A @ meas over all measurement angles θ is 0:

            mean_θ [cos(2θ - 2g)] = 0

        This implies:
            mean_θ [Δc/c(θ)] = 0

        regardless of grain angles. This is a "sanity nail" that catches
        bugs in tensor construction.

    Test:
        For multiple random grain configurations, verify mean(delta_c) ≈ 0.
    """
    results = []
    all_passed = True
    tol = 1e-12  # Strict but CI-safe (60 points × 50 grains × trig)

    test_seeds = [seed, seed+1, seed+2, seed+3, seed+4]

    if verbose:
        print("\n" + "=" * 70)
        print("TEST D: Mean Signal = 0 (Traceless Tensor)")
        print("=" * 70)
        print(f"\n  Verifying: mean_θ[Δc/c(θ)] = 0 for any grain configuration")
        print(f"\n  {'seed':<6} {'mean(Δc/c)':<15} {'status':<6}")
        print("  " + "-" * 35)

    for s in test_seeds:
        rng = np.random.default_rng(s)
        grain_angles = rng.uniform(0, np.pi, M)

        _, delta_c = generate_cavity_signal(grain_angles, delta)
        mean_signal = np.mean(delta_c)

        passed = abs(mean_signal) < tol
        if not passed:
            all_passed = False

        results.append({'seed': s, 'mean': mean_signal, 'passed': passed})

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  {s:<6} {mean_signal:<+15.2e} {status:<6}")

    if verbose:
        print("  " + "-" * 35)
        print(f"\n  RESULT: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 70)

    return {'results': results, 'all_passed': all_passed}


# =============================================================================
# VERIFICATION: Rotation matrix form
# =============================================================================

def verify_rotation_matrix_form(verbose: bool = True) -> bool:
    """
    Verify that (A, B) transformation is indeed a 2D rotation by 2φ.

    The transformation matrix is:
        [A']   [cos(2φ)  -sin(2φ)] [A]
        [B'] = [sin(2φ)   cos(2φ)] [B]

    This is R(+2φ) = standard counter-clockwise rotation by 2φ.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("VERIFICATION: Rotation Matrix Structure")
        print("=" * 70)

    A, B = 0.3, 0.4
    phi = np.pi / 5

    A1, B1 = rotate_coefficients(A, B, phi)

    R = np.array([
        [np.cos(2*phi), -np.sin(2*phi)],
        [np.sin(2*phi), np.cos(2*phi)]
    ])
    v_rot = R @ np.array([A, B])
    A2, B2 = v_rot

    match = np.allclose([A1, B1], [A2, B2])
    det_ok = np.isclose(np.linalg.det(R), 1.0)
    orthogonal = np.allclose(R.T @ R, np.eye(2))

    if verbose:
        print(f"\nTest: A={A}, B={B}, φ={np.degrees(phi):.1f}°")
        print(f"\nR(2φ) = [[cos, -sin], [sin, cos]]:")
        print(f"  [{R[0,0]:+.4f}  {R[0,1]:+.4f}]")
        print(f"  [{R[1,0]:+.4f}  {R[1,1]:+.4f}]")
        print(f"\nFormula vs matrix: match = {match}")
        print(f"det(R) = 1: {det_ok}")
        print(f"R^T R = I: {orthogonal}")
        print("=" * 70)

    return match and det_ok and orthogonal


# =============================================================================
# MAIN: RUN ALL TESTS
# =============================================================================

def run_all_tests(verbose: bool = True) -> Dict:
    """Run all tests and return summary."""

    print("\n" + "=" * 70)
    print("TEST 1A-5: COMPLETE TEST SUITE")
    print("=" * 70)

    results = {}

    # Test 1: Rotation transformation
    results['rotation'] = test_rotation_transformation(verbose=verbose)

    # Test 2: Analytic coefficients
    results['analytic'] = test_analytic_coefficients(verbose=verbose)

    # Test 3: Group composition
    results['composition'] = test_group_composition(verbose=verbose)

    # Test 4: Linear scaling
    results['scaling'] = test_linear_scaling(verbose=verbose)

    # Test 5: Signal shift equivalence
    results['shift'] = test_signal_shift_equivalence(verbose=verbose)

    # Test 6: Multiple seeds
    results['seeds'] = test_multiple_seeds(verbose=verbose)

    # Test A: Periodicity and parity (tripwire tests)
    results['periodicity_parity'] = test_periodicity_and_parity(verbose=verbose)

    # Test C: 3D cubic kernel with Rz rotation
    results['cubic_kernel'] = test_cubic_kernel_rotation(verbose=verbose)

    # Test D: Mean signal = 0 (traceless tensor)
    results['mean_zero'] = test_mean_signal_zero(verbose=verbose)

    # Matrix verification
    results['matrix'] = verify_rotation_matrix_form(verbose=verbose)

    # Summary
    all_passed = all([
        results['rotation']['all_passed'],
        results['analytic']['passed'],
        results['composition']['all_passed'],
        results['scaling']['all_passed'],
        results['shift']['all_passed'],
        results['seeds']['all_passed'],
        results['periodicity_parity']['all_passed'],
        results['cubic_kernel']['all_passed'],
        results['mean_zero']['all_passed'],
        results['matrix']
    ])

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (Rotation transformation): {'PASS' if results['rotation']['all_passed'] else 'FAIL'}")
    print(f"  Test 2 (Analytic coefficients):   {'PASS' if results['analytic']['passed'] else 'FAIL'}")
    print(f"  Test 3 (Group composition):       {'PASS' if results['composition']['all_passed'] else 'FAIL'}")
    print(f"  Test 4 (Linear scaling):          {'PASS' if results['scaling']['all_passed'] else 'FAIL'}")
    print(f"  Test 5 (Signal shift):            {'PASS' if results['shift']['all_passed'] else 'FAIL'}")
    print(f"  Test 6 (Multiple seeds):          {'PASS' if results['seeds']['all_passed'] else 'FAIL'}")
    print(f"  Test A (Periodicity/parity):      {'PASS' if results['periodicity_parity']['all_passed'] else 'FAIL'}")
    print(f"  Test C (3D cubic kernel):         {'PASS' if results['cubic_kernel']['all_passed'] else 'FAIL'}")
    print(f"  Test D (Mean signal = 0):         {'PASS' if results['mean_zero']['all_passed'] else 'FAIL'}")
    print(f"  Matrix verification:              {'PASS' if results['matrix'] else 'FAIL'}")
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if all_passed:
        print(f"\n  The 2ω signal structure is fully validated:")
        print(f"    - Coefficients match analytic formulas")
        print(f"    - Rotation transformation is correct (spin-2)")
        print(f"    - Group composition holds")
        print(f"    - Linear scaling confirmed")
        print(f"    - Robust across configurations")
        print(f"    - Periodicity and parity verified (tripwire tests)")
        print(f"    - 3D cubic kernel: 4ω harmonic, R(4α) symmetry")
        print(f"    - Traceless tensor confirmed (mean = 0)")

    print("=" * 70)

    return {'results': results, 'all_passed': all_passed}


if __name__ == "__main__":
    run_all_tests(verbose=True)
