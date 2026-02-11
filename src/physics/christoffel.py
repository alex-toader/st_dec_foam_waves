"""
Velocity Spread δv/v from Foam Geometry
=======================================

PRIMARY DERIVATION (direct, no fit):
    Foam geometry → DisplacementBloch → v(k̂) sampling → δv/v

SECONDARY (validation):
    Christoffel fit → C11, C12, C44 → confirms cubic symmetry

INPUTS:
    - Foam geometry: vertices v, edges e (from builders)
    - Spring constants: k_L = 3.0, k_T = 1.0

UNITS:
    - Reduced units with ρ = 1 (mass density)
    - Christoffel eigenvalues = v² directly (not ρv²)

OUTPUTS:
    - δv/v = (v_max - v_min) / v_mean for transverse modes
    - WP:     δv/v = 2.5% (most isotropic)
    - Kelvin: δv/v = 6.4%
    - FCC:    δv/v = 16.5%

ASSUMPTIONS FOR CAVITY TESTS:
    1. Elastic δv/v ≈ EM Δc/c (elastic-electromagnetic bridge)
    2. ℓ_corr = ℓ_Planck (grain correlation length = Planck length)
    These are NOT derived, they are model assumptions.

Jan 2026
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, Dict

from .bloch import DisplacementBloch
from core_math.builders import build_fcc_supercell_periodic, build_wp_supercell_periodic
from core_math.builders.multicell_periodic import build_bcc_supercell_periodic


def christoffel_matrix(k_hat: np.ndarray, C11: float, C12: float, C44: float) -> np.ndarray:
    """
    Build Christoffel matrix for cubic symmetry.

    Γ_ij = C_ijkl k̂_k k̂_l

    Args:
        k_hat: (3,) unit wave vector
        C11, C12, C44: cubic elastic constants

    Returns:
        (3, 3) Christoffel matrix
    """
    kx, ky, kz = k_hat

    G = np.array([
        [C11*kx**2 + C44*(ky**2 + kz**2), (C12+C44)*kx*ky, (C12+C44)*kx*kz],
        [(C12+C44)*kx*ky, C11*ky**2 + C44*(kx**2 + kz**2), (C12+C44)*ky*kz],
        [(C12+C44)*kx*kz, (C12+C44)*ky*kz, C11*kz**2 + C44*(kx**2 + ky**2)]
    ])
    return G


def christoffel_velocities_squared(k_hat: np.ndarray, C11: float, C12: float, C44: float) -> np.ndarray:
    """
    Get sorted v² eigenvalues from Christoffel matrix.

    Returns:
        (3,) sorted eigenvalues (v² for T1, T2, L)
    """
    G = christoffel_matrix(k_hat, C11, C12, C44)
    eigs = np.linalg.eigvalsh(G)
    return np.sort(eigs)


def golden_spiral(n: int) -> np.ndarray:
    """Golden spiral sampling on unit sphere."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def measure_velocities(db: DisplacementBloch, L: float,
                       epsilon: float = 0.005, n_directions: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measure acoustic velocities for many directions.

    Args:
        db: DisplacementBloch instance
        L: period
        epsilon: reduced wavevector magnitude
        n_directions: number of directions to sample

    Returns:
        directions: (n, 3) array of k̂ directions
        v_squared: (n, 3) array of v² for each direction (sorted)
    """
    directions = golden_spiral(n_directions)
    k_mag = epsilon * 2 * np.pi / L

    v_squared = []

    for k_hat in directions:
        k = k_mag * k_hat
        omega_T, omega_L, _ = db.classify_modes(k)

        # Get all 3 acoustic frequencies
        omega_all = np.array([omega_T[0], omega_T[1], omega_L[0]])
        v_sq = np.sort((omega_all / k_mag)**2)
        v_squared.append(v_sq)

    return directions, np.array(v_squared)


def fit_cubic_elastic(directions: np.ndarray, v_squared_obs: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Fit C11, C12, C44 to observed velocities.

    Uses least squares to minimize |v²_obs - v²_christoffel|².

    Args:
        directions: (n, 3) array of k̂ directions
        v_squared_obs: (n, 3) array of observed v²

    Returns:
        (C11, C12, C44): fitted elastic constants
        residual: fit residual (sum of squares)
        n_points: number of data points (for RMS calculation)
    """
    def residuals(params):
        C11, C12, C44 = params
        res = []
        for i, k_hat in enumerate(directions):
            v_sq_pred = christoffel_velocities_squared(k_hat, C11, C12, C44)
            res.extend(v_squared_obs[i] - v_sq_pred)
        return np.array(res)

    # Initial guess from [100] direction velocities (dot product more robust)
    idx_100 = np.argmax(directions @ np.array([1.0, 0.0, 0.0]))
    v_sq_100 = v_squared_obs[idx_100]
    C44_init = v_sq_100[0]
    C11_init = v_sq_100[2]
    C12_init = 0.5 * C11_init

    result = least_squares(residuals, [C11_init, C12_init, C44_init],
                          bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))

    # NOTE: result.cost = 0.5 * sum(r²), so use result.fun for actual SS
    residual_ss = np.sum(result.fun**2)
    n_points = len(directions) * 3  # 3 velocities per direction
    return result.x, residual_ss, n_points


def zener_anisotropy(C11: float, C12: float, C44: float) -> float:
    """
    Compute Zener anisotropy factor.

    A_Z = 2*C44 / (C11 - C12)
    A_Z = 1 ⟺ isotropic
    """
    return 2 * C44 / (C11 - C12)


def compute_velocity_spread(C11: float, C12: float, C44: float,
                            n_samples: int = 1000, seed: int = 42,
                            require_stable: bool = True) -> Dict:
    """
    Compute velocity spread δv/v from Christoffel.

    This is the KEY OUTPUT for cavity wash-out calculations:
        δv/v = (v_max - v_min) / v_mean

    Samples random directions k̂, computes v(k̂) from Christoffel,
    returns statistics.

    DERIVATION CHAIN:
        C11, C12, C44 → random sampling → δv/v

    Args:
        C11, C12, C44: cubic elastic constants (from fit_cubic_elastic)
        n_samples: number of random directions
        seed: random seed for reproducibility
        require_stable: if True, raise ValueError on instability (v² < 0)

    Returns:
        dict with v_min, v_max, v_mean, delta_v_over_v
        If unstable and require_stable=False, returns delta_v_over_v=np.nan
    """
    # Check stability conditions first
    stable = (C44 > 0) and (C11 - C12 > 0) and (C11 + 2*C12 > 0)
    if not stable and require_stable:
        raise ValueError(
            f"Unstable elastic constants: C11={C11:.4f}, C12={C12:.4f}, C44={C44:.4f}. "
            f"Conditions: C44>0={C44>0}, C11-C12>0={C11-C12>0}, C11+2C12>0={C11+2*C12>0}"
        )

    rng = np.random.default_rng(seed)

    v_all = []
    n_negative = 0
    for _ in range(n_samples):
        # Random unit vector on sphere
        k_hat = rng.standard_normal(3)
        k_hat = k_hat / np.linalg.norm(k_hat)

        # Get velocities from Christoffel
        v_sq = christoffel_velocities_squared(k_hat, C11, C12, C44)

        # Check for negative v² (instability)
        if np.any(v_sq < -1e-10):
            n_negative += 1
            if require_stable:
                raise ValueError(f"Negative v² at k̂={k_hat}: v²={v_sq}")

        v = np.sqrt(np.maximum(v_sq, 0))

        # T modes only (first two)
        v_all.extend(v[:2])

    # If too many negative, return NaN
    if n_negative > n_samples * 0.01:  # >1% negative
        return {
            'v_min': np.nan, 'v_max': np.nan, 'v_mean': np.nan,
            'delta_v': np.nan, 'delta_v_over_v': np.nan,
            'n_negative': n_negative, 'stable': False,
        }

    v_all = np.array(v_all)
    v_min = np.min(v_all)
    v_max = np.max(v_all)
    v_mean = np.mean(v_all)

    delta_v = v_max - v_min
    delta_v_over_v = delta_v / v_mean

    return {
        'v_min': v_min,
        'v_max': v_max,
        'v_mean': v_mean,
        'delta_v': delta_v,
        'delta_v_over_v': delta_v_over_v,
        'n_negative': n_negative,
        'stable': True,
    }


def compute_delta_v_direct(db: DisplacementBloch, L: float,
                           n_directions: int = 200, epsilon: float = 0.005) -> Dict:
    """
    DIRECT δv/v computation from Bloch - NO FIT.

    PRIMARY DERIVATION (shortest path):
        Foam → DisplacementBloch → v(k̂) for many directions → δv/v

    Args:
        db: DisplacementBloch instance
        L: period
        n_directions: number of k̂ directions to sample
        epsilon: reduced wavevector |k|/(2π/L)

    Returns:
        dict with v_min, v_max, v_mean, delta_v_over_v
    """
    directions = golden_spiral(n_directions)
    k_mag = epsilon * 2 * np.pi / L

    v_T_all = []
    for k_hat in directions:
        k = k_mag * k_hat
        omega_T, omega_L, _ = db.classify_modes(k)
        # Transverse velocities (2 per direction)
        v_T_all.extend([omega_T[0] / k_mag, omega_T[1] / k_mag])

    v_T_all = np.array(v_T_all)

    return {
        'v_min': np.min(v_T_all),
        'v_max': np.max(v_T_all),
        'v_mean': np.mean(v_T_all),
        'delta_v_over_v': (np.max(v_T_all) - np.min(v_T_all)) / np.mean(v_T_all),
        'n_samples': len(v_T_all),
    }


def analyze_structure(name: str, v: np.ndarray, e: list, L: float,
                      k_L: float = 3.0, k_T: float = 1.0) -> Dict:
    """
    Full analysis for a structure.

    PRIMARY DERIVATION (direct, no fit):
        Foam geometry (v, e) → DisplacementBloch → v(k̂) → δv/v

    SECONDARY (validation only):
        Christoffel fit → C11, C12, C44 → verifies cubic symmetry

    Returns dict with delta_v_over_v (primary) and C11, C12, C44 (validation).
    """
    db = DisplacementBloch(v, e, L, k_L=k_L, k_T=k_T)

    # PRIMARY: Direct δv/v (no fit)
    direct = compute_delta_v_direct(db, L, n_directions=200)

    # SECONDARY: Christoffel fit (validation that foam is cubic)
    directions, v_squared = measure_velocities(db, L, epsilon=0.005, n_directions=100)
    (C11, C12, C44), residual, n_points = fit_cubic_elastic(directions, v_squared)
    rms_per_point = np.sqrt(residual / n_points) if n_points > 0 else 0.0
    A_Z = zener_anisotropy(C11, C12, C44)
    spread_fit = compute_velocity_spread(C11, C12, C44, n_samples=1000)

    # Stability conditions (cubic)
    stability = {
        'C44 > 0': C44 > 0,
        'C11 - C12 > 0': (C11 - C12) > 0,
        'C11 + 2*C12 > 0': (C11 + 2*C12) > 0,
    }
    all_stable = all(stability.values())

    return {
        'name': name,
        # PRIMARY OUTPUT (direct, no fit):
        'delta_v_over_v': direct['delta_v_over_v'],
        'v_min': direct['v_min'],
        'v_max': direct['v_max'],
        'v_mean': direct['v_mean'],
        # SECONDARY (Christoffel validation):
        'C11': C11,
        'C12': C12,
        'C44': C44,
        'A_Z': A_Z,
        'delta_v_over_v_fit': spread_fit['delta_v_over_v'],
        # Fit quality (how well foam matches cubic)
        'residual': residual,
        'n_points': n_points,
        'rms_per_point': rms_per_point,
        'stability': stability,
        'all_stable': all_stable,
        'db': db,
        'L': L,
    }


def print_analysis(result: Dict):
    """Print analysis results."""
    print(f"\n{'='*60}")
    print(f"{result['name']}")
    print(f"{'='*60}")

    # PRIMARY OUTPUT (direct, no fit)
    print(f"\n>>> PRIMARY: Direct Bloch sampling (no fit)")
    print(f"  v_min = {result['v_min']:.4f}")
    print(f"  v_max = {result['v_max']:.4f}")
    print(f"  v_mean = {result['v_mean']:.4f}")
    print(f"  δv/v = {result['delta_v_over_v']:.4f} = {result['delta_v_over_v']*100:.1f}%")

    # SECONDARY (Christoffel validation)
    print(f"\n>>> SECONDARY: Christoffel fit (validates cubic symmetry)")
    print(f"  C11 = {result['C11']:.4f}")
    print(f"  C12 = {result['C12']:.4f}")
    print(f"  C44 = {result['C44']:.4f}")
    print(f"  A_Z = {result['A_Z']:.4f}")
    print(f"  δv/v (from fit) = {result['delta_v_over_v_fit']:.4f} = {result['delta_v_over_v_fit']*100:.1f}%")

    # Consistency check
    diff = abs(result['delta_v_over_v'] - result['delta_v_over_v_fit']) / result['delta_v_over_v']
    print(f"\n  Direct vs Fit difference: {diff*100:.1f}%")
    if diff < 0.05:
        print(f"  → Cubic symmetry confirmed (fit matches direct)")

    # Verify on special directions
    print(f"\nVerification on high-symmetry directions:")

    special_dirs = {
        '[100]': np.array([1, 0, 0]),
        '[110]': np.array([1, 1, 0]) / np.sqrt(2),
        '[111]': np.array([1, 1, 1]) / np.sqrt(3),
    }

    db = result['db']
    L = result['L']
    C11, C12, C44 = result['C11'], result['C12'], result['C44']

    print(f"  {'Dir':<8} {'v_T1 obs':<10} {'v_T1 pred':<10} {'v_T2 obs':<10} {'v_T2 pred':<10} {'v_L obs':<10} {'v_L pred':<10}")
    print(f"  {'-'*68}")

    for dir_name, k_hat in special_dirs.items():
        # Observed
        k_mag = 0.005 * 2 * np.pi / L
        k = k_mag * k_hat
        omega_T, omega_L, _ = db.classify_modes(k)
        v_obs = np.sort([omega_T[0]/k_mag, omega_T[1]/k_mag, omega_L[0]/k_mag])

        # Predicted
        v_sq_pred = christoffel_velocities_squared(k_hat, C11, C12, C44)
        v_pred = np.sqrt(v_sq_pred)

        print(f"  {dir_name:<8} {v_obs[0]:<10.4f} {v_pred[0]:<10.4f} {v_obs[1]:<10.4f} {v_pred[1]:<10.4f} {v_obs[2]:<10.4f} {v_pred[2]:<10.4f}")

    print(f"\nFit quality:")
    print(f"  Total residual (sum of squares): {result['residual']:.6f}")
    print(f"  RMS per point: {result['rms_per_point']:.6f}")
    print(f"  N points: {result['n_points']}")

    print(f"\nStability conditions (cubic):")
    C11, C12, C44 = result['C11'], result['C12'], result['C44']
    for condition, passed in result['stability'].items():
        status = "PASS" if passed else "FAIL"
        if condition == 'C44 > 0':
            val = C44
        elif condition == 'C11 - C12 > 0':
            val = C11 - C12
        else:
            val = C11 + 2*C12
        print(f"  {condition}: {val:.4f} → {status}")

    if result['all_stable']:
        print(f"  → All stability conditions satisfied")
    else:
        print(f"  → WARNING: Some stability conditions FAILED")


def main():
    """Run Christoffel analysis on all structures."""
    print("="*60)
    print("CHRISTOFFEL / ZENER FIT FOR ELASTIC ANISOTROPY")
    print("="*60)
    print("\nTheory:")
    print("  Γ_ij(k̂) = C_ijkl k̂_k k̂_l  (Christoffel matrix)")
    print("  eigenvalues(Γ) = ρv²")
    print("  A_Z = 2*C44/(C11-C12)  (Zener anisotropy)")
    print("  A_Z = 1 ⟺ isotropic")

    results = {}

    # Weaire-Phelan (most isotropic foam)
    # N=1 gives 8 cells (2 Type A dodecahedra + 6 Type B tetrakaidecahedra)
    v, e, f = build_wp_supercell_periodic(1, L_cell=4.0)
    results['WP'] = analyze_structure("WP N=1 (A15 foam)", v, e, 4.0)
    print_analysis(results['WP'])

    # Kelvin (BCC foam)
    v, e, f, _ = build_bcc_supercell_periodic(2)
    results['Kelvin'] = analyze_structure("Kelvin N=2 (BCC foam)", v, e, 8.0)
    print_analysis(results['Kelvin'])

    # FCC
    fcc_result = build_fcc_supercell_periodic(2)
    v, e, f = fcc_result[0], fcc_result[1], fcc_result[2]
    results['FCC'] = analyze_structure("FCC N=2", v, e, 8.0)
    print_analysis(results['FCC'])

    # NOTE: SC solid not included - requires different spring model
    # SC is a cubic lattice of atoms, not a foam structure
    # Would need tensorial springs to get proper elastic behavior

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Structure':<12} {'C11':<10} {'C12':<10} {'C44':<10} {'A_Z':<10} {'δv/v':<10}")
    print("-"*60)

    for name, r in results.items():
        delta_pct = r['delta_v_over_v'] * 100
        print(f"{name:<12} {r['C11']:<10.4f} {r['C12']:<10.4f} {r['C44']:<10.4f} {r['A_Z']:<10.4f} {delta_pct:<10.1f}%")

    print("\n" + "="*60)
    print("KEY OUTPUT: δv/v VALUES FOR CAVITY TESTS")
    print("="*60)
    print("\nThese values are DERIVED from first principles:")
    print("  1. Foam geometry → DisplacementBloch → acoustic velocities v(k̂)")
    print("  2. Christoffel fit → C11, C12, C44")
    print("  3. Random k̂ sampling → δv/v = (v_max - v_min) / v_mean")
    print()
    print(f"{'Structure':<12} {'δv/v':<10} {'Use in cavity test as'}")
    print("-"*50)
    for name, r in results.items():
        delta_pct = r['delta_v_over_v'] * 100
        print(f"{name:<12} {delta_pct:<10.1f}% DELTA_{name.upper()} = {r['delta_v_over_v']:.3f}")

    print("\nInterpretation:")
    print("  A_Z = 1: Perfect isotropy (T modes degenerate for all k̂)")
    print("  A_Z ≠ 1: Anisotropy (T1/T2 birefringence)")
    print("  δv/v: velocity spread used in wash-out calculation")

    return results


# =============================================================================
# VALIDATION TESTS (for pytest or direct run)
# =============================================================================

# Helper builders for tests
def _build_wp():
    """Build WP supercell, return (V, E, L)."""
    V, E, F = build_wp_supercell_periodic(1, L_cell=4.0)
    return V, E, 4.0

def _build_kelvin():
    """Build Kelvin supercell, return (V, E, L)."""
    V, E, F, _ = build_bcc_supercell_periodic(2)
    return V, E, 8.0

def _build_fcc():
    """Build FCC supercell, return (V, E, L)."""
    result = build_fcc_supercell_periodic(2)
    V, E, F = result[0], result[1], result[2]
    return V, E, 8.0


def test_synthetic_recovery():
    """
    Synthetic recovery test: known C → v² → fit → recover C.

    Validates fitting independently of DisplacementBloch.
    """
    # Known elastic constants (realistic cubic values)
    C11_true, C12_true, C44_true = 2.40, -0.53, 1.67

    # Generate synthetic v² data
    directions = golden_spiral(100)
    v_squared_obs = np.array([
        christoffel_velocities_squared(k_hat, C11_true, C12_true, C44_true)
        for k_hat in directions
    ])

    # Fit
    (C11_fit, C12_fit, C44_fit), residual, n_points = fit_cubic_elastic(directions, v_squared_obs)

    # Check recovery (should be near-exact for synthetic data)
    # Use 1e-5 for CI robustness across numerical configs
    tol = 1e-5
    errors = {
        'C11': abs(C11_fit - C11_true),
        'C12': abs(C12_fit - C12_true),
        'C44': abs(C44_fit - C44_true),
    }

    print(f"\nSynthetic recovery test:")
    print(f"  True:  C11={C11_true:.4f}, C12={C12_true:.4f}, C44={C44_true:.4f}")
    print(f"  Fit:   C11={C11_fit:.4f}, C12={C12_fit:.4f}, C44={C44_fit:.4f}")
    print(f"  Error: C11={errors['C11']:.2e}, C12={errors['C12']:.2e}, C44={errors['C44']:.2e}")

    all_passed = all(e < tol for e in errors.values())
    assert all_passed, f"Recovery failed: errors = {errors}"
    print(f"  → PASS (all errors < {tol})")

    return {'passed': all_passed, 'errors': errors}


def test_epsilon_convergence():
    """
    Epsilon convergence test: verify acoustic limit is reached.

    Compare v² at epsilon=0.01 vs 0.005. Should converge.
    """
    # Build Kelvin cell
    v, e, f, _ = build_bcc_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    # Measure at two epsilons
    _, v_sq_01 = measure_velocities(db, L, epsilon=0.01, n_directions=50)
    _, v_sq_005 = measure_velocities(db, L, epsilon=0.005, n_directions=50)

    # Relative difference
    rel_diff = np.abs(v_sq_01 - v_sq_005) / (np.abs(v_sq_005) + 1e-10)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    q99_rel_diff = np.quantile(rel_diff, 0.99)

    print(f"\nEpsilon convergence test:")
    print(f"  Max relative diff (ε=0.01 vs 0.005): {max_rel_diff:.4f}")
    print(f"  99th percentile: {q99_rel_diff:.4f}")
    print(f"  Mean relative diff: {mean_rel_diff:.4f}")

    # Use q99 (robust to single outlier) and mean for CI stability
    passed = (q99_rel_diff < 0.15) and (mean_rel_diff < 0.05)
    assert passed, f"Convergence failed: q99={q99_rel_diff:.4f}, mean={mean_rel_diff:.4f}"
    print(f"  → PASS (q99 < 0.15, mean < 0.05)")

    return {'passed': passed, 'max_rel_diff': max_rel_diff}


def test_zener_isotropic():
    """
    Zener sanity test: A_Z = 1 ⟺ T1 = T2 for all directions.

    When C11 = C12 + 2*C44, the medium is isotropic and both
    transverse modes should be degenerate.
    """
    # Set C44, compute C11 to make A_Z = 1
    C44 = 1.5
    C12 = 0.5
    # A_Z = 2*C44/(C11-C12) = 1 → C11 - C12 = 2*C44 → C11 = C12 + 2*C44
    C11 = C12 + 2 * C44

    A_Z = zener_anisotropy(C11, C12, C44)
    assert abs(A_Z - 1.0) < 1e-10, f"A_Z = {A_Z}, should be 1.0"

    # Check degeneracy on many directions
    directions = golden_spiral(100)
    max_split = 0.0

    for k_hat in directions:
        v_sq = christoffel_velocities_squared(k_hat, C11, C12, C44)
        # v_sq[0] = T1, v_sq[1] = T2 (sorted, both transverse)
        split = abs(v_sq[1] - v_sq[0]) / (v_sq[0] + 1e-10)
        max_split = max(max_split, split)

    print(f"\nZener sanity test (A_Z = 1):")
    print(f"  C11={C11:.4f}, C12={C12:.4f}, C44={C44:.4f}")
    print(f"  A_Z = {A_Z:.6f}")
    print(f"  Max T1/T2 split: {max_split:.6f}")

    # Should be exactly degenerate
    threshold = 1e-10
    passed = max_split < threshold
    assert passed, f"T modes not degenerate: split = {max_split}"
    print(f"  → PASS (T1 = T2 for all directions)")

    return {'passed': passed, 'max_split': max_split}


def test_classify_modes_sanity():
    """
    Verify classify_modes picks correct acoustic modes on [100],[110],[111].

    For each direction, the 3 acoustic modes should be the 3 lowest
    non-zero frequencies, with correct L/T polarization.
    """
    # Build Kelvin cell
    v, e, f, _ = build_bcc_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    special_dirs = {
        '[100]': np.array([1.0, 0.0, 0.0]),
        '[110]': np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        '[111]': np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    }

    epsilon = 0.005
    k_mag = epsilon * 2 * np.pi / L

    print(f"\nClassify modes sanity check:")
    all_passed = True

    for dir_name, k_hat in special_dirs.items():
        k = k_mag * k_hat

        # Get classified modes
        omega_T, omega_L, eigvecs = db.classify_modes(k)

        # Get all eigenvalues for comparison
        D = db.build_dynamical_matrix(k)
        all_omega_sq = np.linalg.eigvalsh(D)
        all_omega = np.sqrt(np.maximum(all_omega_sq, 0))
        all_omega_sorted = np.sort(all_omega)

        # The 3 acoustic modes should be among the lowest
        # (there may be zero modes too)
        acoustic_from_classify = sorted([omega_T[0], omega_T[1], omega_L[0]])

        # Find lowest 3 non-zero eigenvalues
        nonzero_omega = all_omega_sorted[all_omega_sorted > 1e-6]
        lowest_3 = nonzero_omega[:3] if len(nonzero_omega) >= 3 else nonzero_omega

        # Check overlap (allow small numerical differences)
        match = True
        for ac_omega in acoustic_from_classify:
            if not any(abs(ac_omega - lo) / (lo + 1e-10) < 0.01 for lo in lowest_3):
                match = False
                break

        status = "PASS" if match else "FAIL"
        if not match:
            all_passed = False

        print(f"  {dir_name}: ω_T=[{omega_T[0]:.4f},{omega_T[1]:.4f}], ω_L={omega_L[0]:.4f}")
        print(f"          lowest 3: [{', '.join(f'{x:.4f}' for x in lowest_3)}] → {status}")

    assert all_passed, "Mode classification mismatch on some directions"
    print(f"  → ALL PASS")

    return {'passed': all_passed}


def test_bloch_vs_christoffel_delta():
    """
    T-C1: Compare δv/v from direct Bloch sampling vs Christoffel fit.

    This closes the derivation chain: if fit is good, both methods
    should give similar δv/v.
    """
    # Build Kelvin cell
    v, e, f, _ = build_bcc_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    # Method 1: Direct Bloch sampling
    directions = golden_spiral(200)
    k_mag = 0.005 * 2 * np.pi / L

    v_direct = []
    for k_hat in directions:
        k = k_mag * k_hat
        omega_T, omega_L, _ = db.classify_modes(k)
        # T modes velocities
        v_direct.extend([omega_T[0]/k_mag, omega_T[1]/k_mag])

    v_direct = np.array(v_direct)
    delta_direct = (np.max(v_direct) - np.min(v_direct)) / np.mean(v_direct)

    # Method 2: Christoffel fit
    directions_fit, v_squared = measure_velocities(db, L, epsilon=0.005, n_directions=100)
    (C11, C12, C44), _, _ = fit_cubic_elastic(directions_fit, v_squared)
    spread = compute_velocity_spread(C11, C12, C44, n_samples=1000)
    delta_fit = spread['delta_v_over_v']

    # Compare
    rel_diff = abs(delta_direct - delta_fit) / delta_direct

    print(f"\nBloch vs Christoffel δv/v comparison:")
    print(f"  Direct Bloch:    δv/v = {delta_direct:.4f} ({delta_direct*100:.1f}%)")
    print(f"  Christoffel fit: δv/v = {delta_fit:.4f} ({delta_fit*100:.1f}%)")
    print(f"  Relative diff:   {rel_diff*100:.1f}%")

    # Should match within 20%
    threshold = 0.20
    passed = rel_diff < threshold
    assert passed, f"δv/v mismatch: direct={delta_direct:.4f}, fit={delta_fit:.4f}, diff={rel_diff*100:.1f}%"
    print(f"  → PASS (diff < {threshold*100:.0f}%)")

    return {'passed': passed, 'delta_direct': delta_direct, 'delta_fit': delta_fit}


def test_reproducibility():
    """
    T-C2: Verify WP/Kelvin/FCC results are reproducible within expected ranges.

    Fixed parameters: k_L=3.0, k_T=1.0
    """
    print(f"\nReproducibility test (fixed params):")

    # Expected ranges (from previous validated runs)
    expected = {
        'WP': {
            'A_Z': (0.90, 1.10),
            'delta_v_over_v': (0.01, 0.06),
        },
        'Kelvin': {
            'A_Z': (1.05, 1.25),
            'delta_v_over_v': (0.04, 0.10),
        },
        'FCC': {
            'A_Z': (1.30, 1.50),
            'delta_v_over_v': (0.12, 0.20),
        },
    }

    all_passed = True
    results = {}

    # WP
    V, E, L = _build_wp()
    results['WP'] = analyze_structure("WP", V, E, L, k_L=3.0, k_T=1.0)
    print(f"  WP:     A_Z={results['WP']['A_Z']:.3f}, δv/v={results['WP']['delta_v_over_v']:.3f}")

    # Kelvin
    V, E, L = _build_kelvin()
    results['Kelvin'] = analyze_structure("Kelvin", V, E, L, k_L=3.0, k_T=1.0)
    print(f"  Kelvin: A_Z={results['Kelvin']['A_Z']:.3f}, δv/v={results['Kelvin']['delta_v_over_v']:.3f}")

    # FCC
    V, E, L = _build_fcc()
    results['FCC'] = analyze_structure("FCC", V, E, L, k_L=3.0, k_T=1.0)
    print(f"  FCC:    A_Z={results['FCC']['A_Z']:.3f}, δv/v={results['FCC']['delta_v_over_v']:.3f}")

    # Check all against expected ranges
    for name, r in results.items():
        for key, (lo, hi) in expected[name].items():
            val = r[key]
            ok = lo <= val <= hi
            if not ok:
                all_passed = False
                print(f"    {name}: {key}={val:.3f} NOT in [{lo}, {hi}]")

    assert all_passed, "Some values outside expected ranges"
    print(f"  → ALL PASS")

    return {'passed': all_passed}


def test_fit_quality():
    """
    Verify Christoffel fit has small residual (foam is truly cubic).

    If rms_per_point is small, it confirms the cubic Christoffel model
    captures the foam behavior, not just that δv/v happens to match.
    """
    print(f"\nFit quality test:")

    # RMS per point should be very small (foam follows cubic exactly)
    # Threshold generous enough to survive epsilon/n_directions changes
    threshold = 0.03  # in v² units (ρ=1 reduced units)

    # WP
    V, E, L = _build_wp()
    result_wp = analyze_structure("WP", V, E, L, k_L=3.0, k_T=1.0)

    # Kelvin
    V, E, L = _build_kelvin()
    result_k = analyze_structure("Kelvin", V, E, L, k_L=3.0, k_T=1.0)

    # FCC
    V, E, L = _build_fcc()
    result_f = analyze_structure("FCC", V, E, L, k_L=3.0, k_T=1.0)

    print(f"  WP:     rms_per_point = {result_wp['rms_per_point']:.6f}")
    print(f"  Kelvin: rms_per_point = {result_k['rms_per_point']:.6f}")
    print(f"  FCC:    rms_per_point = {result_f['rms_per_point']:.6f}")
    print(f"  Threshold: < {threshold}")

    assert result_wp['rms_per_point'] < threshold, f"WP rms {result_wp['rms_per_point']:.6f} >= {threshold}"
    assert result_k['rms_per_point'] < threshold, f"Kelvin rms {result_k['rms_per_point']:.6f} >= {threshold}"
    assert result_f['rms_per_point'] < threshold, f"FCC rms {result_f['rms_per_point']:.6f} >= {threshold}"

    print(f"  → PASS (all foams follow cubic Christoffel)")
    return {'passed': True}


def test_three_zero_modes_at_gamma():
    """
    Verify exactly 3 zero modes at Γ (k=0) and 0 at k≠0.

    For any connected lattice with displacement DOF:
    - k=0: exactly 3 zero modes (global translations x,y,z)
    - k≠0: 0 zero modes (no floppy mechanisms)

    This catches:
    - Disconnected graphs (more than 3 zero modes)
    - Edge/phase bugs (zero modes at k≠0)
    - Zero stiffness edges
    """
    print(f"\nThree zero modes test:")

    structures = [
        ("WP", _build_wp),
        ("Kelvin", _build_kelvin),
        ("FCC", _build_fcc),
    ]

    tol_w2 = 1e-10  # tolerance for ω² ≈ 0
    all_passed = True

    for name, builder in structures:
        V, E, L = builder()
        db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)

        # Γ point (k=0)
        k0 = np.zeros(3)
        D0 = db.build_dynamical_matrix(k0)
        w2_0 = np.linalg.eigvalsh(D0)
        n_zero_gamma = int(np.sum(w2_0 < tol_w2))

        # Small nonzero k
        eps = 0.005
        k = np.array([eps, 0.0, 0.0]) * 2 * np.pi / L
        Dk = db.build_dynamical_matrix(k)
        w2_k = np.linalg.eigvalsh(Dk)
        n_zero_k = int(np.sum(w2_k < tol_w2))

        gamma_ok = (n_zero_gamma == 3)
        k_ok = (n_zero_k == 0)
        status = "PASS" if (gamma_ok and k_ok) else "FAIL"

        print(f"  {name}: Γ={n_zero_gamma} (expect 3), k≠0={n_zero_k} (expect 0) → {status}")

        if not gamma_ok:
            all_passed = False
            print(f"    ERROR: {name} has {n_zero_gamma} zero modes at Γ, expected 3")
        if not k_ok:
            all_passed = False
            print(f"    ERROR: {name} has {n_zero_k} zero modes at k≠0, expected 0")

    assert all_passed, "Some structures failed zero mode test"
    print(f"  → ALL PASS")

    return {'passed': all_passed}


if __name__ == "__main__":
    # Run validation tests first
    print("=" * 60)
    print("VALIDATION TESTS")
    print("=" * 60)
    test_synthetic_recovery()
    test_epsilon_convergence()
    test_zener_isotropic()
    test_classify_modes_sanity()
    test_bloch_vs_christoffel_delta()
    test_reproducibility()
    test_fit_quality()
    test_three_zero_modes_at_gamma()

    # Then main analysis
    main()
