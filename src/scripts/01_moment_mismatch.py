#!/usr/bin/env python3
"""
QW-1: MOMENT MISMATCH ANALYSIS FOR FOAM EDGE DIRECTIONS
========================================================

Computes the 4th order moment mismatch for edge direction distributions.

INPUTS
------

Internal (from model):
  - Foam geometry (WP, Kelvin/BCC, FCC) from builders
  - Edge directions n̂ = (v_j - v_i) / |v_j - v_i| with minimum image convention
  - Periodic box size L: WP L=N×4, Kelvin/FCC L=4×N
  - δv/v values from Bloch analysis (0_b_outputs.md): WP 2.5%, Kelvin 6.4%, FCC 16.5%

External:
  - None (pure geometry analysis)

OUTPUTS
-------

  - Q4 mismatch ||Q4 - Q4_iso||: WP 0.055, Kelvin 0.091, FCC 0.243
  - I4 = <n_x⁴+n_y⁴+n_z⁴>: WP 0.66, Kelvin 0.50, FCC 0.33 (iso=0.6)
  - Q2 mismatch = 0 for all (cubic symmetry preserved)
  - KEY FINDING: Q4 mismatch ranking = δv/v ranking (WP < Kelvin < FCC)
  - Edge direction isotropy predicts acoustic anisotropy

VALIDATION (run with --test)
----------------------------

  - T1: Flip invariance (n → -n) for even moments
  - T2: Rotation invariance of Q4_mismatch (Frobenius norm is SO(3) scalar)
  - T3: Monte Carlo isotropy baseline (N=100k: I4→0.6, cubic→0.2)
  - T4: Periodic unwrap correctness (42/192 BCC edges cross boundary)

PHYSICS
-------

For isotropic edge distribution in 3D, the 4th moment tensor satisfies:
    <n_i n_j n_k n_l>_iso = (1/15)(δ_ij δ_kl + δ_ik δ_jl + δ_il δ_jk)

A useful scalar invariant is:
    I4 = <n_x^4 + n_y^4 + n_z^4>

For isotropic: I4 = 3/5 = 0.6
For edges along axes: I4 = 1 (maximum)
For edges along body diagonals: I4 = 1/3 (minimum)

The mismatch Δ = I4 - 0.6 measures deviation from isotropy.

MINIMUM IMAGE CONVENTION
------------------------

Vertices are wrapped to [0, L). Edges crossing periodic boundaries have
diff components near ±L. We apply:

    diff = diff - L * round(diff / L)

This wraps each component to [-L/2, L/2], ensuring correct edge vectors.

Jan 2026
"""

import numpy as np
from pathlib import Path
import sys

# Path setup: src/scripts/moment_mismatch.py -> parents[1] = src
_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from core_math.builders import (
    build_wp_supercell_periodic,
    build_bcc_supercell_periodic,
    build_fcc_supercell_periodic,
)


# =============================================================================
# EDGE DIRECTION ANALYSIS (with minimum image convention)
# =============================================================================

def get_edge_directions(V: np.ndarray, E: np.ndarray, L: float = None) -> tuple:
    """
    Compute normalized edge direction vectors with minimum image convention.

    Args:
        V: (N_v, 3) vertex positions (wrapped to [0, L))
        E: (N_e, 2) edge indices
        L: periodic box size (required for correct unwrapping)

    Returns:
        directions: (N_e, 3) unit vectors along each edge
        lengths: (N_e,) edge lengths after unwrapping
    """
    E = np.asarray(E, dtype=int)
    diff = V[E[:, 1]] - V[E[:, 0]]

    # Apply minimum image convention (wrap to nearest periodic image)
    if L is not None:
        diff = diff - L * np.round(diff / L)

    lengths = np.linalg.norm(diff, axis=1)

    # Avoid division by zero for degenerate edges
    lengths_safe = np.maximum(lengths, 1e-12)

    return diff / lengths_safe[:, np.newaxis], lengths


def compute_I4(directions: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Compute the 4th moment invariant I4 = <n_x^4 + n_y^4 + n_z^4>.

    For isotropic: I4 = 3/5 = 0.6
    """
    n4_sum = np.sum(directions**4, axis=1)  # (N,)

    if weights is None:
        return np.mean(n4_sum)
    else:
        return np.average(n4_sum, weights=weights)


def compute_Q4_tensor(directions: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Compute full 4th moment tensor Q_{ijkl} = <n_i n_j n_k n_l>.

    Returns as (3,3,3,3) array.
    Uses einsum for efficiency.
    """
    n = directions  # (N, 3)

    if weights is None:
        # Unweighted: Q = mean over einsum
        Q = np.einsum('ni,nj,nk,nl->ijkl', n, n, n, n) / len(n)
    else:
        # Weighted: Q = weighted average
        w = weights / weights.sum()  # normalize
        Q = np.einsum('n,ni,nj,nk,nl->ijkl', w, n, n, n, n)

    return Q


def isotropic_Q4_tensor() -> np.ndarray:
    """
    Return the isotropic 4th moment tensor.

    Q^iso_{ijkl} = (1/15)(δ_ij δ_kl + δ_ik δ_jl + δ_il δ_jk)
    """
    Q = np.zeros((3, 3, 3, 3))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Q[i,j,k,l] = (
                        (1 if i==j else 0) * (1 if k==l else 0) +
                        (1 if i==k else 0) * (1 if j==l else 0) +
                        (1 if i==l else 0) * (1 if j==k else 0)
                    ) / 15.0

    return Q


def compute_Q4_mismatch(directions: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Compute Frobenius norm of Q4 - Q4_iso.
    """
    Q = compute_Q4_tensor(directions, weights)
    Q_iso = isotropic_Q4_tensor()

    return np.linalg.norm(Q - Q_iso)


def compute_Q2_tensor(directions: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Compute 2nd moment tensor Q_{ij} = <n_i n_j>.

    For isotropic: Q^iso = (1/3) δ_ij
    """
    if weights is None:
        return np.mean(directions[:, :, np.newaxis] * directions[:, np.newaxis, :], axis=0)
    else:
        w = weights / weights.sum()
        return np.einsum('n,ni,nj->ij', w, directions, directions)


def compute_Q2_mismatch(directions: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Compute Frobenius norm of Q2 - (1/3)I.
    """
    Q = compute_Q2_tensor(directions, weights)
    Q_iso = np.eye(3) / 3

    return np.linalg.norm(Q - Q_iso)


def compute_cubic_term(directions: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Compute <n_x^2 n_y^2 + n_y^2 n_z^2 + n_z^2 n_x^2>.

    For isotropic: this = 3 × (1/15) = 1/5 = 0.2
    """
    n2 = directions**2
    cubic = n2[:, 0] * n2[:, 1] + n2[:, 1] * n2[:, 2] + n2[:, 2] * n2[:, 0]

    if weights is None:
        return np.mean(cubic)
    else:
        return np.average(cubic, weights=weights)


# =============================================================================
# ANALYSIS FOR EACH STRUCTURE
# =============================================================================

def analyze_structure(name: str, builder_func, N: int, L: float,
                      verbose: bool = True) -> dict:
    """
    Analyze edge direction moments for a given structure.

    Args:
        name: display name
        builder_func: builder function
        N: supercell size
        L: periodic box size (must match builder)
        verbose: print results
    """
    # Build structure
    result = builder_func(N=N)
    V, E = result[0], result[1]

    # Get edge directions with minimum image convention
    directions, lengths = get_edge_directions(V, E, L=L)

    # Verify unwrapping worked (after minimum image, no edge should exceed L/2)
    max_len = lengths.max()
    if max_len > 0.49 * L:
        print(f"  WARNING: max edge length {max_len:.2f} > 0.49*L = {0.49*L:.2f} (unwrap may have failed)")

    # Compute moments (unweighted)
    I4 = compute_I4(directions)
    Q2_mismatch = compute_Q2_mismatch(directions)
    Q4_mismatch = compute_Q4_mismatch(directions)
    cubic_term = compute_cubic_term(directions)

    # Compute moments (length-weighted)
    I4_w = compute_I4(directions, weights=lengths)
    Q4_mismatch_w = compute_Q4_mismatch(directions, weights=lengths)
    cubic_term_w = compute_cubic_term(directions, weights=lengths)

    # Sanity checks
    # I4 range: [1/3, 1] where 1/3 = body diagonals, 1 = axes, 0.6 = isotropic
    assert 0.3 < I4 < 1.1, f"I4 = {I4} out of physical range [1/3, 1]"
    # Q2 mismatch should be small for cubic structures (warning, not assert)
    if Q2_mismatch > 0.1:
        print(f"  WARNING: Q2 mismatch {Q2_mismatch:.4f} unusually large for cubic structure")

    if verbose:
        print(f"\n{name} (N={N}, L={L}):")
        print(f"  Vertices: {len(V)}, Edges: {len(E)}")
        print(f"  Edge lengths: min={lengths.min():.3f}, max={lengths.max():.3f}, mean={lengths.mean():.3f}")
        print(f"\n  UNWEIGHTED:")
        print(f"    I4 = {I4:.6f}  (iso = 0.600, Δ = {I4 - 0.6:+.6f})")
        print(f"    Q2 mismatch = {Q2_mismatch:.6f}")
        print(f"    Q4 mismatch = {Q4_mismatch:.6f}")
        print(f"    cubic term  = {cubic_term:.6f}  (iso = 0.200)")
        print(f"\n  LENGTH-WEIGHTED:")
        print(f"    I4 = {I4_w:.6f}  (iso = 0.600, Δ = {I4_w - 0.6:+.6f})")
        print(f"    Q4 mismatch = {Q4_mismatch_w:.6f}")
        print(f"    cubic term  = {cubic_term_w:.6f}  (iso = 0.200)")

    return {
        'name': name,
        'N': N,
        'L': L,
        'n_vertices': len(V),
        'n_edges': len(E),
        'edge_lengths': lengths,
        # Unweighted
        'I4': I4,
        'delta_I4': I4 - 0.6,
        'Q2_mismatch': Q2_mismatch,
        'Q4_mismatch': Q4_mismatch,
        'cubic_term': cubic_term,
        # Length-weighted
        'I4_weighted': I4_w,
        'Q4_mismatch_weighted': Q4_mismatch_w,
        'cubic_term_weighted': cubic_term_w,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_analysis(verbose: bool = True, N_all: int = 2) -> dict:
    """
    Run moment mismatch analysis for all structures.

    Args:
        verbose: print detailed output
        N_all: supercell size for all structures (default: 2 for consistency)
    """

    if verbose:
        print("=" * 70)
        print("QW-1: MOMENT MISMATCH ANALYSIS (with minimum image convention)")
        print("=" * 70)
        print(f"\nUsing N={N_all} for all structures (consistent comparison)")
        print("Isotropic reference: I4 = 0.6, cubic term = 0.2")

    results = {}

    # Analyze each structure with consistent N
    # WP: L = N * L_cell (L_cell = 4.0 default)
    results['WP'] = analyze_structure(
        'Weaire-Phelan', build_wp_supercell_periodic,
        N=N_all, L=N_all * 4.0, verbose=verbose
    )

    # BCC/Kelvin: L = 4.0 * N
    results['Kelvin'] = analyze_structure(
        'Kelvin (BCC)', build_bcc_supercell_periodic,
        N=N_all, L=4.0 * N_all, verbose=verbose
    )

    # FCC: L = 4.0 * N
    results['FCC'] = analyze_structure(
        'FCC', build_fcc_supercell_periodic,
        N=N_all, L=4.0 * N_all, verbose=verbose
    )

    # Add known δv/v values from Bloch analysis (from 0_b_outputs.md)
    results['WP']['delta_v_v'] = 0.025  # 2.5%
    results['Kelvin']['delta_v_v'] = 0.064  # 6.4%
    results['FCC']['delta_v_v'] = 0.165  # 16.5%

    # Summary tables
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY: UNWEIGHTED")
        print("=" * 70)
        print(f"\n{'Structure':<15} {'Q4 mismatch':>12} {'δv/v':>10} {'cubic term':>12}")
        print("-" * 55)

        for name in ['WP', 'Kelvin', 'FCC']:
            r = results[name]
            print(f"{r['name']:<15} {r['Q4_mismatch']:>12.6f} {r['delta_v_v']*100:>9.1f}% {r['cubic_term']:>12.6f}")

        print("-" * 55)
        print(f"{'Isotropic':<15} {'0.000000':>12} {'0.0':>9}% {'0.200000':>12}")

        print("\n" + "=" * 70)
        print("SUMMARY: LENGTH-WEIGHTED")
        print("=" * 70)
        print(f"\n{'Structure':<15} {'Q4 mismatch':>12} {'δv/v':>10} {'cubic term':>12}")
        print("-" * 55)

        for name in ['WP', 'Kelvin', 'FCC']:
            r = results[name]
            print(f"{r['name']:<15} {r['Q4_mismatch_weighted']:>12.6f} {r['delta_v_v']*100:>9.1f}% {r['cubic_term_weighted']:>12.6f}")

        print("-" * 55)
        print(f"{'Isotropic':<15} {'0.000000':>12} {'0.0':>9}% {'0.200000':>12}")

        # Ranking comparison
        print("\n" + "=" * 70)
        print("RANKING COMPARISON")
        print("=" * 70)

        print("\nBy Q4 mismatch (UNWEIGHTED):")
        q4_sorted = sorted(['WP', 'Kelvin', 'FCC'], key=lambda x: results[x]['Q4_mismatch'])
        print(f"  {' < '.join(q4_sorted)}")
        print(f"  ({results[q4_sorted[0]]['Q4_mismatch']:.4f} < {results[q4_sorted[1]]['Q4_mismatch']:.4f} < {results[q4_sorted[2]]['Q4_mismatch']:.4f})")

        print("\nBy Q4 mismatch (LENGTH-WEIGHTED):")
        q4_w_sorted = sorted(['WP', 'Kelvin', 'FCC'], key=lambda x: results[x]['Q4_mismatch_weighted'])
        print(f"  {' < '.join(q4_w_sorted)}")
        print(f"  ({results[q4_w_sorted[0]]['Q4_mismatch_weighted']:.4f} < {results[q4_w_sorted[1]]['Q4_mismatch_weighted']:.4f} < {results[q4_w_sorted[2]]['Q4_mismatch_weighted']:.4f})")

        print("\nBy δv/v (acoustic anisotropy from Bloch):")
        dv_sorted = sorted(['WP', 'Kelvin', 'FCC'], key=lambda x: results[x]['delta_v_v'])
        print(f"  {' < '.join(dv_sorted)}")
        print(f"  ({results[dv_sorted[0]]['delta_v_v']*100:.1f}% < {results[dv_sorted[1]]['delta_v_v']*100:.1f}% < {results[dv_sorted[2]]['delta_v_v']*100:.1f}%)")

        # Key finding
        print("\n" + "=" * 70)
        print("KEY FINDING")
        print("=" * 70)

        if q4_sorted == dv_sorted:
            print("\n✓ UNWEIGHTED ranking matches δv/v ranking!")
            print(f"  Edge Q4 mismatch predicts acoustic anisotropy.")
        elif q4_w_sorted == dv_sorted:
            print("\n✓ LENGTH-WEIGHTED ranking matches δv/v ranking!")
            print(f"  Length-weighted edge statistics predict acoustic anisotropy.")
        else:
            print("\n⚠️  Neither weighting scheme matches δv/v ranking.")
            print(f"\n  Unweighted: {' < '.join(q4_sorted)}")
            print(f"  Weighted:   {' < '.join(q4_w_sorted)}")
            print(f"  δv/v:       {' < '.join(dv_sorted)}")
            print("\n  This implies: connectivity/topology matters, not just edge directions.")
            print("  The Bloch dynamical matrix captures more than edge statistics.")

        print("\n" + "=" * 70)

    return results


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def run_validation_tests(verbose: bool = True) -> bool:
    """
    Run validation tests for the moment mismatch implementation.

    Tests:
        T1: Flip invariance (n → -n) for even moments
        T2: Rotation invariance of Q4_mismatch
        T3: Monte Carlo isotropy baseline
        T4: Periodic unwrap correctness

    Returns:
        True if all tests pass
    """
    all_passed = True

    if verbose:
        print("=" * 70)
        print("VALIDATION TESTS")
        print("=" * 70)

    # -------------------------------------------------------------------------
    # T1: Flip invariance (n → -n)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n--- T1: Flip Invariance (n → -n) ---")

    V, E, _ = build_wp_supercell_periodic(N=2)
    L = 8.0
    dirs, lengths = get_edge_directions(V, E, L=L)

    I4_orig = compute_I4(dirs)
    I4_flip = compute_I4(-dirs)
    cubic_orig = compute_cubic_term(dirs)
    cubic_flip = compute_cubic_term(-dirs)
    Q4_orig = compute_Q4_mismatch(dirs)
    Q4_flip = compute_Q4_mismatch(-dirs)

    t1_pass = (
        abs(I4_orig - I4_flip) < 1e-12 and
        abs(cubic_orig - cubic_flip) < 1e-12 and
        abs(Q4_orig - Q4_flip) < 1e-12
    )
    all_passed &= t1_pass

    if verbose:
        print(f"  I4:         diff = {abs(I4_orig - I4_flip):.2e}")
        print(f"  cubic_term: diff = {abs(cubic_orig - cubic_flip):.2e}")
        print(f"  Q4_mismatch: diff = {abs(Q4_orig - Q4_flip):.2e}")
        print(f"  T1 {'PASS' if t1_pass else 'FAIL'}")

    # -------------------------------------------------------------------------
    # T2: Rotation invariance of Q4_mismatch
    # -------------------------------------------------------------------------
    if verbose:
        print("\n--- T2: Rotation Invariance ---")

    np.random.seed(42)

    # Generate random directions
    N_test = 500
    dirs_rand = np.random.randn(N_test, 3)
    dirs_rand = dirs_rand / np.linalg.norm(dirs_rand, axis=1, keepdims=True)

    mismatch_original = compute_Q4_mismatch(dirs_rand)

    # Random rotation matrix via QR decomposition
    A = np.random.randn(3, 3)
    R, _ = np.linalg.qr(A)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1  # ensure proper rotation

    dirs_rotated = dirs_rand @ R.T
    mismatch_rotated = compute_Q4_mismatch(dirs_rotated)

    t2_pass = abs(mismatch_original - mismatch_rotated) < 1e-10
    all_passed &= t2_pass

    if verbose:
        print(f"  Q4_mismatch original: {mismatch_original:.10f}")
        print(f"  Q4_mismatch rotated:  {mismatch_rotated:.10f}")
        print(f"  Difference: {abs(mismatch_original - mismatch_rotated):.2e}")
        print(f"  T2 {'PASS' if t2_pass else 'FAIL'}")

    # -------------------------------------------------------------------------
    # T3: Monte Carlo isotropy baseline
    # -------------------------------------------------------------------------
    if verbose:
        print("\n--- T3: Monte Carlo Isotropy Baseline ---")

    np.random.seed(123)
    N_mc = 100000

    # Uniform on sphere via normalized Gaussians
    z = np.random.randn(N_mc, 3)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)

    I4_mc = compute_I4(z)
    cubic_mc = compute_cubic_term(z)
    Q4_mc = compute_Q4_mismatch(z)

    # Expected: I4 = 0.6, cubic = 0.2, Q4_mismatch ~ 1/sqrt(N)
    expected_std = 1 / np.sqrt(N_mc)

    t3_pass = (
        abs(I4_mc - 0.6) < 5 * expected_std and
        abs(cubic_mc - 0.2) < 5 * expected_std and
        Q4_mc < 30 * expected_std  # Q4 Frobenius norm has larger prefactor
    )
    all_passed &= t3_pass

    if verbose:
        print(f"  N = {N_mc}")
        print(f"  I4 = {I4_mc:.6f} (expected 0.600, tol ~{5*expected_std:.4f})")
        print(f"  cubic = {cubic_mc:.6f} (expected 0.200, tol ~{5*expected_std:.4f})")
        print(f"  Q4_mismatch = {Q4_mc:.6f} (expected ~{expected_std:.4f})")
        print(f"  T3 {'PASS' if t3_pass else 'FAIL'}")

    # -------------------------------------------------------------------------
    # T4: Periodic unwrap correctness
    # -------------------------------------------------------------------------
    if verbose:
        print("\n--- T4: Periodic Unwrap Correctness ---")

    V_bcc, E_bcc, *_ = build_bcc_supercell_periodic(N=2)
    L_bcc = 8.0
    E_arr = np.asarray(E_bcc)

    diff_raw = V_bcc[E_arr[:, 1]] - V_bcc[E_arr[:, 0]]
    diff_fixed = diff_raw - L_bcc * np.round(diff_raw / L_bcc)

    lengths_raw = np.linalg.norm(diff_raw, axis=1)
    lengths_fixed = np.linalg.norm(diff_fixed, axis=1)

    # Count boundary-crossing edges
    n_crossing = np.sum(np.any(np.abs(diff_raw) > L_bcc / 2, axis=1))

    # After fix, all edges should be short (< L/2)
    t4_pass = (
        n_crossing > 0 and  # there are crossing edges
        lengths_fixed.max() < 0.49 * L_bcc and  # all fixed lengths are short
        lengths_raw.max() > L_bcc / 2  # raw lengths are long (proves issue exists)
    )
    all_passed &= t4_pass

    if verbose:
        print(f"  BCC N=2: {len(E_bcc)} edges, {n_crossing} cross boundary")
        print(f"  Raw max length:   {lengths_raw.max():.3f} (should be > L/2 = {L_bcc/2})")
        print(f"  Fixed max length: {lengths_fixed.max():.3f} (should be < 0.49*L = {0.49*L_bcc:.1f})")
        print(f"  T4 {'PASS' if t4_pass else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print(f"VALIDATION: {'ALL TESTS PASS' if all_passed else 'SOME TESTS FAILED'}")
        print("=" * 70)

    return all_passed


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run validation tests only
        success = run_validation_tests(verbose=True)
        sys.exit(0 if success else 1)
    else:
        # Run main analysis
        run_analysis(verbose=True)
        print("\n(Run with --test for validation tests)")
