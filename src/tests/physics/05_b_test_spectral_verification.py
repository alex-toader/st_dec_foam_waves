#!/usr/bin/env python3
"""
SPECTRAL VERIFICATION TESTS
===========================

Advanced verification tests for plane-wave embedding and acoustic mode analysis.

Tests that L plane-wave remains CONCENTRATED (not hybridized) and that the
Schur complement S selectively acts on longitudinal modes.

STRUCTURES TESTED: C15, Kelvin, FCC, WP (all 4)

TESTS (12 total)
----------------
Per structure (×4):
- L concentrated (max_a > 0.7-0.8): L plane-wave weight in single mode
- S selects longitudinal: ||S u_L|| >> ||S u_T|| (>5-10×)
- G structure: block-diagonal, G_LL >> G_TT, T nearly degenerate

EXPECTED OUTPUT (Jan 2026)
--------------------------
    12 passed in 0.51s

PHYSICS
-------

The key question: Is "exactly 2 modes" due to GAPPING or HYBRIDIZATION?

Answer: GAPPING. The L plane-wave mode remains concentrated (max_a ≈ 0.88)
but is pushed to very high frequency (ω² ≈ 300-400). This is simpler and
more defensable than claiming hybridization.

Jan 2026
"""

import numpy as np

from physics.bloch import DisplacementBloch
from physics.bath import (
    build_vertex_laplacian_bloch,
    build_divergence_operator_bloch,
    compute_discrete_schur,
)
from core_math_v2.builders import build_fcc_supercell_periodic, build_wp_supercell_periodic
from core_math_v2.builders.multicell_periodic import build_bcc_supercell_periodic
from core_math_v2.builders.c15_periodic import build_c15_supercell_periodic


# =============================================================================
# CONSTANTS (duplicated from 05_a for self-containment)
# =============================================================================

ZERO_TOL = 1e-10
NORM_TOL = 1e-14

DEFAULT_L_C15 = 4.0
DEFAULT_L_KELVIN = 8.0
DEFAULT_L_FCC = 8.0
DEFAULT_L_WP = 4.0
DEFAULT_K_L = 3.0
DEFAULT_K_T = 1.0
DEFAULT_LAMBDA_BATH = 10.0

MAX_A_CONCENTRATED = 0.8
DIV_ZERO_TOL = 1e-20

K_DIR_111 = np.array([1, 1, 1]) / np.sqrt(3)


# =============================================================================
# CORE FUNCTIONS (duplicated from 05_a for self-containment)
# =============================================================================

def build_planewave_embedding(db, k):
    """Build plane-wave embedding matrix U (3V × 3)."""
    V = db.V
    positions = db.vertices

    def planewave(v):
        u = np.zeros(3*V, dtype=complex)
        for i in range(V):
            phase = np.exp(1j * np.dot(k, positions[i]))
            u[3*i:3*i+3] = v * phase
        return u

    e_x = np.array([1, 0, 0])
    e_y = np.array([0, 1, 0])
    e_z = np.array([0, 0, 1])

    return np.column_stack([planewave(e_x), planewave(e_y), planewave(e_z)])


def build_D_eff(db, k, lambda_bath=0.0):
    """Build D_eff(k) = D(k) + λ² S(k)."""
    D = db.build_dynamical_matrix(k)

    if lambda_bath > 0:
        A = build_vertex_laplacian_bloch(db, k)
        B = build_divergence_operator_bloch(db, k)
        S = compute_discrete_schur(A, B)
        D_eff = D + lambda_bath**2 * S
    else:
        D_eff = D

    return D_eff


# =============================================================================
# SPECTRAL WEIGHT FUNCTIONS
# =============================================================================

def build_planewave_L(db, k):
    """Build the longitudinal plane-wave vector u_L (polarization k̂)."""
    V = db.V
    positions = db.vertices
    k_mag = np.linalg.norm(k)

    if k_mag < NORM_TOL:
        k_hat = np.array([1, 0, 0])
    else:
        k_hat = k / k_mag

    u_L = np.zeros(3*V, dtype=complex)
    for i in range(V):
        phase = np.exp(1j * np.dot(k, positions[i]))
        u_L[3*i:3*i+3] = k_hat * phase

    return u_L


def build_planewave_T(db, k):
    """Build two transverse plane-wave vectors u_T1, u_T2 (orthogonal to k̂)."""
    V = db.V
    positions = db.vertices
    k_mag = np.linalg.norm(k)

    if k_mag < NORM_TOL:
        k_hat = np.array([1, 0, 0])
    else:
        k_hat = k / k_mag

    if abs(k_hat[0]) < 0.9:
        perp = np.array([1, 0, 0])
    else:
        perp = np.array([0, 1, 0])

    t1 = np.cross(k_hat, perp)
    t1 = t1 / np.linalg.norm(t1)

    t2 = np.cross(k_hat, t1)
    t2 = t2 / np.linalg.norm(t2)

    u_T1 = np.zeros(3*V, dtype=complex)
    u_T2 = np.zeros(3*V, dtype=complex)

    for i in range(V):
        phase = np.exp(1j * np.dot(k, positions[i]))
        u_T1[3*i:3*i+3] = t1 * phase
        u_T2[3*i:3*i+3] = t2 * phase

    return u_T1, u_T2


def spectral_weight_test(db, k, lambda_bath, n_scan=None):
    """
    T1: Spectral weight spreading test.

    For the longitudinal plane-wave u_L:
        a_j = |⟨w_j, u_L⟩|² / (||w_j||² ||u_L||²)

    Returns:
        dict with max_a, participation_ratio, sum_a, top_5
    """
    D_eff = build_D_eff(db, k, lambda_bath)
    eigenvalues, eigenvectors = np.linalg.eigh(D_eff)

    n_modes = len(eigenvalues)
    if n_scan is None:
        n_scan = n_modes
    n_scan = min(n_scan, n_modes)

    u_L = build_planewave_L(db, k)
    norm_uL = np.sqrt(np.real(np.vdot(u_L, u_L)))

    weights = []
    for j in range(n_scan):
        w_j = eigenvectors[:, j]
        norm_w = np.sqrt(np.real(np.vdot(w_j, w_j)))

        if norm_w < NORM_TOL:
            weights.append(0.0)
            continue

        overlap = np.vdot(w_j, u_L)
        a_j = (np.abs(overlap)**2) / (norm_w**2 * norm_uL**2)
        weights.append(a_j)

    weights = np.array(weights)

    max_a = np.max(weights)
    sum_a = np.sum(weights)

    if np.sum(weights**2) > DIV_ZERO_TOL:
        PR = sum_a**2 / np.sum(weights**2)
    else:
        PR = 0

    idx_sorted = np.argsort(weights)[::-1]
    top_5 = []
    for i in range(min(5, n_scan)):
        j = idx_sorted[i]
        top_5.append({
            'j': j,
            'omega_sq': eigenvalues[j],
            'a_j': weights[j],
        })

    return {
        'max_a': max_a,
        'participation_ratio': PR,
        'sum_a': sum_a,
        'top_5': top_5,
    }


def check_S_action(db, k, lambda_bath):
    """
    Test G: Verify ||S u_L|| >> ||S u_T||.

    The Schur complement S should selectively hit L, not T.
    """
    if lambda_bath <= 0:
        return {'status': 'skipped', 'reason': 'λ=0, no S term'}

    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)
    S = compute_discrete_schur(A, B)

    u_L = build_planewave_L(db, k)
    u_T1, u_T2 = build_planewave_T(db, k)

    norm_uL = np.sqrt(np.real(np.vdot(u_L, u_L)))
    norm_T1 = np.sqrt(np.real(np.vdot(u_T1, u_T1)))
    norm_T2 = np.sqrt(np.real(np.vdot(u_T2, u_T2)))

    S_uL = S @ u_L
    S_T1 = S @ u_T1
    S_T2 = S @ u_T2

    norm_S_uL = np.sqrt(np.real(np.vdot(S_uL, S_uL)))
    norm_S_T1 = np.sqrt(np.real(np.vdot(S_T1, S_T1)))
    norm_S_T2 = np.sqrt(np.real(np.vdot(S_T2, S_T2)))

    ratio_L = norm_S_uL / norm_uL if norm_uL > NORM_TOL else 0
    ratio_T1 = norm_S_T1 / norm_T1 if norm_T1 > NORM_TOL else 0
    ratio_T2 = norm_S_T2 / norm_T2 if norm_T2 > NORM_TOL else 0

    # Use reasonable floor (1e-6) to avoid division by tiny numbers
    # If both T ratios are < 1e-6, the ratio is effectively infinite (L dominates)
    RATIO_FLOOR = 1e-6
    return {
        'ratio_L': ratio_L,
        'ratio_T1': ratio_T1,
        'ratio_T2': ratio_T2,
        'ratio_L_over_T': ratio_L / max(ratio_T1, ratio_T2, RATIO_FLOOR),
    }


# =============================================================================
# TEST F: RECONSTRUCTION IDENTITY
# =============================================================================

def check_G_structure(db, k, lambda_bath):
    """
    Test F: Effective 3×3 Hamiltonian structure.

    Verify that the G matrix in the L/T basis shows correct physics:
    1. G is approximately block-diagonal in L/T basis (off-diagonal < 1% of diagonal)
    2. With bath (λ > 0): G_LL >> G_TT (L mode is massive due to Schur complement)
    3. G_TT shows correct transverse degeneracy

    The L/T basis is constructed from k-direction:
    - L: polarization parallel to k (longitudinal)
    - T1, T2: polarizations perpendicular to k (transverse)

    NOTE: This test requires k ≠ 0 (cannot define L/T at Γ point).
    """
    k_mag = np.linalg.norm(k)
    if k_mag < NORM_TOL:
        raise ValueError("check_G_structure requires k ≠ 0 (L/T undefined at Γ)")

    D_eff = build_D_eff(db, k, lambda_bath)
    V = db.V
    positions = db.vertices

    # Build L/T polarization vectors
    k_hat = k / k_mag

    # Find orthogonal T1, T2
    if abs(k_hat[0]) < 0.9:
        perp = np.array([1, 0, 0])
    else:
        perp = np.array([0, 1, 0])
    T1 = np.cross(k_hat, perp)
    T1 = T1 / np.linalg.norm(T1)
    T2 = np.cross(k_hat, T1)

    def planewave(polarization):
        u = np.zeros(3 * V, dtype=complex)
        for i in range(V):
            phase = np.exp(1j * np.dot(k, positions[i]))
            u[3 * i:3 * i + 3] = polarization * phase
        return u

    # Build U in (L, T1, T2) basis
    U_LT = np.column_stack([planewave(k_hat), planewave(T1), planewave(T2)])

    # Orthonormalize via Cholesky decomposition
    # UtU = L @ L† where L is lower triangular (numpy convention)
    # We need U_ortho such that U_ortho† @ U_ortho = I
    # Setting U_ortho = U @ inv(L†), we get:
    #   U_ortho† @ U_ortho = inv(L†)† @ U† @ U @ inv(L†)
    #                      = inv(L) @ UtU @ inv(L†)
    #                      = inv(L) @ L @ L† @ inv(L†) = I
    # Note: inv(L)† = inv(L†) is a standard identity for invertible matrices
    UtU = U_LT.conj().T @ U_LT
    L_chol = np.linalg.cholesky(UtU)
    # Compute inv(L†) via solve for numerical stability
    # solve(L†, I) gives inv(L†) directly
    inv_L_dag = np.linalg.solve(L_chol.conj().T, np.eye(3))
    U_ortho = U_LT @ inv_L_dag

    # Build G in L/T basis
    G = U_ortho.conj().T @ D_eff @ U_ortho

    # Extract components
    G_LL = np.real(G[0, 0])  # Longitudinal eigenvalue
    G_TT = G[1:, 1:]  # 2×2 transverse block
    G_off = G[0, 1:]  # L-T off-diagonal

    eigenvalues_G = np.linalg.eigvalsh(G)
    eigenvalues_TT = np.linalg.eigvalsh(G_TT)

    # Off-diagonal magnitude relative to diagonal
    off_diag_rel = np.max(np.abs(G_off)) / max(G_LL, np.max(np.abs(eigenvalues_TT)))

    # T degeneracy: (max - min) / mean
    T_spread = (eigenvalues_TT[1] - eigenvalues_TT[0]) / np.mean(eigenvalues_TT) if np.mean(eigenvalues_TT) > 1e-10 else 0

    return {
        'G_eigenvalues': np.sort(np.real(eigenvalues_G)),
        'G_LL': G_LL,
        'G_TT_eigenvalues': eigenvalues_TT,
        'G_TT_mean': np.mean(eigenvalues_TT),
        'L_over_T_ratio': G_LL / np.mean(eigenvalues_TT) if np.mean(eigenvalues_TT) > 1e-10 else float('inf'),
        'off_diagonal_relative': off_diag_rel,
        'T_degeneracy_spread': T_spread,
    }


# =============================================================================
# STRUCTURE BUILDERS (duplicated from 05_a)
# =============================================================================

def _build_kelvin():
    """Build Kelvin N=2, return (db, k)."""
    v, e, f, _ = build_bcc_supercell_periodic(2)
    db = DisplacementBloch(v, e, DEFAULT_L_KELVIN, k_L=DEFAULT_K_L, k_T=DEFAULT_K_T)
    k = 0.02 * (2 * np.pi / db.L) * K_DIR_111
    return db, k


def _build_fcc():
    """Build FCC N=2, return (db, k)."""
    result = build_fcc_supercell_periodic(2)
    v, e = result[0], result[1]
    db = DisplacementBloch(v, e, DEFAULT_L_FCC, k_L=DEFAULT_K_L, k_T=DEFAULT_K_T)
    k = 0.02 * (2 * np.pi / db.L) * K_DIR_111
    return db, k


def _build_wp():
    """Build WP (Weaire-Phelan) N=1, return (db, k)."""
    v, e, f = build_wp_supercell_periodic(1, L_cell=4.0)
    db = DisplacementBloch(v, e, DEFAULT_L_WP, k_L=DEFAULT_K_L, k_T=DEFAULT_K_T)
    k = 0.02 * (2 * np.pi / db.L) * K_DIR_111
    return db, k


def _build_c15():
    """Build C15 (Laves) N=1, return (db, k)."""
    v, e, f, _ = build_c15_supercell_periodic(N=1, L_cell=4.0)
    db = DisplacementBloch(v, e, DEFAULT_L_C15, k_L=DEFAULT_K_L, k_T=DEFAULT_K_T)
    k = 0.02 * (2 * np.pi / db.L) * K_DIR_111
    return db, k


# =============================================================================
# PYTEST TESTS
# =============================================================================

def test_kelvin_L_concentrated_not_hybridized():
    """Verify L plane-wave remains concentrated (max_a > 0.8) with bath."""
    db, k = _build_kelvin()

    sw = spectral_weight_test(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)
    print(f"\nKelvin λ={DEFAULT_LAMBDA_BATH}: L spectral weight max_a = {sw['max_a']:.3f}, PR = {sw['participation_ratio']:.2f}")

    assert sw['max_a'] > MAX_A_CONCENTRATED, f"L should be concentrated (max_a > {MAX_A_CONCENTRATED}), got {sw['max_a']:.3f}"
    assert sw['participation_ratio'] < 2, f"L should not spread (PR < 2), got {sw['participation_ratio']:.2f}"


def test_kelvin_S_selects_longitudinal():
    """Verify S (Schur complement) acts on L, not T."""
    db, k = _build_kelvin()
    tg = check_S_action(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)

    print(f"\nKelvin: ||S u_L||/||S u_T|| = {tg['ratio_L_over_T']:.1f}×")
    assert tg['ratio_L_over_T'] > 10, f"S should select L over T by >10×, got {tg['ratio_L_over_T']:.1f}×"


def test_fcc_L_concentrated_not_hybridized():
    """Verify L plane-wave remains concentrated (max_a > 0.8) with bath for FCC."""
    db, k = _build_fcc()

    sw = spectral_weight_test(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)
    print(f"\nFCC λ={DEFAULT_LAMBDA_BATH}: L spectral weight max_a = {sw['max_a']:.3f}, PR = {sw['participation_ratio']:.2f}")

    assert sw['max_a'] > MAX_A_CONCENTRATED, f"L should be concentrated (max_a > {MAX_A_CONCENTRATED}), got {sw['max_a']:.3f}"
    assert sw['participation_ratio'] < 2, f"L should not spread (PR < 2), got {sw['participation_ratio']:.2f}"


def test_fcc_S_selects_longitudinal():
    """Verify S (Schur complement) acts on L, not T for FCC."""
    db, k = _build_fcc()
    tg = check_S_action(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)

    print(f"\nFCC: ||S u_L||/||S u_T|| = {tg['ratio_L_over_T']:.1f}×")
    assert tg['ratio_L_over_T'] > 10, f"S should select L over T by >10×, got {tg['ratio_L_over_T']:.1f}×"


def test_wp_L_concentrated_not_hybridized():
    """Verify L plane-wave remains concentrated with bath for WP.

    WP is more complex, so we allow slightly lower max_a (>0.7 vs >0.8).
    """
    db, k = _build_wp()

    sw = spectral_weight_test(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)
    print(f"\nWP λ={DEFAULT_LAMBDA_BATH}: L spectral weight max_a = {sw['max_a']:.3f}, PR = {sw['participation_ratio']:.2f}")

    assert sw['max_a'] > 0.7, f"L should be concentrated (max_a > 0.7), got {sw['max_a']:.3f}"
    assert sw['participation_ratio'] < 3, f"L should not spread much (PR < 3), got {sw['participation_ratio']:.2f}"


def test_wp_S_selects_longitudinal():
    """Verify S (Schur complement) acts on L, not T for WP structure.

    WP is more complex (non-affine) than Kelvin/FCC, so we allow a relaxed
    threshold of >5 instead of >10.
    """
    db, k = _build_wp()
    tg = check_S_action(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)

    print(f"\nWP: ||S u_L||/||S u_T|| = {tg['ratio_L_over_T']:.1f}×")
    assert tg['ratio_L_over_T'] > 5, f"S should select L over T by >5×, got {tg['ratio_L_over_T']:.1f}×"


def test_G_structure_kelvin():
    """Test F: Verify effective 3×3 Hamiltonian structure in L/T basis.

    With bath, the G matrix should show:
    1. Block-diagonal structure (off-diagonal < 1% of diagonal)
    2. G_LL >> G_TT (L mode massive due to Schur complement)
    3. T modes nearly degenerate (spread < 10%)
    """
    db, k = _build_kelvin()
    result = check_G_structure(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)

    print(f"\nKelvin G structure (λ={DEFAULT_LAMBDA_BATH}):")
    print(f"  G eigenvalues: {result['G_eigenvalues']}")
    print(f"  G_LL: {result['G_LL']:.2f}")
    print(f"  G_TT eigenvalues: {result['G_TT_eigenvalues']}")
    print(f"  L/T ratio: {result['L_over_T_ratio']:.1f}×")
    print(f"  Off-diagonal (rel): {result['off_diagonal_relative']:.2e}")
    print(f"  T degeneracy spread: {result['T_degeneracy_spread']:.2e}")

    # Test 1: Block-diagonal structure (off-diagonal < 1% of diagonal)
    assert result['off_diagonal_relative'] < 0.01, \
        f"G should be block-diagonal (off-diag < 1%), got {result['off_diagonal_relative']:.2e}"

    # Test 2: L mode massive relative to T (at least 10× with bath λ=10)
    assert result['L_over_T_ratio'] > 10, \
        f"G_LL should be >> G_TT (>10×), got {result['L_over_T_ratio']:.1f}×"

    # Test 3: T modes nearly degenerate (spread < 10%)
    assert result['T_degeneracy_spread'] < 0.1, \
        f"T modes should be nearly degenerate (spread < 10%), got {result['T_degeneracy_spread']:.2e}"


def test_G_structure_fcc():
    """Test F: Verify effective 3×3 Hamiltonian structure for FCC."""
    db, k = _build_fcc()
    result = check_G_structure(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)

    print(f"\nFCC G structure (λ={DEFAULT_LAMBDA_BATH}):")
    print(f"  G eigenvalues: {result['G_eigenvalues']}")
    print(f"  L/T ratio: {result['L_over_T_ratio']:.1f}×")
    print(f"  Off-diagonal (rel): {result['off_diagonal_relative']:.2e}")
    print(f"  T degeneracy spread: {result['T_degeneracy_spread']:.2e}")

    assert result['off_diagonal_relative'] < 0.01, \
        f"G should be block-diagonal (off-diag < 1%), got {result['off_diagonal_relative']:.2e}"
    assert result['L_over_T_ratio'] > 10, \
        f"G_LL should be >> G_TT (>10×), got {result['L_over_T_ratio']:.1f}×"
    assert result['T_degeneracy_spread'] < 0.1, \
        f"T modes should be nearly degenerate (spread < 10%), got {result['T_degeneracy_spread']:.2e}"


def test_G_structure_wp():
    """Test F: Verify effective 3×3 Hamiltonian structure for WP.

    WP is more complex, so we allow slightly relaxed thresholds:
    - L/T ratio > 5 (vs > 10 for Kelvin/FCC)
    - T spread < 20% (vs < 10% for Kelvin/FCC)
    """
    db, k = _build_wp()
    result = check_G_structure(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)

    print(f"\nWP G structure (λ={DEFAULT_LAMBDA_BATH}):")
    print(f"  G eigenvalues: {result['G_eigenvalues']}")
    print(f"  L/T ratio: {result['L_over_T_ratio']:.1f}×")
    print(f"  Off-diagonal (rel): {result['off_diagonal_relative']:.2e}")
    print(f"  T degeneracy spread: {result['T_degeneracy_spread']:.2e}")

    assert result['off_diagonal_relative'] < 0.05, \
        f"G should be approximately block-diagonal (off-diag < 5%), got {result['off_diagonal_relative']:.2e}"
    assert result['L_over_T_ratio'] > 5, \
        f"G_LL should be >> G_TT (>5×), got {result['L_over_T_ratio']:.1f}×"
    assert result['T_degeneracy_spread'] < 0.2, \
        f"T modes should be nearly degenerate (spread < 20%), got {result['T_degeneracy_spread']:.2e}"


def test_c15_L_concentrated_not_hybridized():
    """Verify L plane-wave remains concentrated with bath for C15."""
    db, k = _build_c15()

    sw = spectral_weight_test(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)
    print(f"\nC15 λ={DEFAULT_LAMBDA_BATH}: L spectral weight max_a = {sw['max_a']:.3f}, PR = {sw['participation_ratio']:.2f}")

    assert sw['max_a'] > 0.7, f"L should be concentrated (max_a > 0.7), got {sw['max_a']:.3f}"
    assert sw['participation_ratio'] < 3, f"L should not spread much (PR < 3), got {sw['participation_ratio']:.2f}"


def test_c15_S_selects_longitudinal():
    """Verify S (Schur complement) acts on L, not T for C15."""
    db, k = _build_c15()
    tg = check_S_action(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)

    print(f"\nC15: ||S u_L||/||S u_T|| = {tg['ratio_L_over_T']:.1f}×")
    assert tg['ratio_L_over_T'] > 5, f"S should select L over T by >5×, got {tg['ratio_L_over_T']:.1f}×"


def test_G_structure_c15():
    """Test F: Verify effective 3×3 Hamiltonian structure for C15."""
    db, k = _build_c15()
    result = check_G_structure(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)

    print(f"\nC15 G structure (λ={DEFAULT_LAMBDA_BATH}):")
    print(f"  G eigenvalues: {result['G_eigenvalues']}")
    print(f"  L/T ratio: {result['L_over_T_ratio']:.1f}×")
    print(f"  Off-diagonal (rel): {result['off_diagonal_relative']:.2e}")
    print(f"  T degeneracy spread: {result['T_degeneracy_spread']:.2e}")

    assert result['off_diagonal_relative'] < 0.05, \
        f"G should be approximately block-diagonal (off-diag < 5%), got {result['off_diagonal_relative']:.2e}"
    assert result['L_over_T_ratio'] > 5, \
        f"G_LL should be >> G_TT (>5×), got {result['L_over_T_ratio']:.1f}×"
    assert result['T_degeneracy_spread'] < 0.2, \
        f"T modes should be nearly degenerate (spread < 20%), got {result['T_degeneracy_spread']:.2e}"


if __name__ == "__main__":
    print("=" * 70)
    print("SPECTRAL VERIFICATION TESTS")
    print("=" * 70)

    for name, builder in [("C15", _build_c15), ("Kelvin", _build_kelvin), ("FCC", _build_fcc), ("WP", _build_wp)]:
        db, k = builder()
        print(f"\n{name}:")

        sw = spectral_weight_test(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)
        print(f"  L spectral weight: max_a = {sw['max_a']:.3f}, PR = {sw['participation_ratio']:.2f}")

        tg = check_S_action(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)
        if tg.get('status') != 'skipped':
            print(f"  S selectivity: ||S u_L||/||S u_T|| = {tg['ratio_L_over_T']:.1f}×")

        # Test F: G structure
        result = check_G_structure(db, k, lambda_bath=DEFAULT_LAMBDA_BATH)
        print(f"  G structure: L/T = {result['L_over_T_ratio']:.1f}×, off-diag = {result['off_diagonal_relative']:.2e}")
