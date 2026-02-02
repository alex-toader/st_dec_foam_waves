#!/usr/bin/env python3
"""
k-Space Bath → P_L Projection Tests
====================================

Tests that local bath produces P_L via Schur complement in k-space.

STRUCTURES TESTED: C15, Kelvin, WP, FCC (all 4)

Theory (continuum):
  Bath DOF: scalar field p(x) on vertices
  Bath energy: (1/2)|∇p|² → (1/2)k²|p̃|² in k-space
  Coupling: p(∇·u) → p̃·(ik·ũ) in k-space

  Integrating out p:
    E_eff = (1/2)ũ†·[Schur complement]·ũ
    Schur = C† A⁻¹ C
         = (ik)† (1/k²) (ik)
         = k_i k_j / k²
         = P_L (longitudinal projector)

This explains "exactly 2 modes":
  - Medium has 3 DOF per point (vector u)
  - Bath stiffens longitudinal component (1 mode becomes gapped/optical)
  - Left with 2 acoustic transverse modes (T1, T2)

TESTS (16 total)
----------------
- Continuum identity (T-B01)
- Divergence scaling, Hermiticity
- Longitudinal selectivity (C15, WP, Kelvin, FCC)
- Dynamic bath tests (kernel, sign flip, static comparison)
- Square penalty tests (PSD, stability, stiffening, scaling)

EXPECTED OUTPUT (Jan 2026)
--------------------------
    16 passed in 0.52s

Jan 2026
"""

import numpy as np
import pytest
from typing import Dict

from physics import build_kelvin_supercell_periodic
from physics.bloch import DisplacementBloch
from physics.bath import (
    continuum_P_L,
    bath_schur_continuum,
    build_vertex_laplacian_bloch,
    build_divergence_operator_bloch,
    compute_discrete_schur,
    build_PL_full,
    # Dynamic bath functions
    compute_dynamic_kernel,
    compute_dynamic_schur,
    D_eff_dynamic,
    get_bath_critical_frequency,
    # Square penalty bath functions
    compute_square_penalty_schur,
    D_eff_square_penalty,
    verify_square_penalty_psd,
)
from core_math_v2.builders import build_fcc_supercell_periodic, build_wp_supercell_periodic
from core_math_v2.builders.c15_periodic import build_c15_supercell_periodic


# =============================================================================
# TEST 1: CONTINUUM IDENTITY
# =============================================================================

def test_continuum_identity():
    """
    T-B01: Bath Schur complement = P_L (analytically).

    This is trivially true in continuum:
      Schur = k⊗k/k² = k̂⊗k̂ = P_L
    """
    # Test several directions
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        np.array([0.3, 0.5, 0.8]),
    ]

    for k in directions:
        k = k / np.linalg.norm(k) * 0.1  # normalize then scale
        P_L = continuum_P_L(k)
        Schur = bath_schur_continuum(k)
        diff = np.linalg.norm(Schur - P_L)
        assert diff < 1e-14, f"Schur != P_L at k={k}: diff={diff}"


# =============================================================================
# DISCRETE SCHUR TESTS (KEY TESTS)
# =============================================================================

def run_discrete_bath_schur(db: DisplacementBloch, L: float, name: str,
                             epsilon: float = 0.02, n_directions: int = 10) -> Dict:
    """
    TEST 4: Discrete Schur complement S = B†A⁻¹B ≈ α·P_L

    This is the KEY test that shows local bath → P_L on lattice.
    """
    np.random.seed(42)

    # Test directions
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    ]
    for _ in range(n_directions - 3):
        v = np.random.randn(3)
        directions.append(v / np.linalg.norm(v))

    k_mag = epsilon * 2 * np.pi / L

    results = []

    for k_hat in directions[:n_directions]:
        k = k_mag * k_hat

        # Build discrete operators
        A = build_vertex_laplacian_bloch(db, k)
        B = build_divergence_operator_bloch(db, k)
        S = compute_discrete_schur(A, B)
        P_L = build_PL_full(db, k)

        # Correlation via quadratic forms on random vectors
        n_samples = 50
        qf_S = []
        qf_PL = []
        for _ in range(n_samples):
            u = np.random.randn(3 * db.V) + 1j * np.random.randn(3 * db.V)
            u /= np.linalg.norm(u)
            qf_S.append(np.real(np.vdot(u, S @ u)))
            qf_PL.append(np.real(np.vdot(u, P_L @ u)))

        qf_S = np.array(qf_S)
        qf_PL = np.array(qf_PL)

        # Correlation
        if np.std(qf_PL) > 1e-10 and np.std(qf_S) > 1e-10:
            corr = np.corrcoef(qf_S, qf_PL)[0, 1]
        else:
            corr = 0.0

        # Find best scale α such that S ≈ α·P_L
        norm_PL_sq = np.real(np.trace(P_L.conj().T @ P_L))
        if norm_PL_sq > 1e-10:
            alpha = np.real(np.trace(S.conj().T @ P_L)) / norm_PL_sq
        else:
            alpha = 0.0

        # Relative error after scaling
        diff = S - alpha * P_L
        norm_S = np.linalg.norm(S, 'fro')
        rel_err = np.linalg.norm(diff, 'fro') / max(norm_S, 1e-10)

        results.append({
            'k_hat': k_hat,
            'correlation': corr,
            'rel_error': rel_err,
            'alpha': alpha,
        })

    # Summary statistics
    mean_corr = np.mean([r['correlation'] for r in results])
    mean_err = np.mean([r['rel_error'] for r in results])

    # Verdict
    if mean_corr > 0.95 and mean_err < 0.2:
        verdict = "PASS"
    elif mean_corr > 0.8:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {
        'name': name,
        'mean_correlation': mean_corr,
        'mean_rel_error': mean_err,
        'results': results,
        'verdict': verdict,
    }


def run_uniform_bath_schur(db: DisplacementBloch, L: float, name: str,
                            epsilon: float = 0.02, n_directions: int = 10) -> Dict:
    """
    TEST 4b: Uniform bath Schur (single effective DOF).

    This projects the bath onto its uniform mode, which should give
    the correct continuum behavior S ≈ α·P_L.
    """
    V = db.V

    np.random.seed(42)

    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    ]
    for _ in range(n_directions - 3):
        v = np.random.randn(3)
        directions.append(v / np.linalg.norm(v))

    k_mag = epsilon * 2 * np.pi / L

    results = []

    for k_hat in directions[:n_directions]:
        k = k_mag * k_hat

        # Build full operators
        A_full = build_vertex_laplacian_bloch(db, k)
        B_full = build_divergence_operator_bloch(db, k)

        # Uniform mode projector
        uniform = np.ones(V, dtype=complex) / np.sqrt(V)

        # Project A onto uniform mode
        A_eff = np.vdot(uniform, A_full @ uniform)
        B_eff = uniform.conj() @ B_full

        # Schur complement (rank-1)
        if np.abs(A_eff) > 1e-12:
            S = np.outer(B_eff.conj(), B_eff) / A_eff
        else:
            S = np.zeros((3*V, 3*V), dtype=complex)

        P_L = build_PL_full(db, k)

        # Correlation via quadratic forms
        n_samples = 50
        qf_S = []
        qf_PL = []
        for _ in range(n_samples):
            u = np.random.randn(3 * V) + 1j * np.random.randn(3 * V)
            u /= np.linalg.norm(u)
            qf_S.append(np.real(np.vdot(u, S @ u)))
            qf_PL.append(np.real(np.vdot(u, P_L @ u)))

        qf_S = np.array(qf_S)
        qf_PL = np.array(qf_PL)

        if np.std(qf_PL) > 1e-10 and np.std(qf_S) > 1e-10:
            corr = np.corrcoef(qf_S, qf_PL)[0, 1]
        else:
            corr = 0.0

        norm_PL_sq = np.real(np.trace(P_L.conj().T @ P_L))
        if norm_PL_sq > 1e-10:
            alpha = np.real(np.trace(S.conj().T @ P_L)) / norm_PL_sq
        else:
            alpha = 0.0

        diff = S - alpha * P_L
        norm_S = np.linalg.norm(S, 'fro')
        rel_err = np.linalg.norm(diff, 'fro') / max(norm_S, 1e-10)

        results.append({
            'k_hat': k_hat,
            'correlation': corr,
            'rel_error': rel_err,
            'alpha': alpha,
        })

    mean_corr = np.mean([r['correlation'] for r in results])
    mean_err = np.mean([r['rel_error'] for r in results])

    if mean_corr > 0.95 and mean_err < 0.2:
        verdict = "PASS"
    elif mean_corr > 0.8:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {
        'name': name,
        'mean_correlation': mean_corr,
        'mean_rel_error': mean_err,
        'results': results,
        'verdict': verdict,
    }


def run_planewave_schur(db: DisplacementBloch, L: float, name: str,
                        epsilon: float = 0.02, n_directions: int = 10) -> Dict:
    """
    TEST 4′: Plane-wave Rayleigh test (CORRECT TEST).

    Instead of comparing S to P_L as 3V×3V matrices, we test the action
    of S on plane-wave displacement patterns.

    Theory:
      For a plane-wave displacement u_i = v · e^{ik·x_i} with polarization v:
      - If S acts like P_L, then: u†Su ∝ |k·v|²/k² (longitudinal selectivity)
      - Transverse polarization (v ⊥ k): u†Su ≈ 0
      - Longitudinal polarization (v ∥ k): u†Su ≈ α·||u||²
    """
    V = db.V
    positions = db.vertices

    def build_planewave(v: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Build plane-wave displacement u_i = v · e^{ik·x_i}."""
        u = np.zeros(3*V, dtype=complex)
        for i in range(V):
            phase = np.exp(1j * np.dot(k, positions[i]))
            u[3*i:3*i+3] = v * phase
        return u

    np.random.seed(42)

    # Test directions
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    ]
    for _ in range(n_directions - 3):
        v = np.random.randn(3)
        directions.append(v / np.linalg.norm(v))

    k_mag = epsilon * 2 * np.pi / L

    all_results = []

    for k_hat in directions[:n_directions]:
        k = k_mag * k_hat

        # Build Schur complement
        A = build_vertex_laplacian_bloch(db, k)
        B = build_divergence_operator_bloch(db, k)
        S = compute_discrete_schur(A, B)

        # Build orthonormal polarization basis
        if np.abs(k_hat[0]) < 0.9:
            t1 = np.cross(k_hat, np.array([1, 0, 0]))
        else:
            t1 = np.cross(k_hat, np.array([0, 1, 0]))
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(k_hat, t1)
        t2 = t2 / np.linalg.norm(t2)

        # Build plane-wave displacements
        u_L = build_planewave(k_hat, k)
        u_T1 = build_planewave(t1, k)
        u_T2 = build_planewave(t2, k)

        # Compute quadratic forms Q = u†Su / ||u||²
        # Each normalized by its own norm (reviewer fix)
        norm_L = np.linalg.norm(u_L)**2
        norm_T1 = np.linalg.norm(u_T1)**2
        norm_T2 = np.linalg.norm(u_T2)**2

        Q_L = np.real(np.vdot(u_L, S @ u_L)) / norm_L
        Q_T1 = np.real(np.vdot(u_T1, S @ u_T1)) / norm_T1
        Q_T2 = np.real(np.vdot(u_T2, S @ u_T2)) / norm_T2

        # Ratio: longitudinal should dominate
        Q_T_max = max(np.abs(Q_T1), np.abs(Q_T2), 1e-12)
        ratio = np.abs(Q_L) / Q_T_max if Q_T_max > 1e-12 else np.inf

        # PASS criterion
        passed = (ratio > 10) and (np.abs(Q_T1) < 0.1 * np.abs(Q_L)) and (np.abs(Q_T2) < 0.1 * np.abs(Q_L))

        all_results.append({
            'k_hat': k_hat,
            'Q_L': Q_L,
            'Q_T1': Q_T1,
            'Q_T2': Q_T2,
            'ratio': ratio,
            'passed': passed,
        })

    # Summary
    n_passed = sum(1 for r in all_results if r['passed'])
    pass_rate = n_passed / len(all_results)
    mean_ratio = np.mean([r['ratio'] for r in all_results if np.isfinite(r['ratio'])])

    if pass_rate > 0.9 and mean_ratio > 10:
        verdict = "PASS"
    elif pass_rate > 0.5 or mean_ratio > 5:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {
        'name': name,
        'pass_rate': pass_rate,
        'mean_ratio': mean_ratio,
        'results': all_results,
        'verdict': verdict,
    }


# =============================================================================
# VERIFICATIONS A-E
# =============================================================================

def verify_A_cos2_angle(db: DisplacementBloch, L: float, name: str,
                        epsilon: float = 0.02, n_angles: int = 10) -> Dict:
    """VERIFICATION A: cos² angle dependence."""
    V = db.V
    positions = db.vertices

    def build_planewave(v: np.ndarray, k: np.ndarray) -> np.ndarray:
        u = np.zeros(3*V, dtype=complex)
        for i in range(V):
            phase = np.exp(1j * np.dot(k, positions[i]))
            u[3*i:3*i+3] = v * phase
        return u

    k_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    k_mag = epsilon * 2 * np.pi / L
    k = k_mag * k_hat

    t1 = np.cross(k_hat, np.array([1, 0, 0]))
    t1 = t1 / np.linalg.norm(t1)

    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)
    S = compute_discrete_schur(A, B)

    u_L = build_planewave(k_hat, k)
    norm_u = np.linalg.norm(u_L)**2
    Q_L = np.real(np.vdot(u_L, S @ u_L)) / norm_u

    angles = np.linspace(0, np.pi/2, n_angles)
    errors = []

    for theta in angles:
        v = np.cos(theta) * k_hat + np.sin(theta) * t1
        v = v / np.linalg.norm(v)
        u = build_planewave(v, k)
        Q_theta = np.real(np.vdot(u, S @ u)) / norm_u

        cos2 = np.cos(theta)**2
        Q_ratio = Q_theta / Q_L if Q_L > 1e-12 else 0
        error = np.abs(Q_ratio - cos2) * 100
        errors.append(error)

    max_error = np.max(errors)

    if max_error < 5.0:
        verdict = "PASS"
    elif max_error < 10.0:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {'name': name, 'mean_error': np.mean(errors), 'max_error': max_error, 'verdict': verdict}


def verify_B_epsilon_sweep(db: DisplacementBloch, L: float, name: str,
                           epsilon_values: list = [0.005, 0.01, 0.02, 0.04]) -> Dict:
    """VERIFICATION B: ε-sweep (IR behavior)."""
    V = db.V
    positions = db.vertices

    def build_planewave(v: np.ndarray, k: np.ndarray) -> np.ndarray:
        u = np.zeros(3*V, dtype=complex)
        for i in range(V):
            phase = np.exp(1j * np.dot(k, positions[i]))
            u[3*i:3*i+3] = v * phase
        return u

    k_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    t1 = np.cross(k_hat, np.array([1, 0, 0]))
    t1 = t1 / np.linalg.norm(t1)

    Q_L_values = []
    ratios = []

    for eps in epsilon_values:
        k_mag = eps * 2 * np.pi / L
        k = k_mag * k_hat

        A = build_vertex_laplacian_bloch(db, k)
        B = build_divergence_operator_bloch(db, k)
        S = compute_discrete_schur(A, B)

        u_L = build_planewave(k_hat, k)
        u_T = build_planewave(t1, k)
        # Normalize each by its own norm (reviewer fix for consistency)
        norm_L = np.linalg.norm(u_L)**2
        norm_T = np.linalg.norm(u_T)**2

        Q_L = np.real(np.vdot(u_L, S @ u_L)) / norm_L
        Q_T = np.real(np.vdot(u_T, S @ u_T)) / norm_T

        ratio = np.abs(Q_L) / max(np.abs(Q_T), 1e-15)
        Q_L_values.append(Q_L)
        ratios.append(ratio)

    Q_L_mean = np.mean(Q_L_values)
    Q_L_var = (np.max(Q_L_values) - np.min(Q_L_values)) / Q_L_mean * 100

    if Q_L_var < 20 and np.min(ratios) > 1000:
        verdict = "PASS"
    elif Q_L_var < 50 and np.min(ratios) > 100:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {'name': name, 'Q_L_var': Q_L_var, 'min_ratio': np.min(ratios), 'verdict': verdict}


def verify_C_regularization_sweep(db: DisplacementBloch, L: float, name: str,
                                   reg_values: list = [1e-8, 1e-10, 1e-12, 1e-14],
                                   epsilon: float = 0.02) -> Dict:
    """VERIFICATION C: Regularization sweep."""
    V = db.V
    positions = db.vertices

    def build_planewave(v: np.ndarray, k: np.ndarray) -> np.ndarray:
        u = np.zeros(3*V, dtype=complex)
        for i in range(V):
            phase = np.exp(1j * np.dot(k, positions[i]))
            u[3*i:3*i+3] = v * phase
        return u

    k_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    k_mag = epsilon * 2 * np.pi / L
    k = k_mag * k_hat

    t1 = np.cross(k_hat, np.array([1, 0, 0]))
    t1 = t1 / np.linalg.norm(t1)

    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)

    Q_L_values = []

    for reg in reg_values:
        S = compute_discrete_schur(A, B, cutoff=reg)

        u_L = build_planewave(k_hat, k)
        norm_u = np.linalg.norm(u_L)**2

        Q_L = np.real(np.vdot(u_L, S @ u_L)) / norm_u
        Q_L_values.append(Q_L)

    Q_L_var = (np.max(Q_L_values) - np.min(Q_L_values)) / np.mean(Q_L_values) * 100

    if Q_L_var < 1.0:
        verdict = "PASS"
    elif Q_L_var < 5.0:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {'name': name, 'Q_L_var': Q_L_var, 'verdict': verdict}


def verify_D_acoustic_subspace(db: DisplacementBloch, L: float, name: str,
                                epsilon: float = 0.02) -> Dict:
    """VERIFICATION D: T4″ acoustic subspace projection."""
    k_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    k_mag = epsilon * 2 * np.pi / L
    k = k_mag * k_hat

    # Get acoustic eigenvectors from D(k)
    D = db.build_dynamical_matrix(k)
    eigs, vecs = np.linalg.eigh(D)

    # First 3 non-zero eigenvalues are acoustic
    idx_sorted = np.argsort(eigs)
    acoustic_idx = []
    for i in idx_sorted:
        if eigs[i] > 1e-10:
            acoustic_idx.append(i)
        if len(acoustic_idx) == 3:
            break

    U_ac = vecs[:, acoustic_idx]

    # Build Schur
    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)
    S = compute_discrete_schur(A, B)

    # Project S onto acoustic subspace
    S_ac = U_ac.conj().T @ S @ U_ac

    # Eigendecomposition of S_ac
    eigs_ac, vecs_ac = np.linalg.eigh(S_ac)

    # Check spectrum shape: two small, one large
    sorted_eigs = np.sort(np.abs(eigs_ac))
    ratio_large_small = sorted_eigs[2] / max(sorted_eigs[0], 1e-15)

    # Find eigenvector of largest eigenvalue
    idx_max = np.argmax(np.abs(eigs_ac))
    v_max = vecs_ac[:, idx_max]

    # Check which acoustic mode aligns with k̂
    f_L_acoustic = []
    for i in range(3):
        mode = U_ac[:, i]
        f_L = db.longitudinal_fraction(mode, k)
        f_L_acoustic.append(f_L)

    idx_L_mode = np.argmax(f_L_acoustic)
    weight_on_L = np.abs(v_max[idx_L_mode])**2

    if ratio_large_small > 100 and weight_on_L > 0.8:
        verdict = "PASS"
    elif ratio_large_small > 10 and weight_on_L > 0.5:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {
        'name': name,
        'eigenvalues': eigs_ac.tolist(),
        'ratio_large_small': ratio_large_small,
        'weight_on_L': weight_on_L,
        'verdict': verdict,
    }


def verify_E_effective_3x3(db: DisplacementBloch, L: float, name: str,
                           epsilon: float = 0.02) -> Dict:
    """VERIFICATION E: 3×3 effective operator G(k) ≈ α·P_L."""
    V = db.V
    positions = db.vertices

    def build_planewave(v: np.ndarray, k: np.ndarray) -> np.ndarray:
        u = np.zeros(3*V, dtype=complex)
        for i in range(V):
            phase = np.exp(1j * np.dot(k, positions[i]))
            u[3*i:3*i+3] = v * phase
        return u

    k_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    k_mag = epsilon * 2 * np.pi / L
    k = k_mag * k_hat

    # Build embedding matrix U (3V × 3)
    e_x = np.array([1, 0, 0])
    e_y = np.array([0, 1, 0])
    e_z = np.array([0, 0, 1])

    U = np.column_stack([
        build_planewave(e_x, k),
        build_planewave(e_y, k),
        build_planewave(e_z, k),
    ])

    # Build Schur
    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)
    S = compute_discrete_schur(A, B)

    # Compute G(k) = (U†U)⁻¹ U† S U
    UtU = U.conj().T @ U
    UtSU = U.conj().T @ S @ U
    G = np.linalg.solve(UtU, UtSU)

    # Eigendecomposition
    eigs_G, vecs_G = np.linalg.eigh(G)

    # Find α (largest eigenvalue)
    alpha = np.max(np.abs(eigs_G))
    idx_max = np.argmax(np.abs(eigs_G))
    v_max = vecs_G[:, idx_max]

    # Compare to k̂
    alignment = np.abs(np.dot(v_max, k_hat))**2

    # Compare G to α·P_L
    P_L = continuum_P_L(k)
    diff = G - alpha * P_L
    rel_error = np.linalg.norm(diff, 'fro') / np.linalg.norm(G, 'fro')

    if rel_error < 0.1 and alignment > 0.95:
        verdict = "PASS"
    elif rel_error < 0.2 and alignment > 0.8:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {
        'name': name,
        'alpha': alpha,
        'alignment': alignment,
        'rel_error': rel_error,
        'verdict': verdict,
    }


def run_all_verifications(db: DisplacementBloch, L: float, name: str) -> Dict:
    """Run all verification tests A-E."""
    results = {}
    results['A_cos2'] = verify_A_cos2_angle(db, L, name)
    results['B_epsilon'] = verify_B_epsilon_sweep(db, L, name)
    results['C_reg'] = verify_C_regularization_sweep(db, L, name)
    results['D_acoustic'] = verify_D_acoustic_subspace(db, L, name)
    results['E_3x3'] = verify_E_effective_3x3(db, L, name)

    all_pass = all(r['verdict'] == 'PASS' for r in results.values())
    any_fail = any(r['verdict'] == 'FAIL' for r in results.values())

    if all_pass:
        overall = "ALL PASS"
    elif any_fail:
        overall = "SOME FAIL"
    else:
        overall = "MARGINAL"

    results['overall'] = overall
    return results


# =============================================================================
# PYTEST TESTS
# =============================================================================

def test_divergence_scaling():
    """
    T-B19: Divergence operator B(k) scales correctly with |k|.

    The discrete divergence should scale linearly with |k|. This verifies
    the coupling operator behaves correctly.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    k_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    eps1, eps2 = 0.01, 0.02

    # Build plane-wave at two different k magnitudes
    def build_planewave(k):
        x = db.vertices
        phase = np.exp(1j * (x @ k))
        u = np.zeros(3 * db.V, dtype=complex)
        for i in range(db.V):
            u[3*i:3*i+3] = k_hat * phase[i]  # longitudinal polarization
        return u / np.linalg.norm(u)

    k1 = (eps1 * 2 * np.pi / L) * k_hat
    k2 = (eps2 * 2 * np.pi / L) * k_hat

    u1 = build_planewave(k1)
    u2 = build_planewave(k2)

    B1 = build_divergence_operator_bloch(db, k1)
    B2 = build_divergence_operator_bloch(db, k2)

    norm_Bu1 = np.linalg.norm(B1 @ u1)
    norm_Bu2 = np.linalg.norm(B2 @ u2)

    # ||Bu|| should scale as k, so ratio of norms ≈ ratio of k magnitudes
    expected_ratio = eps2 / eps1
    actual_ratio = norm_Bu2 / norm_Bu1

    # Allow 30% deviation from linear scaling
    rel_error = abs(actual_ratio - expected_ratio) / expected_ratio
    assert rel_error < 0.3, \
        f"||Bu|| not scaling as k: expected ratio {expected_ratio:.2f}, got {actual_ratio:.2f}"


def test_vertex_laplacian_hermitian():
    """
    T-B20: Vertex Laplacian A(k) is Hermitian for all k.

    A Hermitian operator satisfies A = A†. This is required because:
    1. A represents real physical energy (bath stiffness)
    2. Eigenvalues must be real for physical interpretation
    3. Pseudoinverse of Hermitian stays Hermitian
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    # Test several k values including non-symmetric directions
    test_ks = [
        np.array([0.0, 0.0, 0.0]),                     # Gamma
        np.array([1.0, 0.0, 0.0]) * 0.02 * 2 * np.pi / L,  # [100]
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2) * 0.02 * 2 * np.pi / L,  # [110]
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3) * 0.02 * 2 * np.pi / L,  # [111]
        np.array([0.3, -0.7, 0.5]) * 0.02 * 2 * np.pi / L,  # arbitrary
    ]

    for k in test_ks:
        A = build_vertex_laplacian_bloch(db, k)

        # A should equal its conjugate transpose
        diff = A - A.conj().T
        herm_error = np.linalg.norm(diff, 'fro')
        A_norm = np.linalg.norm(A, 'fro')

        if A_norm > 1e-12:
            rel_error = herm_error / A_norm
        else:
            rel_error = herm_error  # A is near zero at k=0

        assert rel_error < 1e-10, \
            f"A(k) not Hermitian at k={k}: ||A - A†||/||A|| = {rel_error:.2e}"


@pytest.mark.slow
def test_fcc_longitudinal_selectivity():
    """
    T-B15: FCC structure also shows longitudinal selectivity.

    Cross-check on different structure to avoid single-structure overfitting.
    Marked as slow because it builds a larger structure.
    """
    v, e, f, _ = build_fcc_supercell_periodic(2)
    L = 4.0  # FCC period for N=2
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    test_directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    ]

    epsilon = 0.02
    pass_count = 0

    for k_hat in test_directions:
        k = (epsilon * 2 * np.pi / L) * k_hat

        A = build_vertex_laplacian_bloch(db, k)
        B = build_divergence_operator_bloch(db, k)
        S = compute_discrete_schur(A, B)

        # Plane-wave patterns
        x = db.vertices
        phase = np.exp(1j * (x @ k))

        def plane_wave(vpol):
            u = np.zeros(3 * db.V, dtype=complex)
            for i in range(db.V):
                u[3*i:3*i+3] = vpol * phase[i]
            return u / np.linalg.norm(u)

        # Longitudinal and transverse
        vL = k_hat
        if abs(k_hat[2]) < 0.9:
            vT = np.cross(k_hat, np.array([0.0, 0.0, 1.0]))
        else:
            vT = np.cross(k_hat, np.array([1.0, 0.0, 0.0]))
        vT = vT / np.linalg.norm(vT)

        uL = plane_wave(vL)
        uT = plane_wave(vT)

        eL = np.real(np.vdot(uL, S @ uL))
        eT = np.real(np.vdot(uT, S @ uT))

        ratio = eL / max(eT, 1e-14)
        if ratio > 10.0:
            pass_count += 1

    # Require all directions to pass
    assert pass_count == len(test_directions), \
        f"FCC longitudinal selectivity failed: {pass_count}/{len(test_directions)} passed"


@pytest.mark.slow
def test_c15_longitudinal_selectivity():
    """
    T-B16: C15 structure also shows longitudinal selectivity.

    Cross-check on C15 (Laves foam) - the most isotropic structure.
    """
    v, e, f, _ = build_c15_supercell_periodic(N=1, L_cell=4.0)
    L = 4.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    test_directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    ]

    epsilon = 0.02
    pass_count = 0

    for k_hat in test_directions:
        k = (epsilon * 2 * np.pi / L) * k_hat

        A = build_vertex_laplacian_bloch(db, k)
        B = build_divergence_operator_bloch(db, k)
        S = compute_discrete_schur(A, B)

        # Plane-wave patterns
        x = db.vertices
        phase = np.exp(1j * (x @ k))

        def plane_wave(vpol):
            u = np.zeros(3 * db.V, dtype=complex)
            for i in range(db.V):
                u[3*i:3*i+3] = vpol * phase[i]
            return u / np.linalg.norm(u)

        # Longitudinal and transverse
        vL = k_hat
        if abs(k_hat[2]) < 0.9:
            vT = np.cross(k_hat, np.array([0.0, 0.0, 1.0]))
        else:
            vT = np.cross(k_hat, np.array([1.0, 0.0, 0.0]))
        vT = vT / np.linalg.norm(vT)

        uL = plane_wave(vL)
        uT = plane_wave(vT)

        eL = np.real(np.vdot(uL, S @ uL))
        eT = np.real(np.vdot(uT, S @ uT))

        ratio = eL / max(eT, 1e-14)
        if ratio > 10.0:
            pass_count += 1

    assert pass_count == len(test_directions), \
        f"C15 longitudinal selectivity failed: {pass_count}/{len(test_directions)} passed"


@pytest.mark.slow
def test_wp_longitudinal_selectivity():
    """
    T-B17: WP (Weaire-Phelan) structure also shows longitudinal selectivity.

    Cross-check on WP foam.
    """
    v, e, f = build_wp_supercell_periodic(N=1, L_cell=4.0)
    L = 4.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    test_directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    ]

    epsilon = 0.02
    pass_count = 0

    for k_hat in test_directions:
        k = (epsilon * 2 * np.pi / L) * k_hat

        A = build_vertex_laplacian_bloch(db, k)
        B = build_divergence_operator_bloch(db, k)
        S = compute_discrete_schur(A, B)

        # Plane-wave patterns
        x = db.vertices
        phase = np.exp(1j * (x @ k))

        def plane_wave(vpol):
            u = np.zeros(3 * db.V, dtype=complex)
            for i in range(db.V):
                u[3*i:3*i+3] = vpol * phase[i]
            return u / np.linalg.norm(u)

        # Longitudinal and transverse
        vL = k_hat
        if abs(k_hat[2]) < 0.9:
            vT = np.cross(k_hat, np.array([0.0, 0.0, 1.0]))
        else:
            vT = np.cross(k_hat, np.array([1.0, 0.0, 0.0]))
        vT = vT / np.linalg.norm(vT)

        uL = plane_wave(vL)
        uT = plane_wave(vT)

        eL = np.real(np.vdot(uL, S @ uL))
        eT = np.real(np.vdot(uT, S @ uT))

        ratio = eL / max(eT, 1e-14)
        if ratio > 10.0:
            pass_count += 1

    assert pass_count == len(test_directions), \
        f"WP longitudinal selectivity failed: {pass_count}/{len(test_directions)} passed"


# =============================================================================
# DYNAMIC BATH TESTS
# =============================================================================

def test_dynamic_kernel_hermitian():
    """
    T-B21: Dynamic bath kernel is Hermitian for real ω.

    The dynamic kernel (χω² - A)⁻¹ should be Hermitian since A is Hermitian
    and χω² is real.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    k = np.array([0.1, 0.0, 0.0]) * 2 * np.pi / L
    A = build_vertex_laplacian_bloch(db, k)

    chi = 1.0
    # Use omega relative to omega_crit for robustness across structures
    omega_crit = get_bath_critical_frequency(A, chi)
    omega = 0.1 * omega_crit  # Well below critical frequency

    kernel_inv = compute_dynamic_kernel(A, omega, chi, gamma=0.0)

    # Check Hermiticity
    diff = kernel_inv - kernel_inv.conj().T
    rel_error = np.linalg.norm(diff, 'fro') / np.linalg.norm(kernel_inv, 'fro')

    assert rel_error < 1e-10, f"Dynamic kernel not Hermitian: rel_error = {rel_error}"


def test_dynamic_bath_sign_flip():
    """
    T-B22: Dynamic bath gives correct sign at low frequencies.

    Key physics: For ω < ω_crit = √(λ_min(A)/χ):
      - (χω² - A) is negative definite
      - S(ω) = B†(χω² - A)⁻¹B is negative definite
      - D_eff = D - g²S = D + (positive) → STABLE

    This is the sign flip that solves the static elimination problem.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    k = np.array([0.1, 0.0, 0.0]) * 2 * np.pi / L

    # Build operators
    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)
    D = db.build_dynamical_matrix(k)

    chi = 1.0
    g = 1.0

    # Get critical frequency
    omega_crit = get_bath_critical_frequency(A, chi)
    assert omega_crit > 0, "Critical frequency should be positive"

    # Test at ω << ω_crit (should be stable)
    omega_low = omega_crit * 0.1
    D_eff_low = D_eff_dynamic(D, A, B, omega_low, g, chi)
    eigs_low = np.linalg.eigvalsh(D_eff_low)
    min_eig_low = np.min(eigs_low)

    # All eigenvalues should be non-negative (stable)
    assert min_eig_low > -1e-8, \
        f"D_eff unstable at ω={omega_low:.4f} << ω_crit={omega_crit:.4f}: min_eig = {min_eig_low}"


def test_dynamic_schur_vs_static():
    """
    T-B23: Dynamic Schur reduces to static Schur in appropriate limit.

    As ω → 0 with χ → 0 (keeping χω² finite but small):
      S(ω) = B†(χω² - A)⁻¹B → -B†A⁻¹B = -S_static

    This verifies the sign flip mechanism.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    k = np.array([0.1, 0.1, 0.0]) * 2 * np.pi / L

    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)

    # Static Schur
    S_static = compute_discrete_schur(A, B)

    # Dynamic Schur at very low ω (χω² << λ_min(A))
    chi = 1.0
    omega = 0.0001  # Very small
    S_dynamic = compute_dynamic_schur(A, B, omega, chi)

    # At low ω: (χω² - A)⁻¹ ≈ -A⁻¹
    # So S_dynamic ≈ -S_static
    ratio_matrix = S_dynamic + S_static  # Should be ≈ 0

    # Check relative magnitude
    norm_static = np.linalg.norm(S_static, 'fro')
    norm_diff = np.linalg.norm(ratio_matrix, 'fro')
    rel_diff = norm_diff / norm_static

    assert rel_diff < 0.01, \
        f"S_dynamic ≠ -S_static at low ω: rel_diff = {rel_diff}"


def test_critical_frequency_positive():
    """
    T-B24: Critical frequency is well-defined and positive.

    ω_crit = √(λ_min(A)/χ) should be positive for any finite k.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    chi_values = [0.1, 1.0, 10.0]
    k_values = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.1, 0.1, 0.0]),
        np.array([0.1, 0.1, 0.1]),
    ]

    for chi in chi_values:
        for k in k_values:
            k_scaled = k * 2 * np.pi / L
            A = build_vertex_laplacian_bloch(db, k_scaled)
            omega_crit = get_bath_critical_frequency(A, chi)

            assert omega_crit > 0, \
                f"ω_crit not positive: chi={chi}, k={k}, ω_crit={omega_crit}"
            assert np.isfinite(omega_crit), \
                f"ω_crit not finite: chi={chi}, k={k}, ω_crit={omega_crit}"


def test_dynamic_bath_with_damping():
    """
    T-B25: Dynamic bath with damping produces complex kernel.

    With γ > 0, the kernel (χω² - A + iγω)⁻¹ becomes complex.
    The imaginary part corresponds to dissipation.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    k = np.array([0.1, 0.0, 0.0]) * 2 * np.pi / L
    A = build_vertex_laplacian_bloch(db, k)

    chi = 1.0
    gamma = 0.1
    omega = 0.05

    kernel_inv = compute_dynamic_kernel(A, omega, chi, gamma=gamma)

    # With damping, kernel should have imaginary part
    imag_norm = np.linalg.norm(np.imag(kernel_inv), 'fro')
    total_norm = np.linalg.norm(kernel_inv, 'fro')

    # Imaginary part should be non-negligible
    assert imag_norm / total_norm > 0.01, \
        f"Damped kernel should have imaginary part: im/total = {imag_norm/total_norm}"


# =============================================================================
# SQUARE PENALTY BATH TESTS (Alternative Mechanism)
# =============================================================================

def test_square_penalty_psd():
    """
    T-B26: Square penalty Schur is positive semi-definite.

    This is a mathematical identity: S_eff = κW - κ²W(κW + A)⁻¹W ≥ 0.
    It should ALWAYS hold for any valid A, W, κ > 0.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    k = np.array([0.1, 0.0, 0.0]) * 2 * np.pi / L

    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)

    # Test for several κ values
    for kappa in [0.1, 1.0, 10.0, 100.0]:
        result = verify_square_penalty_psd(A, B, kappa)
        assert result['is_psd'], \
            f"Square penalty not PSD at κ={kappa}: min_eig = {result['min_eigenvalue']}"


def test_square_penalty_stable():
    """
    T-B27: Square penalty gives stable D_eff (all eigenvalues ≥ 0).

    Unlike static linear coupling (which gives instability),
    square penalty always gives D_eff = D + (positive) → stable.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    k = np.array([0.1, 0.0, 0.0]) * 2 * np.pi / L

    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)
    D = db.build_dynamical_matrix(k)

    # Test stability for various κ
    for kappa in [1.0, 10.0, 100.0]:
        D_eff = D_eff_square_penalty(D, A, B, kappa)
        eigs = np.linalg.eigvalsh(D_eff)
        min_eig = np.min(eigs)

        assert min_eig > -1e-8, \
            f"D_eff unstable at κ={kappa}: min_eig = {min_eig}"


def test_square_penalty_stiffens_L():
    """
    T-B28: Square penalty stiffens longitudinal mode.

    With sufficient κ, the L mode becomes stiffer (higher velocity).
    NOTE: This is STIFFENING (ω_L ~ c_L·k, acoustic), NOT a true gap (ω_L(k→0) → ω_0).
    The penalty D_pen ~ k² means ω_L² ∝ k² still, just with larger coefficient.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    k = np.array([0.1, 0.0, 0.0]) * 2 * np.pi / L

    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)
    D = db.build_dynamical_matrix(k)

    # Without penalty
    eigs_bare = np.sort(np.linalg.eigvalsh(D))
    # First 3 non-zero are acoustic modes
    acoustic_bare = eigs_bare[eigs_bare > 1e-10][:3]

    # With strong penalty (κ = 100 for clear gap)
    kappa = 100.0
    D_eff = D_eff_square_penalty(D, A, B, kappa)
    eigs_eff = np.sort(np.linalg.eigvalsh(D_eff))
    acoustic_eff = eigs_eff[eigs_eff > 1e-10][:3]

    # L mode should be stiffened (largest acoustic eigenvalue increases)
    # The penalty increases L velocity - any increase confirms the mechanism
    stiffening_ratio = acoustic_eff[2] / acoustic_bare[2]

    assert stiffening_ratio > 1.01, \
        f"L mode not stiffened by penalty: ratio = {stiffening_ratio}"


def test_square_penalty_kappa_scaling():
    """
    T-B29: Square penalty scales correctly with κ.

    As κ → ∞, the penalty approaches Bu = p (hard constraint on divergence).
    The penalty contribution should grow with κ then saturate.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    k = np.array([0.1, 0.0, 0.0]) * 2 * np.pi / L

    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)

    # Compute penalty norm for increasing κ
    kappa_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    norms = []

    for kappa in kappa_values:
        D_penalty = compute_square_penalty_schur(A, B, kappa)
        norms.append(np.linalg.norm(D_penalty, 'fro'))

    # Should be monotonically increasing (penalty grows with κ)
    for i in range(len(norms) - 1):
        assert norms[i+1] >= norms[i] * 0.99, \
            f"Penalty should grow with κ: norm({kappa_values[i+1]}) < norm({kappa_values[i]})"

    # But should saturate (ratio decreases)
    ratio_small = norms[1] / norms[0]  # κ=1 / κ=0.1
    ratio_large = norms[4] / norms[3]  # κ=1000 / κ=100

    assert ratio_large < ratio_small, \
        f"Penalty should saturate: ratio_large={ratio_large} >= ratio_small={ratio_small}"


def test_both_mechanisms_give_stability():
    """
    T-B30: Both dynamic bath AND square penalty give stable D_eff.

    This confirms we have two independent paths to L suppression.
    """
    v, e, f, _ = build_kelvin_supercell_periodic(2)
    L = 8.0
    db = DisplacementBloch(v, e, L, k_L=3.0, k_T=1.0)

    k = np.array([0.1, 0.0, 0.0]) * 2 * np.pi / L

    A = build_vertex_laplacian_bloch(db, k)
    B = build_divergence_operator_bloch(db, k)
    D = db.build_dynamical_matrix(k)

    # Method 1: Dynamic bath (at low ω)
    chi = 1.0
    g = 1.0
    omega_crit = get_bath_critical_frequency(A, chi)
    omega_low = omega_crit * 0.1

    D_eff_dynamic_result = D_eff_dynamic(D, A, B, omega_low, g, chi)
    eigs_dynamic = np.linalg.eigvalsh(D_eff_dynamic_result)
    min_eig_dynamic = np.min(eigs_dynamic)

    # Method 2: Square penalty
    kappa = 10.0
    D_eff_penalty = D_eff_square_penalty(D, A, B, kappa)
    eigs_penalty = np.linalg.eigvalsh(D_eff_penalty)
    min_eig_penalty = np.min(eigs_penalty)

    # Both should be stable
    assert min_eig_dynamic > -1e-8, \
        f"Dynamic bath not stable: min_eig = {min_eig_dynamic}"
    assert min_eig_penalty > -1e-8, \
        f"Square penalty not stable: min_eig = {min_eig_penalty}"

    # Both should gap L (largest acoustic eigenvalue > smallest)
    acoustic_dynamic = np.sort(eigs_dynamic[eigs_dynamic > 1e-10])[:3]
    acoustic_penalty = np.sort(eigs_penalty[eigs_penalty > 1e-10])[:3]

    # L/T ratio should be > 1 for both
    ratio_dynamic = acoustic_dynamic[2] / acoustic_dynamic[0]
    ratio_penalty = acoustic_penalty[2] / acoustic_penalty[0]

    assert ratio_dynamic > 1.2, f"Dynamic: L/T ratio too small: {ratio_dynamic}"
    assert ratio_penalty > 1.2, f"Square penalty: L/T ratio too small: {ratio_penalty}"


def run_mode_classification(db: DisplacementBloch, L: float, name: str,
                              epsilon: float = 0.02, n_directions: int = 20) -> Dict:
    """
    TEST 2: P_L ansatz correctly identifies longitudinal mode (sanity check).

    This tests that using P_L = k̂⊗k̂ as a classifier works:
    - Compute f_L = u†P_L u / u†u for each acoustic mode
    - Mode with highest f_L should be longitudinal

    NOTE: This does NOT test discrete Schur! It only validates that
    the P_L ansatz is a good classifier. See Test 4 for actual Schur test.

    Args:
        db: DisplacementBloch instance
        L: period
        name: structure name
        epsilon: reduced wavevector magnitude
        n_directions: number of directions to test

    Returns:
        dict with test results
    """
    print(f"\n{'='*70}")
    print(f"TEST 2: MODE CLASSIFICATION (P_L ansatz) [{name}]")
    print("=" * 70)

    print("\nSanity check: P_L = k̂⊗k̂ identifies L mode via f_L = u†P_L u / u†u")
    print("NOTE: This does NOT test discrete Schur - see Test 4 for that.")

    # Generate test directions
    np.random.seed(42)
    directions = []

    # Add principal directions
    directions.append(np.array([1.0, 0.0, 0.0]))
    directions.append(np.array([1.0, 1.0, 0.0]) / np.sqrt(2))
    directions.append(np.array([1.0, 1.0, 1.0]) / np.sqrt(3))

    # Add random directions
    for _ in range(n_directions - 3):
        v = np.random.randn(3)
        directions.append(v / np.linalg.norm(v))

    k_mag = epsilon * 2 * np.pi / L

    # For each direction, compare lattice P_L to continuum P_L
    # Using the eigenvector-based test: modes should be classified correctly
    diffs = []

    print(f"\n{'Direction':<25} {'f_L(mode0)':<12} {'f_L(mode1)':<12} {'f_L(mode2)':<12} {'L mode?':<10}")
    print("-" * 75)

    for k_hat in directions[:10]:  # show first 10
        k = k_mag * k_hat
        omega_T, omega_L, f_L_all = db.classify_modes(k)

        # Mode with highest f_L should be longitudinal (f_L > 0.9)
        max_f_L = max(f_L_all)
        classified = "YES" if max_f_L > 0.9 else "NO"

        k_str = f"[{k_hat[0]:.2f},{k_hat[1]:.2f},{k_hat[2]:.2f}]"
        print(f"{k_str:<25} {f_L_all[0]:<12.4f} {f_L_all[1]:<12.4f} {f_L_all[2]:<12.4f} {classified:<10}")

        diffs.append(max_f_L)

    mean_max_f_L = np.mean(diffs)

    print(f"\nMean max f_L: {mean_max_f_L:.4f}")

    if mean_max_f_L > 0.95:
        print("→ P_L correctly identifies longitudinal mode (f_L > 0.95)")
        verdict = "PASS"
    elif mean_max_f_L > 0.85:
        print("→ P_L approximately identifies longitudinal mode (f_L > 0.85)")
        verdict = "MARGINAL"
    else:
        print("→ P_L fails to identify longitudinal mode")
        verdict = "FAIL"

    return {
        'name': name,
        'epsilon': epsilon,
        'mean_max_f_L': mean_max_f_L,
        'verdict': verdict,
    }


def run_bath_effect_on_spectrum(db: DisplacementBloch, L: float, name: str,
                                 m_L_values: list = [0.0, 0.1, 0.3, 0.5],
                                 epsilon: float = 0.05) -> Dict:
    """
    Test: Adding m²·P_L gaps the longitudinal mode.

    This demonstrates the "exactly 2 modes" mechanism:
    - Without bath: 3 acoustic modes (2T + 1L)
    - With bath (m²·P_L): L mode gets gap, 2T remain acoustic

    Args:
        db: DisplacementBloch instance
        L: period
        name: structure name
        m_L_values: longitudinal mass values to test
        epsilon: reduced wavevector magnitude

    Returns:
        dict with test results
    """
    print(f"\n{'='*70}")
    print(f"TEST 3: BATH EFFECT ON SPECTRUM ({name})")
    print("=" * 70)

    print("\nMechanism:")
    print("  Without bath: 3 acoustic modes (2T + 1L)")
    print("  With bath (m²·P_L): L mode gets mass, 2T remain acoustic")
    print("  → 'Exactly 2 modes' emergence")

    k_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)  # [111] direction
    k_mag = epsilon * 2 * np.pi / L
    k = k_mag * k_hat

    print(f"\nDirection: [111], ε = {epsilon}")
    print(f"{'m_L':<10} {'ω_T1':<12} {'ω_T2':<12} {'ω_L':<12} {'gap = ω_L/ω_T':<15}")
    print("-" * 55)

    results = []

    for m_L in m_L_values:
        # Build dynamical matrix with mass term
        D = db.build_dynamical_matrix_with_mass(k, m_L=m_L)
        eigs = np.linalg.eigvalsh(D)
        omega = np.sqrt(np.maximum(eigs, 0))

        # Classify modes
        omega_T, omega_L, f_L = db.classify_modes(k)

        # For massive case, recompute with mass
        if m_L > 0:
            # Get eigenvectors to classify
            D_mass = db.build_dynamical_matrix_with_mass(k, m_L=m_L)
            eigs_m, vecs_m = np.linalg.eigh(D_mass)
            omega_m = np.sqrt(np.maximum(eigs_m, 0))

            # Classify by f_L
            f_L_m = [db.longitudinal_fraction(vecs_m[:, i], k) for i in range(6)]

            # Find L mode (highest f_L among first 3 nonzero)
            nonzero_mask = omega_m > 1e-8
            acoustic_idx = np.where(nonzero_mask)[0][:3]

            f_L_acoustic = [f_L_m[i] for i in acoustic_idx]
            idx_L = acoustic_idx[np.argmax(f_L_acoustic)]
            idx_T = [i for i in acoustic_idx if i != idx_L]

            omega_T1 = omega_m[idx_T[0]] if len(idx_T) > 0 else 0
            omega_T2 = omega_m[idx_T[1]] if len(idx_T) > 1 else 0
            omega_L_val = omega_m[idx_L]
        else:
            omega_T1, omega_T2 = omega_T[0], omega_T[1]
            omega_L_val = omega_L[0]

        gap = omega_L_val / max(omega_T1, 1e-10)

        print(f"{m_L:<10.2f} {omega_T1:<12.6f} {omega_T2:<12.6f} {omega_L_val:<12.6f} {gap:<15.4f}")

        results.append({
            'm_L': m_L,
            'omega_T1': omega_T1,
            'omega_T2': omega_T2,
            'omega_L': omega_L_val,
            'gap': gap,
        })

    # Check if gap increases with m_L
    gaps = [r['gap'] for r in results]
    gap_increases = all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))

    print(f"\nGap increases with m_L: {'YES' if gap_increases else 'NO'}")

    if gap_increases and gaps[-1] > 2.0:
        print("→ Bath successfully gaps longitudinal mode")
        verdict = "PASS"
    else:
        print("→ Bath effect unclear")
        verdict = "MARGINAL"

    return {
        'name': name,
        'results': results,
        'gap_increases': gap_increases,
        'verdict': verdict,
    }


def main():
    """Run Issue 4 tests."""
    print("=" * 70)
    print("ISSUE 4: k-SPACE BATH → P_L PROJECTION")
    print("=" * 70)

    print("""
GOAL: Show that local bath produces P_L via Schur complement

Physics:
  - Bath = scalar pressure field p on vertices
  - Bath energy = (1/2)(∇p)² = (1/2)k²|p̃|² in k-space
  - Coupling = p(∇·u) = p̃·(ik·ũ) in k-space
  - Schur complement = C†A⁻¹C = (ik)†(1/k²)(ik) = k⊗k/k² = P_L

This explains "exactly 2 modes":
  - 3 DOF (vector u) - 1 (longitudinal via P_L) = 2 transverse modes
    """)

    results = {}

    # Test 1: Continuum identity (trivially true by construction)
    test_continuum_identity()
    results['continuum'] = {'verdict': 'EXACT (trivial)'}

    # Build structures
    print("\nBuilding Kelvin N=2...")
    v_k, e_k, f_k, _ = build_kelvin_supercell_periodic(2)
    db_kelvin = DisplacementBloch(v_k, e_k, 8.0, k_L=3.0, k_T=1.0)

    print("Building FCC N=2...")
    v_f, e_f, f_f, _ = build_fcc_supercell_periodic(2)
    db_fcc = DisplacementBloch(v_f, e_f, 8.0, k_L=3.0, k_T=1.0)

    # Test 2: Mode classification (sanity check)
    results['kelvin_classify'] = run_mode_classification(db_kelvin, 8.0, "Kelvin N=2")
    results['fcc_classify'] = run_mode_classification(db_fcc, 8.0, "FCC N=2")

    # Test 3: Bath effect on spectrum (ansatz validation)
    results['kelvin_spectrum'] = run_bath_effect_on_spectrum(db_kelvin, 8.0, "Kelvin N=2")
    results['fcc_spectrum'] = run_bath_effect_on_spectrum(db_fcc, 8.0, "FCC N=2")

    # Test 4: DISCRETE SCHUR (THE KEY TEST)
    results['kelvin_schur'] = run_discrete_bath_schur(db_kelvin, 8.0, "Kelvin N=2")
    results['fcc_schur'] = run_discrete_bath_schur(db_fcc, 8.0, "FCC N=2")

    # Test 4b: UNIFORM BATH (single DOF projection) - WRONG TEST
    results['kelvin_uniform'] = run_uniform_bath_schur(db_kelvin, 8.0, "Kelvin N=2")
    results['fcc_uniform'] = run_uniform_bath_schur(db_fcc, 8.0, "FCC N=2")

    # Test 4′: PLANE-WAVE RAYLEIGH TEST (CORRECT TEST!)
    results['kelvin_planewave'] = run_planewave_schur(db_kelvin, 8.0, "Kelvin N=2")
    results['fcc_planewave'] = run_planewave_schur(db_fcc, 8.0, "FCC N=2")

    # VERIFICATIONS A-E (robustness checks for T4′)
    results['kelvin_verify'] = run_all_verifications(db_kelvin, 8.0, "Kelvin N=2")
    results['fcc_verify'] = run_all_verifications(db_fcc, 8.0, "FCC N=2")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Test':<45} {'Result':<15}")
    print("-" * 60)
    print(f"{'T1: Continuum Schur = P_L':<45} {'EXACT (trivial)':<15}")
    print(f"{'T2: Kelvin P_L classifies L mode':<45} {results['kelvin_classify']['verdict']:<15}")
    print(f"{'T2: FCC P_L classifies L mode':<45} {results['fcc_classify']['verdict']:<15}")
    print(f"{'T3: Kelvin m²·P_L gaps L':<45} {results['kelvin_spectrum']['verdict']:<15}")
    print(f"{'T3: FCC m²·P_L gaps L':<45} {results['fcc_spectrum']['verdict']:<15}")
    print(f"{'T4: Kelvin discrete S ≈ α·P_L (full matrix)':<45} {results['kelvin_schur']['verdict']:<15} (wrong test)")
    print(f"{'T4: FCC discrete S ≈ α·P_L (full matrix)':<45} {results['fcc_schur']['verdict']:<15} (wrong test)")
    print(f"{'T4′: Kelvin plane-wave Rayleigh':<45} {results['kelvin_planewave']['verdict']:<15} ← KEY TEST")
    print(f"{'T4′: FCC plane-wave Rayleigh':<45} {results['fcc_planewave']['verdict']:<15} ← KEY TEST")
    print(f"{'Verifications A-E: Kelvin':<45} {results['kelvin_verify']['overall']:<15}")
    print(f"{'Verifications A-E: FCC':<45} {results['fcc_verify']['overall']:<15}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Determine overall verdict based on Test 4′ AND verifications
    t4p_kelvin = results['kelvin_planewave']['verdict']
    t4p_fcc = results['fcc_planewave']['verdict']
    verify_kelvin = results['kelvin_verify']['overall']
    verify_fcc = results['fcc_verify']['overall']

    if (t4p_kelvin == "PASS" and t4p_fcc == "PASS" and
        verify_kelvin == "ALL PASS" and verify_fcc == "ALL PASS"):
        overall = "ISSUE 4 RESOLVED (ALL VERIFICATIONS PASS)"
        detail = "Local bath → P_L: All tests and verifications PASS!"
    elif t4p_kelvin == "PASS" and t4p_fcc == "PASS":
        overall = "ISSUE 4 RESOLVED"
        detail = "Local bath → P_L: S acts as longitudinal projector on plane-wave patterns!"
    elif t4p_kelvin in ["PASS", "MARGINAL"] and t4p_fcc in ["PASS", "MARGINAL"]:
        overall = "ISSUE 4 PARTIALLY RESOLVED"
        detail = "S has partial longitudinal selectivity on plane-wave patterns"
    else:
        overall = "ISSUE 4 NOT RESOLVED"
        detail = "S does NOT show longitudinal selectivity - may need B(k) with weights"

    print(f"""
{overall}

{detail}

TEST HIERARCHY:
  T1: Continuum identity Schur = P_L         [trivial, by construction]
  T2: P_L ansatz classifies L mode           [sanity check]
  T3: Adding m²P_L gaps L mode               [ansatz validation]
  T4: Discrete S = B†A⁻¹B ≈ α·P_L            [KEY TEST - local bath → P_L]

PHYSICS (if T4 passes):
  1. LOCAL BATH: scalar p on vertices with energy (1/2)p†A(k)p
     - A(k) = vertex Laplacian (local in real space)
  2. LOCAL COUPLING: B(k)u = discrete divergence
  3. SCHUR COMPLEMENT: S = B†A⁻¹B
     - A⁻¹ is nonlocal (Green's function) but comes from local A
  4. RESULT: S ≈ α·P_L → stiffens longitudinal mode

"EXACTLY 2 MODES" MECHANISM:
  - 3 DOF per point (vector u)
  - Bath stiffens longitudinal (1 mode becomes gapped/optical)
  - Left with 2 acoustic transverse modes
    """)

    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
