#!/usr/bin/env python3
"""
L-Mode Mechanisms: Constraint vs Stiffening
============================================

Two approaches to handle longitudinal modes in the foam model:

FIR C (CONSTRAINT): Bu = 0
    - Eliminates L completely via projection onto ker(B)
    - Result: exactly 2 DOF (Maxwell-like)
    - Implementation: SVD-based kernel projection

FIR B (STIFFENING): D_eff = D + g²B†A⁺B
    - Stiffens L via bath coupling (adiabatic limit)
    - Result: 3 DOF with v_L >> v_T (bulk-modulus-like soft constraint)
    - Implementation: Schur complement with pseudoinverse
    - Note: analogous in spirit to massive longitudinal response

KEY RELATIONSHIP:
    Fir C = infinite-penalty limit of Fir B
    As μ → ∞ in D + μS, eigenvalues converge to constrained EVP Q†DQ

TESTS:
    Fir C (Constraint):
        C1: rank(B) = V, dim(ker(B)) = 2V
        C2: Constrained modes satisfy Bu = 0
        C3: 2 acoustic branches remain (not 3)
        C4: Modes are purely transverse (k̂·u = 0)

    Fir B (Stiffening):
        B1: S = B†A⁺B is positive semi-definite
        B2: L-mode stiffened, T-modes preserved (canonical basis)
        B3: v_L/v_T > 5 across k values
        B4: Near-field > far-field (distance-suppressed)
        B5: Penalty → Constraint convergence (Fir B → Fir C limit)

Jan 2026
"""

import numpy as np
import pytest

from physics.bloch import DisplacementBloch
from physics.bath import (
    build_vertex_laplacian_bloch,
    build_divergence_operator_bloch,
)
from core_math_v2.builders.c15_periodic import build_c15_supercell_periodic
from core_math_v2.builders.multicell_periodic import build_bcc_supercell_periodic
from core_math_v2.builders.weaire_phelan_periodic import build_wp_supercell_periodic
from core_math_v2.builders.solids_periodic import build_fcc_supercell_periodic


# =============================================================================
# STRUCTURE CONFIGURATIONS
# =============================================================================

# Available foam/tiling structures for testing
STRUCTURES = {
    'C15': {
        'builder': lambda: build_c15_supercell_periodic(N=1, L_cell=4.0),
        'L_cell': 4.0,
        'description': 'C15 Laves foam (136 vertices)',
    },
    'Kelvin': {
        'builder': lambda: build_bcc_supercell_periodic(N=1),
        'L_cell': 4.0,  # L = 4*N = 4
        'description': 'BCC Kelvin foam (48 vertices)',
    },
    'WP': {
        'builder': lambda: build_wp_supercell_periodic(N=1, L_cell=4.0),
        'L_cell': 4.0,
        'description': 'Weaire-Phelan foam (8 cells)',
    },
    'FCC': {
        'builder': lambda: build_fcc_supercell_periodic(N=1),
        'L_cell': 4.0,  # L = 4*N = 4
        'description': 'FCC rhombic dodecahedra tiling (k=3)',
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_foam(structure_name: str, k_mag: float = 0.1):
    """
    Build foam/tiling structure and operators at given k magnitude.

    Args:
        structure_name: One of 'C15', 'Kelvin', 'WP', 'FCC'
        k_mag: Magnitude of k-vector

    Returns:
        db, D, A, B, V, k, L_cell
    """
    config = STRUCTURES[structure_name]
    result = config['builder']()
    vertices, edges = result[0], result[1]
    L_cell = config['L_cell']

    db = DisplacementBloch(vertices=vertices, edges=edges, L=L_cell)
    V = db.V

    k = np.array([k_mag, 0, 0])

    D = db.build_dynamical_matrix(k)
    B = build_divergence_operator_bloch(db, k)
    A = build_vertex_laplacian_bloch(db, k)

    return db, D, A, B, V, k, L_cell


def setup_c15_foam(k_mag=0.1):
    """Build C15 foam and operators at given k magnitude (legacy wrapper)."""
    return setup_foam('C15', k_mag)


def get_kernel_basis(B: np.ndarray, tol: float = 1e-12) -> tuple:
    """
    Get orthonormal basis Q for ker(B) via SVD.

    Returns:
        Q: (3V, dim_ker) orthonormal basis for ker(B)
        rank_B: numerical rank of B
    """
    U, sigma, Vh = np.linalg.svd(B, full_matrices=True)
    threshold = tol * sigma[0] if len(sigma) > 0 and sigma[0] > 0 else tol
    rank_B = np.sum(sigma > threshold)
    Q = Vh[rank_B:, :].conj().T
    return Q, rank_B


def compute_pseudoinverse(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute pseudoinverse of A (assumed PSD), removing near-zero eigenvalues.

    Uses eigvals > threshold (not abs) to avoid sign issues with small negative
    eigenvalues that can appear due to numeric noise in PSD matrices.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    # Relative threshold to handle varying eigenvalue scales
    threshold = tol * np.max(np.abs(eigvals)) if len(eigvals) > 0 else tol
    # Use > threshold (not abs) to exclude small negative eigenvalues from PSD noise
    mask = eigvals > threshold
    return eigvecs[:, mask] @ np.diag(1.0/eigvals[mask]) @ eigvecs[:, mask].conj().T


def get_kernel_basis_symmetric(M: np.ndarray, dim_ker: int = None, tol: float = 1e-12) -> np.ndarray:
    """
    Get orthonormal basis for ker(M) where M is symmetric PSD.

    Args:
        M: Symmetric PSD matrix
        dim_ker: If known, take exactly this many smallest-eigenvalue vectors (robust)
        tol: If dim_ker not given, use threshold tol*max(|λ|)

    Returns:
        Q: Orthonormal basis for kernel
    """
    eigvals, eigvecs = np.linalg.eigh(M)

    if dim_ker is not None:
        # Robust: take exactly dim_ker smallest eigenvalues (PSD: sort by value, not abs)
        idx = np.argsort(eigvals)
        return eigvecs[:, idx[:dim_ker]]
    else:
        # Threshold-based fallback
        threshold = tol * np.max(np.abs(eigvals)) if len(eigvals) > 0 else tol
        mask = np.abs(eigvals) < threshold
        return eigvecs[:, mask]


def subspace_distance(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """
    Compute distance between subspaces spanned by Q1 and Q2.

    Returns ||P1 - P2||_F where Pi = Qi @ Qi† is the projector.
    Result is 0 if subspaces are identical.
    """
    P1 = Q1 @ Q1.conj().T
    P2 = Q2 @ Q2.conj().T
    return np.linalg.norm(P1 - P2, 'fro')


# =============================================================================
# FIR C TESTS: CONSTRAINT Bu = 0
# =============================================================================

class TestFirCConstraint:
    """Tests for constraint-based L-mode elimination (Bu = 0)."""

    def test_C1_kernel_dimension(self):
        """
        T-L01: rank(B) = V, dim(ker(B)) = 2V.

        The divergence operator B: ℝ³ⱽ → ℝⱽ should have rank V,
        leaving a 2V-dimensional kernel (the transverse subspace).
        """
        db, D, A, B, V, k, L_cell = setup_c15_foam(k_mag=0.1)

        Q, rank_B = get_kernel_basis(B)
        dim_ker = Q.shape[1]

        assert rank_B == V, f"Expected rank(B) = {V}, got {rank_B}"
        assert dim_ker == 2 * V, f"Expected dim(ker(B)) = {2*V}, got {dim_ker}"

    def test_C2_constraint_satisfied(self):
        """
        T-L02: Constrained modes satisfy ||Bu|| < 1e-12.

        Modes reconstructed from ker(B) should exactly satisfy Bu = 0.
        """
        db, D, A, B, V, k, L_cell = setup_c15_foam(k_mag=0.1)

        Q, rank_B = get_kernel_basis(B)

        # Solve reduced EVP
        D_reduced = Q.conj().T @ D @ Q
        eigvals, eigvecs_reduced = np.linalg.eigh(D_reduced)

        # Reconstruct full modes: u = Q @ y
        modes_full = Q @ eigvecs_reduced

        # Check ||Bu|| for all modes
        max_Bu_norm = 0.0
        for i in range(modes_full.shape[1]):
            u = modes_full[:, i]
            Bu_norm = np.linalg.norm(B @ u) / np.linalg.norm(u)
            max_Bu_norm = max(max_Bu_norm, Bu_norm)

        assert max_Bu_norm < 1e-12, f"Max ||Bu||/||u|| = {max_Bu_norm}, expected < 1e-12"

    def test_C3_two_acoustic_branches(self):
        """
        T-L03: Constraint leaves 2 acoustic branches, not 3.

        Compare unconstrained (3 acoustic) vs constrained (2 branches).
        Identify T/L by ||Bu||/||u|| criterion (not eigenvalue order).
        """
        db, D, A, B, V, k, L_cell = setup_c15_foam(k_mag=0.05)
        k_mag = 0.05

        # Unconstrained: get 3 acoustic modes and identify T by ||Bu||
        eigvals_D, eigvecs_D = np.linalg.eigh(D)
        idx = np.argsort(eigvals_D)

        # Compute ||Bu||/||u|| for 3 lowest modes
        Bu_norms = []
        for i in range(3):
            u = eigvecs_D[:, idx[i]]
            Bu_norm = np.linalg.norm(B @ u) / np.linalg.norm(u)
            omega = np.sqrt(max(eigvals_D[idx[i]], 0))
            Bu_norms.append((i, omega, Bu_norm))

        # T modes = 2 with smallest ||Bu||, L mode = 1 with largest ||Bu||
        Bu_norms.sort(key=lambda x: x[2])
        T_modes_unconstrained = Bu_norms[:2]
        v_T_unconstrained = [m[1] / k_mag for m in T_modes_unconstrained]

        # Constrained: should have 2 acoustic modes
        Q, rank_B = get_kernel_basis(B)
        D_reduced = Q.conj().T @ D @ Q
        eigvals_constrained = np.linalg.eigvalsh(D_reduced)
        eigvals_constrained = np.sort(eigvals_constrained)
        omega_constrained = np.sqrt(np.maximum(eigvals_constrained[:2], 0))
        v_constrained = omega_constrained / k_mag

        # Constrained should have 2 velocities
        assert len(v_constrained) == 2
        # Constrained velocities should match T-mode velocities (identified by ||Bu||)
        assert np.allclose(sorted(v_constrained), sorted(v_T_unconstrained), rtol=0.1)

    def test_C4_modes_transverse(self):
        """
        T-L04: Constrained modes are approximately transverse (k̂·u ≈ 0).

        Continuum proxy: in the continuum limit, Bu=0 implies k·u=0 (transverse).
        On discrete lattice, this is approximate. Threshold 0.05 allows lattice effects.

        NOTE: The discrete criterion ||Bu||/||u|| (tested in C2) is the rigorous one;
        this k̂·u test is a continuum-limit sanity check.
        """
        db, D, A, B, V, k, L_cell = setup_c15_foam(k_mag=0.1)
        k_hat = k / np.linalg.norm(k)

        Q, rank_B = get_kernel_basis(B)
        D_reduced = Q.conj().T @ D @ Q
        eigvals, eigvecs_reduced = np.linalg.eigh(D_reduced)

        # Get lowest modes (acoustic)
        idx = np.argsort(eigvals)
        modes_full = Q @ eigvecs_reduced[:, idx[:2]]

        # Check k̂·u content for each mode (continuum proxy, RMS to avoid outlier sensitivity)
        for i in range(2):
            u = modes_full[:, i]
            # Reshape to (V, 3) and compute RMS of k̂·u
            u_3d = u.reshape(-1, 3)
            k_dot_u = np.abs(u_3d @ k_hat)
            rms_k_dot_u = np.sqrt(np.mean(k_dot_u**2)) / np.linalg.norm(u)

            assert rms_k_dot_u < 0.05, f"Mode {i}: RMS |k̂·u|/||u|| = {rms_k_dot_u}, expected < 0.05"


# =============================================================================
# FIR B TESTS: STIFFENING D_eff = D + g²S
# =============================================================================

class TestFirBStiffening:
    """Tests for stiffening-based L-mode suppression (adiabatic bath)."""

    def test_B1_schur_positive_semidefinite(self):
        """
        T-L05: S = B†A⁺B is positive semi-definite.

        This ensures D_eff = D + g²S is stable (no negative eigenvalues from S).
        """
        db, D, A, B, V, k, L_cell = setup_c15_foam(k_mag=0.01)

        A_pinv = compute_pseudoinverse(A)
        S = B.conj().T @ A_pinv @ B

        eigvals_S = np.linalg.eigvalsh(S)
        min_eigval = np.min(eigvals_S)

        # Allow small numerical noise
        assert min_eigval > -1e-10, f"S has negative eigenvalue: {min_eigval}"

    def test_B2_stiffening_canonical_basis(self):
        """
        T-L06: L-mode stiffened, T-modes preserved (canonical basis method).

        Use canonical basis from diagonalizing S in acoustic subspace to handle
        degeneracy. L-mode (high S eigenvalue) should be stiffened, T-modes unchanged.
        """
        db, D, A, B, V, k, L_cell = setup_c15_foam(k_mag=0.1)

        A_pinv = compute_pseudoinverse(A)
        S = B.conj().T @ A_pinv @ B

        g = 1.0
        D_eff = D + g**2 * S

        # Get 3 acoustic eigenvectors of D (degenerate subspace)
        eigvals_D, eigvecs_D = np.linalg.eigh(D)
        idx = np.argsort(eigvals_D)
        U = eigvecs_D[:, idx[:3]]  # 3V × 3 matrix

        # Project S into acoustic subspace: S_3 = U† S U (3×3)
        S_3 = U.conj().T @ S @ U

        # Diagonalize S_3 to get canonical basis
        eigvals_S3, eigvecs_S3 = np.linalg.eigh(S_3)
        idx_S = np.argsort(eigvals_S3)

        # Canonical modes in full space
        canonical_modes = U @ eigvecs_S3[:, idx_S]

        # Compute ratios using Rayleigh quotient
        # After sorting by S eigenvalue: i=0,1 have small S (T-like), i=2 has largest S (L-like)
        T_ratios = []
        L_ratios = []
        for i in range(3):
            mode = canonical_modes[:, i]
            norm_sq = np.real(mode.conj() @ mode)
            omega2_D = np.real(mode.conj() @ D @ mode) / norm_sq
            omega2_Deff = np.real(mode.conj() @ D_eff @ mode) / norm_sq
            ratio = np.sqrt(omega2_Deff / omega2_D) if omega2_D > 1e-20 else np.inf

            # L = mode with LARGEST S eigenvalue (i=2 after sort)
            # T = modes with smaller S eigenvalues (i=0,1)
            if i < 2:
                T_ratios.append(ratio)
            else:  # i == 2: largest S eigenvalue → L mode
                L_ratios.append(ratio)

        # T modes should be unchanged (ratio ≈ 1)
        assert all(r < 1.1 for r in T_ratios), f"T-mode ratios {T_ratios} not < 1.1"
        # L mode should be stiffened (ratio > 2)
        assert all(r > 2.0 for r in L_ratios), f"L-mode ratios {L_ratios} not > 2.0"

    def test_B3_velocity_ratio(self):
        """
        T-L07: v_L/v_T > 5 across multiple k values.

        The stiffening should produce a consistent velocity ratio.
        Identify T/L by ||Bu||/||u|| criterion in lowest N_search modes.

        NOTE: We search in N_search modes because strong stiffening
        can push L outside the lowest 3. Guard ensures L stays acoustic.
        """
        L_cell = 4.0
        result = build_c15_supercell_periodic(N=1, L_cell=L_cell)
        vertices, edges = result[0], result[1]
        db = DisplacementBloch(vertices=vertices, edges=edges, L=L_cell)

        g = 1.0
        k_vals = [0.05, 0.1, 0.15, 0.2]
        N_search = 10  # Search window (enough for stiffened L, avoids optical)
        v_max_acoustic = 50.0  # Guard: v > 50 likely optical, not stiffened L
        velocity_ratios = []

        for k_mag in k_vals:
            k = np.array([k_mag, 0, 0])

            D = db.build_dynamical_matrix(k)
            A = build_vertex_laplacian_bloch(db, k)
            B = build_divergence_operator_bloch(db, k)

            A_pinv = compute_pseudoinverse(A)
            S = B.conj().T @ A_pinv @ B
            D_eff = D + g**2 * S

            eigvals_Deff, eigvecs_Deff = np.linalg.eigh(D_eff)
            idx = np.argsort(eigvals_Deff)

            # Identify T/L by ||Bu|| in lowest N_search modes
            Bu_norms = []
            for i in range(min(N_search, len(eigvals_Deff))):
                mode = eigvecs_Deff[:, idx[i]]
                Bu_norm = np.linalg.norm(B @ mode) / np.linalg.norm(mode)
                omega = np.sqrt(max(eigvals_Deff[idx[i]], 0))
                v = omega / k_mag if k_mag > 0 else 0
                # Only consider modes in acoustic range
                if v < v_max_acoustic:
                    Bu_norms.append((i, eigvals_Deff[idx[i]], Bu_norm))

            # Guard: ensure we found acoustic candidates
            assert Bu_norms, f"No acoustic candidates found at k={k_mag} (v_max={v_max_acoustic})"

            # T = min ||Bu|| (most transverse), L = max ||Bu|| (most longitudinal)
            T_mode = min(Bu_norms, key=lambda x: x[2])
            L_mode = max(Bu_norms, key=lambda x: x[2])

            omega_T = np.sqrt(max(T_mode[1], 0))
            omega_L = np.sqrt(max(L_mode[1], 0))
            v_T = omega_T / k_mag
            v_L = omega_L / k_mag
            ratio = v_L / v_T if v_T > 1e-10 else np.inf
            velocity_ratios.append(ratio)

        min_ratio = min(velocity_ratios)
        assert min_ratio > 5.0, f"Min v_L/v_T = {min_ratio}, expected > 5.0"

    def test_B4_near_field_decay(self):
        """
        T-L08: A⁺ has stronger coupling at short distances than long (finite-size proxy).

        On a finite periodic cell, the pseudoinverse A⁺ is not uniform in space.
        This test verifies that short-distance entries dominate long-distance ones,
        consistent with a distance-decaying (not instantaneous) interaction.

        NOTE: This is a sanity check on the finite-size sample, not a proof of
        Coulomb-like decay. The continuum limit would require larger cells.
        """
        L_cell = 4.0
        result = build_c15_supercell_periodic(N=1, L_cell=L_cell)
        vertices, edges = result[0], result[1]
        db = DisplacementBloch(vertices=vertices, edges=edges, L=L_cell)
        V = db.V

        k = np.array([0.0, 0, 0])
        A = build_vertex_laplacian_bloch(db, k)

        A_pinv = compute_pseudoinverse(A)

        # Compute distances and A⁻¹ values from vertex 0
        # Use L_cell-scaled thresholds for portability across cell sizes
        ref = 0
        pos_ref = vertices[ref]

        near_threshold = 0.25 * L_cell  # near: dist < 0.25*L
        far_threshold = 0.5 * L_cell    # far: dist > 0.5*L

        near_vals = []
        far_vals = []

        for j in range(V):
            if j == ref:
                continue
            delta = vertices[j] - pos_ref
            delta = delta - L_cell * np.round(delta / L_cell)
            dist = np.linalg.norm(delta)
            val = np.abs(A_pinv[ref, j])

            if dist < near_threshold:
                near_vals.append(val)
            elif dist > far_threshold:
                far_vals.append(val)

        mean_near = np.mean(near_vals) if near_vals else 0
        mean_far = np.mean(far_vals) if far_vals else 1

        ratio = mean_near / mean_far if mean_far > 0 else np.inf

        # Modest threshold: near > far by at least 50%
        assert ratio > 1.5, f"Near/far ratio = {ratio}, expected > 1.5"

    def test_B5_subspace_equivalence(self):
        """
        T-L09: ker(B) = ker(S) as subspaces (Fir C ↔ Fir B connection).

        The constraint Bu=0 and the penalty S = B†A⁺B should annihilate
        the same subspace. This is the mathematical foundation for
        Fir C being the hard-constraint limit of Fir B.

        Method: Verify mutual inclusion:
          (a) ker(B) ⊆ ker(S): ||S @ Q_B|| small
          (b) ker(S) ⊆ ker(B): ||B @ Q_S|| small
        """
        db, D, A, B, V, k, L_cell = setup_c15_foam(k_mag=0.1)

        A_pinv = compute_pseudoinverse(A)
        S = B.conj().T @ A_pinv @ B

        # Get kernel of B via SVD (dimension = 2V)
        Q_B, rank_B = get_kernel_basis(B)
        dim_ker_B = Q_B.shape[1]

        # Get low-eigenvalue subspace of S (dim = 2V)
        # Using known dimension is robust even if S spectrum has no clean gap
        Q_S = get_kernel_basis_symmetric(S, dim_ker=dim_ker_B)

        # (a) Verify ker(B) ⊆ ker(S): S @ Q_B should be small
        S_Q_B = S @ Q_B
        norm_S_Q_B = np.linalg.norm(S_Q_B, 'fro') / np.linalg.norm(Q_B, 'fro')
        assert norm_S_Q_B < 1e-7, f"||S @ Q_B||/||Q_B|| = {norm_S_Q_B}, expected < 1e-7"

        # (b) Verify ker(S) ⊆ ker(B): B @ Q_S should be small
        B_Q_S = B @ Q_S
        norm_B_Q_S = np.linalg.norm(B_Q_S, 'fro') / np.linalg.norm(Q_S, 'fro')
        assert norm_B_Q_S < 1e-7, f"||B @ Q_S||/||Q_S|| = {norm_B_Q_S}, expected < 1e-7"

        # Additionally verify that reduced dynamics match
        D_B = Q_B.conj().T @ D @ Q_B
        D_S = Q_S.conj().T @ D @ Q_S

        eigvals_D_B = np.sort(np.linalg.eigvalsh(D_B))
        eigvals_D_S = np.sort(np.linalg.eigvalsh(D_S))

        # First few eigenvalues should match (1e-7 for cross-BLAS stability)
        rel_diff = np.max(np.abs(eigvals_D_B[:10] - eigvals_D_S[:10]) / (np.abs(eigvals_D_B[:10]) + 1e-20))
        assert rel_diff < 1e-7, f"Eigenvalue mismatch: max rel diff = {rel_diff}"


# =============================================================================
# COMBINED TEST: FIR B ↔ FIR C RELATIONSHIP
# =============================================================================

class TestFirBCRelationship:
    """Tests verifying the relationship between Fir B and Fir C."""

    def test_penalty_subspace_convergence(self):
        """
        T-L10: Low-energy subspace of D+μS converges to ker(B) as μ → ∞.

        As penalty μ increases, modes with Su ≠ 0 get pushed to high energy.
        The low-energy subspace approaches ker(B) (the Fir C constraint space).

        This equivalence holds because ker(S)=ker(B) when A⁺ is the pseudoinverse
        on the bath subspace.

        Method: Compare projector onto lowest 2V eigenvectors of D+μS
        with Q_B (ground truth for Fir C). Distance should decrease with μ.
        """
        db, D, A, B, V, k, L_cell = setup_c15_foam(k_mag=0.1)

        A_pinv = compute_pseudoinverse(A)
        S = B.conj().T @ A_pinv @ B

        # Target subspace: ker(B) directly (ground truth, no estimation needed)
        Q_B, _ = get_kernel_basis(B)
        dim_target = Q_B.shape[1]  # Should be 2V

        mu_values = [1e1, 1e2, 1e3, 1e4, 1e5]
        distances = []

        for mu in mu_values:
            D_mu = D + mu * S
            eigvals, eigvecs = np.linalg.eigh(D_mu)
            idx = np.argsort(eigvals)

            # Take lowest dim_target eigenvectors
            Q_mu = eigvecs[:, idx[:dim_target]]

            # Compute subspace distance to Q_B (ground truth)
            dist = subspace_distance(Q_mu, Q_B)
            normalized_dist = dist / np.sqrt(dim_target)
            distances.append(normalized_dist)

        # Check overall convergence (not strict monotonic - allows small numeric noise)
        # 1. Final distance should be small (0.02 allows variation across setups)
        assert distances[-1] < 0.02, f"Final subspace distance = {distances[-1]}, expected < 0.02"

        # 2. Significant improvement from first to last (main criterion)
        improvement = distances[0] / distances[-1] if distances[-1] > 0 else np.inf
        assert improvement > 5, f"Improvement ratio = {improvement}, expected > 5"

        # 3. Mostly decreasing (allow at most 1 violation due to numeric noise)
        violations = sum(1 for i in range(len(distances)-1) if distances[i+1] >= distances[i])
        assert violations <= 1, f"Too many non-decreasing steps: {violations}, distances={distances}"


# =============================================================================
# MULTI-STRUCTURE TESTS
# =============================================================================

# Foam structures (Plateau: 3 faces/edge, tetravalent vertices)
FOAM_LIST = ['C15', 'Kelvin', 'WP']

# All structures including tilings
ALL_STRUCTURES = ['C15', 'Kelvin', 'WP', 'FCC']


class TestMultiStructure:
    """
    Tests run across multiple foam/tiling structures.

    Validates that L-mode mechanisms work generically, not just on C15.
    """

    @pytest.mark.parametrize("structure", FOAM_LIST)
    def test_kernel_dimension(self, structure):
        """
        T-L11: rank(B) = V, dim(ker(B)) = 2V across FOAM structures.

        The divergence operator B: ℝ³ⱽ → ℝⱽ should have rank V
        on Plateau foam structures. FCC tiling excluded (different topology).
        """
        db, D, A, B, V, k, L_cell = setup_foam(structure, k_mag=0.1)

        Q, rank_B = get_kernel_basis(B)
        dim_ker = Q.shape[1]

        assert rank_B == V, f"{structure}: Expected rank(B) = {V}, got {rank_B}"
        assert dim_ker == 2 * V, f"{structure}: Expected dim(ker(B)) = {2*V}, got {dim_ker}"

    @pytest.mark.parametrize("structure", ALL_STRUCTURES)
    def test_constraint_satisfied(self, structure):
        """
        T-L12: Constrained modes satisfy ||Bu|| < 1e-12 across structures.
        """
        db, D, A, B, V, k, L_cell = setup_foam(structure, k_mag=0.1)

        Q, rank_B = get_kernel_basis(B)

        # Solve reduced EVP
        D_reduced = Q.conj().T @ D @ Q
        eigvals, eigvecs_reduced = np.linalg.eigh(D_reduced)

        # Reconstruct full modes: u = Q @ y
        modes_full = Q @ eigvecs_reduced

        # Check ||Bu|| for all modes
        max_Bu_norm = 0.0
        for i in range(modes_full.shape[1]):
            u = modes_full[:, i]
            Bu_norm = np.linalg.norm(B @ u) / np.linalg.norm(u)
            max_Bu_norm = max(max_Bu_norm, Bu_norm)

        assert max_Bu_norm < 1e-12, f"{structure}: Max ||Bu||/||u|| = {max_Bu_norm}, expected < 1e-12"

    @pytest.mark.parametrize("structure", ALL_STRUCTURES)
    def test_schur_positive_semidefinite(self, structure):
        """
        T-L13: S = B†A⁺B is positive semi-definite across structures.
        """
        db, D, A, B, V, k, L_cell = setup_foam(structure, k_mag=0.01)

        A_pinv = compute_pseudoinverse(A)
        S = B.conj().T @ A_pinv @ B

        eigvals_S = np.linalg.eigvalsh(S)
        min_eigval = np.min(eigvals_S)

        # Allow small numerical noise
        assert min_eigval > -1e-10, f"{structure}: S has negative eigenvalue: {min_eigval}"

    @pytest.mark.parametrize("structure", ALL_STRUCTURES)
    def test_subspace_equivalence(self, structure):
        """
        T-L14: ker(B) = ker(S) via mutual inclusion across structures.
        """
        db, D, A, B, V, k, L_cell = setup_foam(structure, k_mag=0.1)

        A_pinv = compute_pseudoinverse(A)
        S = B.conj().T @ A_pinv @ B

        # Get kernel of B via SVD
        Q_B, rank_B = get_kernel_basis(B)
        dim_ker_B = Q_B.shape[1]

        # Get low-eigenvalue subspace of S
        Q_S = get_kernel_basis_symmetric(S, dim_ker=dim_ker_B)

        # (a) Verify ker(B) ⊆ ker(S)
        S_Q_B = S @ Q_B
        norm_S_Q_B = np.linalg.norm(S_Q_B, 'fro') / np.linalg.norm(Q_B, 'fro')
        assert norm_S_Q_B < 1e-7, f"{structure}: ||S @ Q_B||/||Q_B|| = {norm_S_Q_B}"

        # (b) Verify ker(S) ⊆ ker(B)
        B_Q_S = B @ Q_S
        norm_B_Q_S = np.linalg.norm(B_Q_S, 'fro') / np.linalg.norm(Q_S, 'fro')
        assert norm_B_Q_S < 1e-7, f"{structure}: ||B @ Q_S||/||Q_S|| = {norm_B_Q_S}"


# =============================================================================
# MAIN (for standalone execution)
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
