"""
k-Space Bath and Schur Complement
=================================

Implements the bath mechanism that produces longitudinal mode suppression
via Schur complement in k-space.

Theory:
  Bath DOF: scalar field p(x) on vertices with inertia
  Lagrangian: L = (1/2)χ|ṗ|² - (1/2)p†Ap + g·p†Bu

  Static elimination (WRONG):
    Minimizing E over p gives D_eff = D - g²S where S = B†A⁻¹B
    This has NEGATIVE sign → instability, not gapping!

  Dynamic elimination (CORRECT):
    In frequency domain: D_eff(ω) = D - g²B†(χω² - A)⁻¹B
    For ω < ω_crit = √(λ_min(A)/χ):
      - (χω² - A) is negative definite
      - The correction becomes POSITIVE → stable, L gapped!

Physical interpretation:
  - D_eff(ω) = D + Σ(ω) is standard self-energy physics
  - Low ω: adiabatic limit, bath stiffens L mode
  - High ω: resonance with bath modes
  - Adding damping γ: L becomes overdamped/evanescent (near-field only)

Result: "exactly 2 modes"
  - 3 DOF per point (vector u)
  - Bath suppresses longitudinal (gapped or overdamped)
  - Left with 2 acoustic transverse modes

Jan 2026
"""

import numpy as np

from .bloch import DisplacementBloch
from .constants import (
    ZERO_K_THRESHOLD,
    PSEUDOINVERSE_CUTOFF,
)


def continuum_P_L(k: np.ndarray) -> np.ndarray:
    """
    Continuum longitudinal projector P_L = k̂⊗k̂.

    Args:
        k: (3,) wave vector

    Returns:
        P_L: (3, 3) matrix
    """
    k_mag = np.linalg.norm(k)
    if k_mag < ZERO_K_THRESHOLD:
        return np.zeros((3, 3))
    k_hat = k / k_mag
    return np.outer(k_hat, k_hat)


def bath_schur_continuum(k: np.ndarray) -> np.ndarray:
    """
    Bath Schur complement in k-space (continuum theory).

    Bath energy: (1/2)k²|p|²
    Coupling: p·(ik·u)

    Schur = C† A⁻¹ C = (ik)† (1/k²) (ik) = k⊗k / k²

    Args:
        k: (3,) wave vector

    Returns:
        Schur: (3, 3) matrix
    """
    k_mag = np.linalg.norm(k)
    if k_mag < ZERO_K_THRESHOLD:
        return np.zeros((3, 3))

    # C = ik (coupling to divergence)
    # A = k² (bath stiffness from Laplacian)
    # Schur = C† A⁻¹ C = (ik)†(1/k²)(ik) = k⊗k/k²
    return np.outer(k, k) / k_mag**2


# ---------------------------------------------------------------------
# DISCRETE BATH OPERATORS
# ---------------------------------------------------------------------

def build_vertex_laplacian_bloch(db: DisplacementBloch, k: np.ndarray) -> np.ndarray:
    """
    Build Bloch-twisted graph Laplacian on vertices.

    A(k) = D - W(k)
    where D is degree matrix, W(k) is weighted adjacency with Bloch phases.

    For edge (i,j) with crossing n_ij:
        W(k)[i,j] = e^{ik·(n_ij·L)}
        W(k)[j,i] = e^{-ik·(n_ij·L)}

    A(k)[i,i] = degree(i)
    A(k)[i,j] = -W(k)[i,j]  for neighbors

    This is the discrete Laplacian: in k→0 limit, A(k) ~ k² (up to scaling).

    Args:
        db: DisplacementBloch instance (has vertices, edges, crossings, L)
        k: (3,) wave vector

    Returns:
        A: (V, V) complex Hermitian matrix
    """
    V = db.V
    L = db.L
    A = np.zeros((V, V), dtype=complex)

    # Build adjacency and degree
    for e_idx, (i, j) in enumerate(db.edges):
        n = db.crossings[e_idx]
        phase = np.exp(1j * np.dot(k, n * L))

        # Off-diagonal: -phase
        A[i, j] -= phase
        A[j, i] -= np.conj(phase)

        # Diagonal: +1 for each connection
        A[i, i] += 1.0
        A[j, j] += 1.0

    return A


def build_divergence_operator_bloch(db: DisplacementBloch, k: np.ndarray) -> np.ndarray:
    """
    Build Bloch-twisted divergence operator B(k): 3V → V.

    Discrete divergence at vertex i:
        (Bu)_i = Σ_{j~i} ê_ij · (u_j - u_i) · phase_ij

    where ê_ij is the unit vector along edge i→j.

    In continuum limit: B ~ ik·u (divergence).

    Args:
        db: DisplacementBloch instance
        k: (3,) wave vector

    Returns:
        B: (V, 3V) complex matrix
    """
    V = db.V
    L = db.L
    B = np.zeros((V, 3*V), dtype=complex)

    for e_idx, (i, j) in enumerate(db.edges):
        e_hat = db.edge_vectors[e_idx]  # unit vector along edge
        n = db.crossings[e_idx]
        phase = np.exp(1j * np.dot(k, n * L))

        # Contribution to divergence at vertex i from edge (i,j):
        # (Bu)_i += ê · (u_j·phase - u_i)
        for a in range(3):
            # u_j contribution (with phase)
            B[i, 3*j + a] += e_hat[a] * phase
            # u_i contribution (negative)
            B[i, 3*i + a] -= e_hat[a]

        # Contribution to divergence at vertex j from edge (j,i):
        # (Bu)_j += (-ê) · (u_i·conj(phase) - u_j)
        for a in range(3):
            # u_i contribution (with conj phase, negative edge direction)
            B[j, 3*i + a] += (-e_hat[a]) * np.conj(phase)
            # u_j contribution (negative of negative = positive edge dir)
            B[j, 3*j + a] -= (-e_hat[a])

    return B


def compute_discrete_schur(A: np.ndarray, B: np.ndarray,
                           cutoff: float = PSEUDOINVERSE_CUTOFF) -> np.ndarray:
    """
    Compute Schur complement S = B† A⁻¹ B.

    Uses thresholded pseudoinverse for numerical stability
    (handles near-zero eigenvalues at k→0).

    NOTE: This is a HARD CUTOFF pseudoinverse, not Tikhonov/ridge regularization.
    Eigenvalues with |λ| < cutoff are treated as zero (1/λ → 0).
    Ridge would use 1/(λ+α) for all λ, which behaves differently.

    Args:
        A: (V, V) bath stiffness matrix (Hermitian)
        B: (V, 3V) coupling matrix
        cutoff: eigenvalue threshold - values below this are zeroed

    Returns:
        S: (3V, 3V) Schur complement matrix
    """
    # Use eigendecomposition for stable pseudoinverse
    eigvals, eigvecs = np.linalg.eigh(A)

    # Thresholded pseudoinverse: zero out near-zero eigenvalues
    inv_eigvals = np.where(np.abs(eigvals) > cutoff,
                           1.0 / eigvals,
                           0.0)

    # A_pinv = V @ diag(1/λ) @ V†
    A_pinv = eigvecs @ np.diag(inv_eigvals) @ eigvecs.conj().T

    # Schur = B† A⁻¹ B
    S = B.conj().T @ A_pinv @ B

    return S


def build_PL_full(db: DisplacementBloch, k: np.ndarray) -> np.ndarray:
    """
    Build full P_L projector in 3V space (block diagonal k̂⊗k̂).

    Args:
        db: DisplacementBloch instance
        k: (3,) wave vector

    Returns:
        P_L: (3V, 3V) real symmetric matrix
    """
    return db.build_longitudinal_projector(k)


# ---------------------------------------------------------------------
# DYNAMIC BATH (frequency-dependent)
# ---------------------------------------------------------------------

def compute_dynamic_kernel(A: np.ndarray, omega: float, chi: float,
                           gamma: float = 0.0,
                           cutoff: float = PSEUDOINVERSE_CUTOFF) -> np.ndarray:
    """
    Compute dynamic bath kernel (χω² - A + iγω)⁻¹.

    For dynamic bath with inertia χ and optional damping γ:
      L = (1/2)χ|ṗ|² - (1/2)p†Ap + g·p†Bu

    In frequency domain, eliminating p gives kernel (χω² - A)⁻¹.
    With damping: (χω² - A + iγω)⁻¹.

    Key physics:
      - For ω < ω_crit = √(λ_min(A)/χ): kernel is negative definite
      - This gives POSITIVE correction to D_eff (sign flip!)
      - Standard coupled oscillator / self-energy physics

    Args:
        A: (V, V) bath stiffness matrix (Hermitian, positive semi-definite)
        omega: frequency
        chi: bath inertia (mass-like parameter)
        gamma: damping coefficient (default 0 = no damping)
        cutoff: eigenvalue threshold for pseudoinverse

    Returns:
        kernel_inv: (V, V) complex matrix (χω² - A + iγω)⁻¹
    """
    V = A.shape[0]

    # Build kernel matrix: χω² - A + iγω
    kernel = chi * omega**2 * np.eye(V) - A
    if gamma > 0:
        kernel = kernel + 1j * gamma * omega * np.eye(V)

    # Compute pseudoinverse via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(A)

    # Kernel eigenvalues: χω² - λ_A + iγω
    kernel_eigvals = chi * omega**2 - eigvals
    if gamma > 0:
        kernel_eigvals = kernel_eigvals + 1j * gamma * omega

    # Invert with cutoff
    inv_eigvals = np.where(np.abs(kernel_eigvals) > cutoff,
                           1.0 / kernel_eigvals,
                           0.0)

    # Reconstruct inverse
    kernel_inv = eigvecs @ np.diag(inv_eigvals) @ eigvecs.conj().T

    return kernel_inv


def compute_dynamic_schur(A: np.ndarray, B: np.ndarray,
                          omega: float, chi: float,
                          gamma: float = 0.0,
                          cutoff: float = PSEUDOINVERSE_CUTOFF) -> np.ndarray:
    """
    Compute dynamic Schur complement S(ω) = B†(χω² - A + iγω)⁻¹B.

    This is the frequency-dependent self-energy from integrating out
    the bath DOF with inertia.

    Args:
        A: (V, V) bath stiffness matrix
        B: (V, 3V) coupling matrix
        omega: frequency
        chi: bath inertia
        gamma: damping coefficient (default 0)
        cutoff: eigenvalue threshold

    Returns:
        S: (3V, 3V) complex matrix
    """
    kernel_inv = compute_dynamic_kernel(A, omega, chi, gamma, cutoff)
    return B.conj().T @ kernel_inv @ B


def D_eff_dynamic(D: np.ndarray, A: np.ndarray, B: np.ndarray,
                  omega: float, g: float, chi: float,
                  gamma: float = 0.0,
                  cutoff: float = PSEUDOINVERSE_CUTOFF) -> np.ndarray:
    """
    Compute effective dynamical matrix D_eff(ω) = D - g²B†(χω² - A + iγω)⁻¹B.

    This is the central result for dynamic bath:
      - Static elimination gives D - g²S (wrong sign → unstable)
      - Dynamic bath gives D - g²S(ω) where S(ω) flips sign at low ω

    For ω < ω_crit = √(λ_min(A)/χ):
      - (χω² - A) is negative definite
      - S(ω) is negative definite
      - -g²S(ω) is POSITIVE → D_eff = D + (positive) → STABLE

    Physical interpretation (for physicists):
      - D_eff(ω) = D + Σ(ω) where Σ(ω) = -g²B†(χω² - A)⁻¹B
      - Low ω: adiabatic limit, Σ > 0 (stiffening)
      - High ω: resonance with bath modes
      - Equivalent to integrating out fast DOF with retarded response

    Args:
        D: (3V, 3V) elastic dynamical matrix
        A: (V, V) bath stiffness matrix
        B: (V, 3V) coupling matrix
        omega: frequency
        g: coupling strength
        chi: bath inertia
        gamma: damping coefficient (default 0)
        cutoff: eigenvalue threshold

    Returns:
        D_eff: (3V, 3V) effective dynamical matrix
    """
    S_omega = compute_dynamic_schur(A, B, omega, chi, gamma, cutoff)
    return D - g**2 * S_omega


def get_bath_critical_frequency(A: np.ndarray, chi: float,
                                 cutoff: float = PSEUDOINVERSE_CUTOFF) -> float:
    """
    Compute critical frequency ω_crit = √(λ_min(A)/χ).

    Below this frequency, the dynamic bath gives positive correction
    (stiffening). Above, the system enters the bath resonance region.

    Args:
        A: (V, V) bath stiffness matrix
        chi: bath inertia

    Returns:
        omega_crit: critical frequency
    """
    eigvals = np.linalg.eigvalsh(A)
    # Find smallest nonzero eigenvalue
    nonzero_eigvals = eigvals[eigvals > cutoff]
    if len(nonzero_eigvals) == 0:
        return 0.0
    lambda_min = np.min(nonzero_eigvals)
    return np.sqrt(lambda_min / chi)


# ---------------------------------------------------------------------
# SQUARE PENALTY BATH (Alternative Mechanism)
# ---------------------------------------------------------------------
#
# This is an ALTERNATIVE to dynamic bath that also gives positive sign.
#
# Physical interpretation: "divergence is expensive" (bulk modulus / incompressibility)
#
# Energy functional:
#   E[u,p] = (1/2)u†Du + (κ/2)(Bu - p)†W(Bu - p) + (1/2)p†Ap
#
# Here:
#   - W ≥ 0: weight matrix (typically identity)
#   - A ≥ 0: bath stiffness (vertex Laplacian)
#   - κ > 0: penalty strength
#   - p: auxiliary field (not Lagrange multiplier)
#
# Key difference from linear coupling:
#   - Linear: g·p†Bu → gives NEGATIVE Schur (instability)
#   - Square: (Bu - p)² → gives POSITIVE penalty (stability)
#
# Eliminating p (minimizing over p):
#   p* = (κW + A)⁻¹ κW Bu
#
# Effective energy:
#   E_eff = (1/2)u†Du + (1/2)(Bu)†[κW - κ²W(κW + A)⁻¹W](Bu)
#
# The bracket [κW - κ²W(κW + A)⁻¹W] is POSITIVE SEMI-DEFINITE by construction.
# This is a "screened penalty" - finite κ gives finite stiffness, κ→∞ gives constraint.
#
# Comparison of mechanisms:
#   | Mechanism      | Sign     | Physics                    |
#   |----------------|----------|----------------------------|
#   | Dynamic bath   | + at ω<ω_c | Retarded response, inertia |
#   | Square penalty | + always | Divergence is expensive    |
#
# Both give L suppression, but through different physics.


def compute_square_penalty_schur(A: np.ndarray, B: np.ndarray,
                                  kappa: float,
                                  W: np.ndarray = None,
                                  cutoff: float = PSEUDOINVERSE_CUTOFF) -> np.ndarray:
    """
    Compute effective Schur from square penalty: S_eff = κW - κ²W(κW + A)⁻¹W.

    This is the contribution to D_eff from the (Bu - p)² penalty term.
    It is POSITIVE SEMI-DEFINITE by construction.

    E[u,p] = (1/2)u†Du + (κ/2)(Bu - p)†W(Bu - p) + (1/2)p†Ap

    After eliminating p:
      E_eff = (1/2)u†Du + (1/2)(Bu)† S_eff (Bu)
      S_eff = κW - κ²W(κW + A)⁻¹W

    Args:
        A: (V, V) bath stiffness matrix (Hermitian, PSD)
        B: (V, 3V) coupling matrix (divergence operator)
        kappa: penalty strength (κ > 0)
        W: (V, V) weight matrix (default: identity)
        cutoff: eigenvalue threshold for pseudoinverse

    Returns:
        D_penalty: (3V, 3V) penalty contribution to effective D (PSD)
    """
    V = A.shape[0]

    if W is None:
        W = np.eye(V, dtype=complex)

    # Compute (κW + A)⁻¹
    M = kappa * W + A

    # Use eigendecomposition for stable inversion
    eigvals, eigvecs = np.linalg.eigh(M)
    inv_eigvals = np.where(np.abs(eigvals) > cutoff,
                           1.0 / eigvals,
                           0.0)
    M_inv = eigvecs @ np.diag(inv_eigvals) @ eigvecs.conj().T

    # S_eff = κW - κ²W M⁻¹ W
    S_eff = kappa * W - kappa**2 * W @ M_inv @ W

    # Project onto displacement space: D_penalty = B† S_eff B
    D_penalty = B.conj().T @ S_eff @ B

    return D_penalty


def D_eff_square_penalty(D: np.ndarray, A: np.ndarray, B: np.ndarray,
                          kappa: float,
                          W: np.ndarray = None,
                          cutoff: float = PSEUDOINVERSE_CUTOFF) -> np.ndarray:
    """
    Compute effective dynamical matrix with square penalty bath.

    D_eff = D + B† S_eff B

    where S_eff = κW - κ²W(κW + A)⁻¹W is POSITIVE SEMI-DEFINITE.

    This gives L stiffening without frequency dependence (unlike dynamic bath).

    Args:
        D: (3V, 3V) elastic dynamical matrix
        A: (V, V) bath stiffness matrix
        B: (V, 3V) coupling matrix
        kappa: penalty strength
        W: (V, V) weight matrix (default: identity)
        cutoff: eigenvalue threshold

    Returns:
        D_eff: (3V, 3V) effective dynamical matrix (D + positive correction)
    """
    D_penalty = compute_square_penalty_schur(A, B, kappa, W, cutoff)
    return D + D_penalty


def verify_square_penalty_psd(A: np.ndarray, B: np.ndarray,
                               kappa: float,
                               W: np.ndarray = None,
                               cutoff: float = PSEUDOINVERSE_CUTOFF) -> dict:
    """
    Verify that square penalty Schur complement is positive semi-definite.

    This should ALWAYS pass - it's a mathematical identity that
    S_eff = κW - κ²W(κW + A)⁻¹W ≥ 0.

    Args:
        A: (V, V) bath stiffness matrix
        B: (V, 3V) coupling matrix
        kappa: penalty strength
        W: (V, V) weight matrix (default: identity)
        cutoff: eigenvalue threshold

    Returns:
        dict with 'is_psd', 'min_eigenvalue', 'condition_number'
    """
    D_penalty = compute_square_penalty_schur(A, B, kappa, W, cutoff)

    eigvals = np.linalg.eigvalsh(D_penalty)
    min_eig = np.min(eigvals)
    max_eig = np.max(eigvals)

    # PSD means all eigenvalues ≥ 0 (with numerical tolerance)
    is_psd = min_eig > -cutoff

    cond = max_eig / max(min_eig, cutoff) if min_eig > cutoff else np.inf

    return {
        'is_psd': is_psd,
        'min_eigenvalue': min_eig,
        'max_eigenvalue': max_eig,
        'condition_number': cond,
    }
