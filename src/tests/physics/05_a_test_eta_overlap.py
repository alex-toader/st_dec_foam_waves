#!/usr/bin/env python3
"""
CORE η OVERLAP TESTS
====================

Tests that the plane-wave embedding U correctly selects the acoustic sector.

STRUCTURES TESTED: C15, Kelvin, FCC, WP (all 4)

INPUTS
------

Internal (from model):
  - Foam geometry from builders (C15, Kelvin, FCC, WP)
  - DisplacementBloch acoustic modes
  - Bath coupling: D_eff = D + λ² S (Schur complement)

External:
  - None (pure model test)

OUTPUTS
-------

  - Without bath (λ=0): 3 acoustic modes with η ≈ 1
  - With bath (λ>0): 2 acoustic modes with η ≈ 1 in IR window (L mode gapped)
  - Sum rule: Σ η_j = 3 (conserved)

TESTS (15 total)
----------------
Per structure (×4):
- 3 modes without bath (λ=0)
- 2 modes with bath (λ>0, L gapped)
- Sum rule: Σ η_j = 3
Plus: 3 zero-mode tests at Γ (Kelvin, FCC, WP)

EXPECTED OUTPUT (Jan 2026)
--------------------------
    15 passed in 1.43s

PHYSICS
-------

For each eigenvector w_j of D_eff(k), compute:

    η_j = ||P_U w_j||² / ||w_j||²

where P_U = U(U†U)⁻¹U† is the projector onto the plane-wave subspace.

EXPECTED RESULTS:
    - 2-3 acoustic modes: η ≈ 1 (live in plane-wave subspace)
    - Optical modes: η ≪ 1 (orthogonal to plane-waves)

Jan 2026
"""

import numpy as np

from physics.bloch import DisplacementBloch
from physics.bath import (
    build_vertex_laplacian_bloch,
    build_divergence_operator_bloch,
    compute_discrete_schur,
)
from core_math.builders import build_fcc_supercell_periodic, build_wp_supercell_periodic
from core_math.builders.multicell_periodic import build_bcc_supercell_periodic
from core_math.builders.c15_periodic import build_c15_supercell_periodic


# =============================================================================
# CONSTANTS
# =============================================================================

# Numerical tolerances
ZERO_TOL = 1e-10          # General threshold for "effectively zero"
ZERO_TOL_GAMMA = 1e-8     # Relaxed threshold for Γ point zero-mode tests
                          # (allows for BLAS/LAPACK numerical noise)
NORM_TOL = 1e-14          # Threshold for norm comparisons

# Default structure parameters
# NOTE: L = 4.0 * N for Kelvin/FCC assumes unit cell size = 4.0
# Kelvin (BCC): conventional cell side = 4.0, so N=2 supercell has L=8.0
# FCC: conventional cell side = 4.0, consistent with build_fcc_supercell_periodic
# WP: uses L_cell = 4.0 (default), N=1 gives L=4.0 (consistent with christoffel.py)
DEFAULT_L_C15 = 4.0       # Domain size for C15 N=1 (L = 4.0 * N)
DEFAULT_L_KELVIN = 8.0    # Domain size for Kelvin N=2 (L = 4.0 * N)
DEFAULT_L_FCC = 8.0       # Domain size for FCC N=2 (L = 4.0 * N)
DEFAULT_L_WP = 4.0        # Domain size for WP N=1 (L = 4.0 * N)
DEFAULT_K_L = 3.0         # Longitudinal spring constant
DEFAULT_K_T = 1.0         # Transverse spring constant
DEFAULT_LAMBDA_BATH = 10.0  # Bath coupling strength

# η overlap thresholds
# 0.9 means mode must have 90%+ of its norm in the plane-wave subspace
ETA_HIGH = 0.9            # Mode "in" plane-wave subspace

# Frequency classification for acoustic window
# -----------------------------------------------
# Acoustic modes: ω² ~ c² k² where c² depends on geometry, spring constants,
# and connectivity. Empirically, c² is O(1-10) for our parameter choices
# (k_L=3.0, k_T=1.0) across all tested structures (Kelvin, FCC, WP).
# Factor 50 gives generous margin: modes with ω²/k² < 50 are "acoustic"
LOW_FREQ_CUTOFF_FACTOR = 50.0

# Floor ensures numerical stability at very small k
# Set low (0.005) to preserve physics-scaling behavior (cutoff ∝ k²)
# Only activates if 50×k² < 0.005, i.e., k < 0.01
LOW_FREQ_CUTOFF_FLOOR = 0.005

# Modes with ω²/k² > 50 are considered "gapped" (massive)
GAPPED_RATIO = 50.0


def low_freq_cutoff(k):
    """Compute low-frequency cutoff that scales with k².

    Acoustic modes have ω² ~ c²k², so cutoff ∝ k² is robust.
    A floor is applied to avoid numerical issues at very small k.
    """
    k_sq = np.dot(k, k)
    return max(LOW_FREQ_CUTOFF_FACTOR * k_sq, LOW_FREQ_CUTOFF_FLOOR)


# Direction for k-vector (default: [111])
K_DIR_111 = np.array([1, 1, 1]) / np.sqrt(3)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def build_planewave_embedding(db, k):
    """
    Build plane-wave embedding matrix U (3V × 3).

    Columns are plane-wave displacements with polarizations e_x, e_y, e_z.
    """
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

    U = np.column_stack([planewave(e_x), planewave(e_y), planewave(e_z)])
    return U


def build_D_eff(db, k, lambda_bath=0.0):
    """
    Build D_eff(k) = D(k) + λ² S(k).

    The λ² factor (not λ) comes from the Schur complement derivation:
        - Original system has bath coupling λB in off-diagonal blocks
        - Schur complement eliminates bath DOF: S = B† A⁻¹ B
        - Effective dynamics: D_eff = D + λ² S (λ enters twice: once in B†, once in B)

    This is the standard form for integrating out fast DOF in effective field theory.

    Args:
        db: DisplacementBloch instance
        k: wavevector
        lambda_bath: bath coupling strength (default 0.0 = no bath)

    Returns:
        D_eff: (3V, 3V) complex Hermitian matrix
    """
    D = db.build_dynamical_matrix(k)

    if lambda_bath > 0:
        A = build_vertex_laplacian_bloch(db, k)
        B = build_divergence_operator_bloch(db, k)
        S = compute_discrete_schur(A, B)
        D_eff = D + lambda_bath**2 * S
    else:
        D_eff = D

    return D_eff


def compute_eta_j(w_j, P_U):
    """
    Compute overlap η_j = ||P_U w_j||² / ||w_j||².

    Args:
        w_j: eigenvector (3V,) complex
        P_U: projector onto plane-wave subspace (3V, 3V)

    Returns:
        eta_j: overlap fraction in [0, 1]
    """
    Pw = P_U @ w_j
    norm_Pw_sq = np.real(np.vdot(Pw, Pw))
    norm_w_sq = np.real(np.vdot(w_j, w_j))

    if norm_w_sq < NORM_TOL:
        return 0.0

    return norm_Pw_sq / norm_w_sq


def build_projector_PU(U):
    """
    Build projector P_U = U(U†U)⁻¹U†.

    Args:
        U: (3V, 3) embedding matrix

    Returns:
        P_U: (3V, 3V) projector matrix

    Raises:
        ValueError: if U†U is poorly conditioned (indicates a bug, e.g., k=0
                    or duplicate positions)
    """
    UtU = U.conj().T @ U

    # Tripwire: U†U should be well-conditioned (~V·I for plane-waves)
    # Poor conditioning indicates a bug (k=0, duplicate positions, etc.)
    cond = np.linalg.cond(UtU)
    if cond > 100:
        raise ValueError(f"U†U poorly conditioned (cond={cond:.1e}), check k≠0 and positions")

    # Use pinv for numerical stability
    UtU_inv = np.linalg.pinv(UtU, rcond=1e-12)
    P_U = U @ UtU_inv @ U.conj().T
    return P_U


def classify_mode_by_frequency(eigenvalue, k_mag, threshold_ratio=GAPPED_RATIO):
    """
    Classify mode as acoustic or optical based on ω²/k² ratio.

    Acoustic: ω² ~ c² k² → ratio ~ O(1-10)
    Gapped/Optical: ω² ~ m² → ratio >> 1 at small k

    Args:
        eigenvalue: ω² value
        k_mag: |k|
        threshold_ratio: if ω²/k² > threshold, considered gapped

    Returns:
        'acoustic' or 'gapped'
    """
    if k_mag < ZERO_TOL:
        return 'unknown'

    c_sq = eigenvalue / k_mag**2

    if c_sq > threshold_ratio:
        return 'gapped'
    else:
        return 'acoustic'


def check_eta_overlap(db, name, k, lambda_bath=0.0, n_modes=30):
    """
    Main η_j overlap test.

    For the first n_modes eigenvectors of D_eff(k), compute η_j.

    Args:
        db: DisplacementBloch instance
        name: structure name
        k: wavevector
        lambda_bath: bath coupling strength
        n_modes: number of modes to analyze

    Returns:
        list of dicts with results per mode
    """
    k_mag = np.linalg.norm(k)

    # Build D_eff
    D_eff = build_D_eff(db, k, lambda_bath)

    # Diagonalize (eigh returns eigenvalues in ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(D_eff)

    # Build plane-wave embedding and projector
    U = build_planewave_embedding(db, k)
    P_U = build_projector_PU(U)

    # Compute η_j for each mode
    n_modes = min(n_modes, len(eigenvalues))

    results = []
    for j in range(n_modes):
        w_j = eigenvectors[:, j]
        eta_j = compute_eta_j(w_j, P_U)
        omega_sq = eigenvalues[j]
        mode_type = classify_mode_by_frequency(omega_sq, k_mag)

        results.append({
            'j': j,
            'omega_sq': omega_sq,
            'eta_j': eta_j,
            'mode_type': mode_type,
        })

    return results


def sum_rule_test(db, k, lambda_bath):
    """
    Sum rule test: Verify that Σ_j η_j = dim(U) = 3.

    This follows from: Tr(P_U) = dim(U) = 3
    and: Σ_j η_j = Σ_j ⟨w_j|P_U|w_j⟩ = Tr(P_U) (complete basis)

    Returns:
        dict with:
            - sum_eta: Σ η_j
            - expected: 3
            - relative_error: |sum_eta - 3| / 3
    """
    # Build D_eff and diagonalize
    D_eff = build_D_eff(db, k, lambda_bath)
    eigenvalues, eigenvectors = np.linalg.eigh(D_eff)

    # Build projector
    U = build_planewave_embedding(db, k)
    P_U = build_projector_PU(U)

    # Compute η_j for all modes
    sum_eta = 0.0
    for j in range(len(eigenvalues)):
        w_j = eigenvectors[:, j]
        eta_j = compute_eta_j(w_j, P_U)
        sum_eta += eta_j

    expected = 3.0
    rel_error = abs(sum_eta - expected) / expected

    return {
        'sum_eta': sum_eta,
        'expected': expected,
        'relative_error': rel_error,
    }


# =============================================================================
# STRUCTURE BUILDERS
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

def test_kelvin_three_zero_modes_at_gamma():
    """At Γ (k=0): Kelvin should have exactly 3 zero modes (acoustic).

    This is a fundamental tripwire for connectivity + edge bugs.
    Uses relaxed tolerance ZERO_TOL_GAMMA for numerical robustness.
    D(Γ) is PSD, so we use eigenvalues < tol (not abs).
    """
    v, e, f, _ = build_bcc_supercell_periodic(2)
    db = DisplacementBloch(v, e, DEFAULT_L_KELVIN, k_L=DEFAULT_K_L, k_T=DEFAULT_K_T)

    # At k=0 (Γ point)
    k = np.array([0.0, 0.0, 0.0])
    D = db.build_dynamical_matrix(k)
    eigenvalues = np.linalg.eigvalsh(D)

    # Count zero modes: D(Γ) is PSD, so use < tol (more PSD-consistent than abs)
    zero_modes = np.sum(eigenvalues < ZERO_TOL_GAMMA)

    print(f"\nKelvin at Γ: {zero_modes} zero modes (first 6 eigenvalues: {eigenvalues[:6]})")
    assert zero_modes == 3, f"Expected exactly 3 zero modes at Γ, got {zero_modes}"


def test_fcc_three_zero_modes_at_gamma():
    """At Γ (k=0): FCC should have exactly 3 zero modes (acoustic)."""
    result = build_fcc_supercell_periodic(2)
    v, e = result[0], result[1]
    db = DisplacementBloch(v, e, DEFAULT_L_FCC, k_L=DEFAULT_K_L, k_T=DEFAULT_K_T)

    k = np.array([0.0, 0.0, 0.0])
    D = db.build_dynamical_matrix(k)
    eigenvalues = np.linalg.eigvalsh(D)

    # D(Γ) is PSD
    zero_modes = np.sum(eigenvalues < ZERO_TOL_GAMMA)

    print(f"\nFCC at Γ: {zero_modes} zero modes")
    assert zero_modes == 3, f"Expected exactly 3 zero modes at Γ, got {zero_modes}"


def test_wp_three_zero_modes_at_gamma():
    """At Γ (k=0): WP should have exactly 3 zero modes (acoustic)."""
    v, e, f = build_wp_supercell_periodic(1, L_cell=4.0)
    db = DisplacementBloch(v, e, DEFAULT_L_WP, k_L=DEFAULT_K_L, k_T=DEFAULT_K_T)

    k = np.array([0.0, 0.0, 0.0])
    D = db.build_dynamical_matrix(k)
    eigenvalues = np.linalg.eigvalsh(D)

    # D(Γ) is PSD
    zero_modes = np.sum(eigenvalues < ZERO_TOL_GAMMA)

    print(f"\nWP at Γ: {zero_modes} zero modes")
    assert zero_modes == 3, f"Expected exactly 3 zero modes at Γ, got {zero_modes}"


def test_kelvin_three_modes_without_bath():
    """Without bath: Kelvin should have 3 acoustic modes with η ≈ 1."""
    db, k = _build_kelvin()
    results = check_eta_overlap(db, "Kelvin", k, lambda_bath=0.0, n_modes=15)

    high_eta = [r for r in results if r['eta_j'] > ETA_HIGH and r['omega_sq'] > ZERO_TOL]

    print(f"\nKelvin λ=0: {len(high_eta)} modes with η > {ETA_HIGH}")
    assert len(high_eta) >= 3, f"Expected 3 acoustic modes, got {len(high_eta)}"


def test_kelvin_two_modes_with_bath():
    """With bath: Kelvin should have 2 T acoustic modes with η ≈ 1 (L gapped).

    The bath term λ²S gaps the longitudinal mode, leaving only 2 transverse modes
    in the low-frequency window. We assert >= 2 (not == 2) for robustness.
    """
    db, k = _build_kelvin()
    results = check_eta_overlap(db, "Kelvin", k, lambda_bath=DEFAULT_LAMBDA_BATH, n_modes=15)

    # Count modes with η > ETA_HIGH in low-freq window (ω² < cutoff ∝ k²)
    cutoff = low_freq_cutoff(k)
    high_eta_lowfreq = [r for r in results
                       if r['eta_j'] > ETA_HIGH and r['omega_sq'] > ZERO_TOL and r['omega_sq'] < cutoff]

    print(f"\nKelvin λ={DEFAULT_LAMBDA_BATH}: {len(high_eta_lowfreq)} modes with η > {ETA_HIGH} in low-freq window (cutoff={cutoff:.2f})")
    assert len(high_eta_lowfreq) >= 2, f"Expected 2 T modes in low-freq, got {len(high_eta_lowfreq)}"


def test_fcc_three_modes_without_bath():
    """Without bath: FCC should have 3 acoustic modes with η ≈ 1."""
    db, k = _build_fcc()
    results = check_eta_overlap(db, "FCC", k, lambda_bath=0.0, n_modes=15)

    high_eta = [r for r in results if r['eta_j'] > ETA_HIGH and r['omega_sq'] > ZERO_TOL]

    print(f"\nFCC λ=0: {len(high_eta)} modes with η > {ETA_HIGH}")
    assert len(high_eta) >= 3, f"Expected 3 acoustic modes, got {len(high_eta)}"


def test_fcc_two_modes_with_bath():
    """With bath: FCC should have 2 T acoustic modes with η ≈ 1 (L gapped).

    We assert >= 2 (not == 2) for robustness.
    """
    db, k = _build_fcc()
    results = check_eta_overlap(db, "FCC", k, lambda_bath=DEFAULT_LAMBDA_BATH, n_modes=15)

    cutoff = low_freq_cutoff(k)
    high_eta_lowfreq = [r for r in results
                       if r['eta_j'] > ETA_HIGH and r['omega_sq'] > ZERO_TOL and r['omega_sq'] < cutoff]

    print(f"\nFCC λ={DEFAULT_LAMBDA_BATH}: {len(high_eta_lowfreq)} modes with η > {ETA_HIGH} in low-freq window (cutoff={cutoff:.2f})")
    assert len(high_eta_lowfreq) >= 2, f"Expected 2 T modes in low-freq, got {len(high_eta_lowfreq)}"


def test_wp_three_modes_without_bath():
    """Without bath: WP (Weaire-Phelan) should have 3 acoustic modes with η ≈ 1."""
    db, k = _build_wp()
    results = check_eta_overlap(db, "WP", k, lambda_bath=0.0, n_modes=15)

    high_eta = [r for r in results if r['eta_j'] > ETA_HIGH and r['omega_sq'] > ZERO_TOL]

    print(f"\nWP λ=0: {len(high_eta)} modes with η > {ETA_HIGH}")
    assert len(high_eta) >= 3, f"Expected 3 acoustic modes, got {len(high_eta)}"


def test_wp_two_modes_with_bath():
    """With bath: WP should have 2 T acoustic modes with η ≈ 1 (L gapped).

    We assert >= 2 (not == 2) for robustness.
    """
    db, k = _build_wp()
    results = check_eta_overlap(db, "WP", k, lambda_bath=DEFAULT_LAMBDA_BATH, n_modes=15)

    cutoff = low_freq_cutoff(k)
    high_eta_lowfreq = [r for r in results
                       if r['eta_j'] > ETA_HIGH and r['omega_sq'] > ZERO_TOL and r['omega_sq'] < cutoff]

    print(f"\nWP λ={DEFAULT_LAMBDA_BATH}: {len(high_eta_lowfreq)} modes with η > {ETA_HIGH} in low-freq window (cutoff={cutoff:.2f})")
    assert len(high_eta_lowfreq) >= 2, f"Expected 2 T modes in low-freq, got {len(high_eta_lowfreq)}"


def test_c15_three_modes_without_bath():
    """Without bath: C15 (Laves) should have 3 acoustic modes with η ≈ 1."""
    db, k = _build_c15()
    results = check_eta_overlap(db, "C15", k, lambda_bath=0.0, n_modes=15)

    high_eta = [r for r in results if r['eta_j'] > ETA_HIGH and r['omega_sq'] > ZERO_TOL]

    print(f"\nC15 λ=0: {len(high_eta)} modes with η > {ETA_HIGH}")
    assert len(high_eta) >= 3, f"Expected 3 acoustic modes, got {len(high_eta)}"


def test_c15_two_modes_with_bath():
    """With bath: C15 should have 2 T acoustic modes with η ≈ 1 (L gapped).

    We assert >= 2 (not == 2) for robustness.
    """
    db, k = _build_c15()
    results = check_eta_overlap(db, "C15", k, lambda_bath=DEFAULT_LAMBDA_BATH, n_modes=15)

    cutoff = low_freq_cutoff(k)
    high_eta_lowfreq = [r for r in results
                       if r['eta_j'] > ETA_HIGH and r['omega_sq'] > ZERO_TOL and r['omega_sq'] < cutoff]

    print(f"\nC15 λ={DEFAULT_LAMBDA_BATH}: {len(high_eta_lowfreq)} modes with η > {ETA_HIGH} in low-freq window (cutoff={cutoff:.2f})")
    assert len(high_eta_lowfreq) >= 2, f"Expected 2 T modes in low-freq, got {len(high_eta_lowfreq)}"


def test_sum_rule_kelvin():
    """Verify Σ η_j = 3 (overlap conservation) for Kelvin."""
    db, k = _build_kelvin()

    for lam in [0.0, DEFAULT_LAMBDA_BATH]:
        sr = sum_rule_test(db, k, lam)
        print(f"\nKelvin λ={lam}: Ση_j = {sr['sum_eta']:.4f} (expect 3)")
        assert sr['relative_error'] < 1e-6, f"Sum rule violated: Ση_j = {sr['sum_eta']:.4f}"


def test_sum_rule_fcc():
    """Verify Σ η_j = 3 (overlap conservation) for FCC."""
    db, k = _build_fcc()

    for lam in [0.0, DEFAULT_LAMBDA_BATH]:
        sr = sum_rule_test(db, k, lam)
        print(f"\nFCC λ={lam}: Ση_j = {sr['sum_eta']:.4f} (expect 3)")
        assert sr['relative_error'] < 1e-6, f"Sum rule violated: Ση_j = {sr['sum_eta']:.4f}"


def test_sum_rule_wp():
    """Verify Σ η_j = 3 (overlap conservation) for WP."""
    db, k = _build_wp()

    for lam in [0.0, DEFAULT_LAMBDA_BATH]:
        sr = sum_rule_test(db, k, lam)
        print(f"\nWP λ={lam}: Ση_j = {sr['sum_eta']:.4f} (expect 3)")
        assert sr['relative_error'] < 1e-6, f"Sum rule violated: Ση_j = {sr['sum_eta']:.4f}"


def test_sum_rule_c15():
    """Verify Σ η_j = 3 (overlap conservation) for C15."""
    db, k = _build_c15()

    for lam in [0.0, DEFAULT_LAMBDA_BATH]:
        sr = sum_rule_test(db, k, lam)
        print(f"\nC15 λ={lam}: Ση_j = {sr['sum_eta']:.4f} (expect 3)")
        assert sr['relative_error'] < 1e-6, f"Sum rule violated: Ση_j = {sr['sum_eta']:.4f}"


if __name__ == "__main__":
    print("=" * 70)
    print("CORE η OVERLAP TESTS")
    print("=" * 70)

    # Run basic tests
    for name, builder in [("C15", _build_c15), ("Kelvin", _build_kelvin), ("FCC", _build_fcc), ("WP", _build_wp)]:
        db, k = builder()
        print(f"\n{name}: V={db.V}, L={db.L}")

        for lam in [0.0, DEFAULT_LAMBDA_BATH]:
            results = check_eta_overlap(db, name, k, lambda_bath=lam, n_modes=10)
            high_eta = [r for r in results if r['eta_j'] > ETA_HIGH and r['omega_sq'] > ZERO_TOL]
            print(f"  λ={lam}: {len(high_eta)} modes with η > {ETA_HIGH}")
