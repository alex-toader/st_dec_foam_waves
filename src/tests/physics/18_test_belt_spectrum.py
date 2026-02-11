"""
Tests for belt_spectrum module — Discrete Ring Modes (M2).

Verifies FEM operators, standard Nyquist/Brillouin spectral properties,
and integration with foam cell geometry from cell_topology (M0).

Sections:
  1. Uniform ring — analytic reference (FEM + lumped mass)
  2. m=2 existence — Nyquist limit N >= 2m
  3. Exclusion — Nyquist limit for m=4
  4. Mode gap — trigonometric identity
  5-6. Integration — real foam circuits (Kelvin, C15)
  7. Sine dispersion — exact and non-uniform

Feb 2026
"""

import sys
sys.path.insert(0, '/Users/alextoader/Sites/physics_ai/ST_8/src')

import numpy as np
import pytest

from core_math.analysis.belt_spectrum import (
    get_circuit_segments,
    build_laplacian_1d,
    build_mass_matrix_1d,
    compute_mode_spectrum,
    has_m2,
    max_supported_mode,
    compute_belt_spectrum,
)
from core_math.analysis.cell_topology import (
    get_cell_geometry,
    find_simple_cycles,
    equatorial_test,
    find_best_belt,
    circuit_holonomy,
)
from core_math.builders.multicell_periodic import (
    build_bcc_supercell_periodic, generate_bcc_centers
)
from core_math.builders.c15_periodic import (
    build_c15_supercell_periodic, get_c15_points
)


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def kelvin_mesh():
    N = 2
    v, e, f, cfi = build_bcc_supercell_periodic(N)
    centers = np.array(generate_bcc_centers(N))
    L = 4 * N
    return v, e, f, cfi, centers, L


@pytest.fixture(scope='module')
def c15_mesh():
    N = 1
    L_cell = 4.0
    v, e, f, cfi = build_c15_supercell_periodic(N, L_cell)
    centers = get_c15_points(N, L_cell)
    L = N * L_cell
    return v, e, f, cfi, centers, L


@pytest.fixture(scope='module')
def kelvin_cell0(kelvin_mesh):
    v, e, f, cfi, centers, L = kelvin_mesh
    face_data, adj = get_cell_geometry(0, centers[0], v, f, cfi, L)
    return face_data, adj, centers[0]


@pytest.fixture(scope='module')
def kelvin_best_belt(kelvin_cell0):
    face_data, adj, cc = kelvin_cell0
    result = find_best_belt(face_data, adj, cc)
    assert result is not None
    circuit, _ = result
    return circuit, face_data, adj


@pytest.fixture(scope='module')
def c15_z12_cell(c15_mesh):
    v, e, f, cfi, centers, L = c15_mesh
    for ci in range(len(cfi)):
        if len(cfi[ci]) == 12:
            face_data, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            return face_data, adj, centers[ci]
    pytest.skip("No Z12 cell found")


@pytest.fixture(scope='module')
def c15_z16_cell(c15_mesh):
    v, e, f, cfi, centers, L = c15_mesh
    for ci in range(len(cfi)):
        if len(cfi[ci]) == 16:
            face_data, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            return face_data, adj, centers[ci]
    pytest.skip("No Z16 cell found")


# ═══════════════════════════════════════════════════════════════
# 1. UNIFORM RING — ANALYTIC REFERENCE
# ═══════════════════════════════════════════════════════════════

class TestUniformRing:
    """Verify FEM operators and spectrum on uniform ring (analytic solution known)."""

    def test_laplacian_symmetric(self):
        """K is symmetric on non-uniform ring (FEM assembly guarantee)."""
        segments = np.array([1.0, 3.0, 0.7, 2.1, 1.4])
        K = build_laplacian_1d(segments)
        np.testing.assert_allclose(K, K.T, atol=1e-15)

    def test_laplacian_psd_random(self):
        """K is positive semi-definite on random segments."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            N = rng.randint(3, 12)
            segments = rng.uniform(0.5, 3.0, size=N)
            K = build_laplacian_1d(segments)
            M = build_mass_matrix_1d(segments)
            m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
            H = m_inv_sqrt @ K @ m_inv_sqrt
            evals = np.linalg.eigvalsh(H)
            assert all(ev >= -1e-10 for ev in evals), \
                f"Negative eigenvalue {min(evals)} on N={N}"

    def test_laplacian_eigenvalues_N6(self):
        """Uniform 6-ring h=1: eigenvalues = 4sin²(mπ/N)/h² exact."""
        N = 6
        h = 1.0
        segments = np.ones(N) * h
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        evals = np.sort(np.linalg.eigvalsh(H))

        # FEM: K = (1/h) tridiag(-1,2,-1), M = hI
        # H = M^{-1/2} K M^{-1/2} = K/h → eigenvalues = 4sin²(mπ/N)/h²
        expected = sorted([4 * np.sin(m * np.pi / N)**2 / h**2
                          for m in range(N)])

        np.testing.assert_allclose(evals, expected, atol=1e-10)

    def test_laplacian_eigenvalues_N8(self):
        """Uniform 8-ring h=1.5: eigenvalues = 4sin²(mπ/N)/h² exact."""
        N = 8
        h = 1.5
        segments = np.ones(N) * h
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        evals = np.sort(np.linalg.eigvalsh(H))

        expected = sorted([4 * np.sin(m * np.pi / N)**2 / h**2
                          for m in range(N)])

        np.testing.assert_allclose(evals, expected, atol=1e-10)

    def test_zero_mode_exists(self):
        """Stiffness K has a zero eigenvalue (constant null space).

        Tests K directly (not the generalized H = M^{-1/2} K M^{-1/2}),
        since null(K) = span(1) is a property of the stiffness matrix alone.
        """
        for N in [3, 5, 6, 8, 10]:
            segments = np.ones(N)
            K = build_laplacian_1d(segments)
            evals = np.sort(np.linalg.eigvalsh(K))
            assert abs(evals[0]) < 1e-10, f"N={N}: zero mode missing"

    def test_mass_matrix_diagonal(self):
        """Mass matrix is diagonal with m_k = (h_{k-1} + h_k)/2."""
        segments = np.array([1.0, 1.5, 2.0, 1.2])
        M = build_mass_matrix_1d(segments)
        assert M.shape == (4, 4)
        for k in range(4):
            expected = (segments[(k-1) % 4] + segments[k]) / 2.0
            assert abs(M[k, k] - expected) < 1e-12
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert abs(M[i, j]) < 1e-15

    def test_eigenvectors_M_orthonormal(self):
        """Eigenvectors are M-orthonormal: φ^T M φ = I."""
        N = 6
        segments = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_diag = np.diag(M)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(m_diag))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        _, evecs_psi = np.linalg.eigh(H)
        evecs_phi = m_inv_sqrt @ evecs_psi
        check = evecs_phi.T @ M @ evecs_phi
        np.testing.assert_allclose(check, np.eye(N), atol=1e-10)


# ═══════════════════════════════════════════════════════════════
# 2. m=2 EXISTENCE THEOREM: iff N >= 5
# ═══════════════════════════════════════════════════════════════

class TestM2Existence:
    """m=2 exists iff N >= 5. Core result from T7."""

    def test_N3_no_m2(self):
        """N=3 ring: m=2 impossible (max mode = 1)."""
        assert max_supported_mode(3) == 1

    def test_N4_nyquist_m2(self):
        """N=4: m=2 is Nyquist mode (degenerate, 1D not 2D)."""
        assert max_supported_mode(4) == 2
        segments = np.ones(4)
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        evals, evecs = np.linalg.eigh(H)
        # m=2 on N=4 is the staggered mode: (-1)^k, 4 zero crossings
        idx = np.argsort(evals)
        evecs = evecs[:, idx]
        last_mode = evecs[:, -1]
        zc = sum(1 for k in range(4) if last_mode[k] * last_mode[(k+1) % 4] < 0)
        assert zc == 4

    def test_N5_has_m2(self):
        """N=5: first ring with clean (non-Nyquist) m=2."""
        assert max_supported_mode(5) == 2
        segments = np.ones(5)
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        _, evecs = np.linalg.eigh(H)
        all_zc = [sum(1 for k in range(5) if evecs[k, c] * evecs[(k+1) % 5, c] < 0)
                   for c in range(5)]
        assert 4 in all_zc, "m=2 mode (4 zero-crossings) not found on N=5"

    def test_N6_to_N10_all_have_m2(self):
        """N=6..10: all have m=2 (4 zero crossings in some eigenvector)."""
        for N in range(6, 11):
            segments = np.ones(N)
            K = build_laplacian_1d(segments)
            M = build_mass_matrix_1d(segments)
            m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
            H = m_inv_sqrt @ K @ m_inv_sqrt
            _, evecs = np.linalg.eigh(H)
            all_zc = [sum(1 for k in range(N)
                         if evecs[k, c] * evecs[(k+1) % N, c] < 0)
                       for c in range(N)]
            assert 4 in all_zc, f"N={N}: m=2 mode (4 zero-crossings) not found"


# ═══════════════════════════════════════════════════════════════
# 3. EXCLUSION: m=4 iff N >= 9 (T19)
# ═══════════════════════════════════════════════════════════════

class TestExclusion:
    """Nyquist limits for higher modes: m=4 requires N >= 9 (clean)."""

    def test_max_mode_values(self):
        """max_supported_mode gives correct Nyquist limit."""
        assert max_supported_mode(3) == 1
        assert max_supported_mode(4) == 2
        assert max_supported_mode(5) == 2
        assert max_supported_mode(6) == 3
        assert max_supported_mode(7) == 3
        assert max_supported_mode(8) == 4
        assert max_supported_mode(9) == 4
        assert max_supported_mode(10) == 5

    def test_N6_no_m4(self):
        """N=6: m=4 impossible → two m=2 cannot coexist.

        Uses generalized eigenproblem H = M^{-1/2} K M^{-1/2} for consistency.
        """
        assert max_supported_mode(6) < 4
        N = 6
        segments = np.ones(N)
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        _, evecs = np.linalg.eigh(H)
        all_zc = [sum(1 for k in range(N) if evecs[k, c] * evecs[(k+1) % N, c] < 0)
                   for c in range(N)]
        assert 8 not in all_zc, "m=4 (8 zero-crossings) should not exist on N=6"

    def test_N8_no_clean_m4(self):
        """N=8: m=4 is Nyquist (non-degenerate, 1D not 2D eigenspace).

        Uses generalized eigenproblem H for consistency with rest of suite.
        """
        assert max_supported_mode(8) == 4
        N = 8
        segments = np.ones(N)
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        evals = np.sort(np.linalg.eigvalsh(H))
        # Nyquist eigenvalue is non-degenerate (multiplicity 1, not a pair)
        tol = 1e-10
        mult = np.sum(np.abs(evals - evals[-1]) < tol)
        assert mult == 1, \
            f"N=8: Nyquist m=4 should have multiplicity 1, got {mult}"

    def test_N9_has_clean_m4(self):
        """N=9: m=4 is clean (2D eigenspace, not Nyquist).

        Uses generalized eigenproblem H for consistency with rest of suite.
        """
        assert max_supported_mode(9) == 4
        N = 9
        segments = np.ones(N)
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        _, evecs = np.linalg.eigh(H)
        all_zc = [sum(1 for k in range(N) if evecs[k, c] * evecs[(k+1) % N, c] < 0)
                   for c in range(N)]
        assert 8 in all_zc, "m=4 (8 zero-crossings) should exist on N=9"


# ═══════════════════════════════════════════════════════════════
# 4. MODE GAP (T25)
# ═══════════════════════════════════════════════════════════════

class TestModeGap:
    """Adiabatic protection: relative gap Δω/ω₁ = 2cos(π/N) - 1."""

    def test_gap_formula_uniform(self):
        """On uniform ring, gap matches 2cos(π/N)-1 exactly.

        Uses single (evals, evecs) pair from eigh, sorted together,
        so freqs[col] corresponds to evecs[:, col].
        """
        for N in [5, 6, 7, 8, 10]:
            segments = np.ones(N)
            K = build_laplacian_1d(segments)
            M = build_mass_matrix_1d(segments)
            m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
            H = m_inv_sqrt @ K @ m_inv_sqrt

            evals, evecs = np.linalg.eigh(H)
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]
            freqs = np.sqrt(np.maximum(evals, 0))

            omega_1 = freqs[1]
            # Find first mode with 4 zero crossings (m=2)
            omega_2 = None
            for col in range(N):
                zc = sum(1 for k in range(N)
                         if evecs[k, col] * evecs[(k+1) % N, col] < 0)
                if zc == 4:
                    omega_2 = freqs[col]
                    break
            assert omega_2 is not None, f"N={N}: no m=2 mode found"

            gap = (omega_2 - omega_1) / omega_1
            expected_gap = 2 * np.cos(np.pi / N) - 1
            assert abs(gap - expected_gap) < 1e-8, \
                f"N={N}: gap={gap:.6f}, expected={expected_gap:.6f}"

    def test_gap_monotone_with_N(self):
        """Relative gap Δω/ω₁ = 2cos(π/N)-1 increases with N.

        N=5: 0.618, N=6: 0.732, N=8: 0.848, N=10: 0.902.
        Approaches 1 as N→∞ (continuum: ω₂/ω₁ → 2).
        """
        gaps = []
        Ns = [5, 6, 7, 8, 10]
        for N in Ns:
            gap = 2 * np.cos(np.pi / N) - 1
            gaps.append(gap)
        for i in range(len(gaps) - 1):
            assert gaps[i] < gaps[i + 1], \
                f"Gap at N={Ns[i]} ({gaps[i]:.3f}) should be < N={Ns[i+1]} ({gaps[i+1]:.3f})"


# ═══════════════════════════════════════════════════════════════
# 5. FOAM CELL TESTS — m=2 on real circuits (integration)
# ═══════════════════════════════════════════════════════════════

class TestFoamM2:
    """Integration tests: m=2 exists on real foam cell equatorial circuits.

    These use find_best_belt and circuit_holonomy from cell_topology
    as circuit selection pipeline. They verify the full M0+M2 chain.
    """

    def test_kelvin_best_belt_has_m2(self, kelvin_best_belt):
        """Kelvin best belt has m=2."""
        circuit, face_data, adj = kelvin_best_belt
        assert has_m2(circuit, face_data, adj)

    def test_kelvin_best_belt_spectrum(self, kelvin_best_belt):
        """Kelvin best belt: spectrum is valid (PSD, m=2 present)."""
        circuit, face_data, adj = kelvin_best_belt
        spec = compute_mode_spectrum(circuit, face_data, adj)
        assert spec is not None
        N = spec['N']
        assert N >= 5
        assert all(spec['eigenvalues'][i] >= -1e-10 for i in range(N))
        assert 4 in spec['zero_crossings']

    def test_kelvin_segments_positive(self, kelvin_best_belt):
        """All circuit segments are positive."""
        circuit, face_data, adj = kelvin_best_belt
        segs = get_circuit_segments(circuit, face_data, adj)
        assert all(s > 0 for s in segs)

    def test_kelvin_all_2pi_circuits_have_m2(self, kelvin_cell0):
        """Integration/regression: m=2 present on all Ω≈2π circuits from pipeline.

        Uses equatorial_test + circuit_holonomy as Ω≈2π filter,
        then checks m=2 existence on each selected circuit.
        """
        face_data, adj, cc = kelvin_cell0
        n_f = len(face_data)
        cycles = find_simple_cycles(adj, n_f, max_length=8)

        n_2pi = 0
        n_m2 = 0
        for cyc in cycles:
            eq = equatorial_test(cyc, face_data, adj, cc, n_f)
            if not (eq and eq['is_equatorial']):
                continue
            hol = circuit_holonomy(cyc, face_data, adj)
            if hol is None or abs(hol - 2 * np.pi) > 0.3:
                continue
            n_2pi += 1
            if has_m2(cyc, face_data, adj):
                n_m2 += 1

        assert n_2pi > 0, "No Ω=2π circuits found on Kelvin"
        assert n_m2 == n_2pi, \
            f"m=2 missing on {n_2pi - n_m2}/{n_2pi} Ω=2π circuits"

    def test_c15_z12_best_belt_has_m2(self, c15_z12_cell):
        """C15 Z12 best belt has m=2."""
        face_data, adj, cc = c15_z12_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, _ = result
        assert has_m2(circuit, face_data, adj)

    def test_c15_z16_best_belt_has_m2(self, c15_z16_cell):
        """C15 Z16 best belt has m=2."""
        face_data, adj, cc = c15_z16_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, _ = result
        assert has_m2(circuit, face_data, adj)


# ═══════════════════════════════════════════════════════════════
# 6. FULL BELT SPECTRUM — mesh geometry (integration)
# ═══════════════════════════════════════════════════════════════

class TestBeltSpectrum:
    """Integration: compute_belt_spectrum using face areas and edge couplings."""

    def test_kelvin_belt_spectrum(self, kelvin_best_belt):
        """Kelvin: full belt spectrum has valid structure."""
        circuit, face_data, adj = kelvin_best_belt
        spec = compute_belt_spectrum(circuit, face_data, adj)
        assert spec is not None
        N = spec['N']
        assert len(spec['areas']) == N
        assert all(a > 0 for a in spec['areas'])
        assert all(k >= 0 for k in spec['kappas'])
        assert all(spec['eigenvalues'][i] >= -1e-10 for i in range(N))
        assert 4 in spec['zero_crossings']

    def test_kelvin_belt_dispersion(self, kelvin_best_belt):
        """Kelvin belt: dispersion ratio ω²(m=2)/ω²(m=1) close to uniform."""
        circuit, face_data, adj = kelvin_best_belt
        spec = compute_belt_spectrum(circuit, face_data, adj)
        if spec is None:
            pytest.skip("No belt spectrum")
        N = spec['N']
        evals = spec['eigenvalues']

        omega1_sq = evals[1]
        if omega1_sq < 1e-10:
            pytest.skip("Zero ω₁")

        # Find m=2 by zero-crossings
        m2_candidates = [i for i in range(N)
                         if spec['zero_crossings'][i] == 4]
        assert len(m2_candidates) > 0
        omega2_sq = evals[m2_candidates[0]]

        ratio = omega2_sq / omega1_sq
        expected_ratio = np.sin(2 * np.pi / N)**2 / np.sin(np.pi / N)**2

        assert abs(ratio - expected_ratio) < 0.5, \
            f"Dispersion ratio {ratio:.3f} far from uniform {expected_ratio:.3f}"

    def test_c15_z12_belt_spectrum(self, c15_z12_cell):
        """C15 Z12: belt spectrum valid, m=2 present."""
        face_data, adj, cc = c15_z12_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, _ = result
        spec = compute_belt_spectrum(circuit, face_data, adj)
        assert spec is not None
        assert 4 in spec['zero_crossings']

    def test_c15_z16_belt_spectrum(self, c15_z16_cell):
        """C15 Z16: belt spectrum valid, m=2 present."""
        face_data, adj, cc = c15_z16_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, _ = result
        spec = compute_belt_spectrum(circuit, face_data, adj)
        assert spec is not None
        assert 4 in spec['zero_crossings']


# ═══════════════════════════════════════════════════════════════
# 7. SINE DISPERSION — exact on uniform ring
# ═══════════════════════════════════════════════════════════════

class TestSineDispersion:
    """Generalized eigenvalues ω²_m = 4 sin²(mπ/N) / h² on uniform ring."""

    def test_dispersion_exact_N6(self):
        """N=6 uniform: all eigenvalues match 4sin²(mπ/N)/h²."""
        N = 6
        h = 1.0
        segments = np.ones(N) * h
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        evals = np.sort(np.linalg.eigvalsh(H))

        for m in range(N):
            expected = 4 * np.sin(m * np.pi / N)**2 / h**2
            diffs = [abs(ev - expected) for ev in evals]
            assert min(diffs) < 1e-10, \
                f"m={m}: no eigenvalue near {expected:.6f}"

    def test_dispersion_exact_N7_h2(self):
        """N=7 uniform h=2.0: eigenvalues = 4sin²(mπ/N)/h²."""
        N = 7
        h = 2.0
        segments = np.ones(N) * h
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        evals = np.sort(np.linalg.eigvalsh(H))

        expected = sorted([4 * np.sin(m * np.pi / N)**2 / h**2
                          for m in range(N)])
        np.testing.assert_allclose(evals, expected, atol=1e-10)

    def test_dispersion_nonuniform_deviates(self):
        """Non-uniform ring: eigenvalues deviate from uniform sin² formula."""
        N = 6
        segments = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        K = build_laplacian_1d(segments)
        M = build_mass_matrix_1d(segments)
        m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        H = m_inv_sqrt @ K @ m_inv_sqrt
        evals = np.sort(np.linalg.eigvalsh(H))

        h_mean = np.mean(segments)
        expected_uniform = sorted([4 * np.sin(m * np.pi / N)**2 / h_mean**2
                                   for m in range(N)])

        max_diff = max(abs(evals[i] - expected_uniform[i]) for i in range(N))
        assert max_diff > 0.01, \
            "Non-uniform ring should deviate from uniform formula"
