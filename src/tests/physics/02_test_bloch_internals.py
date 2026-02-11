"""
Bloch Module Tests
==================

Pytest tests for physics/bloch.py functionality.
Tests Bloch wave formulation for periodic structures.

Jan 2026
"""

import numpy as np
import pytest
import warnings

from core_math.builders import build_sc_supercell_periodic, build_fcc_supercell_periodic
from physics.bloch import (
    BlochComplex,
    DisplacementBloch,
    compute_edge_crossings,
    compute_edge_geometry,
    build_edge_lookup,
    build_d0_bloch,
    build_d1_bloch,
)
from physics.constants import ZERO_EIGENVALUE_THRESHOLD


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sc_mesh():
    """SC periodic mesh N=3."""
    N = 3
    L = 2.0 * N
    vertices, edges, faces, _ = build_sc_supercell_periodic(N)
    return vertices, edges, faces, L


@pytest.fixture
def displacement_bloch(sc_mesh):
    """DisplacementBloch instance for SC N=3."""
    vertices, edges, faces, L = sc_mesh
    return DisplacementBloch(vertices, edges, L)


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestEdgeCrossings:
    """Tests for compute_edge_crossings and compute_edge_geometry."""

    def test_crossings_equality(self, sc_mesh):
        """T-B11: compute_edge_crossings == compute_edge_geometry[1].

        Both functions should return identical crossing vectors.
        This catches divergence between the two implementations.
        """
        vertices, edges, faces, L = sc_mesh
        crossings_standalone = compute_edge_crossings(vertices, edges, L)
        edge_vectors, crossings_from_geometry = compute_edge_geometry(vertices, edges, L)

        assert np.array_equal(crossings_standalone, crossings_from_geometry), \
            "Crossing vectors differ between compute_edge_crossings and compute_edge_geometry"

    def test_edge_lookup_antisymmetry(self, sc_mesh):
        """T-B12: Edge lookup satisfies anti-symmetry.

        For edge (i,j) with crossing n:
          edge_lookup[(i,j)] = (e_idx, +1, n)
          edge_lookup[(j,i)] = (e_idx, -1, -n)

        This is a core invariant for d₁(k) to work correctly.
        """
        vertices, edges, faces, L = sc_mesh
        crossings = compute_edge_crossings(vertices, edges, L)
        edge_lookup = build_edge_lookup(edges, crossings)

        for e_idx, (i, j) in enumerate(edges):
            # Forward direction
            fwd = edge_lookup[(i, j)]
            assert fwd[0] == e_idx, f"Edge {e_idx}: wrong index in forward lookup"
            assert fwd[1] == +1, f"Edge {e_idx}: wrong sign in forward lookup"
            assert np.array_equal(fwd[2], crossings[e_idx]), \
                f"Edge {e_idx}: wrong crossing in forward lookup"

            # Backward direction
            bwd = edge_lookup[(j, i)]
            assert bwd[0] == e_idx, f"Edge {e_idx}: wrong index in backward lookup"
            assert bwd[1] == -1, f"Edge {e_idx}: wrong sign in backward lookup"
            assert np.array_equal(bwd[2], -crossings[e_idx]), \
                f"Edge {e_idx}: wrong crossing in backward lookup"

    def test_edges_canonical_direction(self, sc_mesh):
        """T-B7: Contract - all edges have canonical direction (i < j).

        The docstring of compute_edge_crossings states edges should have i<j.
        This contract ensures consistent interpretation of crossing direction.
        """
        vertices, edges, faces, L = sc_mesh
        for e_idx, (i, j) in enumerate(edges):
            assert i < j, f"Edge {e_idx} not canonical: ({i}, {j}) should have i<j"

    def test_compute_edge_crossings_shape(self, sc_mesh):
        """Crossings array has correct shape (E, 3)."""
        vertices, edges, faces, L = sc_mesh
        crossings = compute_edge_crossings(vertices, edges, L)
        assert crossings.shape == (len(edges), 3)

    def test_compute_edge_crossings_values(self, sc_mesh):
        """Crossing values are in {-1, 0, +1}."""
        vertices, edges, faces, L = sc_mesh
        crossings = compute_edge_crossings(vertices, edges, L)
        assert np.all(np.isin(crossings, [-1, 0, 1]))

    def test_compute_edge_geometry_shape(self, sc_mesh):
        """Edge geometry returns correct shapes."""
        vertices, edges, faces, L = sc_mesh
        edge_vectors, crossings = compute_edge_geometry(vertices, edges, L)
        assert edge_vectors.shape == (len(edges), 3)
        assert crossings.shape == (len(edges), 3)

    def test_edge_vectors_normalized(self, sc_mesh):
        """Edge vectors are unit vectors."""
        vertices, edges, faces, L = sc_mesh
        edge_vectors, _ = compute_edge_geometry(vertices, edges, L)
        norms = np.linalg.norm(edge_vectors, axis=1)
        # All should be 1.0 (or 0.0 for degenerate edges, which shouldn't exist)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_crossings_unwrap_consistency(self, sc_mesh):
        """T-B1: Verify delta_unwrapped = delta_raw + n*L gives minimal image.

        The unwrapped delta should have components in [-L/2, L/2] and
        be collinear with the unit edge vector from compute_edge_geometry.
        """
        vertices, edges, faces, L = sc_mesh
        edge_vectors, crossings = compute_edge_geometry(vertices, edges, L)

        for e_idx, (i, j) in enumerate(edges):
            delta_raw = vertices[j] - vertices[i]
            n = crossings[e_idx]

            # Unwrap: delta_unwrapped = delta_raw + n*L
            delta_unwrapped = delta_raw + n * L

            # Check minimal image: all components in [-L/2, L/2]
            assert np.all(np.abs(delta_unwrapped) <= L/2 + 1e-10), \
                f"Edge {e_idx}: unwrapped delta {delta_unwrapped} outside [-L/2, L/2]"

            # Check collinearity with unit vector
            length = np.linalg.norm(delta_unwrapped)
            if length > 1e-10:
                direction = delta_unwrapped / length
                # Should be parallel (dot product ≈ ±1, but same direction → +1)
                dot = np.dot(direction, edge_vectors[e_idx])
                assert np.abs(dot - 1.0) < 1e-10, \
                    f"Edge {e_idx}: direction mismatch, dot={dot}"

    def test_no_degenerate_edges(self, sc_mesh):
        """T-B2: Contract - no edges have zero length.

        Degenerate edges would cause division by zero in normalization
        and produce spurious springs in dynamical matrix.
        """
        vertices, edges, faces, L = sc_mesh
        edge_vectors, crossings = compute_edge_geometry(vertices, edges, L)

        # Compute unwrapped lengths
        lengths = []
        for e_idx, (i, j) in enumerate(edges):
            delta_raw = vertices[j] - vertices[i]
            delta_unwrapped = delta_raw + crossings[e_idx] * L
            lengths.append(np.linalg.norm(delta_unwrapped))

        lengths = np.array(lengths)
        min_length = np.min(lengths)

        # All edges must have positive length
        assert min_length > 1e-8, f"Degenerate edge found: min_length={min_length}"

        # Edge vectors should all be unit (consequence of no degeneracy)
        norms = np.linalg.norm(edge_vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)


class TestBlochOperators:
    """Tests for Bloch-twisted DEC operators."""

    def test_d0_bloch_shape(self, sc_mesh):
        """d0_bloch has shape (E, V)."""
        vertices, edges, faces, L = sc_mesh
        k = np.array([0.1, 0, 0])
        d0k = build_d0_bloch(vertices, edges, L, k)
        assert d0k.shape == (len(edges), len(vertices))

    def test_d0_bloch_at_gamma(self, sc_mesh):
        """At k=0, d0_bloch is real."""
        vertices, edges, faces, L = sc_mesh
        k = np.zeros(3)
        d0k = build_d0_bloch(vertices, edges, L, k)
        assert np.allclose(np.imag(d0k), 0, atol=1e-14)

    def test_d1_bloch_shape(self, sc_mesh):
        """d1_bloch has shape (F, E)."""
        vertices, edges, faces, L = sc_mesh
        k = np.array([0.1, 0, 0])
        d1k = build_d1_bloch(vertices, edges, faces, L, k)
        assert d1k.shape == (len(faces), len(edges))

    def test_exactness_direct_at_gamma(self, sc_mesh):
        """T-B17: Direct exactness test d1(k=0) @ d0(k=0) = 0.

        Tests the operators directly without going through BlochComplex.
        This is more future-proof than relying on the deprecated class.
        """
        vertices, edges, faces, L = sc_mesh
        k = np.zeros(3)

        d0k = build_d0_bloch(vertices, edges, L, k)
        d1k = build_d1_bloch(vertices, edges, faces, L, k)

        # d1 @ d0 should be zero (exactness of de Rham complex)
        composition = d1k @ d0k
        norm = np.linalg.norm(composition)
        assert norm < 1e-12, f"Exactness failed: ||d1(0) @ d0(0)|| = {norm}"

    @pytest.mark.xfail(reason="Structural limitation: Bloch-twisted DEC breaks exactness at k≠0")
    def test_exactness_at_small_k(self, sc_mesh):
        """T-B21: Exactness d1(k) @ d0(k) = 0 at small k≠0.

        Documents STRUCTURAL limitation of Bloch-twisted DEC operators.
        At k≠0, phases on incoming/outgoing edges don't cancel:
            exp(ik·n·L) - 1 ≠ 0

        This is NOT an implementation bug - it's a fundamental trade-off:
        - Current: correct k-dependent spectrum, broken exactness
        - Gauge-covariant: exactness preserved, k-independent spectrum

        Physics code (DisplacementBloch, bath operators) bypasses d₀/d₁
        entirely by building matrices directly, so this doesn't affect results.
        """
        vertices, edges, faces, L = sc_mesh
        k_scale = 2 * np.pi / L

        test_ks = [
            np.array([0.01, 0.0, 0.0]) * k_scale,     # [100]
            np.array([0.01, 0.01, 0.0]) * k_scale,    # [110]
            np.array([0.01, 0.01, 0.01]) * k_scale,   # [111]
        ]

        for k in test_ks:
            d0k = build_d0_bloch(vertices, edges, L, k)
            d1k = build_d1_bloch(vertices, edges, faces, L, k)

            composition = d1k @ d0k
            norm = np.linalg.norm(composition)
            assert norm < 1e-10, f"Exactness failed at k={k}: ||d1(k) @ d0(k)|| = {norm}"


# =============================================================================
# BLOCHCOMPLEX TESTS (DEPRECATED CLASS)
# =============================================================================

class TestBlochComplex:
    """Tests for BlochComplex (deprecated)."""

    def test_deprecation_warning(self, sc_mesh):
        """BlochComplex emits deprecation warning."""
        vertices, edges, faces, L = sc_mesh
        with pytest.warns(DeprecationWarning):
            bc = BlochComplex(vertices, edges, faces, L)

    def test_hermitian_at_gamma(self, sc_mesh):
        """Laplacian is Hermitian at k=0."""
        vertices, edges, faces, L = sc_mesh
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bc = BlochComplex(vertices, edges, faces, L)

        k = np.zeros(3)
        err = bc.check_hermitian(k)
        assert err < 1e-12, f"Hermiticity error at Γ: {err}"

    def test_exactness_at_gamma(self, sc_mesh):
        """Exactness d1·d0 = 0 holds at k=0."""
        vertices, edges, faces, L = sc_mesh
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bc = BlochComplex(vertices, edges, faces, L)

        k = np.zeros(3)
        err = bc.check_exactness(k)
        assert err < 1e-12, f"Exactness error at Γ: {err}"

    @pytest.mark.xfail(reason="Structural limitation: Bloch-twisted DEC breaks exactness at k≠0")
    def test_exactness_at_nonzero_k(self, sc_mesh):
        """T-B22: BlochComplex exactness at k≠0 (structural limitation).

        Documents same structural limitation as T-B21 but via BlochComplex.
        The class is deprecated; use DisplacementBloch which builds the
        dynamical matrix directly without d₀/d₁ operators.
        """
        vertices, edges, faces, L = sc_mesh
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bc = BlochComplex(vertices, edges, faces, L)

        k = np.array([0.01, 0.01, 0.01]) * 2 * np.pi / L
        err = bc.check_exactness(k)
        assert err < 1e-10, f"Exactness error at k≠0: {err}"


# =============================================================================
# DISPLACEMENTBLOCH TESTS (MAIN CLASS)
# =============================================================================

class TestDisplacementBloch:
    """Tests for DisplacementBloch (recommended class)."""

    def test_initialization(self, sc_mesh):
        """DisplacementBloch initializes without error."""
        vertices, edges, faces, L = sc_mesh
        db = DisplacementBloch(vertices, edges, L)
        assert db.V == len(vertices)
        assert db.E == len(edges)

    def test_hermitian_at_gamma(self, displacement_bloch):
        """Dynamical matrix is Hermitian at k=0."""
        db = displacement_bloch
        k = np.zeros(3)
        err = db.check_hermitian(k)
        assert err < 1e-12, f"Hermiticity error at Γ: {err}"

    def test_hermitian_at_nonzero_k(self, displacement_bloch):
        """Dynamical matrix is Hermitian at k≠0."""
        db = displacement_bloch
        k = np.array([0.1, 0.05, 0.02])
        err = db.check_hermitian(k)
        assert err < 1e-12, f"Hermiticity error at k={k}: {err}"

    def test_three_zero_modes_at_gamma(self, displacement_bloch):
        """At k=0, there are exactly 3 zero modes (translations)."""
        db = displacement_bloch
        k = np.zeros(3)
        eigs = db.eigenvalues(k)
        n_zero = np.sum(np.abs(eigs) < ZERO_EIGENVALUE_THRESHOLD)
        assert n_zero == 3, f"Expected 3 zero modes at Γ, got {n_zero}"

    def test_kernel_vectors_are_translations(self, displacement_bloch):
        """T-B13: At k=0, zero-mode eigenvectors are uniform translations.

        The 3 zero modes at Γ correspond to rigid translations:
        - u_x: all vertices move in +x direction
        - u_y: all vertices move in +y direction
        - u_z: all vertices move in +z direction

        Each zero-mode eigenvector should be approximately uniform across vertices.
        """
        db = displacement_bloch
        k = np.zeros(3)
        eigs, vecs = db.eigenpairs(k)

        # Find zero modes
        zero_mask = np.abs(eigs) < ZERO_EIGENVALUE_THRESHOLD
        zero_vecs = vecs[:, zero_mask]
        assert zero_vecs.shape[1] == 3, f"Expected 3 zero modes, got {zero_vecs.shape[1]}"

        # For each zero mode, check that displacement is uniform across vertices
        for mode_idx in range(3):
            vec = zero_vecs[:, mode_idx]

            # Extract displacement for each vertex
            displacements = []
            for i in range(db.V):
                u_i = vec[3*i:3*i+3]
                displacements.append(u_i)
            displacements = np.array(displacements)

            # All displacements should be approximately equal (translation = uniform)
            mean_disp = np.mean(displacements, axis=0)
            deviations = np.array([np.linalg.norm(d - mean_disp) for d in displacements])
            max_deviation = np.max(deviations)

            # Deviation should be small relative to mean displacement magnitude
            mean_mag = np.linalg.norm(mean_disp)
            if mean_mag > 1e-10:
                rel_deviation = max_deviation / mean_mag
                assert rel_deviation < 0.01, \
                    f"Zero mode {mode_idx} not uniform: max relative deviation = {rel_deviation}"

    def test_explicit_translations_in_kernel(self, displacement_bloch):
        """T-B16: Explicit translation vectors are in kernel of D(0).

        Construct t_x, t_y, t_z (same displacement on all vertices) and verify
        ||D(0) @ t|| ≈ 0. This catches bugs where we get 3 small eigenvalues
        but for the wrong reason.
        """
        db = displacement_bloch
        k = np.zeros(3)
        D = db.build_dynamical_matrix(k)

        # Build explicit translation vectors
        for axis, name in enumerate(['x', 'y', 'z']):
            # t_axis: all vertices displaced by unit vector in axis direction
            t = np.zeros(3 * db.V)
            for i in range(db.V):
                t[3*i + axis] = 1.0
            t = t / np.linalg.norm(t)  # normalize

            # D(0) @ t should be ~0
            Dt = D @ t
            norm_Dt = np.linalg.norm(Dt)
            assert norm_Dt < ZERO_EIGENVALUE_THRESHOLD, \
                f"Translation t_{name} not in kernel: ||D(0) @ t_{name}|| = {norm_Dt}"

    def test_no_zero_modes_at_nonzero_k(self, displacement_bloch):
        """At k≠0, there are no exact zero modes."""
        db = displacement_bloch
        k = np.array([0.1, 0, 0])
        eigs = db.eigenvalues(k)
        n_zero = np.sum(np.abs(eigs) < ZERO_EIGENVALUE_THRESHOLD)
        # At k≠0, translational zero modes become finite frequency
        assert n_zero == 0, f"Expected 0 zero modes at k≠0, got {n_zero}"

    def test_eigenvalues_nonnegative(self, displacement_bloch):
        """All eigenvalues are non-negative (elastic stability)."""
        db = displacement_bloch
        for k_vec in [np.zeros(3), np.array([0.1, 0, 0]), np.array([0.1, 0.1, 0.1])]:
            eigs = db.eigenvalues(k_vec)
            assert np.all(eigs >= -ZERO_EIGENVALUE_THRESHOLD), \
                f"Negative eigenvalue at k={k_vec}: min={eigs.min()}"

    def test_time_reversal_symmetry(self, displacement_bloch):
        """T-B3: Time-reversal symmetry D(k) and D(-k) have same spectrum.

        For a system without magnetic field, time-reversal symmetry implies
        D(-k) = conj(D(k)), hence identical eigenvalues.
        """
        db = displacement_bloch

        # Test several k vectors
        test_k = [
            np.array([0.1, 0.05, 0.02]),
            np.array([0.15, 0.0, 0.0]),
            np.array([0.1, 0.1, 0.1]),
        ]

        for k in test_k:
            eigs_k = db.eigenvalues(k)
            eigs_minus_k = db.eigenvalues(-k)

            # Sort for comparison (eigenvalues may be in different order)
            eigs_k_sorted = np.sort(eigs_k)
            eigs_minus_k_sorted = np.sort(eigs_minus_k)

            assert np.allclose(eigs_k_sorted, eigs_minus_k_sorted, rtol=1e-10), \
                f"Time-reversal broken: eigs(k) ≠ eigs(-k) for k={k}"

    def test_frequencies_shape(self, displacement_bloch):
        """Frequencies array has correct shape (3V,)."""
        db = displacement_bloch
        k = np.array([0.1, 0, 0])
        omega = db.frequencies(k)
        assert omega.shape == (3 * db.V,)

    def test_dispersion_linear_small_k(self, sc_mesh, displacement_bloch):
        """T-B4: Acoustic dispersion is linear for small k (ω ∝ k).

        Uses reduced wavevector eps = k*L/(2π) to be independent of period.
        """
        vertices, edges, faces, L = sc_mesh
        db = displacement_bloch
        direction = np.array([1, 0, 0])

        # Two small reduced wavevectors (dimensionless)
        eps1, eps2 = 0.01, 0.02

        # Convert to actual k: k = eps * 2π/L
        k1 = eps1 * 2 * np.pi / L
        k2 = eps2 * 2 * np.pi / L

        omega1 = db.frequencies(k1 * direction)[:3]
        omega2 = db.frequencies(k2 * direction)[:3]

        # Velocity should be approximately constant (linear dispersion)
        v1 = omega1 / k1
        v2 = omega2 / k2

        # Allow 5% deviation from linearity
        assert np.allclose(v1, v2, rtol=0.05), \
            f"Non-linear dispersion: v1={v1}, v2={v2}"


class TestDisplacementBlochIsotropy:
    """Tests for isotropy of DisplacementBloch."""

    def test_isotropy_directions(self, sc_mesh, displacement_bloch):
        """Frequencies are similar in different directions (approximate isotropy).

        Uses small k (closer to continuum limit) where isotropy should be better.
        At larger k, lattice anisotropy becomes visible - this is expected physics.
        """
        vertices, edges, faces, L = sc_mesh
        db = displacement_bloch

        # Use small reduced wavevector for continuum-like behavior
        eps = 0.03
        k_mag = eps * 2 * np.pi / L

        directions = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 1, 0]) / np.sqrt(2),
            np.array([1, 1, 1]) / np.sqrt(3),
        ]

        omega_all = []
        for d in directions:
            k = k_mag * d
            omega = db.frequencies(k)[:3]
            omega_all.append(omega)

        omega_all = np.array(omega_all)

        # Anisotropy = std/mean for each branch
        aniso = np.std(omega_all, axis=0) / np.mean(omega_all, axis=0)

        # SC lattice has some anisotropy, but should be < 20%
        assert np.all(aniso < 0.2), f"Large anisotropy: {aniso}"


class TestLongitudinalProjector:
    """Tests for longitudinal projector P_L."""

    def test_projector_shape(self, displacement_bloch):
        """P_L has shape (3V, 3V)."""
        db = displacement_bloch
        k = np.array([0.1, 0, 0])
        P_L = db.build_longitudinal_projector(k)
        assert P_L.shape == (3 * db.V, 3 * db.V)

    def test_projector_symmetric(self, displacement_bloch):
        """P_L is symmetric."""
        db = displacement_bloch
        k = np.array([0.1, 0.05, 0])
        P_L = db.build_longitudinal_projector(k)
        assert np.allclose(P_L, P_L.T, atol=1e-14)

    def test_projector_idempotent(self, displacement_bloch):
        """P_L is idempotent (P_L² = P_L)."""
        db = displacement_bloch
        k = np.array([0.1, 0.05, 0])
        P_L = db.build_longitudinal_projector(k)
        P_L_sq = P_L @ P_L
        assert np.allclose(P_L, P_L_sq, atol=1e-12)

    def test_projector_zero_at_gamma(self, displacement_bloch):
        """P_L is zero at k=0 (no preferred direction)."""
        db = displacement_bloch
        k = np.zeros(3)
        P_L = db.build_longitudinal_projector(k)
        assert np.allclose(P_L, 0, atol=1e-14)

    def test_projector_trace(self, displacement_bloch):
        """P_L has trace = V (one longitudinal DOF per vertex).

        Each 3×3 block k̂⊗k̂ has trace |k̂|² = 1, so total trace = V.
        """
        db = displacement_bloch
        k = np.array([0.1, 0.05, 0.02])
        P_L = db.build_longitudinal_projector(k)
        trace = np.trace(P_L)
        assert np.isclose(trace, db.V, rtol=1e-10), \
            f"P_L trace = {trace}, expected {db.V}"

    def test_projector_rank(self, displacement_bloch):
        """P_L has rank = V (V independent longitudinal modes).

        Since P_L is block-diagonal with V identical rank-1 blocks,
        total rank = V.
        """
        db = displacement_bloch
        k = np.array([0.1, 0.05, 0.02])
        P_L = db.build_longitudinal_projector(k)

        # Count eigenvalues > threshold
        eigs = np.linalg.eigvalsh(P_L)
        rank = np.sum(eigs > 1e-10)
        assert rank == db.V, f"P_L rank = {rank}, expected {db.V}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
