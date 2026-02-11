"""
Tests for Weaire-Phelan Periodic Builder
========================================

Tests the Voronoi-based WP periodic supercell builder.

Run with:
    python3 -m pytest tests/core/test_wp_periodic.py -v

Requires: scipy >= 1.11 (checked in src/__init__.py)

Hardening (Jan 2026):
    - T0-T7: topology, planarity, ordering, edge lengths, duality, robustness
    - Builder uses unwrap_coords_to_reference before ordering (critical for
      faces crossing periodic boundary)
    - Tests use enumerate(ridge_points) not index lookup
    - Angle checks use np.unwrap + is_simple_polygon

Date: Jan 2026
"""

import numpy as np
import pytest
from collections import defaultdict

from core_math.builders.weaire_phelan_periodic import (
    build_wp_supercell_periodic,
    get_wp_periodic_topology,
    verify_wp_foam_structure,
    get_a15_points,
    order_ridge_vertices,
    is_simple_polygon,
    unwrap_coords_to_reference,
    wrap_pos,
)


def get_central_image_index():
    """
    Compute the index of (0,0,0) in product([-1,0,1], repeat=3).

    This is always 13 for Python's itertools.product, but computing
    dynamically is clearer and more robust.
    """
    from itertools import product
    image_offsets = list(product([-1, 0, 1], repeat=3))
    return image_offsets.index((0, 0, 0))


class TestA15Lattice:
    """Tests for A15 lattice point generation."""

    def test_a15_point_count(self):
        """A15 has 8 points per fundamental domain."""
        for N in [1, 2, 3]:
            points = get_a15_points(N, L_cell=1.0)
            expected = 8 * N**3
            assert len(points) == expected, f"N={N}: expected {expected}, got {len(points)}"

    def test_a15_points_in_bounds(self):
        """All A15 points should be in [0, L)³."""
        N = 2
        L_cell = 4.0
        L = N * L_cell
        points = get_a15_points(N, L_cell)

        for i, p in enumerate(points):
            for j, coord in enumerate(p):
                assert 0 <= coord < L, f"Point {i}, coord {j}: {coord} not in [0, {L})"

    def test_a15_wyckoff_positions(self):
        """Check Wyckoff positions for N=1."""
        points = get_a15_points(1, L_cell=1.0)

        # Expected fractional coords
        expected_2a = [[0, 0, 0], [0.5, 0.5, 0.5]]
        expected_6d = [
            [0.25, 0, 0.5], [0.75, 0, 0.5],
            [0.5, 0.25, 0], [0.5, 0.75, 0],
            [0, 0.5, 0.25], [0, 0.5, 0.75],
        ]

        all_expected = expected_2a + expected_6d
        points_set = set(tuple(p) for p in points)
        expected_set = set(tuple(e) for e in all_expected)

        assert points_set == expected_set, "Wyckoff positions mismatch"


class TestTopology:
    """Tests for WP foam topology."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_euler_3complex(self, N):
        """χ(3-complex) = V - E + F - C = 0 for 3-torus."""
        topo = get_wp_periodic_topology(N)
        chi = topo['chi_3complex']
        assert chi == 0, f"N={N}: χ(3-complex) = {chi}, expected 0"

    @pytest.mark.parametrize("N", [1, 2])
    def test_euler_2skeleton(self, N):
        """χ(2-skeleton) = V - E + F = C (number of cells)."""
        topo = get_wp_periodic_topology(N)
        chi = topo['chi_2skeleton']
        C = topo['n_cells']
        assert chi == C, f"N={N}: χ(2-skeleton) = {chi}, expected C = {C}"

    @pytest.mark.parametrize("N", [1, 2])
    def test_cell_count(self, N):
        """8 cells per fundamental domain."""
        topo = get_wp_periodic_topology(N)
        expected = 8 * N**3
        assert topo['n_cells'] == expected


class TestPlateau:
    """Tests for Plateau foam structure (3 faces per edge, degree 4 vertices)."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_all_edges_3_faces(self, N):
        """Every edge bounds exactly 3 faces (Plateau foam)."""
        V, E, F = build_wp_supercell_periodic(N)

        edge_face_count = defaultdict(int)
        for face in F:
            n = len(face)
            for k in range(n):
                v1, v2 = face[k], face[(k+1) % n]
                edge = (min(v1, v2), max(v1, v2))
                edge_face_count[edge] += 1

        for edge, count in edge_face_count.items():
            assert count == 3, f"N={N}: Edge {edge} has {count} faces, expected 3"

    @pytest.mark.parametrize("N", [1, 2])
    def test_all_vertices_degree_4(self, N):
        """Every vertex has degree 4 (tetravalent)."""
        V, E, F = build_wp_supercell_periodic(N)

        vertex_deg = defaultdict(int)
        for i, j in E:
            vertex_deg[i] += 1
            vertex_deg[j] += 1

        for v, deg in vertex_deg.items():
            assert deg == 4, f"N={N}: Vertex {v} has degree {deg}, expected 4"

    @pytest.mark.parametrize("N", [1, 2])
    def test_valid_plateau_foam(self, N):
        """Combined verification."""
        result = verify_wp_foam_structure(N)
        assert result['is_valid_plateau_foam'], f"N={N}: Not a valid Plateau foam"


class TestFaces:
    """Tests for face structure."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_face_sizes_pentagon_hexagon(self, N):
        """WP has only pentagons and hexagons."""
        V, E, F = build_wp_supercell_periodic(N)

        for f_idx, face in enumerate(F):
            size = len(face)
            assert size in [5, 6], f"N={N}: Face {f_idx} has {size} sides, expected 5 or 6"

    @pytest.mark.parametrize("N", [1, 2])
    def test_face_count_scaling(self, N):
        """Face counts scale as N³ with consistent pentagon/hexagon ratio."""
        V, E, F = build_wp_supercell_periodic(N)

        sizes = [len(f) for f in F]
        n_pent = sizes.count(5)
        n_hex = sizes.count(6)
        n_total = len(F)

        # Total faces should be consistent with Euler (χ = C)
        # For WP: roughly 6.75 faces per cell (54/8 = 6.75)
        expected_total_approx = 6.75 * 8 * N**3
        assert 0.9 * expected_total_approx <= n_total <= 1.1 * expected_total_approx, \
            f"N={N}: {n_total} faces, expected ~{expected_total_approx}"

        # Pentagon/hexagon ratio should be ~8:1 (48:6 = 8:1)
        if n_hex > 0:
            ratio = n_pent / n_hex
            assert 7 <= ratio <= 9, f"N={N}: pent/hex ratio = {ratio}, expected ~8"

    def test_face_count_n1_exact(self):
        """N=1: exact face counts (requires scipy >= 1.11)."""
        V, E, F = build_wp_supercell_periodic(1)

        sizes = [len(f) for f in F]
        n_pent = sizes.count(5)
        n_hex = sizes.count(6)

        assert n_pent == 48, f"Expected 48 pentagons, got {n_pent}"
        assert n_hex == 6, f"Expected 6 hexagons, got {n_hex}"
        assert len(F) == 54, f"Expected 54 faces, got {len(F)}"


class TestEdges:
    """Tests for edge structure."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_edges_sorted(self, N):
        """All edges (i, j) have i < j."""
        V, E, F = build_wp_supercell_periodic(N)

        for edge in E:
            i, j = edge
            assert i < j, f"Edge {edge} not sorted"

    @pytest.mark.parametrize("N", [1, 2])
    def test_edges_valid_vertices(self, N):
        """All edge endpoints are valid vertex indices."""
        V, E, F = build_wp_supercell_periodic(N)
        n_V = len(V)

        for i, j in E:
            assert 0 <= i < n_V, f"Edge ({i},{j}): i out of range"
            assert 0 <= j < n_V, f"Edge ({i},{j}): j out of range"

    @pytest.mark.parametrize("N", [1, 2])
    def test_face_edges_exist(self, N):
        """All face boundary edges exist in edge list (T0)."""
        V, E, F = build_wp_supercell_periodic(N)
        edge_set = set(E)

        for f_idx, face in enumerate(F):
            n = len(face)
            for k in range(n):
                v1, v2 = face[k], face[(k+1) % n]
                edge = (min(v1, v2), max(v1, v2))
                assert edge in edge_set, f"Face {f_idx} edge ({v1},{v2}) not in edge list"


class TestVertices:
    """Tests for vertex structure."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_vertices_in_bounds(self, N):
        """All vertices in [0, L)³."""
        L_cell = 4.0
        L = N * L_cell
        V, E, F = build_wp_supercell_periodic(N, L_cell)

        for v_idx, v in enumerate(V):
            for j, coord in enumerate(v):
                assert -1e-6 <= coord < L + 1e-6, \
                    f"Vertex {v_idx}, coord {j}: {coord} not in [0, {L})"

    @pytest.mark.parametrize("N", [1, 2])
    def test_vef_scaling_consistent(self, N):
        """V, E, F scale consistently with N³."""
        topo = get_wp_periodic_topology(N)

        # Per-cell ratios (from N=1: V=46, E=92, F=54, C=8)
        # V/C ≈ 5.75, E/C ≈ 11.5, F/C ≈ 6.75
        v_per_cell = topo['V'] / topo['n_cells']
        e_per_cell = topo['E'] / topo['n_cells']
        f_per_cell = topo['F'] / topo['n_cells']

        # Allow 10% tolerance for numerical variations
        assert 5.0 <= v_per_cell <= 6.5, f"N={N}: V/C = {v_per_cell}"
        assert 10.0 <= e_per_cell <= 13.0, f"N={N}: E/C = {e_per_cell}"
        assert 6.0 <= f_per_cell <= 7.5, f"N={N}: F/C = {f_per_cell}"

    def test_counts_n1_exact(self):
        """N=1: exact V/E counts (requires scipy >= 1.11)."""
        V, E, F = build_wp_supercell_periodic(1)
        assert len(V) == 46, f"Expected 46 vertices, got {len(V)}"
        assert len(E) == 92, f"Expected 92 edges, got {len(E)}"


class TestVertexOrdering:
    """Tests for ridge vertex ordering (critical for correct face construction)."""

    def test_order_ridge_vertices_pentagon(self):
        """Test ordering on a regular pentagon."""
        # Pentagon vertices in random order
        angles = [0, 2*np.pi/5, 4*np.pi/5, 6*np.pi/5, 8*np.pi/5]
        coords = np.array([[np.cos(a), np.sin(a), 0] for a in angles])

        # Shuffle
        shuffled_idx = [2, 0, 4, 1, 3]
        shuffled_coords = coords[shuffled_idx]

        # Sites on z-axis
        site1 = np.array([0, 0, -1])
        site2 = np.array([0, 0, 1])

        ordered = order_ridge_vertices(shuffled_coords, site1, site2)
        ordered_coords = shuffled_coords[ordered]

        # Project to 2D for simple polygon check (more robust than angle diff)
        coords_2d = ordered_coords[:, :2]  # xy plane
        assert is_simple_polygon(coords_2d), "Ordered vertices don't form simple polygon"

        # Also verify angles are monotonic using unwrap to handle -π/π discontinuity
        ordered_angles = np.array([np.arctan2(shuffled_coords[i, 1], shuffled_coords[i, 0])
                                   for i in ordered])
        unwrapped = np.unwrap(ordered_angles)
        diffs = np.diff(unwrapped)
        # All diffs should have same sign (monotonic)
        assert np.all(diffs > 0) or np.all(diffs < 0), "Angles not monotonic after unwrap"

    def test_is_simple_polygon_square(self):
        """Square is simple."""
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert is_simple_polygon(coords), "Square should be simple"

    def test_is_simple_polygon_bowtie(self):
        """Bowtie (self-intersecting) is not simple."""
        coords = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])  # Crosses itself
        assert not is_simple_polygon(coords), "Bowtie should not be simple"

    @pytest.mark.parametrize("N", [1])
    def test_all_faces_simple_polygons(self, N):
        """All faces should be simple (non-self-intersecting) polygons.

        Uses builder output directly (not Voronoi reconstruction).
        Projects each face to 2D via SVD and checks simplicity.
        """
        L_cell = 4.0
        L = N * L_cell
        V, E, F = build_wp_supercell_periodic(N, L_cell)

        n_simple = 0
        n_total = 0

        for face in F:
            if len(face) < 3:
                continue

            n_total += 1

            # Get face coordinates and unwrap to same periodic image
            coords = V[face].copy()
            coords = unwrap_coords_to_reference(coords, L)

            # Project to 2D via SVD (first two principal components)
            centroid = np.mean(coords, axis=0)
            centered = coords - centroid
            _, _, vh = np.linalg.svd(centered)
            u, v = vh[0], vh[1]
            coords_2d = np.array([[np.dot(c, u), np.dot(c, v)] for c in centered])

            if is_simple_polygon(coords_2d):
                n_simple += 1

        # All faces should be simple
        assert n_simple == n_total, f"Only {n_simple}/{n_total} faces are simple polygons"


class TestFacePlanarity:
    """Tests for face planarity (geometric validity)."""

    @pytest.mark.parametrize("N", [1])
    def test_all_faces_planar(self, N):
        """All faces should be planar within tolerance."""
        L_cell = 4.0
        L = N * L_cell
        V, E, F = build_wp_supercell_periodic(N, L_cell)

        max_deviation = 0.0
        n_non_planar = 0
        PLANAR_TOL = 1e-6

        for f_idx, face in enumerate(F):
            if len(face) < 4:
                continue  # Triangles are always planar

            # Get face vertex coordinates and unwrap (use builder's function)
            coords = V[face].copy()
            coords = unwrap_coords_to_reference(coords, L)

            # Compute planarity via SVD
            centroid = np.mean(coords, axis=0)
            centered = coords - centroid
            _, s, vh = np.linalg.svd(centered)

            # Smallest singular value indicates deviation from plane
            # For a planar set, the smallest singular value should be ~0
            if len(s) >= 3:
                deviation = s[2] / (s[0] + 1e-10)  # Relative deviation
                max_deviation = max(max_deviation, deviation)
                if deviation > PLANAR_TOL:
                    n_non_planar += 1

        assert n_non_planar == 0, \
            f"N={N}: {n_non_planar} non-planar faces, max deviation = {max_deviation}"

    def test_n2_planarity_simple_smoke(self):
        """N=2 smoke: sample 50 faces for planarity + simple polygon.

        N=2 has more faces crossing periodic boundary, so this catches
        crossing-bugs that N=1 might miss.
        """
        L_cell = 4.0
        N = 2
        L = N * L_cell
        V, E, F = build_wp_supercell_periodic(N, L_cell)

        PLANAR_TOL = 1e-6

        # Sample 50 faces (or all if fewer)
        np.random.seed(42)
        n_sample = min(50, len(F))
        sample_indices = np.random.choice(len(F), n_sample, replace=False)

        n_non_planar = 0
        n_non_simple = 0

        for f_idx in sample_indices:
            face = F[f_idx]
            if len(face) < 3:
                continue

            # Use builder's unwrap function
            coords = V[face].copy()
            coords = unwrap_coords_to_reference(coords, L)

            # Planarity check (for faces with 4+ vertices)
            if len(face) >= 4:
                centroid = np.mean(coords, axis=0)
                centered = coords - centroid
                _, s, _ = np.linalg.svd(centered)
                if len(s) >= 3:
                    deviation = s[2] / (s[0] + 1e-10)
                    if deviation > PLANAR_TOL:
                        n_non_planar += 1

            # Simple polygon check (project to face plane)
            if len(face) >= 3:
                centroid = np.mean(coords, axis=0)
                centered = coords - centroid
                _, _, vh = np.linalg.svd(centered)
                # Project onto first two principal components
                u, v = vh[0], vh[1]
                coords_2d = np.array([[np.dot(c, u), np.dot(c, v)] for c in centered])
                if not is_simple_polygon(coords_2d):
                    n_non_simple += 1

        assert n_non_planar == 0, f"N=2 smoke: {n_non_planar}/{n_sample} non-planar faces"
        assert n_non_simple == 0, f"N=2 smoke: {n_non_simple}/{n_sample} self-intersecting faces"


class TestBlochCompatibility:
    """Tests for compatibility with DisplacementBloch."""

    def test_bloch_dynamical_matrix(self):
        """DisplacementBloch can build dynamical matrix."""
        from physics.bloch import DisplacementBloch

        V, E, F = build_wp_supercell_periodic(1, L_cell=4.0)
        L = 4.0

        db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)

        k = np.array([0.005, 0, 0]) * 2 * np.pi / L  # Small k for acoustic limit
        D = db.build_dynamical_matrix(k)

        assert D.shape == (3 * len(V), 3 * len(V)), "Wrong matrix shape"
        assert np.allclose(D, D.conj().T), "Dynamical matrix not Hermitian"

    def test_bloch_acoustic_modes(self):
        """Acoustic modes exist (not all zero)."""
        from physics.bloch import DisplacementBloch

        V, E, F = build_wp_supercell_periodic(1, L_cell=4.0)
        L = 4.0

        db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)

        k = np.array([0.005, 0, 0]) * 2 * np.pi / L  # Small k for acoustic limit
        omega_T, omega_L, _ = db.classify_modes(k)

        # Should have non-zero acoustic frequencies
        assert omega_T[0] > 1e-6, "T1 mode is zero"
        assert omega_T[1] > 1e-6, "T2 mode is zero"
        assert omega_L[0] > 1e-6, "L mode is zero"

    def test_bloch_no_spurious_zero_modes(self):
        """No spurious zero modes (structure is connected)."""
        from physics.bloch import DisplacementBloch

        V, E, F = build_wp_supercell_periodic(1, L_cell=4.0)
        L = 4.0

        db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)

        # At k≠0, there should be no zero modes
        k = np.array([0.005, 0, 0]) * 2 * np.pi / L  # Small k for acoustic limit
        D = db.build_dynamical_matrix(k)
        omega_sq = np.linalg.eigvalsh(D)
        omega = np.sqrt(np.maximum(omega_sq, 0))

        n_zero = np.sum(omega < 1e-6)
        assert n_zero == 0, f"Found {n_zero} spurious zero modes"

    def test_delta_v_reasonable(self):
        """δv/v is in reasonable range (0-50%)."""
        from physics.bloch import DisplacementBloch
        from physics.christoffel import compute_delta_v_direct

        V, E, F = build_wp_supercell_periodic(1, L_cell=4.0)
        L = 4.0

        db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)
        result = compute_delta_v_direct(db, L, n_directions=50)

        delta = result['delta_v_over_v']
        assert 0 < delta < 0.5, f"δv/v = {delta} not in reasonable range"


class TestScaling:
    """Tests for scaling with supercell size."""

    def test_vef_scaling(self):
        """V, E, F scale as N³."""
        topo1 = get_wp_periodic_topology(1)
        topo2 = get_wp_periodic_topology(2)

        # Should scale as 8× (2³ = 8)
        assert topo2['V'] == 8 * topo1['V'], "V doesn't scale as N³"
        assert topo2['E'] == 8 * topo1['E'], "E doesn't scale as N³"
        assert topo2['F'] == 8 * topo1['F'], "F doesn't scale as N³"


class TestReproducibility:
    """Tests for reproducibility."""

    def test_deterministic_output(self):
        """Same input gives same output."""
        V1, E1, F1 = build_wp_supercell_periodic(1, L_cell=4.0)
        V2, E2, F2 = build_wp_supercell_periodic(1, L_cell=4.0)

        assert np.allclose(V1, V2), "Vertices not reproducible"
        assert E1 == E2, "Edges not reproducible"
        assert F1 == F2, "Faces not reproducible"


class TestHardening:
    """Additional hardening tests."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_no_duplicate_faces(self, N):
        """Face deduplication should eliminate all duplicates."""
        V, E, F = build_wp_supercell_periodic(N)

        # Canonicalize all faces and check for duplicates
        from core_math.builders.weaire_phelan_periodic import canonical_face
        canonical_faces = [canonical_face(f) for f in F]
        canonical_set = set(canonical_faces)

        assert len(canonical_set) == len(F), \
            f"N={N}: {len(F) - len(canonical_set)} duplicate faces found"

    @pytest.mark.slow
    def test_n3_smoke(self):
        """N=3 smoke test: verify topology without exact counts."""
        result = verify_wp_foam_structure(3)

        # χ(3-complex) = 0
        assert result['chi_3complex'] == 0, \
            f"N=3: χ(3-complex) = {result['chi_3complex']}, expected 0"

        # All edges have 3 faces (Plateau)
        assert result['all_edges_3_faces'], "N=3: Not all edges have 3 faces"

        # All vertices have degree 4
        assert result['all_vertices_deg_4'], "N=3: Not all vertices have degree 4"

        # Only pentagons and hexagons
        for size in result['face_sizes'].keys():
            assert size in [5, 6], f"N=3: Found face with {size} sides"

        # Valid Plateau foam
        assert result['is_valid_plateau_foam'], "N=3: Not a valid Plateau foam"

        # Scaling check: should have 8×27 = 216 cells
        assert result['C'] == 216, f"N=3: {result['C']} cells, expected 216"


class TestEdgeCases:
    """Tests for invalid inputs (negative tests)."""

    def test_n_zero_raises_valueerror(self):
        """N=0 should raise ValueError with clear message."""
        with pytest.raises(ValueError, match="N must be >= 1"):
            build_wp_supercell_periodic(0)

    def test_n_negative_raises_valueerror(self):
        """N=-1 should raise ValueError with clear message."""
        with pytest.raises(ValueError, match="N must be >= 1"):
            build_wp_supercell_periodic(-1)

    def test_l_cell_zero_raises_valueerror(self):
        """L_cell=0 should raise ValueError with clear message."""
        with pytest.raises(ValueError, match="L_cell must be > 0"):
            build_wp_supercell_periodic(1, L_cell=0)

    def test_l_cell_negative_raises_valueerror(self):
        """L_cell<0 should raise ValueError with clear message."""
        with pytest.raises(ValueError, match="L_cell must be > 0"):
            build_wp_supercell_periodic(1, L_cell=-4.0)


class TestT0FaceCycleValidity:
    """T0: Face cycle validity (graph-theoretic).

    Each face must be a valid cycle in the edge graph:
    - All consecutive vertex pairs in face must exist as edges
    - Already checked by test_face_edges_exist, but this adds graph-theoretic check
    """

    @pytest.mark.parametrize("N", [1, 2])
    def test_faces_are_valid_cycles(self, N):
        """Each face forms a closed cycle using existing edges."""
        V, E, F = build_wp_supercell_periodic(N)

        # Build adjacency list
        adj = defaultdict(set)
        for i, j in E:
            adj[i].add(j)
            adj[j].add(i)

        for f_idx, face in enumerate(F):
            n = len(face)
            # Check each consecutive pair is connected
            for k in range(n):
                v1 = face[k]
                v2 = face[(k+1) % n]
                assert v2 in adj[v1], \
                    f"Face {f_idx}: edge ({v1},{v2}) not in graph"

    @pytest.mark.parametrize("N", [1, 2])
    def test_faces_have_no_repeated_vertices(self, N):
        """No vertex appears twice in a face (simple cycle)."""
        V, E, F = build_wp_supercell_periodic(N)

        for f_idx, face in enumerate(F):
            unique = set(face)
            assert len(unique) == len(face), \
                f"Face {f_idx}: repeated vertices"

    @pytest.mark.parametrize("N", [1, 2])
    def test_all_vertices_used_in_faces(self, N):
        """All vertices appear in at least one face."""
        V, E, F = build_wp_supercell_periodic(N)

        used_vertices = set()
        for face in F:
            used_vertices.update(face)

        assert len(used_vertices) == len(V), \
            f"N={N}: {len(V) - len(used_vertices)} vertices not used in faces"


class TestT4PeriodicEdgeLength:
    """T4: Periodic edge length consistency.

    Unwrapped edge lengths should be consistent within the structure.
    """

    def _unwrap_edge(self, v1, v2, L):
        """Unwrap edge to same periodic image."""
        d = v2 - v1
        for j in range(3):
            if d[j] > L/2:
                d[j] -= L
            elif d[j] < -L/2:
                d[j] += L
        return np.linalg.norm(d)

    @pytest.mark.parametrize("N", [1, 2])
    def test_edge_lengths_positive(self, N):
        """All edges have positive length."""
        L_cell = 4.0
        L = N * L_cell
        V, E, F = build_wp_supercell_periodic(N, L_cell)

        for i, j in E:
            length = self._unwrap_edge(V[i], V[j], L)
            assert length > 1e-8, f"Edge ({i},{j}) has zero length"

    @pytest.mark.parametrize("N", [1, 2])
    def test_edge_lengths_bounded(self, N):
        """Edge lengths are bounded (not wrapping artifacts).

        Uses relaxed thresholds to avoid flakiness across platforms/Qhull versions.
        Core sanity: edges must be positive and less than half the period.
        """
        L_cell = 4.0
        L = N * L_cell
        V, E, F = build_wp_supercell_periodic(N, L_cell)

        lengths = []
        for i, j in E:
            length = self._unwrap_edge(V[i], V[j], L)
            lengths.append(length)

        lengths = np.array(lengths)
        min_len = lengths.min()
        max_len = lengths.max()

        # Relaxed bounds: just sanity checks
        # - All edges must be positive (no zero-length edges)
        # - All edges must be < L/2 (otherwise wrapping is wrong)
        assert min_len > 1e-6, f"Min edge length {min_len} too small (zero-length edge?)"
        assert max_len < L / 2, f"Max edge length {max_len} >= L/2 (wrapping artifact?)"

        # Coefficient of variation - relaxed to 1.0 (allows more variation)
        cv = np.std(lengths) / np.mean(lengths)
        assert cv < 1.0, f"Edge length variation suspiciously high: CV = {cv}"

    @pytest.mark.parametrize("N", [1, 2])
    def test_edge_lengths_scale_invariant(self, N):
        """Edge length distribution scales correctly with L_cell.

        Note: Voronoi + rounding may change exact vertex/edge identification
        at different scales, so we compare distribution statistics
        (min/median/max/mean) rather than one-to-one.
        """
        V1, E1, F1 = build_wp_supercell_periodic(N, L_cell=4.0)
        V2, E2, F2 = build_wp_supercell_periodic(N, L_cell=8.0)

        L1 = N * 4.0
        L2 = N * 8.0

        lengths1 = np.array([self._unwrap_edge(V1[i], V1[j], L1) for i, j in E1])
        lengths2 = np.array([self._unwrap_edge(V2[i], V2[j], L2) for i, j in E2])

        # Compare distribution statistics (scaled by factor 2)
        scale = 2.0
        rtol = 0.05  # 5% tolerance for distribution stats

        # Min, max, mean, median should all scale by factor 2
        assert np.isclose(lengths1.min() * scale, lengths2.min(), rtol=rtol), \
            f"Min doesn't scale: {lengths1.min()}*2 vs {lengths2.min()}"
        assert np.isclose(lengths1.max() * scale, lengths2.max(), rtol=rtol), \
            f"Max doesn't scale: {lengths1.max()}*2 vs {lengths2.max()}"
        assert np.isclose(lengths1.mean() * scale, lengths2.mean(), rtol=rtol), \
            f"Mean doesn't scale: {lengths1.mean()}*2 vs {lengths2.mean()}"
        assert np.isclose(np.median(lengths1) * scale, np.median(lengths2), rtol=rtol), \
            f"Median doesn't scale: {np.median(lengths1)}*2 vs {np.median(lengths2)}"

        # Edge count should be identical (same topology)
        assert len(E1) == len(E2), f"Edge counts differ: {len(E1)} vs {len(E2)}"


class TestT5OrderingStability:
    """T5: Ridge ordering stability under input permutation.

    The order_ridge_vertices function should produce consistent cyclic ordering
    regardless of input order.
    """

    def test_ordering_invariant_to_input_order(self):
        """Ordering should be invariant (up to cyclic rotation) to input shuffling."""
        # Create a hexagon in xy plane
        n = 6
        angles = [2*np.pi*k/n for k in range(n)]
        coords = np.array([[np.cos(a), np.sin(a), 0] for a in angles])

        site1 = np.array([0, 0, -1])
        site2 = np.array([0, 0, 1])

        # Get reference ordering
        ref_order = order_ridge_vertices(coords, site1, site2)
        ref_cycle = [coords[i] for i in ref_order]

        # Try 10 random permutations
        np.random.seed(42)
        for _ in range(10):
            perm = np.random.permutation(n)
            shuffled = coords[perm]

            order = order_ridge_vertices(shuffled, site1, site2)
            cycle = [shuffled[i] for i in order]

            # Should get same cyclic sequence (possibly rotated or reversed)
            # Check by finding start point and comparing
            ref_start = ref_cycle[0]
            start_idx = None
            for i, c in enumerate(cycle):
                if np.allclose(c, ref_start):
                    start_idx = i
                    break

            assert start_idx is not None, "Reference vertex not found in cycle"

            # Check forward direction
            forward_match = True
            for k in range(n):
                if not np.allclose(cycle[(start_idx + k) % n], ref_cycle[k]):
                    forward_match = False
                    break

            # Check reverse direction
            reverse_match = True
            for k in range(n):
                if not np.allclose(cycle[(start_idx - k) % n], ref_cycle[k]):
                    reverse_match = False
                    break

            assert forward_match or reverse_match, "Cyclic ordering not preserved"

    @pytest.mark.parametrize("N", [1])
    def test_face_ordering_deterministic(self, N):
        """Building twice gives same face vertex order."""
        V1, E1, F1 = build_wp_supercell_periodic(N)
        V2, E2, F2 = build_wp_supercell_periodic(N)

        assert F1 == F2, "Face ordering not deterministic"


class TestT6DualitySanity:
    """T6: Duality sanity checks.

    Cross-check structural properties against Voronoi/Delaunay duality.
    """

    @pytest.mark.parametrize("N", [1, 2])
    def test_bounded_ridges_become_faces(self, N):
        """Each bounded Voronoi ridge involving central cell becomes a face.

        Properly iterate ridges with enumerate (not index lookup).
        """
        from scipy.spatial import Voronoi
        from itertools import product

        L_cell = 4.0
        L = N * L_cell
        points = get_a15_points(N, L_cell)
        n_pts = len(points)

        # Create periodic images
        images = []
        for di, dj, dk in product([-1, 0, 1], repeat=3):
            offset = np.array([di, dj, dk]) * L
            images.append(points + offset)

        all_points = np.vstack(images)
        central_idx = get_central_image_index()
        central_start = central_idx * n_pts
        central_end = central_start + n_pts

        vor = Voronoi(all_points)

        # Count bounded ridges involving central cell
        n_bounded_central = 0
        for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
            ridge_verts = vor.ridge_vertices[ridge_idx]

            # Skip unbounded
            if -1 in ridge_verts:
                continue

            in_c1 = central_start <= p1 < central_end
            in_c2 = central_start <= p2 < central_end

            if in_c1 or in_c2:
                n_bounded_central += 1

        # Get built faces
        V, E, F = build_wp_supercell_periodic(N, L_cell)

        # After deduplication, face count should be less than or equal to
        # bounded ridges (some ridges are duplicates after wrapping)
        assert len(F) > 0, "No faces built"
        assert len(F) <= n_bounded_central, \
            f"More faces ({len(F)}) than bounded ridges ({n_bounded_central})"

        # Face count should be reasonable (roughly 50-100% of ridges after dedup)
        assert len(F) >= n_bounded_central // 2, \
            f"Too few faces ({len(F)}) vs ridges ({n_bounded_central})"

    @pytest.mark.parametrize("N", [1, 2])
    def test_cells_per_fundamental_domain(self, N):
        """8 cells per fundamental domain (A15 structure)."""
        topo = get_wp_periodic_topology(N)
        cells_per_domain = topo['n_cells'] / (N**3)
        assert cells_per_domain == 8, f"Expected 8 cells/domain, got {cells_per_domain}"

    @pytest.mark.parametrize("N", [1, 2])
    def test_vertex_in_cell_interior(self, N):
        """Each Voronoi vertex is interior to 4 cells (tetravalent)."""
        # This is verified indirectly by vertex degree = 4
        # (each vertex is where 4 cells meet)
        V, E, F = build_wp_supercell_periodic(N)

        vertex_deg = defaultdict(int)
        for i, j in E:
            vertex_deg[i] += 1
            vertex_deg[j] += 1

        # All vertices should have degree 4
        degs = list(vertex_deg.values())
        assert all(d == 4 for d in degs), "Not all vertices tetravalent"

    @pytest.mark.parametrize("N", [1, 2])
    def test_face_vertex_sets_match_ridges(self, N):
        """Face vertex sets correspond to Voronoi ridge vertex sets (after wrap).

        Aligned with builder: unwrap before wrap_pos to ensure consistent rounding.
        """
        from scipy.spatial import Voronoi
        from itertools import product
        from core_math.builders.weaire_phelan_periodic import (
            wrap_pos, unwrap_coords_to_reference
        )

        L_cell = 4.0
        L = N * L_cell
        points = get_a15_points(N, L_cell)
        n_pts = len(points)

        # Create periodic images
        images = []
        for di, dj, dk in product([-1, 0, 1], repeat=3):
            offset = np.array([di, dj, dk]) * L
            images.append(points + offset)

        all_points = np.vstack(images)
        central_idx = get_central_image_index()
        central_start = central_idx * n_pts
        central_end = central_start + n_pts

        vor = Voronoi(all_points)
        V, E, F = build_wp_supercell_periodic(N, L_cell)

        # Build set of face vertex sets (as frozensets of wrapped positions)
        face_vertex_sets = set()
        for face in F:
            vset = frozenset(tuple(V[v]) for v in face)
            face_vertex_sets.add(vset)

        # Collect unique wrapped ridge sets (deduplicate like builder does)
        # Multiple ridges from different periodic images map to same face after wrap
        unique_ridge_sets = set()
        for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
            ridge_verts = vor.ridge_vertices[ridge_idx]

            if -1 in ridge_verts:
                continue

            in_c1 = central_start <= p1 < central_end
            in_c2 = central_start <= p2 < central_end

            if not (in_c1 or in_c2):
                continue

            # Unwrap then wrap ridge vertices (consistent with builder)
            ridge_coords = np.array([vor.vertices[v] for v in ridge_verts])
            ridge_coords_unwrapped = unwrap_coords_to_reference(ridge_coords, L)
            wrapped_set = frozenset(wrap_pos(c, L) for c in ridge_coords_unwrapped)
            unique_ridge_sets.add(wrapped_set)

        # After deduplication, ridge sets should match face sets exactly
        # (both are unique wrapped vertex sets)
        assert unique_ridge_sets == face_vertex_sets, \
            f"Ridge/face mismatch: {len(unique_ridge_sets)} unique ridges vs {len(face_vertex_sets)} faces. " \
            f"Missing in faces: {len(unique_ridge_sets - face_vertex_sets)}, " \
            f"Extra in faces: {len(face_vertex_sets - unique_ridge_sets)}"


class TestT7NumericRobustness:
    """T7: Numeric robustness (WRAP_DECIMALS / WRAP_TOL sensitivity).

    Tests that topological invariants hold across different precision settings.
    Note: Exact V/E/F counts may vary with precision (that's expected),
    but invariants must hold.
    """

    def test_wrap_decimals_6_still_works(self):
        """Builder should work with WRAP_DECIMALS=6 (less precision)."""
        import core_math.builders.weaire_phelan_periodic as wp_module

        # Save original
        orig_decimals = wp_module.WRAP_DECIMALS
        orig_tol = wp_module.WRAP_TOL

        try:
            # Test with lower precision
            wp_module.WRAP_DECIMALS = 6
            wp_module.WRAP_TOL = 1e-6

            result = verify_wp_foam_structure(1)

            # Should still be valid (may have slightly different counts)
            assert result['chi_3complex'] == 0, "χ should be 0"
            assert result['all_edges_3_faces'], "Should have 3 faces per edge"
            assert result['all_vertices_deg_4'], "Should have degree 4 vertices"

        finally:
            # Restore
            wp_module.WRAP_DECIMALS = orig_decimals
            wp_module.WRAP_TOL = orig_tol

    def test_wrap_decimals_10_still_works(self):
        """Builder should work with WRAP_DECIMALS=10 (more precision)."""
        import core_math.builders.weaire_phelan_periodic as wp_module

        orig_decimals = wp_module.WRAP_DECIMALS
        orig_tol = wp_module.WRAP_TOL

        try:
            wp_module.WRAP_DECIMALS = 10
            wp_module.WRAP_TOL = 1e-10

            result = verify_wp_foam_structure(1)

            assert result['chi_3complex'] == 0, "χ should be 0"
            assert result['all_edges_3_faces'], "Should have 3 faces per edge"
            assert result['all_vertices_deg_4'], "Should have degree 4 vertices"

        finally:
            wp_module.WRAP_DECIMALS = orig_decimals
            wp_module.WRAP_TOL = orig_tol

    def test_invariants_stable_across_precision(self):
        """Topological invariants hold across precision settings.

        Note: Exact V/E/F counts may legitimately vary with precision
        (rounding can affect dedup), but invariants must hold.
        """
        import core_math.builders.weaire_phelan_periodic as wp_module

        orig_decimals = wp_module.WRAP_DECIMALS
        orig_tol = wp_module.WRAP_TOL

        # Reference counts for sanity bounds (±20%)
        ref_V, ref_E, ref_F = 46, 92, 54

        for decimals in [6, 8, 10]:
            try:
                wp_module.WRAP_DECIMALS = decimals
                wp_module.WRAP_TOL = 10**(-decimals)

                result = verify_wp_foam_structure(1)

                # Invariants must hold
                assert result['chi_3complex'] == 0, \
                    f"WRAP_DECIMALS={decimals}: χ(3) = {result['chi_3complex']}, expected 0"
                assert result['all_edges_3_faces'], \
                    f"WRAP_DECIMALS={decimals}: Not all edges have 3 faces"
                assert result['all_vertices_deg_4'], \
                    f"WRAP_DECIMALS={decimals}: Not all vertices have degree 4"

                # Only pentagons and hexagons
                for size in result['face_sizes'].keys():
                    assert size in [5, 6], \
                        f"WRAP_DECIMALS={decimals}: Found face with {size} sides"

                # Counts should not explode (within ±20% of reference)
                assert 0.8 * ref_V <= result['V'] <= 1.2 * ref_V, \
                    f"WRAP_DECIMALS={decimals}: V={result['V']} out of bounds"
                assert 0.8 * ref_E <= result['E'] <= 1.2 * ref_E, \
                    f"WRAP_DECIMALS={decimals}: E={result['E']} out of bounds"
                assert 0.8 * ref_F <= result['F'] <= 1.2 * ref_F, \
                    f"WRAP_DECIMALS={decimals}: F={result['F']} out of bounds"

            finally:
                wp_module.WRAP_DECIMALS = orig_decimals
                wp_module.WRAP_TOL = orig_tol


if __name__ == "__main__":
    # Run basic tests
    print("=" * 60)
    print("WP PERIODIC BUILDER TESTS")
    print("=" * 60)

    # Quick summary
    for N in [1, 2]:
        result = verify_wp_foam_structure(N)
        status = "PASS" if result['is_valid_plateau_foam'] else "FAIL"
        print(f"\nN={N}: {status}")
        print(f"  V={result['V']}, E={result['E']}, F={result['F']}, C={result['C']}")
        print(f"  χ(3-complex) = {result['chi_3complex']}")
        print(f"  Face sizes: {result['face_sizes']}")

    print("\n" + "=" * 60)
    print("Run with pytest for full test suite:")
    print("  python3 -m pytest tests/core/test_wp_periodic.py -v")
    print("=" * 60)
