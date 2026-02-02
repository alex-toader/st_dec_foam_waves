"""
Tests for C15 Laves periodic foam builder.

Rule 88: Topology counts (V=136, E=272, F=160) are exact for our target environment.

Validates:
    - Plateau foam structure (deg-4, 3 faces/edge, Euler)
    - Correct topology numbers (V, E, F, C)
    - N-scaling (×8 for N doubling)
    - Face distribution (pentagons + hexagons)
    - Tripwire: edge lengths, planarity, simple polygons
    - Voronoi validity: equidistance, bisector property
    - Invariances: permutation, translation

Run with:
    python3 -m pytest tests/core/test_c15_periodic.py -v

Date: Jan 2026
"""

import pytest
import numpy as np
from collections import defaultdict

from core_math_v2.builders.c15_periodic import (
    build_c15_supercell_periodic,
    verify_c15_foam_structure,
    get_c15_points,
    unwrap_coords_to_reference,
    canonical_face,
    WRAP_TOL,
)


# ============================================================================
# Helper functions
# ============================================================================

def is_simple_polygon(coords_2d: np.ndarray) -> bool:
    """Check if a 2D polygon is simple (non-self-intersecting)."""
    n = len(coords_2d)
    if n < 4:
        return True  # Triangle is always simple

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def segments_intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            A, B = coords_2d[i], coords_2d[(i + 1) % n]
            C, D = coords_2d[j], coords_2d[(j + 1) % n]
            if segments_intersect(A, B, C, D):
                return False
    return True


def unwrap_edge(v1, v2, L):
    """Unwrap edge to same periodic image and return length."""
    d = v2 - v1
    for j in range(3):
        if d[j] > L/2:
            d[j] -= L
        elif d[j] < -L/2:
            d[j] += L
    return np.linalg.norm(d)


# ============================================================================
# Structure tests
# ============================================================================

class TestC15Structure:
    """Test C15 lattice point generation."""

    def test_c15_sites_count(self):
        """C15 has 24 sites per unit cell."""
        points = get_c15_points(N=1, L_cell=1.0)
        assert len(points) == 24, f"Expected 24 sites, got {len(points)}"

    def test_c15_sites_in_unit_cell(self):
        """All sites should be in [0, L)³."""
        L = 4.0
        points = get_c15_points(N=1, L_cell=L)
        assert np.all(points >= 0), "Some points have negative coordinates"
        assert np.all(points < L), f"Some points outside [0, {L})"

    def test_c15_sites_unique(self):
        """All sites should be unique (min distance > 0.01 * L_cell)."""
        L_cell = 1.0
        min_dist_threshold = 0.01 * L_cell  # Scale-aware threshold
        points = get_c15_points(N=1, L_cell=L_cell)
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                assert dist > min_dist_threshold, \
                    f"Sites {i} and {j} too close: dist={dist} < {min_dist_threshold}"


# ============================================================================
# Topology tests
# ============================================================================

class TestC15Topology:
    """Test C15 foam topology."""

    def test_n1_topology_exact(self):
        """N=1: V=136, E=272, F=160, C=24."""
        vertices, edges, faces, _ = build_c15_supercell_periodic(N=1)

        V, E, F = len(vertices), len(edges), len(faces)
        C = 24

        assert V == 136, f"V={V}, expected 136"
        assert E == 272, f"E={E}, expected 272"
        assert F == 160, f"F={F}, expected 160"
        assert V - E + F - C == 0, f"χ = {V-E+F-C}, expected 0"

    def test_n2_scaling(self):
        """N=2: scales by 8 (V×8, E×8, F×8, C×8)."""
        v1, e1, f1, _ = build_c15_supercell_periodic(N=1)
        v2, e2, f2, _ = build_c15_supercell_periodic(N=2)

        V1, E1, F1 = len(v1), len(e1), len(f1)
        V2, E2, F2 = len(v2), len(e2), len(f2)

        assert V2 == 8 * V1, f"V2={V2}, expected {8*V1}"
        assert E2 == 8 * E1, f"E2={E2}, expected {8*E1}"
        assert F2 == 8 * F1, f"F2={F2}, expected {8*F1}"

    @pytest.mark.parametrize("N", [1, 2])
    def test_euler_characteristic(self, N):
        """χ(3-complex) = V - E + F - C = 0 for all N."""
        vertices, edges, faces, _ = build_c15_supercell_periodic(N)
        V, E, F = len(vertices), len(edges), len(faces)
        C = 24 * N**3
        chi = V - E + F - C
        assert chi == 0, f"N={N}: χ = {chi}, expected 0"


# ============================================================================
# Plateau foam tests
# ============================================================================

class TestC15PlateauFoam:
    """Test Plateau foam properties."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_all_vertices_degree_4(self, N):
        """Every vertex has exactly 4 edges (tetravalent)."""
        vertices, edges, faces, _ = build_c15_supercell_periodic(N)

        vertex_deg = defaultdict(int)
        for i, j in edges:
            vertex_deg[i] += 1
            vertex_deg[j] += 1

        degrees = set(vertex_deg.values())
        assert degrees == {4}, f"N={N}: Vertex degrees: {degrees}, expected {{4}}"

    @pytest.mark.parametrize("N", [1, 2])
    def test_all_edges_3_faces(self, N):
        """Every edge bounds exactly 3 faces."""
        vertices, edges, faces, _ = build_c15_supercell_periodic(N)

        edge_face = defaultdict(int)
        for face in faces:
            n = len(face)
            for k in range(n):
                v1, v2 = face[k], face[(k+1) % n]
                edge = (min(v1, v2), max(v1, v2))
                edge_face[edge] += 1

        counts = set(edge_face.values())
        assert counts == {3}, f"N={N}: Edge-face counts: {counts}, expected {{3}}"

    def test_valid_plateau_foam(self):
        """Full Plateau foam validation."""
        result = verify_c15_foam_structure(N=1)
        assert result['is_valid_plateau_foam'], "C15 is not a valid Plateau foam"


# ============================================================================
# Face tests
# ============================================================================

class TestC15Faces:
    """Test face properties."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_face_sizes(self, N):
        """Faces are pentagons and hexagons only."""
        vertices, edges, faces, _ = build_c15_supercell_periodic(N)

        face_sizes = set(len(f) for f in faces)
        assert face_sizes == {5, 6}, f"N={N}: Face sizes: {face_sizes}, expected {{5, 6}}"

    def test_pentagon_majority(self):
        """~90% of faces are pentagons."""
        vertices, edges, faces, _ = build_c15_supercell_periodic(N=1)

        n_pent = sum(1 for f in faces if len(f) == 5)
        pct_pent = 100 * n_pent / len(faces)

        assert 85 < pct_pent < 95, f"Pentagon %: {pct_pent:.1f}%, expected ~90%"

    def test_face_counts_n1(self):
        """N=1: 144 pentagons, 16 hexagons."""
        result = verify_c15_foam_structure(N=1)

        assert result['face_sizes'].get(5, 0) == 144, \
            f"Pentagons: {result['face_sizes'].get(5, 0)}, expected 144"
        assert result['face_sizes'].get(6, 0) == 16, \
            f"Hexagons: {result['face_sizes'].get(6, 0)}, expected 16"


# ============================================================================
# Tripwire: Edge length tests
# ============================================================================

class TestC15EdgeLengths:
    """Tripwire: edge length sanity checks."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_edge_lengths_positive(self, N):
        """All edges have positive length (no collapsed edges)."""
        L_cell = 4.0
        L = N * L_cell
        V, E, F, _ = build_c15_supercell_periodic(N, L_cell)

        for i, j in E:
            length = unwrap_edge(V[i], V[j], L)
            assert length > 1e-8, f"Edge ({i},{j}) has zero length"

    @pytest.mark.parametrize("N", [1, 2])
    def test_edge_lengths_bounded(self, N):
        """Edge lengths < L/2 (no wrapping artifacts)."""
        L_cell = 4.0
        L = N * L_cell
        V, E, F, _ = build_c15_supercell_periodic(N, L_cell)

        lengths = [unwrap_edge(V[i], V[j], L) for i, j in E]
        min_len = min(lengths)
        max_len = max(lengths)

        assert min_len > 1e-6, f"Min edge length {min_len} too small"
        assert max_len < L / 2, f"Max edge length {max_len} >= L/2 (wrapping artifact)"


# ============================================================================
# Tripwire: Vertex distance gap (prevents WRAP_TOL merging distinct vertices)
# ============================================================================

class TestC15VertexDistanceGap:
    """Tripwire: ensure WRAP_TOL doesn't accidentally merge distinct vertices."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_vertex_min_distance_gap(self, N):
        """Min distance between distinct vertices >> WRAP_TOL.

        If two vertices are closer than ~100×WRAP_TOL, rounding could merge them,
        corrupting topology. This test ensures a clear gap exists.
        """
        from scipy.spatial import cKDTree

        L_cell = 4.0
        L = N * L_cell
        V, E, F, _ = build_c15_supercell_periodic(N, L_cell)

        # Build KDTree with periodic images for correct min distance
        all_V = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    offset = np.array([di, dj, dk]) * L
                    all_V.append(V + offset)
        all_V = np.vstack(all_V)

        tree = cKDTree(all_V)

        # For each vertex in central cell, find distance to nearest OTHER vertex
        min_distances = []
        for i, v in enumerate(V):
            dists, _ = tree.query(v, k=2)  # k=2: self + nearest neighbor
            min_distances.append(dists[1])  # dists[0] = 0 (self)

        min_dist = min(min_distances)
        gap_ratio = min_dist / WRAP_TOL

        # Gap should be >> 100 to be safe from rounding collisions
        assert gap_ratio > 100, \
            f"N={N}: min vertex distance {min_dist:.2e} only {gap_ratio:.0f}× WRAP_TOL ({WRAP_TOL:.0e})"


# ============================================================================
# Tripwire: Planarity and simple polygon tests
# ============================================================================

class TestC15FacePlanarity:
    """Tripwire: face planarity and simple polygon checks."""

    @pytest.mark.parametrize("N", [1])
    def test_all_faces_planar(self, N):
        """All faces should be planar within tolerance."""
        L_cell = 4.0
        L = N * L_cell
        V, E, F, _ = build_c15_supercell_periodic(N, L_cell)

        PLANAR_TOL = 1e-6
        n_non_planar = 0

        for face in F:
            if len(face) < 4:
                continue  # Triangles are always planar

            coords = V[face].copy()
            coords = unwrap_coords_to_reference(coords, L)

            centroid = np.mean(coords, axis=0)
            centered = coords - centroid
            _, s, _ = np.linalg.svd(centered)

            if len(s) >= 3:
                deviation = s[2] / (s[0] + 1e-10)
                if deviation > PLANAR_TOL:
                    n_non_planar += 1

        assert n_non_planar == 0, f"N={N}: {n_non_planar} non-planar faces"

    @pytest.mark.parametrize("N", [1])
    def test_all_faces_simple_polygons(self, N):
        """All faces should be simple (non-self-intersecting) polygons."""
        L_cell = 4.0
        L = N * L_cell
        V, E, F, _ = build_c15_supercell_periodic(N, L_cell)

        n_non_simple = 0

        for face in F:
            coords = V[face].copy()
            coords = unwrap_coords_to_reference(coords, L)

            # Project to 2D via SVD
            centroid = np.mean(coords, axis=0)
            centered = coords - centroid
            _, _, vh = np.linalg.svd(centered)
            coords_2d = centered @ vh[:2].T

            if not is_simple_polygon(coords_2d):
                n_non_simple += 1

        assert n_non_simple == 0, f"N={N}: {n_non_simple} non-simple polygon faces"

    def test_n2_planarity_simple_smoke(self):
        """N=2 smoke: sample 50 faces for planarity + simple polygon."""
        L_cell = 4.0
        N = 2
        L = N * L_cell
        V, E, F, _ = build_c15_supercell_periodic(N, L_cell)

        PLANAR_TOL = 1e-6

        np.random.seed(42)
        n_sample = min(50, len(F))
        sample_indices = np.random.choice(len(F), n_sample, replace=False)

        n_bad = 0
        for idx in sample_indices:
            face = F[idx]
            coords = V[face].copy()
            coords = unwrap_coords_to_reference(coords, L)

            centroid = np.mean(coords, axis=0)
            centered = coords - centroid
            _, s, vh = np.linalg.svd(centered)

            # Check planarity
            if len(s) >= 3 and s[2] / (s[0] + 1e-10) > PLANAR_TOL:
                n_bad += 1
                continue

            # Check simple polygon
            coords_2d = centered @ vh[:2].T
            if not is_simple_polygon(coords_2d):
                n_bad += 1

        assert n_bad == 0, f"N=2: {n_bad}/{n_sample} sampled faces failed planarity/simple"


# ============================================================================
# Determinism test
# ============================================================================

class TestC15VertexUsage:
    """Test that all vertices are used."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_all_vertices_used_in_faces(self, N):
        """All vertices should appear in at least one face."""
        V, E, F, _ = build_c15_supercell_periodic(N)

        used = set()
        for face in F:
            used.update(face)

        assert len(used) == len(V), \
            f"N={N}: {len(V) - len(used)} vertices not used in faces"

    @pytest.mark.parametrize("N", [1, 2])
    def test_all_vertices_used_in_edges(self, N):
        """All vertices should appear in at least one edge."""
        V, E, F, _ = build_c15_supercell_periodic(N)

        used = set()
        for i, j in E:
            used.add(i)
            used.add(j)

        assert len(used) == len(V), \
            f"N={N}: {len(V) - len(used)} vertices not used in edges"


class TestC15Determinism:
    """Test builder determinism."""

    def test_deterministic_output(self):
        """Same input gives same output."""
        v1, e1, f1, _ = build_c15_supercell_periodic(N=1, L_cell=4.0)
        v2, e2, f2, _ = build_c15_supercell_periodic(N=1, L_cell=4.0)

        assert len(v1) == len(v2), "Vertex count differs"
        assert len(e1) == len(e2), "Edge count differs"
        assert len(f1) == len(f2), "Face count differs"

        # Compare vertex sets (order-independent)
        set1 = {tuple(np.round(v, 8)) for v in v1}
        set2 = {tuple(np.round(v, 8)) for v in v2}
        assert set1 == set2, "Vertex position sets differ"


# ============================================================================
# Robustness tests
# ============================================================================

class TestC15Robustness:
    """Robustness tests for CI stability."""

    def test_topology_invariant_to_L_cell_scale(self):
        """V/E/F counts should be same for different L_cell values."""
        N = 1

        v1, e1, f1, _ = build_c15_supercell_periodic(N, L_cell=1.0)
        v2, e2, f2, _ = build_c15_supercell_periodic(N, L_cell=4.0)
        v3, e3, f3, _ = build_c15_supercell_periodic(N, L_cell=10.0)

        assert len(v1) == len(v2) == len(v3), \
            f"V differs: {len(v1)}, {len(v2)}, {len(v3)}"
        assert len(e1) == len(e2) == len(e3), \
            f"E differs: {len(e1)}, {len(e2)}, {len(e3)}"
        assert len(f1) == len(f2) == len(f3), \
            f"F differs: {len(f1)}, {len(f2)}, {len(f3)}"

    def test_face_set_canonical_deterministic(self):
        """Canonical face set should be identical across runs."""
        v1, e1, f1, _ = build_c15_supercell_periodic(N=1, L_cell=4.0)
        v2, e2, f2, _ = build_c15_supercell_periodic(N=1, L_cell=4.0)

        # Use same canonical_face as builder (single source of truth)
        set1 = {canonical_face(f) for f in f1}
        set2 = {canonical_face(f) for f in f2}

        assert set1 == set2, "Canonical face sets differ between runs"

    def test_invariants_hold_for_multiple_seeds(self):
        """Core invariants should hold regardless of any internal randomness.

        (Currently builder is deterministic, but this guards against
        future changes that might introduce randomness.)
        """
        for _ in range(3):
            result = verify_c15_foam_structure(N=1)

            assert result['chi_3complex'] == 0, "χ ≠ 0"
            assert result['all_edges_3_faces'], "Not all edges have 3 faces"
            assert result['all_vertices_deg_4'], "Not all vertices degree 4"
            assert set(result['face_sizes'].keys()) <= {5, 6}, "Non-pentagon/hexagon faces"


class TestC15VoronoiValidity:
    """Validate that vertices and faces are geometrically correct Voronoi objects."""

    def test_vertex_equidistance(self):
        """Voronoi vertices should be equidistant from their 4 nearest sites.

        For a valid Voronoi vertex in 3D:
        - The 4 closest generator sites are equidistant
        - The 5th closest site is strictly farther
        """
        N, L_cell = 1, 4.0
        L = N * L_cell

        V, E, F, _ = build_c15_supercell_periodic(N, L_cell)
        sites = get_c15_points(N, L_cell)

        # Create periodic images of sites for distance computation
        all_sites = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    offset = np.array([di, dj, dk]) * L
                    all_sites.extend(sites + offset)
        all_sites = np.array(all_sites)

        # Sample 20 vertices
        np.random.seed(42)
        n_sample = min(20, len(V))
        sample_idx = np.random.choice(len(V), n_sample, replace=False)

        n_valid = 0
        for idx in sample_idx:
            vertex = V[idx]

            # Compute distances to all sites
            dists = np.linalg.norm(all_sites - vertex, axis=1)
            sorted_dists = np.sort(dists)

            # First 4 distances should be approximately equal
            d_top4 = sorted_dists[:4]
            d_5th = sorted_dists[4]

            # Check equidistance of top 4 (relative tolerance)
            mean_d = np.mean(d_top4)
            max_dev = np.max(np.abs(d_top4 - mean_d)) / mean_d

            # Check 5th is strictly farther
            gap = (d_5th - mean_d) / mean_d

            if max_dev < 1e-6 and gap > 1e-6:
                n_valid += 1

        # At least 90% should be valid (some boundary cases may have issues)
        assert n_valid >= 0.9 * n_sample, \
            f"Only {n_valid}/{n_sample} vertices are valid Voronoi vertices"

    def test_face_bisector_property(self):
        """Face vertices should lie on the bisector plane of the two generator sites.

        For each face (separating sites p1, p2), every vertex x should satisfy:
        |x - p1|² = |x - p2|² (i.e., equidistant from both sites)
        """
        N, L_cell = 1, 4.0
        L = N * L_cell

        V, E, F, _ = build_c15_supercell_periodic(N, L_cell)
        sites = get_c15_points(N, L_cell)

        # Create periodic images
        all_sites = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    offset = np.array([di, dj, dk]) * L
                    all_sites.extend(sites + offset)
        all_sites = np.array(all_sites)

        # Sample 30 faces
        np.random.seed(43)
        n_sample = min(30, len(F))
        sample_idx = np.random.choice(len(F), n_sample, replace=False)

        n_valid_faces = 0
        for idx in sample_idx:
            face = F[idx]
            face_vertices = V[face]

            # Unwrap face vertices
            face_unwrapped = unwrap_coords_to_reference(face_vertices.copy(), L)
            centroid = np.mean(face_unwrapped, axis=0)

            # Find the two closest sites to the face centroid
            dists_to_centroid = np.linalg.norm(all_sites - centroid, axis=1)
            closest_two = np.argsort(dists_to_centroid)[:2]
            p1, p2 = all_sites[closest_two[0]], all_sites[closest_two[1]]

            # Check bisector property for each vertex
            all_on_bisector = True
            for v in face_unwrapped:
                d1_sq = np.sum((v - p1)**2)
                d2_sq = np.sum((v - p2)**2)
                diff = abs(d1_sq - d2_sq)
                # Relative tolerance
                mean_d_sq = (d1_sq + d2_sq) / 2
                if diff / mean_d_sq > 1e-6:
                    all_on_bisector = False
                    break

            if all_on_bisector:
                n_valid_faces += 1

        # At least 90% should be valid
        assert n_valid_faces >= 0.9 * n_sample, \
            f"Only {n_valid_faces}/{n_sample} faces satisfy bisector property"


class TestC15PermutationInvariance:
    """Voronoi output should be invariant to input point ordering."""

    def test_permutation_invariance_topology(self):
        """V/E/F counts should be identical for permuted input points."""
        N, L_cell = 1, 4.0

        # Original ordering
        points_orig = get_c15_points(N, L_cell)
        v1, e1, f1, _ = build_c15_supercell_periodic(N, L_cell, points=points_orig)

        # Permuted ordering
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(len(points_orig))
        points_perm = points_orig[perm]
        v2, e2, f2, _ = build_c15_supercell_periodic(N, L_cell, points=points_perm)

        # Topology must be identical
        assert len(v1) == len(v2), f"V differs: {len(v1)} vs {len(v2)}"
        assert len(e1) == len(e2), f"E differs: {len(e1)} vs {len(e2)}"
        assert len(f1) == len(f2), f"F differs: {len(f1)} vs {len(f2)}"

    def test_permutation_invariance_geometry(self):
        """Vertex and face sets should be identical for permuted input points."""
        N, L_cell = 1, 4.0

        # Original ordering
        points_orig = get_c15_points(N, L_cell)
        v1, e1, f1, _ = build_c15_supercell_periodic(N, L_cell, points=points_orig)

        # Permuted ordering
        rng = np.random.default_rng(seed=123)
        perm = rng.permutation(len(points_orig))
        points_perm = points_orig[perm]
        v2, e2, f2, _ = build_c15_supercell_periodic(N, L_cell, points=points_perm)

        # Vertex position sets must be identical
        set1 = {tuple(np.round(v, 8)) for v in v1}
        set2 = {tuple(np.round(v, 8)) for v in v2}
        assert set1 == set2, "Vertex position sets differ"

        # Note: face vertex indices may differ, but face vertex POSITIONS should match
        # We compare face geometries by their vertex coordinate sets
        def face_geometry(face, vertices):
            return frozenset(tuple(np.round(vertices[i], 8)) for i in face)

        geom1 = {face_geometry(f, v1) for f in f1}
        geom2 = {face_geometry(f, v2) for f in f2}
        assert geom1 == geom2, "Face geometries differ"


class TestC15TranslationInvariance:
    """Voronoi output should be invariant to translation of seed points."""

    def test_translation_invariance_topology(self):
        """V/E/F counts should be identical for translated seed points.

        Shift all C15 sites by a fractional vector (mod L) and verify
        the topology is unchanged. This proves no origin bias.
        """
        N, L_cell = 1, 4.0
        L = N * L_cell

        # Original
        points_orig = get_c15_points(N, L_cell)
        v1, e1, f1, _ = build_c15_supercell_periodic(N, L_cell, points=points_orig)

        # Translated by fractional vector (0.1, 0.2, 0.3) * L
        shift = np.array([0.1, 0.2, 0.3]) * L
        points_shifted = (points_orig + shift) % L
        v2, e2, f2, _ = build_c15_supercell_periodic(N, L_cell, points=points_shifted)

        # Topology must be identical
        assert len(v1) == len(v2), f"V differs: {len(v1)} vs {len(v2)}"
        assert len(e1) == len(e2), f"E differs: {len(e1)} vs {len(e2)}"
        assert len(f1) == len(f2), f"F differs: {len(f1)} vs {len(f2)}"

    def test_translation_invariance_face_sizes(self):
        """Face size distribution should be identical for translated points."""
        N, L_cell = 1, 4.0
        L = N * L_cell

        # Original
        points_orig = get_c15_points(N, L_cell)
        _, _, f1, _ = build_c15_supercell_periodic(N, L_cell, points=points_orig)

        # Translated by different fractional vector
        shift = np.array([0.37, 0.53, 0.71]) * L
        points_shifted = (points_orig + shift) % L
        _, _, f2, _ = build_c15_supercell_periodic(N, L_cell, points=points_shifted)

        # Face size counts must be identical
        sizes1 = defaultdict(int)
        sizes2 = defaultdict(int)
        for f in f1:
            sizes1[len(f)] += 1
        for f in f2:
            sizes2[len(f)] += 1

        assert dict(sizes1) == dict(sizes2), \
            f"Face size counts differ: {dict(sizes1)} vs {dict(sizes2)}"


class TestC15FaceEdgeConsistency:
    """Verify faces and edges are consistent (T0 tripwire)."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_face_edges_in_edge_list(self, N):
        """Every edge from face boundary must exist in edge list."""
        V, E, F, _ = build_c15_supercell_periodic(N)

        edge_set = set(E)
        missing = []

        for face_idx, face in enumerate(F):
            for k in range(len(face)):
                a, b = face[k], face[(k + 1) % len(face)]
                edge = (min(a, b), max(a, b))
                if edge not in edge_set:
                    missing.append((face_idx, edge))

        assert len(missing) == 0, \
            f"N={N}: {len(missing)} face edges not in edge list: {missing[:5]}..."


class TestC15NoDuplicateFaces:
    """Verify face deduplication is correct."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_no_duplicate_canonical_faces(self, N):
        """All faces should have unique canonical form.

        Uses same canonical_face as builder (single source of truth).
        """
        V, E, F, _ = build_c15_supercell_periodic(N)

        canonical_faces = [canonical_face(f) for f in F]
        unique_canonical = set(canonical_faces)

        assert len(canonical_faces) == len(unique_canonical), \
            f"N={N}: {len(canonical_faces) - len(unique_canonical)} duplicate faces"


# ============================================================================
# Cell-face incidence tests
# ============================================================================

class TestC15CellFaceIncidence:
    """Test cell_face_incidence structure."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_each_face_in_exactly_two_cells(self, N):
        """Every face appears in exactly 2 cells with opposite orientations."""
        _, _, faces, cell_face_inc = build_c15_supercell_periodic(N)

        n_faces = len(faces)
        # Count how many cells reference each face and collect orientations
        face_cells = defaultdict(list)  # face_idx -> [(cell_idx, orient)]
        for cell_idx, cell_faces in enumerate(cell_face_inc):
            for face_idx, orient in cell_faces:
                face_cells[face_idx].append((cell_idx, orient))

        # Every face must appear in exactly 2 cells
        for face_idx in range(n_faces):
            entries = face_cells[face_idx]
            assert len(entries) == 2, \
                f"N={N}: face {face_idx} in {len(entries)} cells, expected 2"

    @pytest.mark.parametrize("N", [1, 2])
    def test_opposite_orientations_pm1(self, N):
        """Two cells sharing a face have orientations +1 and -1."""
        _, _, faces, cell_face_inc = build_c15_supercell_periodic(N)

        face_cells = defaultdict(list)
        for cell_idx, cell_faces in enumerate(cell_face_inc):
            for face_idx, orient in cell_faces:
                face_cells[face_idx].append(orient)

        for face_idx, orients in face_cells.items():
            assert len(orients) == 2, \
                f"N={N}: face {face_idx} has {len(orients)} entries"
            assert set(orients) == {+1, -1}, \
                f"N={N}: face {face_idx} orientations {orients}, expected {{+1, -1}}"

    @pytest.mark.parametrize("N", [1, 2])
    def test_cells_have_enough_faces(self, N):
        """Every cell has at least 12 faces (C15: Z12 or Z16 polyhedra)."""
        _, _, faces, cell_face_inc = build_c15_supercell_periodic(N)

        for cell_idx, cell_faces in enumerate(cell_face_inc):
            n_cell_faces = len(cell_faces)
            assert n_cell_faces >= 12, \
                f"N={N}: cell {cell_idx} has {n_cell_faces} faces, expected >= 12"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
