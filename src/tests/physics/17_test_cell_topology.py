"""
Tests for cell_topology utility module.

Verifies all shared infrastructure functions against known results
from scripts 45 (A5), 52, 55, 59. Each test section references the
original script that established the expected values.

Feb 2026
"""

import sys
sys.path.insert(0, '/Users/alextoader/Sites/physics_ai/ST_8/src')

import numpy as np
import pytest
from collections import defaultdict

from core_math.builders.multicell_periodic import (
    build_bcc_supercell_periodic, generate_bcc_centers
)
from core_math.builders.c15_periodic import (
    build_c15_supercell_periodic, get_c15_points
)
from core_math.analysis.cell_topology import (
    periodic_delta,
    get_cell_geometry,
    build_cell_adjacency,
    find_simple_cycles,
    equatorial_test,
    circuit_holonomy,
    find_best_belt,
    assign_caps_bfs,
    classify_faces_signed,
    get_belt_polygon,
    get_belt_normal,
    point_in_polygon_2d,
    count_segment_belt_crossings,
    count_path_belt_crossings,
)


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def kelvin_mesh():
    N = 2
    v, e, f, cfi = build_bcc_supercell_periodic(N)
    centers = np.array(generate_bcc_centers(N))
    L = 4 * N
    return v, e, f, cfi, centers, L


@pytest.fixture(scope="module")
def c15_mesh():
    N = 2
    L_cell = 4.0
    v, e, f, cfi = build_c15_supercell_periodic(N, L_cell)
    centers = get_c15_points(N, L_cell)
    L = N * L_cell
    return v, e, f, cfi, centers, L


@pytest.fixture(scope="module")
def kelvin_cell0(kelvin_mesh):
    v, e, f, cfi, centers, L = kelvin_mesh
    face_data, adj = get_cell_geometry(0, centers[0], v, f, cfi, L)
    return face_data, adj, centers[0]


@pytest.fixture(scope="module")
def c15_z12_cell(c15_mesh):
    """First Z12 cell in C15."""
    v, e, f, cfi, centers, L = c15_mesh
    for ci in range(len(cfi)):
        if len(cfi[ci]) == 12:
            face_data, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            return face_data, adj, centers[ci], ci
    pytest.skip("No Z12 cell found")


@pytest.fixture(scope="module")
def c15_z16_cell(c15_mesh):
    """First Z16 cell in C15."""
    v, e, f, cfi, centers, L = c15_mesh
    for ci in range(len(cfi)):
        if len(cfi[ci]) == 16:
            face_data, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            return face_data, adj, centers[ci], ci
    pytest.skip("No Z16 cell found")


# ═══════════════════════════════════════════════════════════════
# 1. PERIODIC GEOMETRY
# ═══════════════════════════════════════════════════════════════

class TestPeriodicDelta:
    def test_no_wrap(self):
        d = periodic_delta([1.0, 0, 0], [2.0, 0, 0], 10.0)
        np.testing.assert_allclose(d, [1.0, 0, 0])

    def test_wrap_positive(self):
        d = periodic_delta([1.0, 0, 0], [9.0, 0, 0], 10.0)
        np.testing.assert_allclose(d, [-2.0, 0, 0])

    def test_wrap_negative(self):
        d = periodic_delta([9.0, 0, 0], [1.0, 0, 0], 10.0)
        np.testing.assert_allclose(d, [2.0, 0, 0])

    def test_3d(self):
        d = periodic_delta([0.5, 0.5, 0.5], [9.5, 9.5, 9.5], 10.0)
        np.testing.assert_allclose(d, [-1.0, -1.0, -1.0])

    def test_scalar(self):
        d = periodic_delta(1.0, 9.0, 10.0)
        assert abs(d - (-2.0)) < 1e-12


# ═══════════════════════════════════════════════════════════════
# 2. CELL EXTRACTION
# ═══════════════════════════════════════════════════════════════

class TestGetCellGeometry:
    def test_kelvin_14_faces(self, kelvin_cell0):
        face_data, adj, _ = kelvin_cell0
        assert len(face_data) == 14

    def test_kelvin_face_types(self, kelvin_cell0):
        """Kelvin: 8 hexagons + 6 squares."""
        face_data, adj, _ = kelvin_cell0
        n_sides = sorted([fd['n_sides'] for fd in face_data])
        assert n_sides.count(4) == 6
        assert n_sides.count(6) == 8

    def test_c15_z12_faces(self, c15_z12_cell):
        face_data, adj, _, _ = c15_z12_cell
        assert len(face_data) == 12

    def test_c15_z16_faces(self, c15_z16_cell):
        face_data, adj, _, _ = c15_z16_cell
        assert len(face_data) == 16

    def test_adjacency_symmetric(self, kelvin_cell0):
        face_data, adj, _ = kelvin_cell0
        for i in adj:
            for j in adj[i]:
                assert i in adj[j]

    def test_face_centers_near_cell_center(self, kelvin_cell0):
        face_data, adj, cc = kelvin_cell0
        for fd in face_data:
            dist = np.linalg.norm(fd['center'] - cc)
            assert dist < 5.0  # reasonable for Kelvin cell


class TestBuildCellAdjacency:
    def test_kelvin_adjacency(self, kelvin_mesh):
        v, e, f, cfi, centers, L = kelvin_mesh
        ca, ftc = build_cell_adjacency(cfi, len(cfi))
        # Each Kelvin cell has 14 faces, each shared with a neighbor
        for ci in range(len(cfi)):
            nbrs = ca.get(ci, [])
            assert len(nbrs) == 14  # BCC: all 14 faces shared

    def test_c15_adjacency(self, c15_mesh):
        v, e, f, cfi, centers, L = c15_mesh
        ca, ftc = build_cell_adjacency(cfi, len(cfi))
        # Z12 has 12 neighbors, Z16 has 16
        for ci in range(len(cfi)):
            n_faces = len(cfi[ci])
            nbrs = ca.get(ci, [])
            assert len(nbrs) == n_faces


# ═══════════════════════════════════════════════════════════════
# 3. CIRCUIT FINDING
# ═══════════════════════════════════════════════════════════════

class TestFindSimpleCycles:
    def test_finds_cycles(self, kelvin_cell0):
        face_data, adj, _ = kelvin_cell0
        cycles = find_simple_cycles(adj, len(face_data), max_length=8)
        assert len(cycles) > 0

    def test_cycle_minimum_length_3(self, kelvin_cell0):
        face_data, adj, _ = kelvin_cell0
        cycles = find_simple_cycles(adj, len(face_data), max_length=8)
        for cyc in cycles:
            assert len(cyc) >= 3

    def test_cycle_faces_valid(self, kelvin_cell0):
        face_data, adj, _ = kelvin_cell0
        cycles = find_simple_cycles(adj, len(face_data), max_length=8)
        for cyc in cycles:
            for fi in cyc:
                assert 0 <= fi < len(face_data)


class TestFindBestBelt:
    def test_kelvin_belt_exists(self, kelvin_cell0):
        face_data, adj, cc = kelvin_cell0
        result = find_best_belt(face_data, adj, cc)
        assert result is not None

    def test_kelvin_belt_holonomy_2pi(self, kelvin_cell0):
        """Kelvin belt has Ω ≈ 2π (from T8b/T24)."""
        face_data, adj, cc = kelvin_cell0
        circuit, hol = find_best_belt(face_data, adj, cc)
        assert abs(abs(hol) - 2 * np.pi) < 0.3

    def test_kelvin_belt_length_6(self, kelvin_cell0):
        """Kelvin best belt is N=6 hexagons (from T3 catalog)."""
        face_data, adj, cc = kelvin_cell0
        circuit, hol = find_best_belt(face_data, adj, cc)
        assert len(circuit) == 6

    def test_c15_z12_belt_exists(self, c15_z12_cell):
        face_data, adj, cc, _ = c15_z12_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None

    def test_c15_z12_belt_holonomy_2pi(self, c15_z12_cell):
        face_data, adj, cc, _ = c15_z12_cell
        circuit, hol = find_best_belt(face_data, adj, cc)
        assert abs(abs(hol) - 2 * np.pi) < 0.3


# ═══════════════════════════════════════════════════════════════
# 4. CAP ASSIGNMENT
# ═══════════════════════════════════════════════════════════════

class TestAssignCapsBfs:
    def test_kelvin_two_caps(self, kelvin_cell0):
        face_data, adj, cc = kelvin_cell0
        circuit, _ = find_best_belt(face_data, adj, cc)
        cap1, cap2, belt = assign_caps_bfs(face_data, circuit, adj)
        assert len(cap1) > 0
        assert len(cap2) > 0
        assert len(belt) == len(circuit)

    def test_caps_partition_all_faces(self, kelvin_cell0):
        face_data, adj, cc = kelvin_cell0
        circuit, _ = find_best_belt(face_data, adj, cc)
        cap1, cap2, belt = assign_caps_bfs(face_data, circuit, adj)
        assert len(cap1) + len(cap2) + len(belt) == len(face_data)

    def test_caps_disjoint(self, kelvin_cell0):
        face_data, adj, cc = kelvin_cell0
        circuit, _ = find_best_belt(face_data, adj, cc)
        cap1, cap2, belt = assign_caps_bfs(face_data, circuit, adj)
        assert len(cap1 & cap2) == 0
        assert len(cap1 & belt) == 0
        assert len(cap2 & belt) == 0


class TestClassifyFacesSigned:
    def test_all_faces_classified(self, kelvin_cell0):
        face_data, adj, cc = kelvin_cell0
        circuit, _ = find_best_belt(face_data, adj, cc)
        fc = classify_faces_signed(face_data, circuit, adj)
        assert fc is not None
        assert len(fc) == len(face_data)
        for fi in fc:
            assert fc[fi] in (+1, -1)

    def test_agrees_with_bfs_on_non_belt(self, c15_z12_cell):
        """Signed-distance matches BFS for non-belt faces (script 59 v3: 100%)."""
        face_data, adj, cc, _ = c15_z12_cell
        result = find_best_belt(face_data, adj, cc)
        if result is None:
            pytest.skip("No belt")
        circuit, _ = result
        cap1, cap2, belt = assign_caps_bfs(face_data, circuit, adj)
        fc = classify_faces_signed(face_data, circuit, adj)
        if fc is None:
            pytest.skip("Classification failed")

        for fi in cap1:
            assert fc[fi] == +1
        for fi in cap2:
            assert fc[fi] == -1


# ═══════════════════════════════════════════════════════════════
# 5. BELT GEOMETRY
# ═══════════════════════════════════════════════════════════════

class TestBeltGeometry:
    def test_belt_polygon_shape(self, kelvin_cell0):
        face_data, adj, cc = kelvin_cell0
        circuit, _ = find_best_belt(face_data, adj, cc)
        bp = get_belt_polygon(circuit, face_data, adj)
        assert bp.shape == (len(circuit), 3)

    def test_belt_normal_unit(self, kelvin_cell0):
        face_data, adj, cc = kelvin_cell0
        circuit, _ = find_best_belt(face_data, adj, cc)
        bp = get_belt_polygon(circuit, face_data, adj)
        bn = get_belt_normal(bp)
        assert abs(np.linalg.norm(bn) - 1.0) < 1e-10

    def test_belt_polygon_roughly_planar(self, kelvin_cell0):
        """Belt polygon points should be close to their best-fit plane."""
        face_data, adj, cc = kelvin_cell0
        circuit, _ = find_best_belt(face_data, adj, cc)
        bp = get_belt_polygon(circuit, face_data, adj)
        bn = get_belt_normal(bp)
        centroid = bp.mean(axis=0)
        max_deviation = max(abs(np.dot(p - centroid, bn)) for p in bp)
        assert max_deviation < 0.5  # reasonable for foam cells


# ═══════════════════════════════════════════════════════════════
# 6. INTERSECTION / CROSSING
# ═══════════════════════════════════════════════════════════════

class TestPointInPolygon2d:
    def test_inside_square(self):
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert point_in_polygon_2d((0.5, 0.5), sq) is True

    def test_outside_square(self):
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert point_in_polygon_2d((2.0, 0.5), sq) is False

    def test_inside_triangle(self):
        tri = [(0, 0), (4, 0), (2, 3)]
        assert point_in_polygon_2d((2, 1), tri) is True

    def test_outside_triangle(self):
        tri = [(0, 0), (4, 0), (2, 3)]
        assert point_in_polygon_2d((0, 3), tri) is False


class TestCountSegmentBeltCrossings:
    def test_crossing_segment(self):
        """Segment through a horizontal polygon should count 1."""
        polygon = np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]
        ], dtype=float)
        normal = np.array([0, 0, 1.0])
        p_start = np.array([0, 0, -1.0])
        p_end = np.array([0, 0, 1.0])
        assert count_segment_belt_crossings(p_start, p_end,
                                             polygon, normal) == 1

    def test_non_crossing_segment(self):
        """Segment above the polygon should count 0."""
        polygon = np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]
        ], dtype=float)
        normal = np.array([0, 0, 1.0])
        p_start = np.array([0, 0, 1.0])
        p_end = np.array([0, 0, 2.0])
        assert count_segment_belt_crossings(p_start, p_end,
                                             polygon, normal) == 0

    def test_parallel_segment(self):
        """Segment parallel to polygon plane should count 0."""
        polygon = np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]
        ], dtype=float)
        normal = np.array([0, 0, 1.0])
        p_start = np.array([-2, 0, 0.5])
        p_end = np.array([2, 0, 0.5])
        assert count_segment_belt_crossings(p_start, p_end,
                                             polygon, normal) == 0

    def test_miss_polygon(self):
        """Segment crosses plane but misses polygon → 0."""
        polygon = np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]
        ], dtype=float)
        normal = np.array([0, 0, 1.0])
        p_start = np.array([5, 5, -1.0])
        p_end = np.array([5, 5, 1.0])
        assert count_segment_belt_crossings(p_start, p_end,
                                             polygon, normal) == 0


# ═══════════════════════════════════════════════════════════════
# 7. SMALL-GRAPH REGRESSION (P3)
# ═══════════════════════════════════════════════════════════════

class TestFindSimpleCyclesSmallGraph:
    """Regression tests on known small graphs (P3)."""

    def test_triangle_one_cycle(self):
        """Triangle graph (3 nodes, 3 edges) → exactly 1 cycle."""
        adj = {0: {1: 'a', 2: 'c'}, 1: {0: 'a', 2: 'b'}, 2: {1: 'b', 0: 'c'}}
        cycles = find_simple_cycles(adj, 3, max_length=8)
        assert len(cycles) == 1
        assert len(cycles[0]) == 3

    def test_square_no_diagonal(self):
        """Square (4 nodes, 4 edges, no diagonal) → exactly 1 cycle of length 4."""
        adj = {
            0: {1: 'a', 3: 'd'}, 1: {0: 'a', 2: 'b'},
            2: {1: 'b', 3: 'c'}, 3: {2: 'c', 0: 'd'},
        }
        cycles = find_simple_cycles(adj, 4, max_length=8)
        assert len(cycles) == 1
        assert len(cycles[0]) == 4

    def test_square_with_diagonal(self):
        """Square + 1 diagonal → 3 cycles (two triangles + one square)."""
        adj = {
            0: {1: 'a', 2: 'diag', 3: 'd'},
            1: {0: 'a', 2: 'b'},
            2: {1: 'b', 3: 'c', 0: 'diag'},
            3: {2: 'c', 0: 'd'},
        }
        cycles = find_simple_cycles(adj, 4, max_length=8)
        assert len(cycles) == 3
        lengths = sorted(len(c) for c in cycles)
        assert lengths == [3, 3, 4]

    def test_no_cycles_in_line(self):
        """Linear graph (no cycles)."""
        adj = {0: {1: 'a'}, 1: {0: 'a', 2: 'b'}, 2: {1: 'b'}}
        cycles = find_simple_cycles(adj, 3, max_length=8)
        assert len(cycles) == 0


# ═══════════════════════════════════════════════════════════════
# 8. SHARED-EDGE UNWRAP CONSISTENCY (P5 / T-Geom-1)
# ═══════════════════════════════════════════════════════════════

class TestSharedEdgeUnwrap:
    """
    For each shared edge, the midpoint computed from both adjacent faces
    must coincide (max Δ < 1e-8). Detects unwrap bugs in get_cell_geometry.
    """

    def _check_edge_consistency(self, face_data, adj):
        max_delta = 0.0
        n_checked = 0
        for i in adj:
            for j in adj[i]:
                if j <= i:
                    continue
                e_key = adj[i][j]
                mid_i = face_data[i]['edges'][e_key]
                mid_j = face_data[j]['edges'][e_key]
                delta = np.linalg.norm(mid_i - mid_j)
                max_delta = max(max_delta, delta)
                n_checked += 1
        return max_delta, n_checked

    def test_kelvin_shared_edges(self, kelvin_cell0):
        face_data, adj, _ = kelvin_cell0
        max_delta, n = self._check_edge_consistency(face_data, adj)
        assert n > 0
        assert max_delta < 1e-8, f"max edge midpoint Δ = {max_delta}"

    def test_c15_z12_shared_edges(self, c15_z12_cell):
        face_data, adj, _, _ = c15_z12_cell
        max_delta, n = self._check_edge_consistency(face_data, adj)
        assert n > 0
        assert max_delta < 1e-8, f"max edge midpoint Δ = {max_delta}"

    def test_c15_z16_shared_edges(self, c15_z16_cell):
        face_data, adj, _, _ = c15_z16_cell
        max_delta, n = self._check_edge_consistency(face_data, adj)
        assert n > 0
        assert max_delta < 1e-8, f"max edge midpoint Δ = {max_delta}"


# ═══════════════════════════════════════════════════════════════
# 9. VERTEX ID IN FACE DATA
# ═══════════════════════════════════════════════════════════════

class TestVertexIds:
    """Verify that face_data includes vertex_ids (P1)."""

    def test_kelvin_has_vertex_ids(self, kelvin_cell0):
        face_data, adj, _ = kelvin_cell0
        for fd in face_data:
            assert 'vertex_ids' in fd
            assert len(fd['vertex_ids']) == fd['n_sides']

    def test_c15_z12_has_vertex_ids(self, c15_z12_cell):
        face_data, adj, _, _ = c15_z12_cell
        for fd in face_data:
            assert 'vertex_ids' in fd
            assert len(fd['vertex_ids']) == fd['n_sides']


# ═══════════════════════════════════════════════════════════════
# 10. HOLONOMY COMPLEMENT (P8 / T-REG-4)
# ═══════════════════════════════════════════════════════════════

class TestHolonomyComplement:
    """
    P8 / T-REG-4: Gauss-Bonnet on closed polyhedron.

    Total vertex deficit on any closed polyhedron = 4π (Descartes theorem).
    For each belt: Ω_cap1 + Ω_cap2 = 4π, since vertex assignment partitions
    all vertices between the two caps. Independent consistency check.
    """

    def _total_vertex_deficit(self, face_data):
        """Sum of all vertex deficits on the cell (Descartes theorem → 4π)."""
        all_verts = {}
        for fd in face_data:
            verts = fd['vertices']
            vids = fd['vertex_ids']
            for j, vid in enumerate(vids):
                v = verts[j]
                if vid not in all_verts:
                    all_verts[vid] = []
                prev_v = verts[(j - 1) % len(verts)]
                next_v = verts[(j + 1) % len(verts)]
                e1 = prev_v - v
                e2 = next_v - v
                n1 = np.linalg.norm(e1)
                n2 = np.linalg.norm(e2)
                if n1 < 1e-12 or n2 < 1e-12:
                    continue
                cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                all_verts[vid].append(np.arccos(cos_a))
        return sum(2 * np.pi - sum(angles) for angles in all_verts.values())

    def test_kelvin_descartes_4pi(self, kelvin_cell0):
        """Total vertex deficit = 4π (Descartes theorem)."""
        face_data, adj, _ = kelvin_cell0
        total = self._total_vertex_deficit(face_data)
        assert abs(total - 4 * np.pi) < 1e-8, (
            f"total deficit = {total:.10f}, expected 4π = {4*np.pi:.10f}"
        )

    def test_kelvin_complement_4pi(self, kelvin_cell0):
        """Ω_cap1 + Ω_cap2 ≈ 4π for Kelvin best belt."""
        face_data, adj, cc = kelvin_cell0
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, _ = result
        omega1 = circuit_holonomy(circuit, face_data, adj)
        assert omega1 is not None
        total = self._total_vertex_deficit(face_data)
        omega2 = total - omega1
        assert abs(omega1 + omega2 - 4 * np.pi) < 1e-8
        assert abs(omega1 - 2 * np.pi) < 0.3
        assert abs(omega2 - 2 * np.pi) < 0.3

    def test_c15_z12_complement_4pi(self, c15_z12_cell):
        """Ω_cap1 + Ω_cap2 ≈ 4π for C15 Z12 best belt."""
        face_data, adj, cc, _ = c15_z12_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, _ = result
        omega1 = circuit_holonomy(circuit, face_data, adj)
        assert omega1 is not None
        total = self._total_vertex_deficit(face_data)
        omega2 = total - omega1
        assert abs(omega1 + omega2 - 4 * np.pi) < 1e-8
        assert abs(omega1 - 2 * np.pi) < 0.3
        assert abs(omega2 - 2 * np.pi) < 0.3

    def test_c15_z16_complement_4pi(self, c15_z16_cell):
        """Ω_cap1 + Ω_cap2 ≈ 4π for C15 Z16 best belt."""
        face_data, adj, cc, _ = c15_z16_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, _ = result
        omega1 = circuit_holonomy(circuit, face_data, adj)
        assert omega1 is not None
        total = self._total_vertex_deficit(face_data)
        omega2 = total - omega1
        assert abs(omega1 + omega2 - 4 * np.pi) < 1e-8
        assert abs(omega1 - 2 * np.pi) < 0.3
        assert abs(omega2 - 2 * np.pi) < 0.3


# ═══════════════════════════════════════════════════════════════
# 11. A5 EXCHANGE PARITY (integration test)
# ═══════════════════════════════════════════════════════════════

def _enumerate_paths(adj, start_set, target_set, max_length=10):
    """Enumerate all simple paths from start_set to target_set."""
    all_paths = []
    target_s = set(target_set)

    def dfs(current, path, visited):
        if len(path) > max_length:
            return
        for nb in sorted(adj.get(current, {}).keys()):
            if nb in visited:
                continue
            new_path = path + [nb]
            if nb in target_s:
                all_paths.append(new_path)
            else:
                visited.add(nb)
                dfs(nb, new_path, visited)
                visited.remove(nb)

    for src in sorted(start_set):
        visited = {src}
        dfs(src, [src], visited)

    return all_paths


class TestA5ExchangeParity:
    """
    Integration test: reproduce A5 result on Kelvin cell.

    All cap1→cap2 paths have ODD belt crossings.
    All cap1→cap1 paths have EVEN belt crossings.

    Reference: script 45 (156k+ exchange paths, 125k+ trivial).
    """

    def test_kelvin_exchange_all_odd(self, kelvin_cell0):
        face_data, adj, cc = kelvin_cell0
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result

        belt_polygon = get_belt_polygon(circuit, face_data, adj)
        belt_normal = get_belt_normal(belt_polygon)
        cap1, cap2, belt = assign_caps_bfs(face_data, circuit, adj)

        assert len(cap1) > 0 and len(cap2) > 0

        exchange_paths = _enumerate_paths(adj, cap1, cap2, max_length=10)
        assert len(exchange_paths) > 0

        n_odd = 0
        n_even = 0
        for path in exchange_paths:
            nc = count_path_belt_crossings(
                path, face_data, belt_polygon, belt_normal, belt)
            if nc % 2 == 1:
                n_odd += 1
            else:
                n_even += 1

        assert n_even == 0, (
            f"Exchange: {n_odd} odd, {n_even} even — expected ALL odd"
        )
        assert n_odd == len(exchange_paths)

    def test_kelvin_trivial_all_even(self, kelvin_cell0):
        face_data, adj, cc = kelvin_cell0
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result

        belt_polygon = get_belt_polygon(circuit, face_data, adj)
        belt_normal = get_belt_normal(belt_polygon)
        cap1, cap2, belt = assign_caps_bfs(face_data, circuit, adj)

        # cap1 → cap1 paths (trivial)
        trivial_paths = []
        for src in sorted(cap1):
            targets = cap1 - {src}
            if targets:
                paths = _enumerate_paths(adj, {src}, targets, max_length=10)
                trivial_paths.extend(paths)

        assert len(trivial_paths) > 0

        n_odd = 0
        n_even = 0
        for path in trivial_paths:
            nc = count_path_belt_crossings(
                path, face_data, belt_polygon, belt_normal, belt)
            if nc % 2 == 0:
                n_even += 1
            else:
                n_odd += 1

        assert n_odd == 0, (
            f"Trivial: {n_even} even, {n_odd} odd — expected ALL even"
        )

    def test_c15_z12_exchange_all_odd(self, c15_z12_cell):
        """Same test on C15 Z12 cell."""
        face_data, adj, cc, _ = c15_z12_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result

        belt_polygon = get_belt_polygon(circuit, face_data, adj)
        belt_normal = get_belt_normal(belt_polygon)
        cap1, cap2, belt = assign_caps_bfs(face_data, circuit, adj)

        exchange_paths = _enumerate_paths(adj, cap1, cap2, max_length=12)
        assert len(exchange_paths) > 0

        n_even = 0
        for path in exchange_paths:
            nc = count_path_belt_crossings(
                path, face_data, belt_polygon, belt_normal, belt)
            if nc % 2 == 0:
                n_even += 1

        assert n_even == 0, (
            f"C15 Z12 exchange: {n_even} even out of "
            f"{len(exchange_paths)} — expected ALL odd"
        )
