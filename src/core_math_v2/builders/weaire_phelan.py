"""
Weaire-Phelan Structure - Geometry Only
=======================================

The Weaire-Phelan structure is more efficient than Kelvin for equal-volume cells.
It has 0.3% less surface area than Kelvin foam.

STRUCTURE:
    - 8 cells per fundamental domain
    - 2 Type A cells: Irregular pentagonal dodecahedra (12 pentagonal faces)
    - 6 Type B cells: Tetrakaidecahedra (2 hexagonal + 12 pentagonal faces)

CELL PROPERTIES:
    Type A (dodecahedron):
        V = 20 vertices
        E = 30 edges
        F = 12 faces (all pentagons)
        χ = 20 - 30 + 12 = 2 ✓
        Planarity: ALL faces planar ✓

    Type B (tetrakaidecahedron):
        V = 24 vertices
        E = 36 edges
        F = 14 faces (2 hexagons + 12 pentagons)
        χ = 24 - 36 + 14 = 2 ✓
        Planarity: ALL faces planar ✓ (using aligned geometry, explicit faces)

COMPARISON WITH KELVIN:
    Kelvin: V=24, E=36, F=14 (6 squares + 8 hexagons)
    WP-B:   V=24, E=36, F=14 (2 hexagons + 12 pentagons)

    Same (V, E, F) but DIFFERENT face structure!

NOTE: compute_wp_kappa, compare_wp_kelvin moved to analysis/weaire_phelan_kappa.py

Date: Jan 2026
"""

import numpy as np
from typing import Tuple, List, Dict

from ..spec.constants import EDGE_TOL, EPS_CLOSE, PLANAR_TOL, WP_ROUND


def build_wp_type_a() -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], Dict[tuple, int]]:
    """
    Build Type A cell: Irregular pentagonal dodecahedron (pyritohedron).

    This is a pyritohedron with pyritohedral symmetry Th.
    All 12 faces are irregular pentagons.

    TOPOLOGY:
        V = 20 vertices
        E = 30 edges
        F = 12 faces (pentagons)
        χ = 20 - 30 + 12 = 2

    GEOMETRY (EXPLICIT - foundation-grade):
        8 cube vertices at (±1, ±1, ±1)
        12 axis vertices at cyclic permutations of (0, ±φ, ±1/φ) where φ = golden ratio

    EDGE STRUCTURE (30 edges):
        - 24 edges: cube vertex ↔ axis vertex (each cube vertex connects to 3 axis vertices)
        - 6 edges: axis vertex ↔ axis vertex (connecting "opposite" axis vertices)

    FACE STRUCTURE (12 pentagons):
        Each face has 2 cube vertices and 3 axis vertices.

    Returns:
        vertices: (20, 3) array
        edges: list of 30 edge tuples
        faces: list of 12 pentagonal faces
        v_to_idx: vertex lookup dict
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
    inv_phi = 1 / phi  # ≈ 0.618

    # Build vertices in known structure
    raw_vertices = {}

    # 8 cube vertices at (±1, ±1, ±1) - labeled by signs
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                raw_vertices[('cube', sx, sy, sz)] = (sx, sy, sz)

    # 12 axis vertices at permutations of (0, ±φ, ±1/φ)
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            # Type X: (0, sy*φ, sz*1/φ) - varies in y,z
            raw_vertices[('axis', 'x', s1, s2)] = (0, s1 * phi, s2 * inv_phi)
            # Type Y: (sx*1/φ, 0, sz*φ) - varies in x,z
            raw_vertices[('axis', 'y', s1, s2)] = (s1 * inv_phi, 0, s2 * phi)
            # Type Z: (sx*φ, sy*1/φ, 0) - varies in x,y
            raw_vertices[('axis', 'z', s1, s2)] = (s1 * phi, s2 * inv_phi, 0)

    # Convert to sorted list with index mapping
    def round_coord(c):
        return (round(c[0], 10), round(c[1], 10), round(c[2], 10))

    coord_to_label = {round_coord(v): k for k, v in raw_vertices.items()}
    vertices = sorted(set(round_coord(v) for v in raw_vertices.values()))
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    vertices_arr = np.array(vertices)

    def get_idx(label):
        coord = round_coord(raw_vertices[label])
        return v_to_idx[coord]

    if len(vertices) != 20:
        raise ValueError(f"Expected 20 vertices, got {len(vertices)}")

    # Build edges EXPLICITLY based on pyritohedron structure
    edges = []

    # Each cube vertex (sx, sy, sz) connects to 3 axis vertices:
    #   - axis X with same (sy, sz) signs: (0, sy*φ, sz*1/φ)
    #   - axis Y with same (sx, sz) signs: (sx*1/φ, 0, sz*φ)
    #   - axis Z with same (sx, sy) signs: (sx*φ, sy*1/φ, 0)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                cube_idx = get_idx(('cube', sx, sy, sz))
                # Connect to axis-X vertex
                axis_x_idx = get_idx(('axis', 'x', sy, sz))
                edges.append((min(cube_idx, axis_x_idx), max(cube_idx, axis_x_idx)))
                # Connect to axis-Y vertex
                axis_y_idx = get_idx(('axis', 'y', sx, sz))
                edges.append((min(cube_idx, axis_y_idx), max(cube_idx, axis_y_idx)))
                # Connect to axis-Z vertex
                axis_z_idx = get_idx(('axis', 'z', sx, sy))
                edges.append((min(cube_idx, axis_z_idx), max(cube_idx, axis_z_idx)))

    # 6 axis-axis edges: found by distance (same edge length as cube↔axis edges)
    # These complete the pyritohedron edge structure
    edge_set = set(edges)
    min_dist = float('inf')
    for i in range(20):
        for j in range(i+1, 20):
            d = np.linalg.norm(vertices_arr[i] - vertices_arr[j])
            if d > 0.1 and d < min_dist:
                min_dist = d

    # Add remaining edges (axis-axis) at minimum distance
    # Uses EDGE_TOL from spec/constants.py (not hardcoded)
    for i in range(20):
        for j in range(i+1, 20):
            if (i, j) in edge_set:
                continue
            d = np.linalg.norm(vertices_arr[i] - vertices_arr[j])
            if abs(d - min_dist) < EDGE_TOL:
                edges.append((i, j))

    edges = sorted(set(edges))
    if len(edges) != 30:
        raise ValueError(f"Expected 30 edges, got {len(edges)}")

    # Build faces using cycle-finder, then VERIFY planarity (foundation-grade)
    # The pyritohedron faces are all planar pentagons - verified below
    faces = _find_pentagonal_faces(vertices_arr, edges)

    # Verify each face is planar (this makes it foundation-grade)
    for f_idx, face in enumerate(faces):
        coords = vertices_arr[face]
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        _, s, vh = np.linalg.svd(centered)
        normal = vh[-1]
        deviations = np.abs(centered @ normal)
        max_dev = np.max(deviations)
        if max_dev >= PLANAR_TOL:
            raise ValueError(f"Face {f_idx} not planar: deviation = {max_dev}")

    if len(faces) != 12:
        raise ValueError(f"Expected 12 faces, got {len(faces)}")

    return vertices_arr, edges, faces, v_to_idx


def build_wp_type_b() -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], Dict[tuple, int]]:
    """
    Build Type B cell: Tetrakaidecahedron (14-faced polyhedron).

    This has 2 hexagonal faces and 12 pentagonal faces.
    Different from Kelvin which has 6 squares + 8 hexagons.

    TOPOLOGY:
        V = 24 vertices
        E = 36 edges
        F = 14 faces (2 hexagons + 12 pentagons)
        χ = 24 - 36 + 14 = 2

    GEOMETRY (derived for planar faces):
        - 4 rings of 6 vertices each, ALL ALIGNED (no 30° offset!)
        - Top/bottom hex at z = ±h1, radius r1
        - Upper/lower middle at z = ±h2, radius r2
        - Specific ratio r1=r2, h1/h2 ≈ 3.54 for planar pentagons

    EDGE STRUCTURE:
        - Hex rings (12): top and bottom hexagon edges
        - Hex→middle (12): radial connections
        - Cross middle (12): direct + diagonal connections

    Returns:
        vertices: (24, 3) array
        edges: list of 36 edge tuples
        faces: list of 14 faces (explicit, all planar)
        v_to_idx: vertex lookup dict
    """
    # Geometry parameters for PLANAR faces
    # Derived from constraint: all pentagons must be coplanar
    # Solution: r1 = r2 = 0.5, h1 ≈ 0.974, h2 ≈ 0.275 (scaled by 2 for better size)
    r1 = 1.0       # Top/bottom hexagon radius
    r2 = 1.0       # Middle ring radius (same as r1 for planarity)
    h1 = 1.948     # Top/bottom hexagon height (≈ 0.974 × 2)
    h2 = 0.550     # Middle ring height (≈ 0.275 × 2)

    # Build vertices in known order (NO offset - all rings aligned)
    # IMPORTANT: Uses WP_ROUND from spec/constants.py for rounding precision.
    # Must match between raw_vertices creation and v_to_idx lookup.
    raw_vertices = []

    # 0-5: top hexagon (z = h1)
    for k in range(6):
        angle = k * np.pi / 3
        raw_vertices.append((round(r1 * np.cos(angle), WP_ROUND),
                            round(r1 * np.sin(angle), WP_ROUND), h1))

    # 6-11: bottom hexagon (z = -h1)
    for k in range(6):
        angle = k * np.pi / 3
        raw_vertices.append((round(r1 * np.cos(angle), WP_ROUND),
                            round(r1 * np.sin(angle), WP_ROUND), -h1))

    # 12-17: upper middle ring (z = h2) - ALIGNED, no offset!
    for k in range(6):
        angle = k * np.pi / 3
        raw_vertices.append((round(r2 * np.cos(angle), WP_ROUND),
                            round(r2 * np.sin(angle), WP_ROUND), h2))

    # 18-23: lower middle ring (z = -h2) - ALIGNED, no offset!
    for k in range(6):
        angle = k * np.pi / 3
        raw_vertices.append((round(r2 * np.cos(angle), WP_ROUND),
                            round(r2 * np.sin(angle), WP_ROUND), -h2))

    # Sort vertices for canonical ordering
    vertices = sorted(set(raw_vertices))
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    vertices_arr = np.array(vertices)

    if len(vertices) != 24:
        raise ValueError(f"Expected 24 vertices, got {len(vertices)}")

    # Map from raw index to sorted index
    # Guard: fail fast if rounding mismatch causes lookup failure
    def get_idx(raw_idx):
        vertex = raw_vertices[raw_idx]
        if vertex not in v_to_idx:
            raise ValueError(
                f"raw_vertices[{raw_idx}] = {vertex} not in v_to_idx. "
                f"Rounding mismatch? Check WP_ROUND in spec/constants.py."
            )
        return v_to_idx[vertex]

    # Validate all raw_vertices are in v_to_idx (guard against rounding drift)
    for idx, rv in enumerate(raw_vertices):
        if rv not in v_to_idx:
            raise ValueError(f"raw_vertices[{idx}] = {rv} not found in v_to_idx")

    # Build edges explicitly (36 total, all vertices degree 3)
    edges = []

    # Top hexagon edges (6)
    for k in range(6):
        i, j = get_idx(k), get_idx((k + 1) % 6)
        edges.append((min(i, j), max(i, j)))

    # Bottom hexagon edges (6)
    for k in range(6):
        i, j = get_idx(6 + k), get_idx(6 + (k + 1) % 6)
        edges.append((min(i, j), max(i, j)))

    # Top hex → upper middle (6)
    for k in range(6):
        i, j = get_idx(k), get_idx(12 + k)
        edges.append((min(i, j), max(i, j)))

    # Bottom hex → lower middle (6)
    for k in range(6):
        i, j = get_idx(6 + k), get_idx(18 + k)
        edges.append((min(i, j), max(i, j)))

    # Cross connections upper ↔ lower (12)
    for k in range(6):
        # Direct down: upper[k] → lower[k]
        i, j = get_idx(12 + k), get_idx(18 + k)
        edges.append((min(i, j), max(i, j)))
        # Diagonal: upper[k] → lower[k-1]
        i, j = get_idx(12 + k), get_idx(18 + (k + 5) % 6)
        edges.append((min(i, j), max(i, j)))

    edges = sorted(set(edges))
    if len(edges) != 36:
        raise ValueError(f"Expected 36 edges, got {len(edges)}")

    # Build faces EXPLICITLY (not by cycle finder!)
    # This ensures all faces are planar with the derived geometry
    faces = []

    # 2 hexagonal faces (top and bottom)
    # Top hexagon: vertices 0-5
    faces.append([get_idx(k) for k in range(6)])
    # Bottom hexagon: vertices 6-11 (reverse order for outward normal)
    faces.append([get_idx(6 + k) for k in [0, 5, 4, 3, 2, 1]])

    # 6 top pentagons (around top hex edges)
    # Pentagon containing edge (top[k], top[k+1]):
    # top[k] → top[k+1] → upper[k+1] → lower[k] → upper[k]
    for k in range(6):
        face = [get_idx(k), get_idx((k+1)%6), get_idx(12+(k+1)%6),
                get_idx(18+k), get_idx(12+k)]
        faces.append(face)

    # 6 bottom pentagons (around bottom hex edges)
    # Pentagon containing edge (bottom[k], bottom[k-1]):
    # bottom[k] → lower[k] → upper[k] → lower[k-1] → bottom[k-1]
    for k in range(6):
        face = [get_idx(6+k), get_idx(18+k), get_idx(12+k),
                get_idx(18+(k+5)%6), get_idx(6+(k+5)%6)]
        faces.append(face)

    if len(faces) != 14:
        raise ValueError(f"Expected 14 faces, got {len(faces)}")

    return vertices_arr, edges, faces, v_to_idx


def _find_cycles_for_faces(adj: dict, n_vertices: int) -> List[List[int]]:
    """Find all minimal face cycles (pentagons and hexagons)."""
    seen = set()
    faces = []

    def normalize(face):
        min_idx = face.index(min(face))
        rotated = face[min_idx:] + face[:min_idx]
        reversed_rot = [rotated[0]] + rotated[1:][::-1]
        if tuple(rotated) < tuple(reversed_rot):
            return tuple(rotated)
        return tuple(reversed_rot)

    def dfs(path, target_len):
        if len(path) == target_len:
            if path[0] in adj[path[-1]]:
                face = normalize(path)
                if face not in seen:
                    seen.add(face)
                    faces.append(list(face))
            return

        last = path[-1]
        for next_v in sorted(adj[last]):
            if next_v == path[0] and len(path) < target_len:
                continue
            if next_v in path[1:]:
                continue
            dfs(path + [next_v], target_len)

    # Find 5-cycles (pentagons) and 6-cycles (hexagons)
    for target_len in [5, 6]:
        for start in range(n_vertices):
            for second in sorted(adj[start]):
                if second > start:
                    dfs([start, second], target_len)

    return faces


def _find_pentagonal_faces(vertices: np.ndarray, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Find all pentagonal faces from edge list."""
    from collections import defaultdict

    # Build adjacency
    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    faces = []
    seen = set()

    # Find 5-cycles
    for start in range(len(vertices)):
        for second in adj[start]:
            if second <= start:
                continue
            # Try to find a 5-cycle starting with start -> second
            path = [start, second]
            _find_cycle(adj, path, 5, faces, seen)

    return faces


def _find_cycle(adj: dict, path: List[int], target_len: int,
                faces: List[List[int]], seen: set) -> None:
    """Recursively find cycles of given length."""
    if len(path) == target_len:
        # Check if we can close the cycle
        if path[0] in adj[path[-1]]:
            # Normalize face (start from smallest, go in canonical direction)
            face = _normalize_face(path)
            face_key = tuple(face)
            if face_key not in seen:
                seen.add(face_key)
                faces.append(face)
        return

    last = path[-1]
    for next_v in adj[last]:
        if next_v in path[1:]:  # Can revisit start only at the end
            continue
        if next_v == path[0] and len(path) < target_len:
            continue  # Too early to close
        path.append(next_v)
        _find_cycle(adj, path, target_len, faces, seen)
        path.pop()


def _normalize_face(face: List[int]) -> List[int]:
    """Normalize face to start from smallest index, canonical direction."""
    min_idx = face.index(min(face))
    rotated = face[min_idx:] + face[:min_idx]
    # Choose direction that gives smaller second element
    if rotated[1] > rotated[-1]:
        rotated = [rotated[0]] + rotated[1:][::-1]
    return rotated


def verify_wp_surface_topology(edges: List[Tuple[int, int]],
                                faces: List[List[int]]) -> Tuple[bool, List[str]]:
    """
    Verify WP cell has valid surface topology.

    Checks:
        1. Each edge appears in exactly 2 faces (closed surface)
        2. Each face boundary segment exists as an edge (T0)

    Returns:
        (is_valid, list_of_issues)
    """
    from collections import defaultdict

    edge_face_count = defaultdict(int)
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
    issues = []

    for f_idx, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            v1, v2 = face[k], face[(k + 1) % n]
            edge = (min(v1, v2), max(v1, v2))
            edge_face_count[edge] += 1

            # T0: face segment must exist as edge
            if edge not in edge_set:
                issues.append(f"Face {f_idx} segment ({v1},{v2}) not in edge list")

    # Check each edge appears exactly twice
    for edge in edges:
        e = (min(edge[0], edge[1]), max(edge[0], edge[1]))
        count = edge_face_count.get(e, 0)
        if count != 2:
            issues.append(f"Edge {e} appears in {count} faces (expected 2)")

    return len(issues) == 0, issues


# NOTE: compute_wp_kappa, compare_wp_kelvin moved to analysis/weaire_phelan_kappa.py (layer separation)


if __name__ == "__main__":
    print("=" * 60)
    print("WEAIRE-PHELAN GEOMETRY (pure builders)")
    print("=" * 60)

    print("\nType A (dodecahedron):")
    v, e, f, idx = build_wp_type_a()
    print(f"  V={len(v)}, E={len(e)}, F={len(f)}")
    print(f"  χ = {len(v) - len(e) + len(f)}")
    ok, issues = verify_wp_surface_topology(e, f)
    print(f"  Surface topology valid: {ok}")

    print("\nType B (tetrakaidecahedron):")
    v, e, f, idx = build_wp_type_b()
    print(f"  V={len(v)}, E={len(e)}, F={len(f)}")
    print(f"  χ = {len(v) - len(e) + len(f)}")
    ok, issues = verify_wp_surface_topology(e, f)
    print(f"  Surface topology valid: {ok}")

    print("\nFor κ computation, use: from analysis.weaire_phelan_kappa import compute_wp_kappa")
