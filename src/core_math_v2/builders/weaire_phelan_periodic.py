"""
Periodic Weaire-Phelan Supercell via Voronoi
============================================

N×N×N Weaire-Phelan foam with PERIODIC boundary conditions (3-torus T³).

CONSTRUCTION:
    1. Generate A15 lattice points (space group Pm3n)
    2. Compute Voronoi tessellation with periodic images
    3. Order ridge vertices cyclically in face plane
    4. Extract vertices, edges, faces with wrapping

STRUCTURE:
    8 cells per fundamental domain:
        - 2 Type A (dodecahedra) at Wyckoff 2a
        - 6 Type B (tetrakaidecahedra) at Wyckoff 6d

TOPOLOGY (verified):
    - χ(3-complex) = V - E + F - C = 0 (3-torus T³)
    - χ(2-skeleton) = V - E + F = C (number of cells)
    - Every edge bounds exactly 3 faces (Plateau foam)
    - Every vertex has degree 4 (tetravalent)
    - Faces: pentagons and hexagons only

Date: Jan 2026
"""

import numpy as np
from scipy.spatial import Voronoi
from typing import Tuple, List, Dict, Optional
from itertools import product
from collections import defaultdict

from ..spec.structures import canonical_face as canonical_face_with_orient

# Precision for coordinate wrapping (increased for robustness)
WRAP_DECIMALS = 8
WRAP_TOL = 1e-8


def wrap_coord(x: float, L: float) -> float:
    """Wrap coordinate to [0, L)."""
    result = x % L
    if abs(result) < WRAP_TOL or abs(result - L) < WRAP_TOL:
        result = 0.0
    return result


def wrap_pos(pos: np.ndarray, L: float) -> tuple:
    """Wrap 3D position to canonical form in [0, L)³."""
    return tuple(round(wrap_coord(x, L), WRAP_DECIMALS) for x in pos)


def unwrap_coords_to_reference(coords: np.ndarray, L: float) -> np.ndarray:
    """
    Unwrap periodic coordinates to same image.

    For faces crossing periodic boundary, vertices may be in different
    periodic images (offset by ±L). This function brings them all to
    the same image by using the first vertex as reference.

    CRITICAL for:
        - Stable ordering (atan2 needs continuous coords)
        - Planarity checks
        - Edge construction

    Args:
        coords: (n, 3) array of vertex coordinates
        L: period length

    Returns:
        (n, 3) array with all vertices in same periodic image
    """
    if len(coords) == 0:
        return coords

    unwrapped = coords.copy()
    ref = unwrapped[0]

    for i in range(1, len(unwrapped)):
        for j in range(3):
            diff = unwrapped[i, j] - ref[j]
            if diff > L/2:
                unwrapped[i, j] -= L
            elif diff < -L/2:
                unwrapped[i, j] += L

    return unwrapped


def order_ridge_vertices(ridge_coords: np.ndarray, site1: np.ndarray, site2: np.ndarray) -> List[int]:
    """
    Order ridge vertices cyclically in the face plane.

    Ridge vertices from Voronoi are NOT guaranteed to be in cyclic order.
    This function projects them onto the face plane and sorts by angle.

    Args:
        ridge_coords: (n, 3) array of vertex coordinates
        site1, site2: the two Voronoi sites that share this face

    Returns:
        List of indices into ridge_coords in cyclic order
    """
    n = len(ridge_coords)
    if n < 3:
        return list(range(n))

    # Face normal: direction from site1 to site2
    normal = site2 - site1
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return list(range(n))
    normal = normal / norm_len

    # Centroid of ridge vertices
    centroid = np.mean(ridge_coords, axis=0)

    # Build orthonormal basis in face plane
    # u = any vector perpendicular to normal
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = arbitrary - np.dot(arbitrary, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project vertices onto (u, v) plane and compute angles
    angles = []
    for i, coord in enumerate(ridge_coords):
        rel = coord - centroid
        proj_u = np.dot(rel, u)
        proj_v = np.dot(rel, v)
        angle = np.arctan2(proj_v, proj_u)
        angles.append((angle, i))

    # Sort by angle
    angles.sort(key=lambda x: x[0])
    return [idx for _, idx in angles]


def is_simple_polygon(coords_2d: np.ndarray) -> bool:
    """
    Check if a 2D polygon is simple (non-self-intersecting).

    Args:
        coords_2d: (n, 2) array of 2D coordinates in cyclic order

    Returns:
        True if polygon is simple
    """
    n = len(coords_2d)
    if n < 4:
        return True  # Triangle is always simple

    def ccw(A, B, C):
        """Check if three points are in counter-clockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def segments_intersect(A, B, C, D):
        """Check if segment AB intersects segment CD (properly)."""
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    # Check all non-adjacent edge pairs
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # Adjacent edges (wrap around)
            A, B = coords_2d[i], coords_2d[(i + 1) % n]
            C, D = coords_2d[j], coords_2d[(j + 1) % n]
            if segments_intersect(A, B, C, D):
                return False
    return True


def canonical_face(face: List[int]) -> Optional[tuple]:
    """
    Canonicalize face for deduplication.

    Uses canonical_face from spec/structures.py for consistency.
    Returns None for degenerate faces (< 3 vertices).
    """
    if len(face) < 3:
        return None
    try:
        canon, _ = canonical_face_with_orient(face)
        return canon
    except ValueError:
        return None


def get_a15_points(N: int, L_cell: float = 1.0) -> np.ndarray:
    """
    Generate A15 lattice points for N×N×N supercell.

    A15 Wyckoff positions (space group Pm3n, No. 223):
        2a: (0,0,0), (1/2,1/2,1/2)
        6d: (1/4,0,1/2), (3/4,0,1/2), (1/2,1/4,0),
            (1/2,3/4,0), (0,1/2,1/4), (0,1/2,3/4)
    """
    frac = [
        [0, 0, 0], [0.5, 0.5, 0.5],  # 2a
        [0.25, 0, 0.5], [0.75, 0, 0.5],  # 6d
        [0.5, 0.25, 0], [0.5, 0.75, 0],
        [0, 0.5, 0.25], [0, 0.5, 0.75],
    ]
    points = []
    for i, j, k in product(range(N), repeat=3):
        for f in frac:
            p = [(i + f[0]) * L_cell, (j + f[1]) * L_cell, (k + f[2]) * L_cell]
            points.append(p)
    return np.array(points)


def build_wp_supercell_periodic(N: int, L_cell: float = 4.0) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]]]:
    """
    Build N×N×N Weaire-Phelan supercell with periodic boundary conditions.

    Uses Voronoi tessellation of A15 lattice.

    Args:
        N: supercell size (8N³ cells total)
        L_cell: side of fundamental domain

    Returns:
        vertices: (V, 3) array of unique vertex positions
        edges: list of (i, j) tuples with i < j
        faces: list of vertex index lists

    TOPOLOGY:
        - χ(3-complex) = 0 (3-torus T³)
        - Every edge bounds exactly 3 faces (Plateau foam)
        - Every vertex has degree 4

    Raises:
        ValueError: if N < 1 or L_cell <= 0
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if L_cell <= 0:
        raise ValueError(f"L_cell must be > 0, got {L_cell}")

    L = N * L_cell
    points = get_a15_points(N, L_cell)
    n_pts = len(points)

    # Create 3×3×3 periodic images for Voronoi
    images = []
    image_offset = []
    for di, dj, dk in product([-1, 0, 1], repeat=3):
        offset = np.array([di, dj, dk]) * L
        images.append(points + offset)
        image_offset.append((di, dj, dk))

    all_points = np.vstack(images)
    central_idx = image_offset.index((0, 0, 0))
    central_start = central_idx * n_pts
    central_end = central_start + n_pts

    # Compute Voronoi
    vor = Voronoi(all_points)

    # Collect unique vertices and faces with wrapping
    vertex_dict: Dict[tuple, int] = {}
    vertices: List[np.ndarray] = []
    face_set: Dict[tuple, List[int]] = {}

    def get_vertex_idx(pos: np.ndarray) -> int:
        wrapped = wrap_pos(pos, L)
        if wrapped not in vertex_dict:
            vertex_dict[wrapped] = len(vertices)
            vertices.append(np.array(wrapped))
        return vertex_dict[wrapped]

    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        ridge_verts = vor.ridge_vertices[ridge_idx]

        # Skip unbounded ridges
        if -1 in ridge_verts:
            continue

        # Only process if at least one point is in central cell
        in_c1 = central_start <= p1 < central_end
        in_c2 = central_start <= p2 < central_end

        if not (in_c1 or in_c2):
            continue

        # Get ridge vertex coordinates (before wrapping, for ordering)
        ridge_coords = np.array([vor.vertices[v_idx] for v_idx in ridge_verts])

        # CRITICAL: Unwrap coordinates to same periodic image before ordering
        # Without this, faces crossing periodic boundary have vertices in
        # different images, making atan2 ordering unstable
        ridge_coords_unwrapped = unwrap_coords_to_reference(ridge_coords, L)

        # Order vertices cyclically in face plane using site positions
        site1 = all_points[p1]
        site2 = all_points[p2]
        ordered_indices = order_ridge_vertices(ridge_coords_unwrapped, site1, site2)

        # Map vertices with wrapping, in cyclic order
        # CRITICAL: Use unwrapped coords (consistent with ordering) for mapping
        # get_vertex_idx will wrap to [0,L)³ anyway
        face = []
        for local_idx in ordered_indices:
            pos = ridge_coords_unwrapped[local_idx]
            new_idx = get_vertex_idx(pos)
            face.append(new_idx)

        # Skip degenerate faces (after wrapping, vertices may collapse)
        unique_verts = []
        for v in face:
            if not unique_verts or v != unique_verts[-1]:
                unique_verts.append(v)
        # Also check wrap-around
        if len(unique_verts) > 1 and unique_verts[0] == unique_verts[-1]:
            unique_verts = unique_verts[:-1]
        face = unique_verts

        if len(face) < 3:
            continue

        # Canonicalize and deduplicate
        canon = canonical_face(face)
        if canon and canon not in face_set:
            face_set[canon] = list(canon)

    faces = list(face_set.values())

    # Build edges from faces
    edge_set = set()
    for face in faces:
        n = len(face)
        for k in range(n):
            v1, v2 = face[k], face[(k+1) % n]
            edge = (min(v1, v2), max(v1, v2))
            edge_set.add(edge)

    edges = sorted(edge_set)

    return np.array(vertices), edges, faces


def get_wp_periodic_topology(N: int, L_cell: float = 4.0) -> Dict:
    """
    Compute topology numbers for periodic WP supercell.

    Returns:
        dict with V, E, F, n_cells, chi_2skeleton, chi_3complex
    """
    vertices, edges, faces = build_wp_supercell_periodic(N, L_cell)

    V = len(vertices)
    E = len(edges)
    F = len(faces)
    C = 8 * N**3  # 8 cells per fundamental domain

    return {
        'N': N,
        'n_cells': C,
        'V': V,
        'E': E,
        'F': F,
        'chi_2skeleton': V - E + F,
        'chi_3complex': V - E + F - C,
    }


def verify_wp_foam_structure(N: int, L_cell: float = 4.0) -> Dict:
    """
    Verify WP foam has correct Plateau structure.

    Checks:
        - Every edge bounds exactly 3 faces
        - Every vertex has degree 4
        - χ(3-complex) = 0
    """
    vertices, edges, faces = build_wp_supercell_periodic(N, L_cell)

    V, E, F = len(vertices), len(edges), len(faces)
    C = 8 * N**3

    # Edge-face count
    edge_face = defaultdict(int)
    for face in faces:
        n = len(face)
        for k in range(n):
            v1, v2 = face[k], face[(k+1) % n]
            edge = (min(v1, v2), max(v1, v2))
            edge_face[edge] += 1

    edge_face_counts = list(edge_face.values())
    all_3_faces = all(c == 3 for c in edge_face_counts)

    # Vertex degree
    vertex_deg = defaultdict(int)
    for i, j in edges:
        vertex_deg[i] += 1
        vertex_deg[j] += 1

    vertex_degs = list(vertex_deg.values())
    all_deg_4 = all(d == 4 for d in vertex_degs)

    # Face sizes
    face_sizes = [len(f) for f in faces]
    face_size_counts = defaultdict(int)
    for s in face_sizes:
        face_size_counts[s] += 1

    return {
        'N': N,
        'V': V, 'E': E, 'F': F, 'C': C,
        'chi_3complex': V - E + F - C,
        'all_edges_3_faces': all_3_faces,
        'all_vertices_deg_4': all_deg_4,
        'face_sizes': dict(face_size_counts),
        'is_valid_plateau_foam': all_3_faces and all_deg_4 and (V - E + F - C == 0),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("WEAIRE-PHELAN PERIODIC SUPERCELL (via Voronoi)")
    print("=" * 60)

    for N in [1, 2]:
        print(f"\n--- N = {N} ({8*N**3} cells) ---")

        result = verify_wp_foam_structure(N)

        print(f"V={result['V']}, E={result['E']}, F={result['F']}, C={result['C']}")
        print(f"χ(3-complex) = {result['chi_3complex']}")
        print(f"All edges bound 3 faces: {result['all_edges_3_faces']}")
        print(f"All vertices degree 4: {result['all_vertices_deg_4']}")
        print(f"Face sizes: {result['face_sizes']}")
        print(f"Valid Plateau foam: {result['is_valid_plateau_foam']}")
