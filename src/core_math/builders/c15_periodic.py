"""
Periodic C15 Laves (MgCu2) Supercell via Voronoi
================================================

N×N×N C15 foam with PERIODIC boundary conditions (3-torus T³).

CONSTRUCTION:
    1. Generate C15 lattice points (space group Fd-3m, No. 227)
    2. Compute Voronoi tessellation with periodic images
    3. Order ridge vertices cyclically in face plane
    4. Extract vertices, edges, faces with wrapping

STRUCTURE:
    24 cells per fundamental domain:
        - 8 cells at Wyckoff 8a (diamond sublattice)
        - 16 cells at Wyckoff 16d (pyrochlore sublattice)

TOPOLOGY (verified):
    - χ(3-complex) = V - E + F - C = 0 (3-torus T³)
    - χ(2-skeleton) = V - E + F = C (number of cells)
    - Every edge bounds exactly 3 faces (Plateau foam)
    - Every vertex has degree 4 (tetravalent)
    - Faces: pentagons and hexagons

ISOTROPY:
    C15 has δv/v = 0.93%, which is 2.7× more isotropic than
    Weaire-Phelan (δv/v = 2.53%).

Date: Jan 2026
"""

import numpy as np
from scipy.spatial import Voronoi
from typing import Tuple, List, Dict, Optional
from itertools import product
from collections import defaultdict

from ..spec.structures import canonical_face as canonical_face_with_orient

# Precision for coordinate wrapping
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

    angles.sort(key=lambda x: x[0])
    return [idx for _, idx in angles]


def canonical_face(face: List[int]) -> Optional[tuple]:
    """
    Canonicalize face for deduplication.
    """
    if len(face) < 3:
        return None
    try:
        canon, _ = canonical_face_with_orient(face)
        return canon
    except ValueError:
        return None


def get_c15_points(N: int, L_cell: float = 1.0) -> np.ndarray:
    """
    Generate C15 Laves lattice points for N×N×N supercell.

    C15 (MgCu2-type) Wyckoff positions (space group Fd-3m, No. 227):
        8a:  Diamond sublattice - (0,0,0), (1/4,1/4,1/4) + FCC
        16d: Pyrochlore sublattice - (5/8,5/8,5/8) + permutations + FCC

    Total: 24 sites per unit cell
    """
    # FCC translation vectors (in fractional coordinates)
    fcc_translations = [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ]

    # 8a sites (diamond sublattice)
    sites_8a_base = [[0, 0, 0], [0.25, 0.25, 0.25]]

    # 16d sites (pyrochlore sublattice)
    sites_16d_base = [
        [5/8, 5/8, 5/8],
        [5/8, 3/8, 3/8],
        [3/8, 5/8, 3/8],
        [3/8, 3/8, 5/8],
    ]

    # Generate all fractional positions in unit cell
    frac_positions = []

    # 8a sites
    for base in sites_8a_base:
        for t in fcc_translations:
            pos = [(base[j] + t[j]) % 1.0 for j in range(3)]
            frac_positions.append(pos)

    # 16d sites
    for base in sites_16d_base:
        for t in fcc_translations:
            pos = [(base[j] + t[j]) % 1.0 for j in range(3)]
            frac_positions.append(pos)

    # Remove duplicates using deterministic rounding
    # Round to 8 decimals and use set for dedup (more robust than tolerance checks)
    seen = set()
    unique_frac = []
    for pos in frac_positions:
        # Wrap to [0, 1) and round for deterministic comparison
        wrapped = tuple(round(x % 1.0, 8) for x in pos)
        # Handle edge case: 1.0 rounds to 1.0, should be 0.0
        wrapped = tuple(0.0 if x == 1.0 else x for x in wrapped)
        if wrapped not in seen:
            seen.add(wrapped)
            unique_frac.append(list(wrapped))

    # Generate supercell points
    points = []
    for i, j, k in product(range(N), repeat=3):
        for f in unique_frac:
            p = [(i + f[0]) * L_cell, (j + f[1]) * L_cell, (k + f[2]) * L_cell]
            points.append(p)

    return np.array(points)


def build_c15_supercell_periodic(
    N: int,
    L_cell: float = 4.0,
    points: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], List[List[Tuple[int, int]]]]:
    """
    Build N×N×N C15 Laves supercell with periodic boundary conditions.

    Uses Voronoi tessellation of C15 lattice.

    Args:
        N: supercell size (24N³ cells total)
        L_cell: side of fundamental domain
        points: optional custom points array (for testing permutation invariance)

    Returns:
        vertices: (V, 3) array of unique vertex positions
        edges: list of (i, j) tuples with i < j
        faces: list of vertex index lists
        cell_face_incidence: list of [(face_idx, orientation)] per cell

    TOPOLOGY:
        - χ(3-complex) = 0 (3-torus T³)
        - Every edge bounds exactly 3 faces (Plateau foam)
        - Every vertex has degree 4
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if L_cell <= 0:
        raise ValueError(f"L_cell must be > 0, got {L_cell}")

    L = N * L_cell
    if points is None:
        points = get_c15_points(N, L_cell)
    else:
        points = np.asarray(points, dtype=float).copy()
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

    # face_data[canonical] = {
    #     'face': list (canonical vertex order),
    #     'cells': {cell_idx: orientation}
    # }
    face_data: Dict[tuple, dict] = {}

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

        # Get ridge vertex coordinates
        ridge_coords = np.array([vor.vertices[v_idx] for v_idx in ridge_verts])

        # Unwrap coordinates to same periodic image before ordering
        ridge_coords_unwrapped = unwrap_coords_to_reference(ridge_coords, L)

        # Order vertices cyclically in face plane
        # Normal direction: site1 → site2
        site1 = all_points[p1]
        site2 = all_points[p2]
        ordered_indices = order_ridge_vertices(ridge_coords_unwrapped, site1, site2)

        # Map vertices with wrapping, in cyclic order
        face = []
        for local_idx in ordered_indices:
            pos = ridge_coords_unwrapped[local_idx]
            new_idx = get_vertex_idx(pos)
            face.append(new_idx)

        # Skip degenerate faces
        unique_verts = []
        for v in face:
            if not unique_verts or v != unique_verts[-1]:
                unique_verts.append(v)
        if len(unique_verts) > 1 and unique_verts[0] == unique_verts[-1]:
            unique_verts = unique_verts[:-1]
        face = unique_verts

        if len(face) < 3:
            continue

        # Skip faces with repeated vertices (non-simple cycles after wrapping)
        if len(set(face)) != len(face):
            continue

        # Canonicalize with orientation tracking
        try:
            canon, rel_orient = canonical_face_with_orient(face)
        except ValueError:
            continue

        # Cell indices: Voronoi point index mod n_pts gives cell in fundamental domain
        cell_p1 = p1 % n_pts
        cell_p2 = p2 % n_pts

        if canon not in face_data:
            face_data[canon] = {
                'face': list(canon),
                'cells': {}
            }

        # Orientation convention:
        # - Face vertices ordered with normal from site1 → site2
        # - rel_orient = +1: canonical winding matches face → normal points p1→p2
        #   → Cell p1: outward (+rel_orient), Cell p2: inward (-rel_orient)
        # - rel_orient = -1: canonical winding reversed
        #   → Cell p1: outward (+rel_orient), Cell p2: inward (-rel_orient)
        cells = face_data[canon]['cells']
        if in_c1:
            if cell_p1 in cells:
                assert cells[cell_p1] == rel_orient, \
                    f"Orientation inconsistency: cell {cell_p1}, face {canon[:3]}..."
            cells[cell_p1] = rel_orient
        if in_c2:
            if cell_p2 in cells:
                assert cells[cell_p2] == -rel_orient, \
                    f"Orientation inconsistency: cell {cell_p2}, face {canon[:3]}..."
            cells[cell_p2] = -rel_orient

    # Build face list and canonical-to-index mapping
    faces = []
    canonical_to_face_idx = {}
    for canonical, data in face_data.items():
        canonical_to_face_idx[canonical] = len(faces)
        faces.append(data['face'])

    # Build cell_face_incidence: for each cell, list of (face_idx, orientation)
    n_cells = 24 * N**3
    cell_face_incidence: List[List[Tuple[int, int]]] = [[] for _ in range(n_cells)]
    for canonical, data in face_data.items():
        face_idx = canonical_to_face_idx[canonical]
        for cell_idx, orientation in data['cells'].items():
            cell_face_incidence[cell_idx].append((face_idx, orientation))

    # Build edges from faces
    edge_set = set()
    for face in faces:
        n = len(face)
        for k in range(n):
            v1, v2 = face[k], face[(k+1) % n]
            edge = (min(v1, v2), max(v1, v2))
            edge_set.add(edge)

    edges = sorted(edge_set)

    return np.array(vertices), edges, faces, cell_face_incidence


def get_c15_periodic_topology(N: int, L_cell: float = 4.0) -> Dict:
    """
    Compute topology numbers for periodic C15 supercell.
    """
    vertices, edges, faces, _ = build_c15_supercell_periodic(N, L_cell)

    V = len(vertices)
    E = len(edges)
    F = len(faces)
    C = 24 * N**3  # 24 cells per fundamental domain

    return {
        'N': N,
        'n_cells': C,
        'V': V,
        'E': E,
        'F': F,
        'chi_2skeleton': V - E + F,
        'chi_3complex': V - E + F - C,
    }


def verify_c15_foam_structure(N: int, L_cell: float = 4.0) -> Dict:
    """
    Verify C15 foam has correct Plateau structure.

    Checks:
        - Every edge bounds exactly 3 faces
        - Every vertex has degree 4
        - χ(3-complex) = 0
    """
    vertices, edges, faces, _ = build_c15_supercell_periodic(N, L_cell)

    V, E, F = len(vertices), len(edges), len(faces)
    C = 24 * N**3

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

    # Vertex degree (check ALL vertices have degree info)
    vertex_deg = defaultdict(int)
    for i, j in edges:
        vertex_deg[i] += 1
        vertex_deg[j] += 1

    # Verify all V vertices appear in edges (no isolated vertices)
    all_vertices_have_edges = len(vertex_deg) == V

    vertex_degs = list(vertex_deg.values())
    all_deg_4 = all(d == 4 for d in vertex_degs) and all_vertices_have_edges

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
    print("C15 LAVES PERIODIC SUPERCELL (via Voronoi)")
    print("=" * 60)

    for N in [1]:
        print(f"\n--- N = {N} ({24*N**3} cells) ---")

        result = verify_c15_foam_structure(N)

        print(f"V={result['V']}, E={result['E']}, F={result['F']}, C={result['C']}")
        print(f"χ(3-complex) = {result['chi_3complex']}")
        print(f"All edges bound 3 faces: {result['all_edges_3_faces']}")
        print(f"All vertices degree 4: {result['all_vertices_deg_4']}")
        print(f"Face sizes: {result['face_sizes']}")
        print(f"Valid Plateau foam: {result['is_valid_plateau_foam']}")
