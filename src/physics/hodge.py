"""
Voronoi Dual Hodge Stars for DEC on Foam Complexes
===================================================

Correct DEC Hodge star operators using Voronoi dual geometry.

Standard DEC formula:
    *[σ] = |dual(σ)| / |σ|

For 3D Voronoi foam:
    *₁[e] = dual_face_area(e) / edge_length(e)
    *₂[f] = dual_edge_length(f) / face_area(f)

Where:
    - dual_face(e) = polygon formed by cell centers around edge e
    - dual_edge(f) = segment connecting the two cell centers adjacent to face f

IMPORTANT: Plateau Foam Complex (not raw Voronoi)
-------------------------------------------------
We use a PLATEAU FOAM COMPLEX derived from Voronoi ridges.

The current builders (C15, Kelvin/BCC, WP/A15) produce Plateau foams where:
    - 4 edges per vertex (tetrahedral angle 109.47°)
    - 3 faces per edge (Plateau's first law)
    - 3 cells per edge

NOTE: Not all lattices produce Plateau foams. FCC, for example, has vertices
with 8 edges and does NOT satisfy Plateau structure. Use verify_plateau_structure()
to check any new lattice before assuming Plateau properties.

This is a specific choice of complex. The dual cells are defined with respect
to THIS foam complex, not the classical Delaunay/Voronoi circumcentric dual.

SUPPORTED LATTICES:
    - C15 (Laves): 24 cells/unit, Frank-Kasper polyhedra
    - Kelvin (BCC): 2 cells/unit, truncated octahedra
    - WP (A15): 8 cells/unit, Weaire-Phelan structure

VERIFIED (C15, Kelvin, WP - Jan 2026):
    - Voronoi property: vertices equidistant from adjacent sites (< 1e-8)
    - Plateau structure: 4 edges/vertex, 3 faces/edge
    - Dual orthogonality: |n̂·d̂| > 0.99
    - Exactness: d₁d₀ = 0

COMPARISON with uniform Hodge:
    - build_hodge_stars_uniform: *₁ = a², *₂ = a² (same for all edges/faces)
    - build_hodge_stars_voronoi: *₁[e], *₂[f] vary by local geometry

Jan 2026
"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from itertools import product


# =============================================================================
# COORDINATE WRAPPING UTILITIES
# =============================================================================

WRAP_DECIMALS = 10
WRAP_TOL = 1e-10


def wrap_coord(x: float, L: float) -> float:
    """Wrap coordinate to [0, L)."""
    result = x % L
    if abs(result) < WRAP_TOL or abs(result - L) < WRAP_TOL:
        result = 0.0
    return result


def wrap_pos(pos: np.ndarray, L: float) -> tuple:
    """Wrap 3D position to canonical form in [0, L)³."""
    return tuple(round(wrap_coord(x, L), WRAP_DECIMALS) for x in pos)


def wrap_delta(delta: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Wrap delta vector to [-L/2, L/2] (component-wise for non-cubic)."""
    return delta - L * np.round(delta / L)


def unwrap_coords_to_reference(coords: np.ndarray, L: float) -> np.ndarray:
    """Unwrap periodic coordinates to same image."""
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


# =============================================================================
# FACE PROCESSING
# =============================================================================

def order_ridge_vertices(ridge_coords: np.ndarray, site1: np.ndarray,
                         site2: np.ndarray) -> List[int]:
    """Order ridge vertices cyclically in face plane."""
    n = len(ridge_coords)
    if n < 3:
        return list(range(n))

    normal = site2 - site1
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return list(range(n))
    normal = normal / norm_len

    centroid = np.mean(ridge_coords, axis=0)

    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = arbitrary - np.dot(arbitrary, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

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
    """Canonicalize face for deduplication.

    LEGACY: Not used in current dedup (we use frozenset of edges instead).
    Kept for potential future use or debugging.
    """
    if len(face) < 3:
        return None

    n = len(face)
    rotations = []
    for start in range(n):
        rotations.append(tuple(face[(start + i) % n] for i in range(n)))
    reversed_face = face[::-1]
    for start in range(n):
        rotations.append(tuple(reversed_face[(start + i) % n] for i in range(n)))

    return min(rotations)


# =============================================================================
# LATTICE POINT GENERATORS
# =============================================================================

def get_c15_points(N: int, L_cell: float = 1.0) -> np.ndarray:
    """
    Generate C15 Laves lattice points for N×N×N supercell.

    C15 (MgCu2 structure): 24 atoms per cubic unit cell.
    Voronoi cells: 12-hedra (Frank-Kasper polyhedra).

    Args:
        N: supercell size
        L_cell: unit cell size

    Returns:
        (24*N³, 3) array of points
    """
    fcc_translations = [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ]

    sites_8a_base = [[0, 0, 0], [0.25, 0.25, 0.25]]
    sites_16d_base = [
        [5/8, 5/8, 5/8],
        [5/8, 3/8, 3/8],
        [3/8, 5/8, 3/8],
        [3/8, 3/8, 5/8],
    ]

    frac_positions = []
    for base in sites_8a_base:
        for t in fcc_translations:
            pos = [(base[j] + t[j]) % 1.0 for j in range(3)]
            frac_positions.append(pos)
    for base in sites_16d_base:
        for t in fcc_translations:
            pos = [(base[j] + t[j]) % 1.0 for j in range(3)]
            frac_positions.append(pos)

    seen = set()
    unique_frac = []
    for pos in frac_positions:
        wrapped = tuple(round(x % 1.0, 8) for x in pos)
        wrapped = tuple(0.0 if x == 1.0 else x for x in wrapped)
        if wrapped not in seen:
            seen.add(wrapped)
            unique_frac.append(list(wrapped))

    points = []
    for i, j, k in product(range(N), repeat=3):
        for f in unique_frac:
            p = [(i + f[0]) * L_cell, (j + f[1]) * L_cell, (k + f[2]) * L_cell]
            points.append(p)

    return np.array(points)


def get_bcc_points(N: int, L_cell: float = 1.0) -> np.ndarray:
    """
    Generate BCC lattice points for N×N×N supercell.

    BCC: 2 atoms per cubic unit cell.
    Voronoi cells: truncated octahedra (Kelvin cells).

    Args:
        N: supercell size
        L_cell: unit cell size

    Returns:
        (2*N³, 3) array of points
    """
    # BCC has 2 atoms per unit cell
    frac_positions = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ]

    points = []
    for i, j, k in product(range(N), repeat=3):
        for f in frac_positions:
            p = [(i + f[0]) * L_cell, (j + f[1]) * L_cell, (k + f[2]) * L_cell]
            points.append(p)

    return np.array(points)


def get_a15_points(N: int, L_cell: float = 1.0) -> np.ndarray:
    """
    Generate A15 lattice points for N×N×N supercell.

    A15 (Cr3Si structure): 8 atoms per cubic unit cell.
    Voronoi cells: Weaire-Phelan structure (2 dodecahedra + 6 tetrakaidecahedra).

    Args:
        N: supercell size
        L_cell: unit cell size

    Returns:
        (8*N³, 3) array of points
    """
    # A15 has 8 atoms per unit cell
    # Type A (Wyckoff 2a): 2 atoms
    # Type B (Wyckoff 6c): 6 atoms
    frac_positions = [
        # Type A
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        # Type B
        [0.25, 0.0, 0.5],
        [0.75, 0.0, 0.5],
        [0.5, 0.25, 0.0],
        [0.5, 0.75, 0.0],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.75],
    ]

    points = []
    for i, j, k in product(range(N), repeat=3):
        for f in frac_positions:
            p = [(i + f[0]) * L_cell, (j + f[1]) * L_cell, (k + f[2]) * L_cell]
            points.append(p)

    return np.array(points)


# =============================================================================
# GENERIC FOAM BUILDER WITH DUAL INFO
# =============================================================================

def build_foam_with_dual_info(points: np.ndarray, L: float) -> Dict:
    """
    Build foam complex with full dual structure from lattice points.

    Generic Voronoi-based foam builder. Takes any set of periodic lattice points
    and returns the foam complex with all dual structure information needed for
    correct DEC Hodge star computation.

    IMPORTANT: The resulting foam is NOT guaranteed to be a Plateau foam.
    Use verify_plateau_structure() to check if it satisfies:
        - 4 edges per vertex
        - 3 faces per edge

    Args:
        points: (n_cells, 3) array of cell center positions
        L: cubic box size (points should be in [0, L)³)

    Returns:
        dict with:
            'V': vertices (n_V, 3)
            'E': edges [(i,j), ...] sorted
            'F': faces [[v0,v1,...], ...]
            'L': box size (scalar for cubic)
            'L_vec': box size vector (3,)
            'cell_centers': sites (n_cells, 3)
            'face_to_cells': dict[face_idx] -> (cell_a, cell_b)
            'face_to_cell_shift': dict[face_idx] -> (3,) int array, lattice shift
            'edge_to_cells': dict[edge_tuple] -> set of cell indices
            'edge_to_faces': dict[edge_tuple] -> list of face indices

        The face_to_cell_shift stores the lattice translation from cell_a to cell_b
        in units of L. This is CRITICAL for computing the correct dual edge when
        the two cells are in different periodic images.
    """
    L_vec = np.array([L, L, L])
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

    vor = Voronoi(all_points)

    vertex_dict: Dict[tuple, int] = {}
    vertices: List[np.ndarray] = []
    face_list: List[List[int]] = []
    face_to_cells: Dict[int, Tuple[int, int]] = {}
    face_to_cell_shift: Dict[int, np.ndarray] = {}  # lattice shift from cell_a to cell_b
    # Dedup: edge_key -> (idx, pair_unordered, shift_unordered, area, n_edges, centroid_key)
    seen_faces: Dict[frozenset, Tuple[int, Tuple[int, int], np.ndarray, float, int, tuple]] = {}

    def compute_face_fingerprints(positions: List[np.ndarray], L: float) -> Tuple[float, tuple]:
        """Compute face area and centroid key from unwrapped positions (for dedup).

        Returns:
            area: face area
            centroid_key: tuple of rounded centroid coords (mod L for periodicity)
        """
        if len(positions) < 3:
            return 0.0, (0.0, 0.0, 0.0)
        pos = np.array(positions)
        center = np.mean(pos, axis=0)

        # Area via cross product
        total_cross = np.zeros(3)
        n = len(pos)
        for i in range(n):
            v1 = pos[i] - center
            v2 = pos[(i + 1) % n] - center
            total_cross += np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(total_cross)

        # Centroid key: wrap to [0, L) and round for comparison
        # Handle boundary case: values very close to L should wrap to 0
        def wrap_coord(c):
            wrapped = c % L
            rounded = round(wrapped, 6)
            # If rounded == L (boundary case), wrap to 0
            if rounded >= L:
                rounded = 0.0
            return rounded
        centroid_wrapped = tuple(wrap_coord(c) for c in center)
        return area, centroid_wrapped

    def get_vertex_idx(pos: np.ndarray) -> int:
        wrapped = wrap_pos(pos, L)
        if wrapped not in vertex_dict:
            vertex_dict[wrapped] = len(vertices)
            vertices.append(np.array(wrapped))
        return vertex_dict[wrapped]

    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        ridge_verts = vor.ridge_vertices[ridge_idx]

        if -1 in ridge_verts:
            continue

        in_c1 = central_start <= p1 < central_end
        in_c2 = central_start <= p2 < central_end

        if not (in_c1 or in_c2):
            continue

        ridge_coords = np.array([vor.vertices[v_idx] for v_idx in ridge_verts])

        site1 = all_points[p1]
        site2 = all_points[p2]

        # Initial unwrap for ordering (uses first vertex as reference)
        ridge_coords_unwrapped = unwrap_coords_to_reference(ridge_coords, L)
        ordered_indices = order_ridge_vertices(ridge_coords_unwrapped, site1, site2)

        # SAFER: rebuild positions with incremental unwrap following the order
        # This guarantees consecutive vertices are properly unwrapped
        raw_ordered = [ridge_coords[ordered_indices[0]]]
        for k in range(1, len(ordered_indices)):
            prev_pos = raw_ordered[-1]
            curr_raw = ridge_coords[ordered_indices[k]]
            delta = curr_raw - prev_pos
            delta = delta - L * np.round(delta / L)  # wrap to [-L/2, L/2]
            raw_ordered.append(prev_pos + delta)

        face = []
        for pos in raw_ordered:
            new_idx = get_vertex_idx(pos)
            face.append(new_idx)

        # Skip degenerate
        unique_verts = []
        for v in face:
            if not unique_verts or v != unique_verts[-1]:
                unique_verts.append(v)
        if len(unique_verts) > 1 and unique_verts[0] == unique_verts[-1]:
            unique_verts = unique_verts[:-1]
        face = unique_verts

        if len(face) < 3 or len(set(face)) != len(face):
            continue

        # ROBUST DEDUP KEY: frozenset of edges (invariant under any vertex permutation)
        # This avoids issues with canonical_face when wrap/round produces different orderings
        n_verts = len(face)
        edge_key = frozenset(
            (min(face[i], face[(i+1) % n_verts]), max(face[i], face[(i+1) % n_verts]))
            for i in range(n_verts)
        )

        # Map site indices back to central cell AND compute lattice shift
        cell_a = p1 % n_pts
        cell_b = p2 % n_pts
        # Compute shift: which periodic image is each site in?
        img_a = np.array(image_offset[p1 // n_pts])
        img_b = np.array(image_offset[p2 // n_pts])
        shift_ab = img_b - img_a  # shift from cell_a's image to cell_b's image

        # CRITICAL: shift_ab is anchored to the ORIENTED pair (cell_a, cell_b).
        # Do NOT normalize based on min/max - that would break the geometric meaning.
        # For dedup, use unordered pair but verify shift consistency with orientation.
        cell_pair_unordered = (min(cell_a, cell_b), max(cell_a, cell_b))
        # Convert shift to unordered representation for dedup check only
        if cell_a <= cell_b:
            shift_unordered = shift_ab.copy()
        else:
            shift_unordered = -shift_ab  # flip for consistent dedup key

        # Geometric fingerprints: area + centroid (from raw unwrapped positions)
        # Additional checks beyond edge_key in case of numeric edge collisions
        face_area_fingerprint, centroid_key = compute_face_fingerprints(raw_ordered, L)

        if edge_key not in seen_faces:
            face_idx = len(face_list)
            # Store geometric order (face) for correct boundary orientation
            face_list.append(face)
            # Store unordered pair/shift/area/n_edges/centroid for dedup verification
            seen_faces[edge_key] = (face_idx, cell_pair_unordered, shift_unordered,
                                    face_area_fingerprint, n_verts, centroid_key)
            # But face_to_cells and face_to_cell_shift use ORIENTED pair
            face_to_cells[face_idx] = (cell_a, cell_b)
            face_to_cell_shift[face_idx] = shift_ab
        else:
            # Verify cell pair, shift, area, edge count, AND centroid consistency
            existing_idx, existing_pair, existing_shift, existing_area, existing_n, existing_centroid = seen_faces[edge_key]
            if n_verts != existing_n:
                raise AssertionError(
                    f"Face dedup collision: n_edges {existing_n} vs {n_verts}"
                )
            if cell_pair_unordered != existing_pair:
                raise AssertionError(
                    f"Face dedup collision: cells {existing_pair} vs {cell_pair_unordered}"
                )
            if not np.array_equal(shift_unordered, existing_shift):
                raise AssertionError(
                    f"Face dedup collision: shift {existing_shift} vs {shift_unordered}"
                )
            # Area should match within RELATIVE tolerance (scales with L_cell)
            area_diff = abs(face_area_fingerprint - existing_area)
            max_area = max(existing_area, face_area_fingerprint, 1e-12)
            rel_diff = area_diff / max_area
            if rel_diff > 1e-6:
                raise AssertionError(
                    f"Face dedup collision: area {existing_area:.6f} vs {face_area_fingerprint:.6f} "
                    f"(rel_diff={rel_diff:.2e})"
                )
            # Centroid should match (ultimate hardening against accidental area collision)
            if centroid_key != existing_centroid:
                raise AssertionError(
                    f"Face dedup collision: centroid {existing_centroid} vs {centroid_key}"
                )

    # Build edges from faces
    edge_set = set()
    for face in face_list:
        n = len(face)
        for k in range(n):
            v1, v2 = face[k], face[(k+1) % n]
            edge = (min(v1, v2), max(v1, v2))
            edge_set.add(edge)

    edges = sorted(edge_set)

    # Build edge_to_faces and edge_to_cells
    edge_to_faces_map = defaultdict(list)
    for f_idx, face in enumerate(face_list):
        n = len(face)
        for k in range(n):
            v1, v2 = face[k], face[(k+1) % n]
            edge = (min(v1, v2), max(v1, v2))
            edge_to_faces_map[edge].append(f_idx)

    edge_to_cells: Dict[Tuple[int, int], set] = {}
    for edge in edges:
        cells = set()
        for f_idx in edge_to_faces_map[edge]:
            ca, cb = face_to_cells[f_idx]
            cells.add(ca)
            cells.add(cb)
        edge_to_cells[edge] = cells

    return {
        'V': np.array(vertices),
        'E': edges,
        'F': face_list,
        'L': L,
        'L_vec': L_vec,
        'cell_centers': points,
        'face_to_cells': face_to_cells,
        'face_to_cell_shift': face_to_cell_shift,  # lattice shift for dual edge
        'edge_to_cells': edge_to_cells,
        'edge_to_faces': dict(edge_to_faces_map),
    }


# =============================================================================
# CONVENIENCE BUILDERS FOR SPECIFIC LATTICES
# =============================================================================

def build_c15_with_dual_info(N: int, L_cell: float = 4.0) -> Dict:
    """
    Build C15 (Laves) foam with dual structure.

    C15 produces Frank-Kasper polyhedra. This IS a Plateau foam:
        - 4 edges per vertex
        - 3 faces per edge

    Args:
        N: supercell size (24*N³ cells)
        L_cell: unit cell size

    Returns:
        Dict with V, E, F, L, L_vec, cell_centers, face_to_cells,
        face_to_cell_shift, edge_to_cells, edge_to_faces
    """
    L = N * L_cell
    points = get_c15_points(N, L_cell)
    return build_foam_with_dual_info(points, L)


def build_kelvin_with_dual_info(N: int, L_cell: float = 4.0) -> Dict:
    """
    Build Kelvin foam (BCC Voronoi) with dual structure.

    Kelvin produces truncated octahedra. This IS a Plateau foam:
        - 4 edges per vertex
        - 3 faces per edge
        - 14 faces per cell (8 hexagons + 6 squares)

    Args:
        N: supercell size (2*N³ cells)
        L_cell: unit cell size

    Returns:
        Dict with V, E, F, L, L_vec, cell_centers, face_to_cells,
        face_to_cell_shift, edge_to_cells, edge_to_faces
    """
    L = N * L_cell
    points = get_bcc_points(N, L_cell)
    return build_foam_with_dual_info(points, L)


def build_wp_with_dual_info(N: int, L_cell: float = 4.0) -> Dict:
    """
    Build Weaire-Phelan foam (A15 Voronoi) with dual structure.

    WP produces 2 types of cells per unit:
        - 2 dodecahedra (12 pentagons)
        - 6 tetrakaidecahedra (12 pentagons + 2 hexagons)

    This IS a Plateau foam:
        - 4 edges per vertex
        - 3 faces per edge

    Args:
        N: supercell size (8*N³ cells)
        L_cell: unit cell size

    Returns:
        Dict with V, E, F, L, L_vec, cell_centers, face_to_cells,
        face_to_cell_shift, edge_to_cells, edge_to_faces
    """
    L = N * L_cell
    points = get_a15_points(N, L_cell)
    return build_foam_with_dual_info(points, L)


# =============================================================================
# GEOMETRY COMPUTATIONS
# =============================================================================

def compute_edge_length(V: np.ndarray, edge: Tuple[int, int],
                        L_vec: np.ndarray) -> float:
    """Compute edge length with periodic wrapping."""
    i, j = edge
    delta = V[j] - V[i]
    delta = wrap_delta(delta, L_vec)
    return np.linalg.norm(delta)


def compute_face_area(V: np.ndarray, face: List[int],
                      L_vec: np.ndarray) -> float:
    """Compute face area with periodic wrapping."""
    positions = [V[face[0]].copy()]
    for i in range(1, len(face)):
        delta = V[face[i]] - V[face[i-1]]
        delta = wrap_delta(delta, L_vec)
        positions.append(positions[-1] + delta)
    positions = np.array(positions)

    n = len(positions)
    center = np.mean(positions, axis=0)
    total_cross = np.zeros(3)
    for i in range(n):
        v1 = positions[i] - center
        v2 = positions[(i + 1) % n] - center
        total_cross += np.cross(v1, v2)

    return 0.5 * np.linalg.norm(total_cross)


def compute_dual_edge_length(cell_centers: np.ndarray, cell_a: int, cell_b: int,
                             L_vec: np.ndarray, shift: np.ndarray = None) -> float:
    """
    Compute length of dual edge (segment between two cell centers).

    Args:
        cell_centers: (n_cells, 3) array
        cell_a, cell_b: cell indices
        L_vec: box size vector
        shift: (3,) int array, lattice shift from cell_a to cell_b.
               If None, uses wrap_delta (legacy behavior, may pick wrong image).

    Returns:
        Length of dual edge.
    """
    if shift is not None:
        # Use exact shift from face construction
        delta = cell_centers[cell_b] + shift * L_vec - cell_centers[cell_a]
    else:
        # Legacy: wrap to nearest image (may be incorrect for some faces)
        delta = cell_centers[cell_b] - cell_centers[cell_a]
        delta = wrap_delta(delta, L_vec)
    return np.linalg.norm(delta)


def compute_dual_face_area(V: np.ndarray, edge: Tuple[int, int],
                           cell_centers: np.ndarray, cell_indices: set,
                           L_vec: np.ndarray) -> float:
    """
    Compute area of dual face (polygon formed by cell centers around edge).

    For n_cells==3 (Plateau foam): computes triangle area directly.
    For n_cells>3: uses ConvexHull to handle potential interior points.
    """
    cell_list = list(cell_indices)
    n_cells = len(cell_list)

    if n_cells < 3:
        raise ValueError(f"Edge {edge} has only {n_cells} adjacent cells, need >= 3")

    i, j = edge
    edge_vec = V[j] - V[i]
    edge_vec = wrap_delta(edge_vec, L_vec)
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        raise ValueError(f"Edge {edge} has zero length")
    edge_dir = edge_vec / edge_len

    midpoint = V[i] + 0.5 * edge_vec

    positions = []
    for c_idx in cell_list:
        delta = cell_centers[c_idx] - midpoint
        delta = wrap_delta(delta, L_vec)
        positions.append(delta)
    positions = np.array(positions)

    # Project onto plane perpendicular to edge
    projected = positions - np.outer(positions @ edge_dir, edge_dir)

    # Build orthonormal basis in plane
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(edge_dir, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = arbitrary - np.dot(arbitrary, edge_dir) * edge_dir
    u = u / np.linalg.norm(u)
    w = np.cross(edge_dir, u)

    coords_2d = np.column_stack([projected @ u, projected @ w])

    # For exactly 3 cells (Plateau foam): compute triangle area directly
    # This avoids ConvexHull issues with nearly collinear points
    if n_cells == 3:
        p1, p2, p3 = coords_2d[0], coords_2d[1], coords_2d[2]
        # Triangle area = 0.5 * |cross(p2-p1, p3-p1)| in 2D = 0.5 * |det|
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                         (p3[0] - p1[0]) * (p2[1] - p1[1]))
        # Use relative threshold based on cell scale (L_cell²)
        L_cell = np.min(L_vec)
        area_threshold = 1e-10 * L_cell**2
        if area < area_threshold:
            raise ValueError(
                f"Edge {edge}: dual face has near-zero area ({area:.2e}), "
                f"threshold={area_threshold:.2e}, cells {cell_list} may be collinear. "
                f"coords_2d: p1={p1}, p2={p2}, p3={p3}"
            )
        return area

    # For >3 cells: use ConvexHull
    try:
        hull = ConvexHull(coords_2d)
        ordered_coords = coords_2d[hull.vertices]
    except Exception:
        # Fallback to angle-ordering
        centroid_2d = np.mean(coords_2d, axis=0)
        angles = np.arctan2(coords_2d[:, 1] - centroid_2d[1],
                           coords_2d[:, 0] - centroid_2d[0])
        order = np.argsort(angles)
        ordered_coords = coords_2d[order]

    # Shoelace formula
    area = 0.5 * abs(
        np.dot(ordered_coords[:, 0], np.roll(ordered_coords[:, 1], -1)) -
        np.dot(ordered_coords[:, 1], np.roll(ordered_coords[:, 0], -1))
    )

    return area


# =============================================================================
# HODGE STAR COMPUTATION
# =============================================================================

def build_hodge_stars_voronoi(data: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build correct DEC Hodge stars from Voronoi dual structure.

    Args:
        data: Dict from build_c15_with_dual_info (or similar) containing:
            V, E, F, L_vec, cell_centers, face_to_cells, edge_to_cells

    Returns:
        star1: (n_E,) array - *₁[e] = dual_face_area / edge_length
        star2: (n_F,) array - *₂[f] = dual_edge_length / face_area
    """
    V = data['V']
    E = data['E']
    F = data['F']
    L_vec = data['L_vec']
    cell_centers = data['cell_centers']
    face_to_cells = data['face_to_cells']
    face_to_cell_shift = data.get('face_to_cell_shift', {})  # optional for backward compat
    edge_to_cells = data['edge_to_cells']

    n_E = len(E)
    n_F = len(F)

    # star2: dual_edge_length / face_area
    star2 = np.zeros(n_F)
    for f_idx in range(n_F):
        face = F[f_idx]
        cell_a, cell_b = face_to_cells[f_idx]
        shift = face_to_cell_shift.get(f_idx, None)

        face_area = compute_face_area(V, face, L_vec)
        dual_len = compute_dual_edge_length(cell_centers, cell_a, cell_b, L_vec, shift)

        if face_area < 1e-12:
            raise ValueError(f"Face {f_idx} has zero area")

        star2[f_idx] = dual_len / face_area

    # star1: dual_face_area / edge_length
    star1 = np.zeros(n_E)
    for e_idx, edge in enumerate(E):
        cells = edge_to_cells[edge]

        edge_len = compute_edge_length(V, edge, L_vec)
        dual_area = compute_dual_face_area(V, edge, cell_centers, cells, L_vec)

        if edge_len < 1e-12:
            raise ValueError(f"Edge {e_idx} has zero length")

        star1[e_idx] = dual_area / edge_len

    return star1, star2


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_plateau_structure(data: Dict) -> Dict:
    """
    Verify that the mesh is a valid Plateau foam complex.

    Returns dict with verification results.
    """
    V = data['V']
    E = data['E']
    F = data['F']
    L_vec = data['L_vec']

    results = {}

    # Check vertex degrees (should all be 4)
    # Initialize for ALL vertices to catch isolated vertices
    vertex_deg = {i: 0 for i in range(len(V))}
    for (i, j) in E:
        vertex_deg[i] += 1
        vertex_deg[j] += 1
    degs = list(vertex_deg.values())
    results['edges_per_vertex'] = set(degs)
    results['plateau_vertices'] = (set(degs) == {4})

    # Check faces per edge (should all be 3)
    edge_to_faces = defaultdict(set)
    for f_idx, face in enumerate(F):
        for k in range(len(face)):
            e = (min(face[k], face[(k+1)%len(face)]),
                 max(face[k], face[(k+1)%len(face)]))
            edge_to_faces[e].add(f_idx)
    face_counts = [len(edge_to_faces[e]) for e in E]
    results['faces_per_edge'] = set(face_counts)
    results['plateau_edges'] = (set(face_counts) == {3})

    # Check dual orthogonality (using exact shift for hard test)
    face_to_cell_shift = data.get('face_to_cell_shift', {})
    dots = []
    for f_idx in range(len(F)):
        face = F[f_idx]
        ca, cb = data['face_to_cells'][f_idx]

        # Face normal
        pos = [V[face[0]].copy()]
        for k in range(1, len(face)):
            d = V[face[k]] - V[face[k-1]]
            d = wrap_delta(d, L_vec)
            pos.append(pos[-1] + d)
        pos = np.array(pos)
        ctr = np.mean(pos, axis=0)
        cross = sum(np.cross(pos[k]-ctr, pos[(k+1)%len(pos)]-ctr)
                    for k in range(len(pos)))
        n_hat = cross / (np.linalg.norm(cross) + 1e-12)

        # Dual edge direction - use exact shift if available
        shift = face_to_cell_shift.get(f_idx, None)
        if shift is not None:
            # Exact: cell_b is at centers[cb] + shift * L_vec relative to cell_a
            delta = data['cell_centers'][cb] + shift * L_vec - data['cell_centers'][ca]
        else:
            # Fallback: wrap to nearest image
            delta = data['cell_centers'][cb] - data['cell_centers'][ca]
            delta = wrap_delta(delta, L_vec)
        d_hat = delta / (np.linalg.norm(delta) + 1e-12)

        dots.append(abs(np.dot(n_hat, d_hat)))

    results['dual_orthogonality_min'] = min(dots)
    results['dual_orthogonality_mean'] = np.mean(dots)
    results['orthogonality_ok'] = (min(dots) > 0.99)

    results['all_ok'] = (results['plateau_vertices'] and
                         results['plateau_edges'] and
                         results['orthogonality_ok'])

    # A3: Verify each edge's 3 faces have distinct cell pairs
    edge_cell_pair_ok = True
    edge_cell_pair_issues = []
    for e in E:
        face_indices = edge_to_faces.get(e, set())
        if len(face_indices) != 3:
            continue  # already flagged by plateau_edges check
        cell_pairs = set()
        for f_idx in face_indices:
            ca, cb = data['face_to_cells'][f_idx]
            pair = (min(ca, cb), max(ca, cb))
            cell_pairs.add(pair)
        if len(cell_pairs) != 3:
            edge_cell_pair_ok = False
            edge_cell_pair_issues.append((e, len(cell_pairs)))

    results['edge_face_cell_distinct'] = edge_cell_pair_ok
    results['edge_cell_pair_issues'] = edge_cell_pair_issues
    results['all_ok'] = results['all_ok'] and edge_cell_pair_ok

    return results


def verify_voronoi_property(data: Dict, tol: float = 1e-8) -> Dict:
    """
    Verify bisector property: face vertices lie on bisector plane between adjacent sites.

    CONTEXT: We build a Plateau complex derived from Voronoi ridges (vor.ridge_vertices).
    Faces inherit from ridges, and face_to_cells maps each face to the two sites
    whose ridge it came from.

    This test verifies that the face_to_cells mapping is CORRECT by checking that
    face vertices are equidistant from both mapped sites (bisector property).

    This is a TRULY INDEPENDENT test (not self-confirming):
    - Uses raw distances in site_a's frame (no wrap_delta masking)
    - Catches shift errors that wrap would hide

    For each face vertex x and adjacent cells (site_a, site_b with shift):
        |dist(x, site_a) - dist(x, site_b_shifted)| < tol

    Args:
        data: Dict from build_foam_with_dual_info or convenience builders
        tol: tolerance for equidistance check

    Returns:
        dict with:
            'max_asymmetry': max |d_a - d_b| across all vertices
            'mean_asymmetry': mean asymmetry
            'voronoi_ok': bool, True if max_asymmetry < tol
            'locality_ok': bool, True if max abs component < L/2
            'max_raw_component': max |dx|, |dy|, or |dz|
    """
    V = data['V']
    F = data['F']
    L_vec = data['L_vec']
    L_half = L_vec / 2
    cell_centers = data['cell_centers']
    face_to_cells = data['face_to_cells']
    face_to_cell_shift = data.get('face_to_cell_shift', {})

    asymmetries = []
    raw_max_components = []  # track max abs component (not norm) for locality
    boundary_deviations = []  # track boundary cases (diagnostic, not failure)

    for f_idx, face in enumerate(F):
        ca, cb = face_to_cells[f_idx]
        site_a = cell_centers[ca]
        site_b = cell_centers[cb]

        # Use exact shift if available (makes test truly independent)
        shift = face_to_cell_shift.get(f_idx, None)

        # Unwrap face vertices
        pos = [V[face[0]].copy()]
        for k in range(1, len(face)):
            d = V[face[k]] - V[face[k-1]]
            d = wrap_delta(d, L_vec)
            pos.append(pos[-1] + d)

        for p in pos:
            if shift is not None:
                # TRULY INDEPENDENT TEST: no wrap_delta masking shift errors
                # 1. Find integer shift n_a to bring p into site_a's image
                #    Use floor(x+0.5) instead of round() for stable tie-breaking at .5
                delta_raw = p - site_a
                n_a = np.floor(delta_raw / L_vec + 0.5).astype(int)
                p_shifted = p - n_a * L_vec  # p in site_a's image

                # Boundary check: p_shifted should be within L/2 of site_a
                # This is a DIAGNOSTIC, not a hard failure (vertex can be at L/2 boundary)
                max_dev_a = np.max(np.abs(p_shifted - site_a) - L_half)
                if max_dev_a > 1e-6:  # relaxed tolerance for boundary cases
                    boundary_deviations.append(max_dev_a)

                # 2. Compute raw distances (no wrap!)
                dist_a = np.linalg.norm(p_shifted - site_a)
                # site_b is at site_b + shift*L relative to site_a
                site_b_in_a_frame = site_b + shift * L_vec
                dist_b = np.linalg.norm(p_shifted - site_b_in_a_frame)

                # Track locality
                raw_max_components.append(np.max(np.abs(p_shifted - site_a)))
                raw_max_components.append(np.max(np.abs(p_shifted - site_b_in_a_frame)))
            else:
                # Fallback: use wrap_delta (may mask errors)
                delta_a_raw = p - site_a
                raw_max_components.append(np.max(np.abs(delta_a_raw)))
                delta_a = wrap_delta(delta_a_raw, L_vec)
                dist_a = np.linalg.norm(delta_a)

                delta_b_raw = p - site_b
                raw_max_components.append(np.max(np.abs(delta_b_raw)))
                delta_b = wrap_delta(delta_b_raw, L_vec)
                dist_b = np.linalg.norm(delta_b)

            asymmetries.append(abs(dist_a - dist_b))

    max_asym = max(asymmetries)
    mean_asym = np.mean(asymmetries)
    max_raw_comp = max(raw_max_components)
    # Locality: max abs component should be < L/2 (component-wise, not norm)
    locality_ok = max_raw_comp < np.min(L_half)
    # Boundary diagnostics
    n_boundary_cases = len(boundary_deviations)
    max_boundary_dev = max(boundary_deviations) if boundary_deviations else 0.0

    return {
        'max_asymmetry': max_asym,
        'mean_asymmetry': mean_asym,
        'voronoi_ok': max_asym < tol,
        'locality_ok': locality_ok,  # informational, not a fail criterion
        'max_raw_component': max_raw_comp,  # max |dx|, |dy|, or |dz| before wrap
        'n_boundary_cases': n_boundary_cases,  # diagnostic: vertices near L/2 boundary
        'max_boundary_dev': max_boundary_dev,  # diagnostic: max deviation beyond L/2
    }
