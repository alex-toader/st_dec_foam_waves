"""
Kelvin Cell Geometry
====================

Pure geometric construction of the truncated octahedron (Kelvin cell).
NO physics assumptions. NO hardcoded values.

CONSTRUCTION:
    The Kelvin cell is the Voronoi cell of the BCC lattice.
    Vertices are permutations of (0, ±1, ±2).

    Edges connect vertices at distance √2.
    Faces are 6 squares (coord = ±2) and 8 hexagons (|x|+|y|+|z| = 3).

OUTPUTS (all derived from construction):
    V = 24 vertices
    E = 36 edges
    F = 14 faces (6 squares + 8 hexagons)
    χ = V - E + F = 2 (Euler characteristic of closed surface)

REFERENCE: Weaire & Phelan (1994), Kelvin (1887)
"""

import numpy as np
from itertools import permutations
from typing import Tuple, List, Dict

from ..spec.structures import create_mesh
from ..spec.constants import COMPLEX_SURFACE, EPS_CLOSE


def build_vertices() -> Tuple[np.ndarray, Dict[tuple, int]]:
    """
    Build Kelvin cell vertices.

    CONSTRUCTION:
        Vertices are all permutations of (0, ±1, ±2).
        This gives 3! × 2² = 24 vertices.

    Returns:
        vertices: (24, 3) array of integer coordinates
        v_to_idx: dict mapping coordinate tuple to index

    Example:
        >>> v, idx = build_vertices()
        >>> len(v)
        24
    """
    vertices = []

    # Generate all permutations of (0, ±1, ±2)
    for perm in permutations([0, 1, 2]):
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                v = [0, 0, 0]
                for i, val in enumerate(perm):
                    if val == 0:
                        v[i] = 0
                    elif val == 1:
                        v[i] = s1
                    else:  # val == 2
                        v[i] = s2 * 2
                vertices.append(tuple(v))

    # Remove duplicates and sort for reproducibility
    vertices = sorted(set(vertices))
    v_to_idx = {v: i for i, v in enumerate(vertices)}

    return np.array(vertices, dtype=float), v_to_idx


def build_edges(vertices: np.ndarray) -> List[Tuple[int, int]]:
    """
    Build Kelvin cell edges.

    CONSTRUCTION:
        Two vertices are connected if their Euclidean distance is √2.
        Each vertex has exactly 3 neighbors → 24 × 3 / 2 = 36 edges.

    Args:
        vertices: (V, 3) array from build_vertices()

    Returns:
        edges: list of 36 tuples (i, j) with i < j

    VERIFICATION:
        E = 36 (always, from geometry)
    """
    edges = []
    n_v = len(vertices)

    for i in range(n_v):
        for j in range(i + 1, n_v):
            # Distance squared
            d2 = np.sum((vertices[i] - vertices[j])**2)
            # Connect if distance = √2 (d² = 2)
            if abs(d2 - 2.0) < EPS_CLOSE:
                edges.append((i, j))

    return sorted(edges)


def _order_face_vertices(vertices: np.ndarray,
                         face_idx: List[int],
                         normal: np.ndarray) -> List[int]:
    """
    Order face vertices counter-clockwise when viewed from normal direction.

    Internal helper function.
    """
    coords = vertices[face_idx]
    centroid = coords.mean(axis=0)
    normal = normal / np.linalg.norm(normal)

    # Build local coordinate frame
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Compute angles and sort
    angles = []
    for p in coords:
        dp = p - centroid
        angles.append(np.arctan2(np.dot(dp, v), np.dot(dp, u)))

    order = np.argsort(angles)
    return [face_idx[o] for o in order]


def build_faces(vertices: np.ndarray) -> List[List[int]]:
    """
    Build Kelvin cell faces with proper vertex ordering.

    CONSTRUCTION:
        6 squares: vertices with coordinate = ±2 on one axis
        8 hexagons: vertices with sx*x + sy*y + sz*z = 3 for sx,sy,sz ∈ {±1}

    Args:
        vertices: (V, 3) array from build_vertices()

    Returns:
        faces: list of 14 faces, each face is list of vertex indices in CCW order

    VERIFICATION:
        F = 14 (6 squares + 8 hexagons)
        Each square has 4 vertices
        Each hexagon has 6 vertices
    """
    faces = []

    # 6 squares (at coordinates ±2)
    for axis in range(3):
        for sign in [-2, 2]:
            face_idx = [i for i, v in enumerate(vertices)
                       if abs(v[axis] - sign) < EPS_CLOSE]
            normal = np.zeros(3)
            normal[axis] = sign
            ordered = _order_face_vertices(vertices, face_idx, normal)
            faces.append(ordered)

    # 8 hexagons (at planes sx*x + sy*y + sz*z = 3)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                face_idx = [i for i, v in enumerate(vertices)
                           if abs(sx*v[0] + sy*v[1] + sz*v[2] - 3) < EPS_CLOSE]
                normal = np.array([sx, sy, sz], dtype=float)
                ordered = _order_face_vertices(vertices, face_idx, normal)
                faces.append(ordered)

    return faces


def build_kelvin_cell(strict: bool = True) -> Tuple[np.ndarray, List[Tuple[int, int]],
                                  List[List[int]], Dict[tuple, int]]:
    """
    Build complete Kelvin cell geometry.

    This is the main entry point for cell construction.

    Args:
        strict: If True (default), verify V=24, E=36, F=14, χ=2.
                If False, skip verification (for exploration/audit).

    Returns:
        vertices: (24, 3) array of vertex coordinates
        edges: list of 36 edge tuples (i, j) with i < j
        faces: list of 14 face vertex lists in CCW order
        v_to_idx: dict mapping vertex coordinate tuple to index

    INVARIANTS (verified by construction when strict=True):
        V = 24, E = 36, F = 14
        χ = V - E + F = 2

    Example:
        >>> v, e, f, idx = build_kelvin_cell()
        >>> len(v), len(e), len(f)
        (24, 36, 14)
        >>> len(v) - len(e) + len(f)  # Euler characteristic
        2
    """
    vertices, v_to_idx = build_vertices()
    edges = build_edges(vertices)
    faces = build_faces(vertices)

    # Verify construction (optional)
    if strict:
        V, E, F = len(vertices), len(edges), len(faces)
        if V != 24:
            raise ValueError(f"Expected V=24, got {V}")
        if E != 36:
            raise ValueError(f"Expected E=36, got {E}")
        if F != 14:
            raise ValueError(f"Expected F=14, got {F}")
        if V - E + F != 2:
            raise ValueError(f"Euler characteristic should be 2, got {V - E + F}")

    return vertices, edges, faces, v_to_idx


def get_topology_numbers() -> Dict[str, int]:
    """
    Get the topological numbers of Kelvin cell.

    These are COMPUTED from construction, not hardcoded.

    Returns:
        dict with keys: V, E, F, chi

    FORMULAS:
        V = 24 (permutations of (0,±1,±2))
        E = 36 (vertices at distance √2)
        F = 14 (6 squares + 8 hexagons)
        χ = V - E + F = 2 (Euler characteristic)
    """
    vertices, edges, faces, _ = build_kelvin_cell()

    V = len(vertices)
    E = len(edges)
    F = len(faces)
    chi = V - E + F

    return {'V': V, 'E': E, 'F': F, 'chi': chi}


# =============================================================================
# CONTRACT-COMPLIANT WRAPPERS
# =============================================================================

def build_kelvin_cell_mesh(name: str = "kelvin_cell") -> dict:
    """
    Build single Kelvin cell as contract-compliant mesh dict.

    Single cell = SURFACE (closed 2-manifold, 2 faces/edge).
    """
    V, E, F, _ = build_kelvin_cell(strict=True)
    return create_mesh(V, E, F, COMPLEX_SURFACE, name=name, n_cells=1)


def build_kelvin_foam(n_cells: int = 1, name: str = None) -> dict:
    """
    DEPRECATED: Use build_bcc_foam_periodic() from multicell_periodic.py instead.

    A single Kelvin cell has 2 faces/edge (surface), NOT 3 (foam).
    Real Kelvin foam with 3 faces/edge requires the periodic multicell construction.

    This function is kept for backwards compatibility but raises an error.
    """
    raise NotImplementedError(
        "build_kelvin_foam is DEPRECATED. "
        "A single Kelvin cell has 2 faces/edge (surface), not 3 (foam). "
        "Use build_bcc_foam_periodic(N) from multicell_periodic.py for real foam."
    )


# Self-test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("KELVIN CELL GEOMETRY - VERIFICATION")
    print("=" * 60)

    topo = get_topology_numbers()
    print(f"\nTopology (computed from construction):")
    print(f"  V = {topo['V']} vertices")
    print(f"  E = {topo['E']} edges")
    print(f"  F = {topo['F']} faces")
    print(f"  χ = V - E + F = {topo['chi']}")

    vertices, edges, faces, _ = build_kelvin_cell()

    # Count face types
    n_squares = sum(1 for f in faces if len(f) == 4)
    n_hexagons = sum(1 for f in faces if len(f) == 6)
    print(f"\nFace types:")
    print(f"  {n_squares} squares (4 vertices each)")
    print(f"  {n_hexagons} hexagons (6 vertices each)")

    # Verify edge count from vertex degree
    # Each vertex has degree 3, so sum of degrees = 2E
    degree_sum = 0
    for i in range(len(vertices)):
        degree = sum(1 for e in edges if i in e)
        degree_sum += degree
    print(f"\nEdge verification:")
    print(f"  Sum of vertex degrees = {degree_sum}")
    print(f"  Expected 2E = {2 * len(edges)}")
    print(f"  Match: {degree_sum == 2 * len(edges)}")

    print("\n" + "=" * 60)
    print("All verifications passed.")
    print("=" * 60)
