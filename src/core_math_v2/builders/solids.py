"""
Solid Lattice Cell Construction
===============================

Build Voronoi cells for common crystal lattices.

STRUCTURES:
    - SC (Simple Cubic): Cube cell (V=8, E=12, F=6)
    - FCC (Face-Centered Cubic): Rhombic dodecahedron (V=14, E=24, F=12)

TOPOLOGY in periodic lattice (cells per edge = k):
    - SC cube: k=4 (4 cells share each edge)
    - FCC rhombic: k=3 (3 cells share each edge)
    - BCC Kelvin foam: k=3 (3 faces meet at each edge - Plateau border)

NOTE: FCC and Kelvin both have k=3, but different physics:
    - FCC: crystal tiling (cells per edge)
    - Kelvin: soap film (faces per edge at Plateau border)
    The distinction is physical (packing vs surface tension), not topological.

USE CASE:
    Test if κ-locking discriminates foam from solid (it does via geometry, not k).

Date: Jan 2026
"""

import numpy as np
from typing import Tuple, List, Dict

from ..spec.constants import EPS_CLOSE


def build_sc_cell() -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], Dict[tuple, int]]:
    """
    Build Simple Cubic (SC) Voronoi cell = cube.

    TOPOLOGY:
        V = 8 vertices (cube corners)
        E = 12 edges
        F = 6 faces (squares)
        χ = V - E + F = 8 - 12 + 6 = 2

    GEOMETRY:
        Vertices at (±1, ±1, ±1)
        Edge length = 2

    LATTICE PROPERTY:
        In periodic SC lattice, each edge is shared by 4 cubes
        → k=4 cells per edge in periodic lattice

    Returns:
        vertices: (8, 3) array
        edges: list of 12 edge tuples (i, j) with i < j
        faces: list of 6 faces, CCW from outside
        v_to_idx: dict mapping (x,y,z) tuple to vertex index
    """
    # 8 corners of unit cube at (±1, ±1, ±1)
    vertices = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vertices.append((x, y, z))

    vertices = sorted(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    vertices_arr = np.array(vertices, dtype=float)

    # Edges: connect vertices at distance 2
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            d2 = np.sum((vertices_arr[i] - vertices_arr[j])**2)
            if abs(d2 - 4.0) < EPS_CLOSE:  # edge length = 2
                edges.append((i, j))
    edges = sorted(edges)

    # 6 faces (squares at ±1 on each axis)
    faces = []
    for axis in range(3):
        for sign in [-1, 1]:
            face_idx = [i for i, v in enumerate(vertices) if v[axis] == sign]
            # Order vertices CCW when viewed from outside
            coords = vertices_arr[face_idx]
            centroid = coords.mean(axis=0)
            normal = np.zeros(3)
            normal[axis] = sign

            # Build local frame
            if abs(normal[0]) < 0.9:
                u = np.cross(normal, [1, 0, 0])
            else:
                u = np.cross(normal, [0, 1, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)

            angles = [np.arctan2(np.dot(coords[k] - centroid, v),
                                 np.dot(coords[k] - centroid, u))
                      for k in range(len(face_idx))]
            order = np.argsort(angles)
            faces.append([face_idx[o] for o in order])

    if len(vertices) != 8:
        raise ValueError(f"Expected 8 vertices, got {len(vertices)}")
    if len(edges) != 12:
        raise ValueError(f"Expected 12 edges, got {len(edges)}")
    if len(faces) != 6:
        raise ValueError(f"Expected 6 faces, got {len(faces)}")

    return vertices_arr, edges, faces, v_to_idx


def build_fcc_cell() -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], Dict[tuple, int]]:
    """
    Build FCC Voronoi cell = rhombic dodecahedron.

    TOPOLOGY:
        V = 14 vertices (6 axial + 8 cubic)
        E = 24 edges
        F = 12 faces (rhombi)
        χ = V - E + F = 14 - 24 + 12 = 2

    GEOMETRY:
        6 axial vertices at (±2, 0, 0), (0, ±2, 0), (0, 0, ±2)
        8 cubic vertices at (±1, ±1, ±1)
        Edge length = √3

    LATTICE PROPERTY:
        In periodic FCC lattice, each edge is shared by 3 cells
        → k=3 cells per edge in periodic lattice (same k as foam)

    NOTE: Same k as foam but different physics (crystal packing vs soap film).

    Returns:
        vertices: (14, 3) array
        edges: list of 24 edge tuples (i, j) with i < j
        faces: list of 12 faces, CCW from outside
        v_to_idx: dict mapping (x,y,z) tuple to vertex index
    """
    # 6 axial vertices on coordinate axes
    vertices = [
        (2, 0, 0), (-2, 0, 0),
        (0, 2, 0), (0, -2, 0),
        (0, 0, 2), (0, 0, -2),
    ]
    # 8 cubic vertices at (±1, ±1, ±1)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                vertices.append((sx, sy, sz))

    vertices = sorted(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    vertices_arr = np.array(vertices, dtype=float)

    # Edges: axial to cubic at distance √3
    # |[2,0,0] - [1,1,1]| = √((2-1)² + 1² + 1²) = √3
    edges = []
    for i in range(14):
        for j in range(i + 1, 14):
            d2 = np.sum((vertices_arr[i] - vertices_arr[j])**2)
            if abs(d2 - 3.0) < EPS_CLOSE:  # edge length = √3
                edges.append((i, j))
    edges = sorted(edges)

    # Build faces: 12 rhombic faces
    # Each rhombus has 2 axial vertices + 2 cubic vertices
    faces = []

    # Find axial vertex indices (those with |v| = 2)
    axial_idx = [i for i, v in enumerate(vertices) if abs(np.linalg.norm(v) - 2) < 0.1]

    # Pairs of adjacent axial vertices (form edges of octahedron)
    axial_pairs = []
    for i in range(len(axial_idx)):
        for j in range(i + 1, len(axial_idx)):
            vi = np.array(vertices[axial_idx[i]])
            vj = np.array(vertices[axial_idx[j]])
            # Adjacent if perpendicular (dot = 0)
            if abs(np.dot(vi, vj)) < 0.1:
                axial_pairs.append((axial_idx[i], axial_idx[j]))

    for (a1, a2) in axial_pairs:
        v1, v2 = vertices_arr[a1], vertices_arr[a2]
        # Find cubic vertices adjacent to both
        cubic_adj = []
        for c in range(14):
            if c in axial_idx:
                continue
            d1 = np.sum((vertices_arr[c] - v1)**2)
            d2_val = np.sum((vertices_arr[c] - v2)**2)
            if abs(d1 - 3.0) < EPS_CLOSE and abs(d2_val - 3.0) < EPS_CLOSE:
                cubic_adj.append(c)
        if len(cubic_adj) == 2:
            c1, c2 = cubic_adj
            face = [a1, c1, a2, c2]
            faces.append(face)

    # Order faces CCW when viewed from outside
    def order_face_ccw(f_idx, verts):
        coords = verts[f_idx]
        centroid = coords.mean(axis=0)
        normal = centroid / np.linalg.norm(centroid)  # outward from origin

        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1, 0, 0])
        else:
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        angles = []
        for p in coords:
            dp = p - centroid
            angles.append(np.arctan2(np.dot(dp, v), np.dot(dp, u)))

        order = np.argsort(angles)
        return [f_idx[o] for o in order]

    faces = [order_face_ccw(f, vertices_arr) for f in faces]

    if len(vertices) != 14:
        raise ValueError(f"Expected 14 vertices, got {len(vertices)}")
    if len(edges) != 24:
        raise ValueError(f"Expected 24 edges, got {len(edges)}")
    if len(faces) != 12:
        raise ValueError(f"Expected 12 faces, got {len(faces)}")

    return vertices_arr, edges, faces, v_to_idx


def verify_cell_topology(vertices, edges, faces, name="Cell"):
    """
    Verify basic topological properties of a cell.

    Returns dict with V, E, F, χ, and validity checks.
    """
    V = len(vertices)
    E = len(edges)
    F = len(faces)
    chi = V - E + F

    # Check Euler characteristic
    chi_ok = (chi == 2)

    edge_set = set(edges)

    # Check edge consistency: each edge should appear in exactly 2 faces
    # Also check that every face boundary segment is a valid edge
    edge_face_count = {e: 0 for e in edges}
    bad_face_edges = []

    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            e = (min(v1, v2), max(v1, v2))
            if e not in edge_set:
                bad_face_edges.append((fi, e))
            else:
                edge_face_count[e] += 1

    # For single cell (2-manifold), each edge should have exactly 2 faces
    edges_ok = all(c == 2 for c in edge_face_count.values())
    face_edges_ok = (len(bad_face_edges) == 0)

    return {
        'name': name,
        'V': V,
        'E': E,
        'F': F,
        'chi': chi,
        'chi_ok': chi_ok,
        'edges_ok': edges_ok,
        'face_edges_ok': face_edges_ok,
        'bad_face_edges': bad_face_edges,
        'valid': chi_ok and edges_ok and face_edges_ok
    }


def compute_dihedral_angles(vertices, edges, faces):
    """
    Compute dihedral angle for each edge (angle between adjacent faces).

    For tiling analysis:
        - 360° / dihedral = number of cells meeting at edge
        - SC cube: 90° → 4 cells/edge (k=4)
        - FCC rhombic: 120° → 3 cells/edge (k=3, same as foam but crystal tiling)

    Returns:
        dict with edge -> angle in degrees
    """
    # Build edge -> faces map
    edge_faces = {e: [] for e in edges}
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            e = (min(v1, v2), max(v1, v2))
            if e in edge_faces:
                edge_faces[e].append(fi)

    # Compute face normals (outward from origin)
    face_normals = []
    for face in faces:
        pts = vertices[face]
        centroid = pts.mean(axis=0)
        # Use cross product of first two edges
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        # Make sure it points outward (away from origin)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        face_normals.append(normal)

    # Compute dihedral angles
    dihedrals = {}
    for e, face_list in edge_faces.items():
        if len(face_list) == 2:
            n1 = face_normals[face_list[0]]
            n2 = face_normals[face_list[1]]
            # Dihedral angle = π - angle between normals
            cos_angle = np.clip(np.dot(n1, n2), -1, 1)
            angle_between = np.arccos(cos_angle)
            dihedral = np.pi - angle_between  # interior angle
            dihedrals[e] = np.degrees(dihedral)

    return dihedrals


def verify_geometry(vertices, edges, faces, name="Cell"):
    """
    Verify geometric properties:
    - Edge lengths (should be uniform)
    - Dihedral angles
    - Cells per edge in tiling (from dihedral)
    """
    # Edge lengths
    edge_lengths = []
    for i, j in edges:
        length = np.linalg.norm(vertices[j] - vertices[i])
        edge_lengths.append(length)

    edge_lengths = np.array(edge_lengths)
    uniform = np.allclose(edge_lengths, edge_lengths[0], rtol=EPS_CLOSE)

    # Dihedral angles
    dihedrals = compute_dihedral_angles(vertices, edges, faces)
    dihedral_values = np.array(list(dihedrals.values()))
    dihedral_uniform = np.allclose(dihedral_values, dihedral_values[0], rtol=EPS_CLOSE)

    # Cells per edge in tiling (GEOMETRIC estimate from dihedral, not topological invariant)
    # This tells us how many cells WOULD meet at an edge if this polyhedron tiles space.
    # For actual topological k, use Tr(d₁ᵀd₁)/E from operators.
    avg_dihedral = np.mean(dihedral_values)
    cells_per_edge = 360.0 / avg_dihedral

    return {
        'name': name,
        'edge_length': edge_lengths[0],
        'edge_length_uniform': uniform,
        'dihedral_deg': avg_dihedral,
        'dihedral_uniform': dihedral_uniform,
        'cells_per_edge': cells_per_edge,  # geometric estimate, not topological k
        'k3_like': abs(cells_per_edge - 3) < 0.1,  # informational: would tile with k≈3
    }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SOLID LATTICE CELLS - FULL VERIFICATION")
    print("=" * 60)

    for name, builder in [("SC Cube", build_sc_cell),
                          ("FCC Rhombic Dodecahedron", build_fcc_cell)]:
        v, e, f, idx = builder()

        # Topology check
        topo = verify_cell_topology(v, e, f, name)
        print(f"\n{name}:")
        print(f"  TOPOLOGY:")
        print(f"    V = {topo['V']}, E = {topo['E']}, F = {topo['F']}")
        print(f"    χ = {topo['chi']} {'✓' if topo['chi_ok'] else '✗'}")
        print(f"    Edges in 2 faces: {'✓' if topo['edges_ok'] else '✗'}")
        print(f"    Face edges valid: {'✓' if topo['face_edges_ok'] else '✗'}")
        if topo['bad_face_edges']:
            print(f"    ⚠ Bad edges: {topo['bad_face_edges']}")

        # Geometry check
        geom = verify_geometry(v, e, f, name)
        print(f"  GEOMETRY:")
        print(f"    Edge length = {geom['edge_length']:.4f} {'(uniform)' if geom['edge_length_uniform'] else '(NOT uniform!)'}")
        print(f"    Dihedral = {geom['dihedral_deg']:.2f}° {'(uniform)' if geom['dihedral_uniform'] else '(NOT uniform!)'}")
        print(f"    Cells/edge in tiling = {geom['cells_per_edge']:.1f}")
        print(f"    k=3 (3 cells/edge): {'YES ✓' if geom['k3_like'] else 'NO ✗'}")

    print("\n" + "=" * 60)
    print("COMPARISON: k VALUE BY STRUCTURE")
    print("=" * 60)
    print()
    print("| Structure   | V  | E  | F  | Dihedral | Cells/edge | k=3? |")
    print("|-------------|----|----|----|---------:|:----------:|:----:|")
    print("| SC Cube     |  8 | 12 |  6 |      90° |     4      |  NO  |")
    print("| FCC Rhombic | 14 | 24 | 12 |     120° |     3      | YES  |")
    print("| BCC Kelvin  | 24 | 36 | 14 |     120° |     3      | YES  |")
    print()
    print("NOTE: FCC and Kelvin both have k=3, but different physics:")
    print("  - BCC Kelvin: foam (Plateau borders, surface tension)")
    print("  - FCC: crystal tiling (no surface tension physics)")
